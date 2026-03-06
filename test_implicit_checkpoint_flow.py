"""
Test flow for implicit checkpointing (@checkpoint with exclude=[...] / serialization_config).

Scenarios covered
-----------------
1. @checkpoint             : all public attrs serialized automatically,
                             crash-and-resume via @retry
2. exclude=[...]           : listed attrs skipped; all others checkpointed
3. serialization_config    : "raw" format for bytes payloads

Run (local datastore, no cloud needed):
    python test_implicit_checkpoint_flow.py run
"""

from metaflow import FlowSpec, step, current, checkpoint, retry


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fake_weights(seed=42):
    """Return a deterministic 128-byte blob."""
    return bytes([(seed + i) % 256 for i in range(128)])


def _train_one_epoch(weights, epoch):
    """Mutate one byte to simulate a training step."""
    w = bytearray(weights)
    w[epoch % len(w)] = (w[epoch % len(w)] + 1) % 256
    return bytes(w)


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------

class ImplicitCheckpointTestFlow(FlowSpec):
    """
    Sequential test flow:

        start
          → train_all_fields       (scenario 1: all attrs + crash/resume)
          → train_filtered_fields  (scenario 2: exclude=[...])
          → train_raw_bytes        (scenario 3: serialization_config raw)
          → end
    """

    # ------------------------------------------------------------------
    # start
    # ------------------------------------------------------------------
    @step
    def start(self):
        print("=== ImplicitCheckpointTestFlow starting ===")
        self.next(self.train_all_fields)

    # ------------------------------------------------------------------
    # Scenario 1 – all attrs, crash-and-resume
    # ------------------------------------------------------------------
    @retry(times=1)
    @checkpoint(load_policy="fresh")
    @step
    def train_all_fields(self):
        """
        Attempt 0:
          - self.epoch=0, self.loss=1.0 initialised fresh
          - Runs epochs 0-2, saving after each
          - Deliberately raises RuntimeError after epoch 2

        Attempt 1 (retry):
          - is_loaded=True  → current.checkpoint.load() restores state
          - self.epoch==2, self.loss≈0.729 (0.9**3)
          - Loop continues from epoch 2 and finishes at epoch 4

        Assertions:
          self.epoch == 4
          self.loss < 1.0
          self.completed_scenario_1 == True
        """
        if current.checkpoint.is_loaded:
            print("[attempt %d] Restoring from checkpoint" % current.retry_count)
            current.checkpoint.load()
            print(
                "[attempt %d] Restored: epoch=%d  loss=%.4f"
                % (current.retry_count, self.epoch, self.loss)
            )
            assert self.epoch == 2, "Expected epoch=2 on resume, got %d" % self.epoch
        else:
            print("[attempt 0] Fresh start")
            self.epoch = 0
            self.loss = 1.0

        for epoch in range(self.epoch, 5):
            self.loss = round(self.loss * 0.9, 6)
            self.epoch = epoch
            print("  epoch=%d  loss=%.6f" % (self.epoch, self.loss))
            current.checkpoint.save()

            if current.retry_count == 0 and epoch == 2:
                raise RuntimeError("[intentional crash at epoch 2]")

        assert self.epoch == 4, "Expected epoch=4 at end, got %d" % self.epoch
        assert self.loss < 1.0
        self.completed_scenario_1 = True
        print("--- scenario 1 PASSED ---")
        self.next(self.train_filtered_fields)

    # ------------------------------------------------------------------
    # Scenario 2 – exclude filter
    # ------------------------------------------------------------------
    @checkpoint(exclude=["scratch"], load_policy="none")
    @step
    def train_filtered_fields(self):
        """
        'scratch' is excluded; 'epoch' and 'loss' are checkpointed.

        Verifies:
          - save() returns a checkpoint artifact dict
          - checkpoint manifest includes 'epoch' and 'loss'
          - checkpoint manifest does NOT include 'scratch'
        """
        self.epoch = 0
        self.loss = 1.0
        self.scratch = 42        # excluded → must not appear in checkpoint

        for epoch in range(3):
            self.loss = round(self.loss * 0.85, 6)
            self.epoch = epoch
            ref = current.checkpoint.save()

        # verify manifest fields
        fields_info = ref.get("metadata", {}).get("_implicit_manifest", {}).get("fields", {})
        assert "epoch" in fields_info, "Missing 'epoch' in manifest"
        assert "loss" in fields_info, "Missing 'loss' in manifest"
        assert "scratch" not in fields_info, "'scratch' must not be checkpointed"

        self.completed_scenario_2 = True
        print("--- scenario 2 PASSED ---")
        self.next(self.train_raw_bytes)

    # ------------------------------------------------------------------
    # Scenario 3 – raw bytes serialization
    # ------------------------------------------------------------------
    @checkpoint(
        serialization_config={"weights": "raw"},
        load_policy="none",
    )
    @step
    def train_raw_bytes(self):
        """
        'weights' is a bytes object saved with format='raw'.
        'epoch' is an int saved with the default format='pickle'.

        Verifies that the manifest metadata records the correct formats.
        """
        self.weights = _fake_weights(seed=7)
        self.epoch = 0

        for epoch in range(4):
            self.weights = _train_one_epoch(self.weights, epoch)
            self.epoch = epoch
            ref = current.checkpoint.save()

        assert ref is not None
        implicit_manifest = ref.get("metadata", {}).get("_implicit_manifest", {})
        fields_info = implicit_manifest.get("fields", {})

        assert "weights" in fields_info, "Missing 'weights' in manifest"
        assert "epoch" in fields_info, "Missing 'epoch' in manifest"
        assert fields_info["weights"]["format"] == "raw", (
            "Expected raw format for weights, got %r" % fields_info["weights"]["format"]
        )
        assert fields_info["epoch"]["format"] == "pickle", (
            "Expected pickle format for epoch, got %r" % fields_info["epoch"]["format"]
        )

        self.completed_scenario_3 = True
        print("--- scenario 3 PASSED ---")
        self.next(self.end)

    # ------------------------------------------------------------------
    # end
    # ------------------------------------------------------------------
    @step
    def end(self):
        assert self.completed_scenario_1
        assert self.completed_scenario_2
        assert self.completed_scenario_3
        print("=== All implicit checkpoint scenarios PASSED ===")


if __name__ == "__main__":
    ImplicitCheckpointTestFlow()
