"""
Test flow for implicit checkpointing (@checkpoint with exclude=[...] / serialization_config).

Scenarios covered
-----------------
1. @checkpoint             : all public attrs serialized automatically,
                             crash-and-resume via @retry
2. exclude=[...]           : listed attrs skipped; all others checkpointed
3. serialization_config    : "raw" format for bytes payloads
4. explicit load           : load(reference=ref) restores a specific checkpoint
5. list() name filter      : list(name=...) returns only matching checkpoints
6. user metadata           : metadata= in save() round-trips through the artifact dict
7. complex Python types    : list/dict/str/float/int survive pickle via crash+resume

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
          → explicit_load          (scenario 4: load(reference=ref))
          → list_name_filter       (scenario 5: list(name=...) filtering)
          → user_metadata          (scenario 6: metadata= round-trip)
          → complex_types          (scenario 7: complex types via crash+resume)
          → auto_load_resume       (scenario 8: auto_load=True)
          → inspect_checkpoint     (scenario 9: inspect() without download)
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
        fields_info = (ref.implicit_manifest or {}).get("fields", {})
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
        implicit_manifest = ref.implicit_manifest or {}
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
        self.next(self.explicit_load)

    # ------------------------------------------------------------------
    # Scenario 4 – explicit reference load
    # ------------------------------------------------------------------
    @checkpoint(load_policy="none")
    @step
    def explicit_load(self):
        """
        Saves a checkpoint, clobbers two fields in memory, then restores them
        via current.checkpoint.load(reference=ref).

        Verifies that an explicit reference load correctly deserializes only
        the checkpoint's recorded fields back onto the flow.
        """
        self.val = 99
        self.tag = "before"
        ref = current.checkpoint.save()

        # clobber in-memory state
        self.val = 0
        self.tag = "after"
        assert self.val == 0

        # restore from the explicit reference
        current.checkpoint.load(reference=ref)
        assert self.val == 99, "Expected val=99 after load, got %d" % self.val
        assert self.tag == "before", "Expected tag='before' after load, got %r" % self.tag

        self.completed_scenario_4 = True
        print("--- scenario 4 PASSED ---")
        self.next(self.list_name_filter)

    # ------------------------------------------------------------------
    # Scenario 5 – list() name filter
    # ------------------------------------------------------------------
    @checkpoint(load_policy="none")
    @step
    def list_name_filter(self):
        """
        Saves 3 checkpoints: 2 named 'best', 1 named 'latest'.

        Verifies:
          - list(name='best')   returns exactly 2 entries, all with name='best'
          - list(name='latest') returns exactly 1 entry
          - list()              returns all 3 entries
        """
        self.x = 1
        current.checkpoint.save(name="best")
        self.x = 2
        current.checkpoint.save(name="latest")
        self.x = 3
        current.checkpoint.save(name="best")

        best = current.checkpoint.list(name="best")
        latest = current.checkpoint.list(name="latest")
        all_chkpts = current.checkpoint.list()

        assert len(best) == 2, "Expected 2 'best' checkpoints, got %d" % len(best)
        assert len(latest) == 1, "Expected 1 'latest' checkpoint, got %d" % len(latest)
        assert len(all_chkpts) == 3, "Expected 3 total checkpoints, got %d" % len(all_chkpts)
        for c in best:
            assert c.name == "best", "Wrong name in best list: %r" % c.name

        self.completed_scenario_5 = True
        print("--- scenario 5 PASSED ---")
        self.next(self.user_metadata)

    # ------------------------------------------------------------------
    # Scenario 6 – user metadata round-trip
    # ------------------------------------------------------------------
    @checkpoint(load_policy="none")
    @step
    def user_metadata(self):
        """
        Saves with metadata={"accuracy": 0.95, "tag": "best_so_far"} and a
        custom name, then verifies all fields survive in the returned artifact dict.
        """
        self.epoch = 5
        self.loss = 0.123
        ref = current.checkpoint.save(
            name="my_ckpt",
            metadata={"accuracy": 0.95, "tag": "best_so_far"},
        )

        assert ref.name == "my_ckpt", "Wrong checkpoint name: %r" % ref.name
        meta = ref.metadata or {}
        assert meta.get("accuracy") == 0.95, (
            "Expected accuracy=0.95, got %r" % meta.get("accuracy")
        )
        assert meta.get("tag") == "best_so_far", (
            "Expected tag='best_so_far', got %r" % meta.get("tag")
        )
        assert "_implicit_manifest" not in meta, (
            "_implicit_manifest must not appear in user-visible metadata"
        )

        self.completed_scenario_6 = True
        print("--- scenario 6 PASSED ---")
        self.next(self.complex_types)

    # ------------------------------------------------------------------
    # Scenario 7 – complex Python types via crash-and-resume
    # ------------------------------------------------------------------
    @retry(times=1)
    @checkpoint(load_policy="fresh")
    @step
    def complex_types(self):
        """
        Attempt 0:
          - Stores list, dict, str, float, int on self
          - Saves a checkpoint then crashes intentionally

        Attempt 1 (retry):
          - is_loaded=True → current.checkpoint.load() restores all fields
          - Asserts every field matches the original value exactly
        """
        if current.checkpoint.is_loaded:
            current.checkpoint.load()
            assert self.my_list == [1, 2, 3], "list mismatch: %r" % self.my_list
            assert self.my_dict == {"a": 1, "b": [2, 3]}, (
                "dict mismatch: %r" % self.my_dict
            )
            assert self.my_str == "hello", "str mismatch: %r" % self.my_str
            assert abs(self.my_float - 3.14) < 1e-9, (
                "float mismatch: %r" % self.my_float
            )
            assert self.my_int == 42, "int mismatch: %r" % self.my_int
            self.completed_scenario_7 = True
            print("--- scenario 7 PASSED ---")
        else:
            self.my_list = [1, 2, 3]
            self.my_dict = {"a": 1, "b": [2, 3]}
            self.my_str = "hello"
            self.my_float = 3.14
            self.my_int = 42
            current.checkpoint.save()
            raise RuntimeError("[intentional crash for complex types test]")
        self.next(self.auto_load_resume)

    # ------------------------------------------------------------------
    # Scenario 8 – auto_load=True eliminates is_loaded/load() boilerplate
    # ------------------------------------------------------------------
    @retry(times=1)
    @checkpoint(load_policy="fresh", auto_load=True)
    @step
    def auto_load_resume(self):
        """
        Attempt 0:
          - Sets self.counter = 10, saves a checkpoint, then crashes intentionally.

        Attempt 1 (retry):
          - auto_load=True means the decorator has already called load() before
            this body runs — no is_loaded / load() boilerplate needed.
          - Asserts self.counter == 10 to confirm the restore happened.
        """
        if current.retry_count == 0:
            self.counter = 10
            current.checkpoint.save()
            raise RuntimeError("[intentional crash for auto_load test]")

        # Attempt 1: checkpoint was already loaded by the decorator.
        assert self.counter == 10, (
            "Expected counter=10 after auto_load restore, got %d" % self.counter
        )
        self.completed_scenario_8 = True
        print("--- scenario 8 PASSED ---")
        self.next(self.inspect_checkpoint)

    # ------------------------------------------------------------------
    # Scenario 9 – inspect() returns manifest without downloading files
    # ------------------------------------------------------------------
    @checkpoint(
        serialization_config={"weights": "raw"},
        load_policy="none",
    )
    @step
    def inspect_checkpoint(self):
        """
        Saves one checkpoint then verifies inspect() returns the correct manifest
        via three different reference types (artifact, dict, key string) — all
        without triggering a full checkpoint download.
        """
        self.weights = _fake_weights(seed=3)
        self.epoch = 7
        ref = current.checkpoint.save(name="inspect_me")

        # --- via CheckpointArtifact (direct attribute) ---
        manifest = current.checkpoint.inspect(ref)
        assert manifest is not None, "inspect(artifact) returned None"
        fields = manifest.get("fields", {})
        assert "weights" in fields, "Missing 'weights' in inspected manifest"
        assert "epoch" in fields, "Missing 'epoch' in inspected manifest"
        assert fields["weights"]["format"] == "raw"
        assert fields["epoch"]["format"] == "pickle"

        # --- via dict ---
        manifest_from_dict = current.checkpoint.inspect(ref.to_dict())
        assert manifest_from_dict == manifest, "inspect(dict) mismatch"

        # --- via key string (metadata-only lookup, no file download) ---
        manifest_from_key = current.checkpoint.inspect(ref.key)
        assert manifest_from_key == manifest, "inspect(key) mismatch"

        # --- ref.metadata must still be clean (no _implicit_manifest) ---
        assert "_implicit_manifest" not in (ref.metadata or {}), (
            "_implicit_manifest must not appear in ref.metadata"
        )

        self.completed_scenario_9 = True
        print("--- scenario 9 PASSED ---")
        self.next(self.end)

    # ------------------------------------------------------------------
    # end
    # ------------------------------------------------------------------
    @step
    def end(self):
        assert self.completed_scenario_1
        assert self.completed_scenario_2
        assert self.completed_scenario_3
        assert self.completed_scenario_4
        assert self.completed_scenario_5
        assert self.completed_scenario_6
        assert self.completed_scenario_7
        assert self.completed_scenario_8
        assert self.completed_scenario_9
        print("=== All implicit checkpoint scenarios PASSED ===")


if __name__ == "__main__":
    ImplicitCheckpointTestFlow()
