"""
Test flow for implicit checkpointing.

Scenarios covered
-----------------
1. crash-and-resume          : all public attrs saved; retry restores from checkpoint
2. save / load / list        : explicit load with complex Python types, list() name
                               filtering, metadata round-trip, no _implicit_manifest leak
3. exclude=[...]             : listed attrs skipped
4. include=[...]             : only listed attrs saved; include+exclude conflict raises
5. inspect()                 : returns manifest via artifact/dict/key without download

Run (local datastore, no cloud needed):
    python test_implicit_checkpoint_flow.py run
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from metaflow import FlowSpec, step, current, checkpoint, retry, card

from checkpoint_card import emit_checkpoint_dir_card


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------

class ImplicitCheckpointTestFlow(FlowSpec):

    @step
    def start(self):
        self.next(
            self.crash_and_resume,
            self.save_load_and_list,
            self.exclude_filter,
            self.include_filter,
            self.inspect_checkpoint,
        )

    # ------------------------------------------------------------------
    # Scenario 1 – crash-and-resume
    # ------------------------------------------------------------------
    @retry(times=1)
    @checkpoint(load_policy="fresh")
    @card(id="checkpoint_dir")
    @step
    def crash_and_resume(self):
        """Attempt 0: save epoch=0, crash. Attempt 1: load, verify epoch=0 restored, finish."""
        if current.checkpoint.is_loaded:
            current.checkpoint.load()
            assert self.epoch == 0, "Expected epoch=0 on resume, got %d" % self.epoch
            self.loss = round(self.loss * 0.9, 6)
            self.epoch = 1
            current.checkpoint.save()
        else:
            self.epoch = 0
            self.loss = 1.0
            current.checkpoint.save()
            raise RuntimeError("[intentional crash at epoch 0]")

        assert self.epoch == 1 and self.loss < 1.0
        self.completed_1 = True
        print("--- scenario 1 PASSED ---")
        emit_checkpoint_dir_card()
        self.next(self.join)

    # ------------------------------------------------------------------
    # Scenario 2 – save / load / list / metadata
    # ------------------------------------------------------------------
    @checkpoint(load_policy="none")
    @card(id="checkpoint_dir")
    @step
    def save_load_and_list(self):
        """
        Complex Python types survive pickle round-trip via explicit load.
        list(name=...) filters correctly.
        metadata= survives and _implicit_manifest stays hidden.
        """
        # --- complex types: save, clobber, load(reference), verify ---
        self.my_list = [1, 2, 3]
        self.my_dict = {"a": 1, "b": [2, 3]}
        self.my_str = "hello"
        self.my_float = 3.14
        self.my_int = 42
        ref = current.checkpoint.save()

        self.my_list = []
        self.my_dict = {}
        self.my_str = ""
        self.my_float = 0.0
        self.my_int = 0
        current.checkpoint.load(reference=ref)
        assert self.my_list == [1, 2, 3], "list mismatch"
        assert self.my_dict == {"a": 1, "b": [2, 3]}, "dict mismatch"
        assert self.my_str == "hello", "str mismatch"
        assert abs(self.my_float - 3.14) < 1e-9, "float mismatch"
        assert self.my_int == 42, "int mismatch"

        # --- list(name=...) filtering ---
        self.x = 1
        current.checkpoint.save(name="best")
        self.x = 2
        current.checkpoint.save(name="latest")
        self.x = 3
        current.checkpoint.save(name="best")

        best = current.checkpoint.list(name="best")
        latest = current.checkpoint.list(name="latest")
        assert len(best) == 2, "Expected 2 'best', got %d" % len(best)
        assert len(latest) == 1, "Expected 1 'latest', got %d" % len(latest)
        assert all(c.name == "best" for c in best), "Wrong name in best list"

        # --- metadata round-trip; _implicit_manifest hidden ---
        self.val = 1
        ref_meta = current.checkpoint.save(
            name="meta_ckpt",
            metadata={"accuracy": 0.95, "tag": "best_so_far"},
        )
        assert ref_meta.name == "meta_ckpt"
        meta = ref_meta.metadata or {}
        assert meta.get("accuracy") == 0.95
        assert meta.get("tag") == "best_so_far"
        assert "_implicit_manifest" not in meta, "_implicit_manifest must not appear in metadata"

        self.completed_2 = True
        print("--- scenario 2 PASSED ---")
        emit_checkpoint_dir_card()
        self.next(self.join)

    # ------------------------------------------------------------------
    # Scenario 3 – exclude=[...]
    # ------------------------------------------------------------------
    @checkpoint(exclude=["scratch"], load_policy="none")
    @card(id="checkpoint_dir")
    @step
    def exclude_filter(self):
        """'scratch' excluded; 'epoch' and 'loss' checkpointed."""
        self.epoch = 0
        self.loss = 0.5
        self.scratch = 42

        ref = current.checkpoint.save()
        fields = (ref.implicit_manifest or {}).get("fields", {})
        assert "epoch" in fields and "loss" in fields, "epoch/loss missing from manifest"
        assert "scratch" not in fields, "'scratch' must not be checkpointed"

        self.completed_3 = True
        print("--- scenario 3 PASSED ---")
        emit_checkpoint_dir_card()
        self.next(self.join)

    # ------------------------------------------------------------------
    # Scenario 4 – include=[...]
    # ------------------------------------------------------------------
    @checkpoint(include=["epoch", "loss"], load_policy="none")
    @card(id="checkpoint_dir")
    @step
    def include_filter(self):
        """Only 'epoch' and 'loss' checkpointed; include+exclude conflict raises ValueError."""
        self.epoch = 3
        self.loss = 0.42
        self.scratch = "ignored"

        ref = current.checkpoint.save()
        fields = (ref.implicit_manifest or {}).get("fields", {})
        assert "epoch" in fields and "loss" in fields, "epoch/loss missing"
        assert "scratch" not in fields, "'scratch' must not be checkpointed"
        assert len(fields) == 2, "Expected exactly 2 fields, got %d" % len(fields)

        from metaflow_extensions.obcheckpoint.plugins.machine_learning_utilities.checkpoints.final_api import (
            _get_implicit_fields,
        )
        try:
            _get_implicit_fields(self, exclude=["scratch"], include=["epoch"])
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "mutually exclusive" in str(e)

        self.completed_4 = True
        print("--- scenario 4 PASSED ---")
        emit_checkpoint_dir_card()
        self.next(self.join)

    # ------------------------------------------------------------------
    # Scenario 5 – inspect()
    # ------------------------------------------------------------------
    @checkpoint(load_policy="none")
    @card(id="checkpoint_dir")
    @step
    def inspect_checkpoint(self):
        """inspect() returns manifest via artifact/dict/key without downloading files."""
        self.epoch = 0
        self.loss = 0.5
        ref = current.checkpoint.save(name="inspect_me")

        manifest = current.checkpoint.inspect(ref)
        assert manifest is not None
        fields = manifest.get("fields", {})
        assert "epoch" in fields and "loss" in fields, "epoch/loss missing from manifest"
        assert current.checkpoint.inspect(ref.to_dict()) == manifest, "inspect(dict) mismatch"
        assert current.checkpoint.inspect(ref.key) == manifest, "inspect(key) mismatch"
        assert "_implicit_manifest" not in (ref.metadata or {}), (
            "_implicit_manifest must not appear in ref.metadata"
        )

        self.completed_5 = True
        print("--- scenario 5 PASSED ---")
        emit_checkpoint_dir_card()
        self.next(self.join)

    # ------------------------------------------------------------------
    # join
    # ------------------------------------------------------------------
    @step
    def join(self, inputs):
        assert inputs.crash_and_resume.completed_1
        assert inputs.save_load_and_list.completed_2
        assert inputs.exclude_filter.completed_3
        assert inputs.include_filter.completed_4
        assert inputs.inspect_checkpoint.completed_5
        self.next(self.end)

    # ------------------------------------------------------------------
    # end
    # ------------------------------------------------------------------
    @step
    def end(self):
        print("=== All implicit checkpoint scenarios PASSED ===")


if __name__ == "__main__":
    ImplicitCheckpointTestFlow()
