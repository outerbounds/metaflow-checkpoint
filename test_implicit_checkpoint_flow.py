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

import json

from metaflow import FlowSpec, step, current, checkpoint, retry, card
from metaflow.cards import Markdown, Table


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_tree(keys):
    root = {}
    for key in keys:
        node = root
        for part in key.split("/"):
            node = node.setdefault(part, {})
    return root


def _render_tree(node, prefix=""):
    lines = []
    items = sorted(node.items())
    for i, (name, children) in enumerate(items):
        is_last = i == len(items) - 1
        connector = "└── " if is_last else "├── "
        lines.append(prefix + connector + name)
        extension = "    " if is_last else "│   "
        lines.extend(_render_tree(children, prefix + extension))
    return lines


def _emit_checkpoint_dir_card(card_id="checkpoint_dir"):
    artifacts = current.checkpoint.list()
    c = current.card[card_id]

    if not artifacts:
        c.append(Markdown("*No checkpoints found for this step.*"))
        return

    all_paths = []
    for a in artifacts:
        fields = (a.implicit_manifest or {}).get("fields", {})
        all_paths.append(a.key + "/__implicit_checkpoint__.json")
        for field_info in sorted(fields.values(), key=lambda x: x["filename"]):
            all_paths.append(a.key + "/" + field_info["filename"])
    tree_lines = _render_tree(_make_tree(all_paths))
    c.append(Markdown("## Checkpoint Directory Structure"))
    c.append(Markdown("```\n" + "\n".join(tree_lines) + "\n```"))

    rows = []
    for a in sorted(artifacts, key=lambda x: x.created_on):
        ck_dir = a.key.split("/")[-1]
        fields = sorted((a.implicit_manifest or {}).get("fields", {}).keys())
        rows.append([
            ck_dir,
            a.name or "",
            ck_dir.split(".")[1] if "." in ck_dir else "?",
            a.created_on[:19],
            ", ".join(fields) or "(none)",
            str(a.metadata or {}),
        ])
    c.append(Markdown("## Checkpoint Details"))
    c.append(Table(
        headers=["Checkpoint dir", "Name", "Attempt", "Created", "Fields", "User Metadata"],
        data=rows,
    ))

    c.append(Markdown("## Manifests"))
    for a in sorted(artifacts, key=lambda x: x.created_on):
        manifest = a.implicit_manifest
        ck_dir = a.key.split("/")[-1]
        if manifest is None:
            c.append(Markdown("**%s** — no manifest" % ck_dir))
        else:
            c.append(Markdown("**%s**\n```json\n%s\n```" % (ck_dir, json.dumps(manifest, indent=2))))


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------

class ImplicitCheckpointTestFlow(FlowSpec):

    @step
    def start(self):
        self.next(self.crash_and_resume)

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
        _emit_checkpoint_dir_card()
        self.next(self.save_load_and_list)

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
        _emit_checkpoint_dir_card()
        self.next(self.exclude_filter)

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
        _emit_checkpoint_dir_card()
        self.next(self.include_filter)

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
        _emit_checkpoint_dir_card()
        self.next(self.inspect_checkpoint)

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
        _emit_checkpoint_dir_card()
        self.next(self.end)

    # ------------------------------------------------------------------
    # end
    # ------------------------------------------------------------------
    @step
    def end(self):
        assert self.completed_1
        assert self.completed_2
        assert self.completed_3
        assert self.completed_4
        assert self.completed_5
        print("=== All implicit checkpoint scenarios PASSED ===")


if __name__ == "__main__":
    ImplicitCheckpointTestFlow()
