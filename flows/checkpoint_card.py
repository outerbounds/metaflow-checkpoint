"""Shared helper: emit a checkpoint-directory card for any @checkpoint step."""

import json

from metaflow import current
from metaflow.cards import Markdown, Table


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


def emit_checkpoint_dir_card(card_id="checkpoint_dir"):
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
