from typing import Union, Optional, TYPE_CHECKING
from metaflow.metadata_provider import MetaDatum
from ..datastructures import CheckpointArtifact
from .core import CheckpointReferenceResolver

if TYPE_CHECKING:
    import metaflow


def checkpoint_load_related_metadata(checkpoint: CheckpointArtifact, current_attempt):
    data = {
        "checkpoint-load-key": checkpoint.key,
        "checkpoint-load-pathspec": checkpoint.pathspec,
        "checkpoint-load-attempt": checkpoint.attempt,
        "checkpoint-load-name": checkpoint.name,
    }
    md = []
    for k, v in data.items():
        md.append(
            MetaDatum(
                field=k,
                value=v,
                type="checkpoint",
                tags=[
                    "attempt_id:%s" % str(current_attempt),
                ],
            )
        )
    return md


def trace_lineage(flow, checkpoint: CheckpointArtifact):
    """
    Trace the lineage of the checkpoint by tracing the previous paths.
    """
    from metaflow import Task

    def _loaded_parent_key(
        _checkpoint: CheckpointArtifact, _flow, parents=[], max_depth=20
    ):
        try:
            metadata = Task(
                _checkpoint.pathspec, attempt=_checkpoint.attempt
            ).metadata_dict
        except AttributeError:
            return parents  # No parents were discovered.
        if "checkpoint-load-key" not in metadata:
            return parents
        data = CheckpointReferenceResolver.from_key(
            _flow, metadata.get("checkpoint-load-key")
        )
        if data is None:
            return parents
        if data.key == _checkpoint.key:
            # This means that we are iterating over the same checkpoint again and again
            # because the checkpoint-load-key of the task is bringing it back to the same task.
            # Ideally this shouldn't happen unless there is bug in the `checkpoint-load-key` metadata
            # setter.
            return parents

        parents.append(data)
        if max_depth == len(parents):
            return parents

        return _loaded_parent_key(data, _flow, parents=parents, max_depth=max_depth)

    return _loaded_parent_key(checkpoint, flow, parents=[])
