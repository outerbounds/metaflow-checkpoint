from typing import Union, Optional, TYPE_CHECKING
from metaflow import current

from ..exceptions import (
    TODOException,
)
from .identity_utils import (
    TaskIdentifier,
    ResumeTaskNotFoundError,
    AttemptEnum,
    MAX_HASH_LEN,
)
from collections import namedtuple


if TYPE_CHECKING:
    import metaflow


ResolvedTask = namedtuple(
    "ResolvedTask",
    [
        "flow_name",
        "run_id",
        "step_name",
        "task_id",
        "is_foreach",
        "current_attempt",
        "pathspec",
        "control_task_pathspec",
    ],
)

OriginInfo = namedtuple(
    "OriginInfo", ["origin_run_id", "origin_task_id", "origin_attempt", "is_foreach"]
)


def resolve_pathspec_for_flowspec(
    run: "metaflow.FlowSpec" = None,
):
    if not current.is_running_flow:
        raise ValueError(
            "TODO: set error when no checkpointer(run=self) is set when flow is not running."
        )
    control_task_pathspec = None
    if getattr(current, "parallel", None):
        control_task_pathspec = "/".join(
            [
                current.flow_name,
                current.run_id,
                current.step_name,
                current.parallel.control_task_id,
            ]
        )
    return ResolvedTask(
        current.flow_name,
        current.run_id,
        current.step_name,
        current.task_id,
        run.index is not None,
        current.retry_count,
        current.pathspec,
        control_task_pathspec,
    )


def resolve_storage_backend(run: "metaflow.FlowSpec" = None):
    if not current.is_running_flow:
        raise ValueError(
            "TODO: set error when no checkpointer(run=self) is set when flow is not running."
        )
    return run._datastore._storage_impl


def resolve_task_identifier(
    run: "metaflow.FlowSpec",
    gang_scheduled_task=False,
    gang_schedule_task_idf_index=0,
):
    if not current.is_running_flow:
        raise TODOException(
            "TODO: Error resolving task identifier when flow is not running. "
        )
    if gang_scheduled_task:
        return TaskIdentifier.for_parallel_task_index(run, gang_schedule_task_idf_index)
    return TaskIdentifier.from_flowspec(run)
