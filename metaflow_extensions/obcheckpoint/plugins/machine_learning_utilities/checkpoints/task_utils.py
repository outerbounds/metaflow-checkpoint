# This file is USED!
from typing import Union, Optional, TYPE_CHECKING
from metaflow import current

from metaflow.exception import MetaflowException
from metaflow.datastore import FlowDataStore
import os

from .constants import CHECKPOINT_TASK_IDENTIFIER_ENV_VAR_NAME

if TYPE_CHECKING:
    import metaflow


class UnresolvableDatastoreException(MetaflowException):
    pass


def resolve_task_identifier(
    identifier: Optional[str] = None,
):
    """
    TODO : This function is used and needs to refactored a little.
    """
    if identifier is not None and isinstance(identifier, str):
        return identifier
    elif os.environ.get(CHECKPOINT_TASK_IDENTIFIER_ENV_VAR_NAME, None) is not None:
        return os.environ[CHECKPOINT_TASK_IDENTIFIER_ENV_VAR_NAME]
    # If we have a `@checkpoint` decorator we can have a way to create
    # the `Checkpointer` object from just the `current` object.
    elif current is not None and getattr(current, "checkpoint", None):
        _chckpt = getattr(current, "checkpoint", None)
        return _chckpt.task_identifier
    else:
        raise ValueError(
            "TODO: set error when task_identifier cannot be resolved from the environment."
        )


def resolve_pathspec_and_attempt(
    task: Union["metaflow.Task", str],
):
    from metaflow.client.core import Task

    _pathspec, _current_attempt = None, None
    if isinstance(task, Task):
        _pathspec, _current_attempt = task.pathspec, task.current_attempt
    elif isinstance(task, str):
        if len(task.split("/")) != 4:
            raise ValueError(
                "TODO: set error when pathspec is not a valid task pathspec."
            )
        _task = Task(task)
        _pathspec, _current_attempt = _task.pathspec, _task.current_attempt

    return _pathspec, _current_attempt
