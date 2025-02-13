# THIS FILE IS ALSO USED (Figure where is the best place to put it.)
from typing import Union, Optional, TYPE_CHECKING
from metaflow import current
from metaflow.exception import MetaflowException
from .context import datastore_context

if TYPE_CHECKING:
    import metaflow


class UnresolvableDatastoreException(MetaflowException):
    pass


def init_datastorage_object():
    return datastore_context.get()


def resolve_storage_backend(pathspec: Union[str, "metaflow.Task"] = None):
    """
    This function is ONLY called when the user is calling `Checkpoint.list`.
    What happens when users call list:
        1. The task has completed so the list call is using the `_task_checkpoints` data-artifact to list the checkpoints.
        2. The task is still running or has crashed. This means that in order to list the checkpoints, we need access to the datastore.

    For case `1`, this code path wont even be called because the data-artifact is already a separate object.
    For case `2` is where this code path is important. But since we now expose a `artifact_store_from` context manager, we
    know pre-hand what the datastore needs to be and the datastore context has already been switched.

    SO in turn, one can make the argument that this function is not needed and we can just have a
    users do a `Checkpoint.list` under the context manager if they need to access objects in a different datastore.

    This is a respectible pattern since directly reading the task metadata and trying to do a list call is not a good pattern
    since the creds of metaflow default datastore might not be the same as the datastore the user wants to access.

    There is a function that can help verify if the datastore set in the task metadata
    is the same as the default datastore. If it is not, then we should shout warning messages
    to the user. In case the permissions are the same, nothing wrong happens, if they are not then
    user will have some hint in the logs to help them figure out the issue.
    """
    if isinstance(pathspec, str):
        from metaflow import Task

        task = Task(pathspec)
    else:
        task = pathspec
    validation = datastore_context.current_context_matches_task_metadata(task)
    return validation, datastore_context.get()


class FlowNotRunningException(MetaflowException):
    pass


def storage_backend_from_flow(flow: "metaflow.FlowSpec"):
    if not current.is_running_flow:
        raise FlowNotRunningException
    return datastore_context.get()
