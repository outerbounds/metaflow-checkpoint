# THIS FILE IS ALSO USED (Figure where is the best place to put it.)
from typing import Union, Optional, TYPE_CHECKING

from metaflow.exception import MetaflowException
from metaflow.datastore import FlowDataStore
from .core import resolve_root


if TYPE_CHECKING:
    import metaflow


class UnresolvableDatastoreException(MetaflowException):
    pass


def _get_flow_datastore(task):
    flow_name = task.pathspec.split("/")[0]
    # Resolve datastore type
    ds_type = None
    # We need to set the correct datastore root here so that
    # we can ensure that the card client picks up the correct path to the cards

    meta_dict = task.metadata_dict
    ds_type = meta_dict.get("ds-type", None)

    if ds_type is None:
        raise UnresolvableDatastoreException(task)

    ds_root = meta_dict.get("ds-root", None)

    if ds_root is None:
        ds_root = resolve_root(ds_type)

    # Delay load to prevent circular dep
    from metaflow.plugins import DATASTORES

    storage_impl = [d for d in DATASTORES if d.TYPE == ds_type][0]
    return FlowDataStore(
        flow_name=flow_name,
        environment=None,
        storage_impl=storage_impl,
        # ! ds root cannot be none otherwise `list_content`
        # ! method fails in the datastore abstraction.
        ds_root=ds_root,
    )


def init_datastorage_object():
    from metaflow.plugins import DATASTORES
    from metaflow.metaflow_config import DEFAULT_DATASTORE

    storage_impl = [d for d in DATASTORES if d.TYPE == DEFAULT_DATASTORE][0]
    return storage_impl(storage_impl.get_datastore_root_from_config(print))


def resolve_storage_backend(pathspec: Union[str, "metaflow.Task"] = None):
    from metaflow.client.core import Task

    if isinstance(pathspec, Task):
        return _get_flow_datastore(pathspec)._storage_impl
    elif isinstance(pathspec, str):
        if len(pathspec.split("/")) != 4:
            raise ValueError("Pathspec is not of the correct format.")
        return _get_flow_datastore(Task(pathspec))._storage_impl
    else:
        raise ValueError(
            "Pathspec is of invalid type. It should be either a string or a Task object but got %s"
            % type(pathspec)
        )
