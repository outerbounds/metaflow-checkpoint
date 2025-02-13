from typing import Union, Optional, TYPE_CHECKING
from metaflow.exception import MetaflowException
import sys
from collections import namedtuple

# import context manager
from contextlib import contextmanager
import os
import json

ARTIFACT_STORE_CONFIG_ENV_VAR = "METAFLOW_CHECKPOINT_ARTIFACT_STORE_CONFIG"

if TYPE_CHECKING:
    import metaflow


def will_switch_context(func):
    """
    A decorator that wraps methods in ArtifactStoreContext.
    It takes the return value of the decorated method and passes it to
    self.switch_context(), then returns None.
    """

    def wrapper(self, *args, **kwargs):
        context = func(self, *args, **kwargs)
        self.switch_context(context)
        return None

    return wrapper


DatastoreContextValidation = namedtuple(
    "DatastoreContextInfo", ["is_valid", "needs_external_context", "context_mismatch"]
)


class UnresolvableDatastoreException(MetaflowException):
    pass


class ArtifactStoreContext:

    """
    This class will act as a singleton to switch the datastore so that
    Any checkpoint operations will be done in the correct datastore. This
    can be a useful way of short-circuting a datastore switch between runtime
    And post-runtime retrieval operations.
    """

    current_context = None  # dict

    _MF_DATASTORES = None

    @property
    def MF_DATASTORES(self):
        if self._MF_DATASTORES is None:
            from metaflow.plugins import DATASTORES

            self._MF_DATASTORES = DATASTORES
        return self._MF_DATASTORES

    def __init__(self):
        if ARTIFACT_STORE_CONFIG_ENV_VAR in os.environ:
            self.switch_context(json.loads(os.environ[ARTIFACT_STORE_CONFIG_ENV_VAR]))

    @will_switch_context
    def flow_init_context(self, context):
        if context is None:
            return None
        required_context_keys = ["type", "config"]
        required_config_keys = [
            "root",
        ]
        if not all([k in context for k in required_context_keys]):
            raise ValueError("Context does not have all required keys.")
        if not all([k in context["config"] for k in required_config_keys]):
            raise ValueError("Config does not have all required keys.")
        os.environ[ARTIFACT_STORE_CONFIG_ENV_VAR] = json.dumps(context)
        return context

    def switch_context(self, context):
        self.current_context = context

    def get(self):
        if self.current_context is None:
            return self.default()
        return self._from_current_context()

    def _from_current_context(self):
        # from metaflow.plugins import DATASTORES
        _type = self._runtime_ds_type(self.current_context["type"])
        _config = self.current_context["config"].copy()
        root = _config.pop("root")
        storage_impl = [d for d in self.MF_DATASTORES if d.TYPE == _type][0]
        return storage_impl(root, **_config)

    def _runtime_ds_type(self, ds_type):
        # Hack ; Ideally all datastores should have better
        # abstractions upstream. This will help handle the
        # case where we don't need to vendor the `s3-compatible`
        # datastore in the plugin
        if ds_type == "s3":
            return "s3-compatible"
        return ds_type

    def _task_metadata_ds_type(self, ds_type):
        # Hack ; Ideally all datastores should have better
        # abstractions upstream. This will help handle the
        # case where we don't need to vendor the `s3-compatible`
        # datastore in the plugin
        if ds_type == "s3-compatible":
            return "s3"
        return ds_type

    @will_switch_context
    def context_from_run(self, run: "metaflow.Run", config=None):
        md_keys = ["artifact-store-ds-type", "artifact-store-ds-root"]
        if not all([any([k in t for t in run.system_tags]) for k in md_keys]):
            # When we return None, we switch context to the default datastore.
            return None

        # extract artifact store type and root from system tags
        for t in run.system_tags:
            if t.startswith("artifact-store-ds-type:"):
                ds_type = t.split(":")[1]
                continue
            if t.startswith("artifact-store-ds-root:"):
                ds_root = ":".join(t.split(":")[1:])
                continue

        usable_context = {"type": ds_type, "config": {"root": ds_root}}
        if config is not None:
            usable_context["config"].update(config)

        return usable_context

    @will_switch_context
    def context_from_task(self, task: "metaflow.Task", config=None):
        metadata = task.metadata_dict
        md_keys = ["artifact-store-ds-type", "artifact-store-ds-root"]
        if not all([k in metadata for k in md_keys]):
            # When we return None, we switch context to the default datastore.
            return None

        ds_type = metadata["artifact-store-ds-type"]
        ds_root = metadata["artifact-store-ds-root"]
        usable_context = {"type": ds_type, "config": {"root": ds_root}}
        if config is not None:
            usable_context["config"].update(config)
        return usable_context

    def current_context_matches_task_metadata(self, task: "metaflow.Task"):
        metadata = task.metadata_dict
        md_keys = ["artifact-store-ds-type", "artifact-store-ds-root"]
        default_keys = ["ds-type", "ds-root"]
        task_ds_info = {"type": None, "root": None, "used_external_ds": False}
        if all([k in metadata for k in md_keys]):
            # This means that the task was run using an external artifact store.
            # so we can just return true.
            task_ds_info["type"] = metadata["artifact-store-ds-type"]
            task_ds_info["root"] = metadata["artifact-store-ds-root"]
            task_ds_info["used_external_ds"] = True
        elif all([k in metadata for k in default_keys]):
            # This means that the task was run using the default datastore.
            # so we can just return true.
            task_ds_info["type"] = metadata["ds-type"]
            task_ds_info["root"] = metadata["ds-root"]
        else:
            # We only raise an exeception because we are unable to resolve the datastore. This happens in the following situation:
            # The remote task has been deployed and didn't start execution and the user ctrl+c'd the job.
            # The means that metaflow's `step` command has not been able to register the `ds-type` and `ds-root` task metadata.
            # And in this scenarios even trying to access the checkpoints will make no difference. Since we would have essentially
            # written nothing.
            raise UnresolvableDatastoreException(
                "The datastore for the task was not found in the task metadata. This happens when the task didn't begin execution and "
                "it was killed by the user. Essentially no objects would have been written for this task by the "
                "`@checkpoint`/`@model`/`@huggingface_hub` decorators."
            )

        if self.current_context is not None:
            # This means that we are NOT in the default datastore context.
            # this means that user has explicitly set the datastore context in the flow code (via `@with_artifact_store`)
            # or via the `artifact_store_from` context manager.
            if not all(
                [self.current_context[k] == task_ds_info[k] for k in ["type", "root"]]
            ):
                # "TODO: Explain to the user that the execution context is setup to use an external datastore that is different from the task metadata. "
                # "And that they need to use the `artifact_store_from` context manager to configure the external datastore."
                return DatastoreContextValidation(
                    is_valid=False, needs_external_context=False, context_mismatch=True
                )
        else:
            if task_ds_info["used_external_ds"]:
                # "TODO: Explain to the user that they are using they need to use the `artifact_store_from` context manager to configure the external datastore "
                # "And that currently the client is only setup to access the default datastore."
                return DatastoreContextValidation(
                    is_valid=False, needs_external_context=True, context_mismatch=False
                )
        return DatastoreContextValidation(
            is_valid=True, needs_external_context=False, context_mismatch=False
        )

    def to_task_metadata(self):
        _ds = self.get()
        return {
            "artifact-store-ds-type": self._task_metadata_ds_type(_ds.TYPE),
            "artifact-store-ds-root": _ds.datastore_root,
        }

    def default(self):
        from metaflow.metaflow_config import DEFAULT_DATASTORE

        search_datastore = self._runtime_ds_type(DEFAULT_DATASTORE)

        storage_impl = [d for d in self.MF_DATASTORES if d.TYPE == search_datastore][0]
        return storage_impl(storage_impl.get_datastore_root_from_config(print))


datastore_context = ArtifactStoreContext()


@contextmanager
def artifact_store_from(run=None, task=None, config=None):
    """
    This context manager can be used to switch the artifact store
    for a block of code. This is useful when users maybe accessing
    checkpoints/models from a different datastore using the
    `@with_artifact_store` decorator.
    """
    if run is None and task is None:
        raise ValueError("Either run or task must be provided")
    if run is not None and task is not None:
        raise ValueError("Only one of run or task can be provided")

    try:
        if run is not None:
            datastore_context.context_from_run(run, config)
        else:
            datastore_context.context_from_task(task, config)
        yield
    finally:
        datastore_context.switch_context(None)
