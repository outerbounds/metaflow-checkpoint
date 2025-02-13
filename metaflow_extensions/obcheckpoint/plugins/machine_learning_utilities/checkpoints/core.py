from collections import namedtuple
from hashlib import sha256

import os
import sys

from metaflow import current
from metaflow.exception import MetaflowException
from typing import List, Union, Optional, TYPE_CHECKING
import os
from .exceptions import CheckpointNotAvailableException, CheckpointException
from ..utils import flowspec_utils
from .checkpoint_storage import CheckpointDatastore
from ..datastructures import CheckpointArtifact
from ..datastore.task_utils import (
    init_datastorage_object,
    resolve_storage_backend as resolve_task_storage_backend,
    storage_backend_from_flow,
)
from .constants import (
    CHECKPOINT_TAG_PREFIX,
    MAX_HASH_LEN,
    DEFAULT_NAME,
    CHECKPOINT_UID_ENV_VAR_NAME,
    DEFAULT_STORAGE_FORMAT,
)

if TYPE_CHECKING:
    import metaflow

MAX_HASH_LEN = 12


def _coalesce(*args):
    return next((x for x in args if x is not None), None)


def _coalesce_lambdas(*args):
    for x in args:
        _val = x()
        if _val is not None:
            return _val
    return None


class Checkpointer:
    # TODO : this abstraction is not as well designed as it should be
    # TODO : Figure better abstraction interplay.

    _checkpoint_uid: str

    _checkpoint_datastore: CheckpointDatastore

    @property
    def current_version(self):
        return self._current_version

    def _set_current_version(self, version_id):
        self._current_version = version_id

    def set_root_prefix(self, root_prefix):
        self._checkpoint_datastore.set_root_prefix(root_prefix)

    def __init__(
        self,
        datastore: CheckpointDatastore,
        attempt: int = 0,
        # TODO: [POST RELEASE]: See if attempt can be moved elsewhere
    ) -> None:
        self._checkpoint_datastore = datastore
        self._attempt = attempt
        self._current_version = 0

    def _get_latest_checkpoint(self, safe=True):
        return self._checkpoint_datastore.latest(safe=safe)

    def _update_version(self):
        self._current_version += 1

    def load_metadata(self, version_id: int = None, name=DEFAULT_NAME) -> dict:
        return self._checkpoint_datastore.load_metadata(
            attempt=self._attempt,
            version_id=version_id,
            name=name,
        )

    def artifact_id(self, name: str, version_id: int = None):
        # ! Can only be run when in write mode.
        version_id = self._current_version if version_id is None else version_id
        key_name = self._checkpoint_datastore.create_key_name(
            self._checkpoint_datastore._NAME_ENTROPY,
            self._attempt,
            name,
            version_id,
        )
        return self._checkpoint_datastore.artifact_store.resolve_key_relative_path(
            key_name
        )

    def save(
        self,
        path: str,
        metadata={},
        latest=True,
        name=DEFAULT_NAME,
        storage_format=DEFAULT_STORAGE_FORMAT,
    ) -> CheckpointArtifact:
        _art = self._checkpoint_datastore.save(
            path,
            attempt=self._attempt,
            version_id=self._current_version,
            name=name,
            metadata=metadata,
            set_latest=latest,
            storage_format=storage_format,
        )
        self._update_version()
        return _art

    @classmethod
    def _from_task_object(
        cls,
        task: "metaflow.Task",
    ):
        """
        This will instantiate the Checkpoint Datastore with only access to the checkpoint metadata store.
        Useful when we are making checkpoint list calls post task-runtime.
        """
        _checkpoint_datastore = ReadResolver.from_pathspec(task.pathspec)
        return cls(
            datastore=_checkpoint_datastore,
            attempt=task.current_attempt,
        )

    @classmethod
    def _from_checkpoint_and_storage_backend(
        cls, checkpoint: CheckpointArtifact, storage_backend
    ):
        """
        Used by: `Factory` to load the artifact from native MetaflowArtifactReference objects.
        """
        _checkpoint_datastore = CheckpointDatastore.init_read_store(
            storage_backend, checkpoint_key=checkpoint.key
        )
        obj = cls(
            datastore=_checkpoint_datastore,
            attempt=checkpoint.attempt,
        )
        return obj

    @classmethod
    def _from_key(cls, key: str):
        key_decomp = CheckpointDatastore.decompose_key(key)
        datastore = ReadResolver.from_key(key)
        obj = cls(
            datastore=datastore,
            attempt=key_decomp.attempt,
        )
        obj._set_current_version(key_decomp.version_id)
        return obj

    @classmethod
    def _from_checkpoint(cls, checkpoint: Union[CheckpointArtifact, dict]):
        _chckpt: CheckpointArtifact = CheckpointArtifact.hydrate(checkpoint)
        # TODO [POST-RELEASE]: Suport out of task checkpoints
        datastore = ReadResolver.from_checkpoint(_chckpt)
        obj = cls(
            datastore=datastore,
            attempt=_chckpt.attempt,
        )
        obj._set_current_version(_chckpt.version_id)
        return obj

    def _load_checkpoint(
        self,
        local_path: str,
        version_id: int = None,
        name=DEFAULT_NAME,
        storage_format=None,  # Loading should require an explicit format
    ):
        self._checkpoint_datastore.load(
            local_path,
            version_id=version_id,
            attempt=self._attempt,
            name=name,
            storage_format=storage_format,
        )

    def _list_checkpoints(
        self,
        name: Optional[str] = DEFAULT_NAME,
        attempt: Optional[int] = None,
        within_task: Optional[bool] = True,
    ):
        return self._checkpoint_datastore.list(
            name=name, attempt=attempt, within_task=within_task
        )


class CheckpointLoadPolicy:
    @classmethod
    def fresh(
        cls,
        datastore: CheckpointDatastore,
        flow: "metaflow.FlowSpec",
    ) -> Union[CheckpointArtifact, None]:
        """
        ```python
        @checkpoint(load_policy="fresh")
        ```
        While in `fresh` mode, we want to load the "latest" checkpoint from
        what ever task is executing at the current memoment.

        The behavior is will be such that 1st attempt of any task will not load
        any checkpoint and there after it will load the checkpoint from the previous
        attempt of the task (ala the lastest checkpoint within the task).


        """
        latest_task_checkpoint_lambda = lambda: datastore.latest(
            current_task=True, safe=True
        )
        return latest_task_checkpoint_lambda()

    @classmethod
    def eager(
        cls,
        datastore: CheckpointDatastore,
        flow: "metaflow.FlowSpec",
    ) -> Union[CheckpointArtifact, None]:
        """
        ```python
        @checkpoint(load_policy="eager")
        ```
        While in `eager` mode, we want to load the "latest" checkpoint ever
        written at the "scope" level based on the kind of task that is executing.

        Setting this mode helps "checkpoints leak across executions" for the same task
        there by allowing a way to reboot the state when new executions start.
        """
        latest_task_checkpoint_lambda = lambda: datastore.latest(
            current_task=True, safe=True
        )
        latest_scope_checkpoint_lambda = lambda: datastore.latest(
            current_task=False, safe=True
        )
        return _coalesce_lambdas(
            latest_task_checkpoint_lambda,
            latest_scope_checkpoint_lambda,
        )


class ScopeResolver:
    @classmethod
    def from_namespace(cls):
        from metaflow import get_namespace

        ns = get_namespace()
        if ns is None:
            raise MetaflowException("Cannot resolve checkpoint path without Namespace.")
        return sha256(ns.encode()).hexdigest()[:MAX_HASH_LEN]

    @classmethod
    def from_tags(cls, tags):
        from metaflow import Run

        return sha256(tags[0].split(":")[1].encode()).hexdigest()[:MAX_HASH_LEN]


def warning_message(message, logger=None, ts=False, prefix="[@checkpoint][warning]"):
    msg = "%s %s" % (prefix, message)
    if logger:
        logger(msg, timestamp=ts, bad=True)
    else:
        print(msg, file=sys.stderr)


class ReadResolver:
    """
    Responsible for instantiating the `CheckpointDatastore` during read operations
    based on different context's.
    """

    @classmethod
    def from_pathspec(cls, pathspec):
        # This resolver helps create the datastore when the user is calling `Checkpoint.list`
        # from a notebook or a script
        validation, storage_backend = resolve_task_storage_backend(pathspec=pathspec)
        if not validation.is_valid:
            if validation.needs_external_context:
                warning_message(
                    (
                        "The Task (%s) used an external datastore for storing checkpoints. "
                        "While the current execution context is configured to use the default datastore, "
                        "this means that some of the artifacts might not be accessible. "
                        "Please use the `artifact_store_from` context manager to configure the "
                        "external datastore."
                    )
                    % pathspec
                )
            elif validation.context_mismatch:
                warning_message(
                    (
                        "The current datastore context set via `artifact_store_from` context manager"
                        "doesn't match the artifact store set in the task metadata of task (%s). "
                        "This means that some objects might not be accessible under the context manager."
                        "If the Flow was not using `@with_artifact_store` context manager, "
                        "then remove the `artifact_store_from` context manager in your user code."
                    )
                    % pathspec
                )
        _checkpoint_datastore = CheckpointDatastore.init_read_store(
            storage_backend, pathspec=pathspec
        )
        return _checkpoint_datastore

    @classmethod
    def from_key(cls, checkpoint_key):
        """"""
        storage_backend = init_datastorage_object()
        _checkpoint_datastore = CheckpointDatastore.init_read_store(
            storage_backend, checkpoint_key=checkpoint_key
        )
        return _checkpoint_datastore

    @classmethod
    def from_checkpoint(cls, checkpoint: "CheckpointArtifact"):
        """ """
        storage_backend = init_datastorage_object()
        _checkpoint_datastore = CheckpointDatastore.init_read_store(
            storage_backend,
            checkpoint_key=checkpoint.key,
        )
        return _checkpoint_datastore

    @classmethod
    def from_key_and_run(cls, run, checkpoint_key) -> CheckpointDatastore:
        """ """
        storage_backend = storage_backend_from_flow(
            flow=run,
        )
        _checkpoint_datastore = CheckpointDatastore.init_read_store(
            storage_backend, checkpoint_key=checkpoint_key
        )
        return _checkpoint_datastore


class WriteResolver:
    """
    Responsible for instantiating the `CheckpointDatastore` and the subsequent
    `_checkpointer_uid` which can instantiate the `Checkpointer` object outside
    of the metaflow context.
    """

    ResolverInfo = namedtuple(
        "ResolverInfo", ["flow", "run", "step", "taskid", "taskidf", "scope", "attempt"]
    )

    @classmethod
    def can_resolve_from_envionment(cls):
        if CHECKPOINT_UID_ENV_VAR_NAME in os.environ:
            return True
        return False

    @classmethod
    def decompose_checkpoint_id(cls, checkpoint_id):
        if len(checkpoint_id.split("/")) != 7:
            raise CheckpointException(
                "`%s` environment variable is not in correct format."
                % CHECKPOINT_UID_ENV_VAR_NAME
            )

        flow, run, step, taskid, taskidf, scope, attempt = checkpoint_id.split("/")
        return cls.resolver_info(flow, run, step, taskid, taskidf, scope, attempt)

    @classmethod
    def resolver_info(cls, flow, run, step, taskid, taskidf, scope, attempt):
        return cls.ResolverInfo(flow, run, step, taskid, taskidf, scope, attempt)

    @classmethod
    def construct_checkpoint_id(cls, resolver_info: ResolverInfo):
        return "/".join([str(x) for x in resolver_info])

    @classmethod
    def from_environment(
        cls,
    ):
        # Resolve the full checkpoint datastore from the environment.
        chckpt_uid = os.environ[CHECKPOINT_UID_ENV_VAR_NAME]
        _resolver_info = cls.decompose_checkpoint_id(chckpt_uid)
        pathspec = "%s/%s/%s/%s" % (
            _resolver_info.flow,
            _resolver_info.run,
            _resolver_info.step,
            _resolver_info.taskid,
        )
        storage_backend = init_datastorage_object()
        _checkpoint_datastore = CheckpointDatastore.init_write_store(
            storage_backend,
            pathspec=pathspec,
            scope=_resolver_info.scope,
            task_identifier=_resolver_info.taskidf,
        )
        return _checkpoint_datastore, _resolver_info

    @classmethod
    def from_run(
        cls,
        run: "metaflow.FlowSpec",
        scope: str,
        task_identifier: Optional[str] = None,
        gang_scheduled_task=False,
    ):
        """
        The task-identifier gets computed in the Metaflow main process with the
        i.e. in the decorator. The pathspec we choose to write the metadata store
        depends on if the task is being gang scheduled or not.
        """

        storage_backend = storage_backend_from_flow(
            flow=run,
        )
        identifier = task_identifier
        resolved_pathspec_info = flowspec_utils.resolve_pathspec_for_flowspec(
            run,
        )
        # For gang scheduled tasks with @parallel decorator, all worker
        # tasks will be writing to the pathspec as the control task for
        # the metadata store.
        # The `task_identifier` for all workers tasks in @parallel
        # will be the same as the control task identifier.
        pathspec = resolved_pathspec_info.pathspec
        if gang_scheduled_task:
            pathspec = resolved_pathspec_info.control_task_pathspec

        _checkpoint_datastore = CheckpointDatastore.init_write_store(
            storage_backend,
            pathspec=pathspec,
            scope=scope,
            task_identifier=identifier,
        )

        flow_name, run_id, step_name, task_id = pathspec.split("/")
        return _checkpoint_datastore, cls.resolver_info(
            flow_name,
            run_id,
            step_name,
            task_id,
            identifier,
            scope,
            resolved_pathspec_info.current_attempt,
        )


class CheckpointReferenceResolver:
    """
    Resolve the Metaflow checkpoint object based on the flow artifact reference
    or key; Used for lineage derivation.
    """

    @classmethod
    def from_key(cls, flow, checkpoint_key):
        """
        Used by lineage derivation
        """
        checkpoint_datastore = ReadResolver.from_key_and_run(
            run=flow, checkpoint_key=checkpoint_key
        )
        key_comps = checkpoint_datastore.decompose_key(checkpoint_key)
        _checkpoint = checkpoint_datastore.load_metadata(
            attempt=key_comps.attempt,
            version_id=key_comps.version_id,
            name=key_comps.name,
        )
        if _checkpoint is None:
            raise CheckpointNotAvailableException(
                "Checkpoint with key `%s` not found." % checkpoint_key
            )
        return CheckpointArtifact.hydrate(_checkpoint)
