import os
from typing import Callable, Generator, Iterator, List, Optional, Union

from metaflow.datastore.datastore_storage import DataStoreStorage
from .constants import DEFAULT_NAME, CHECKPOINTS_STORAGE_PREFIX
from ..exceptions import KeyNotCompatibleWithObjectException
from ..utils.identity_utils import pathspec_hash
from ..utils.general import replace_start_and_end_slash
from ..datastore.core import allow_safe, DatastoreInterface, ObjectStorage
from ..datastore.exceptions import (
    DatastoreReadInitException,
    DatastoreWriteInitException,
    DatastoreNotReadyException,
)
from ..datastructures import CheckpointArtifact
from ..datastore.utils import safe_serialize
import json
import re
from datetime import datetime
from collections import namedtuple

ARTIFACT_STORE_NAME = "artifacts"

METADATA_STORE_NAME = "metadata"

ARTIFACT_METADATA_STORE_NAME = "artifact_metadata"


CheckpointsPathComponents = namedtuple(
    "CheckpointsPathComponents",
    [
        "flow_name",
        "step_name",
        "scope",
        "task_identifier",
        "pathspec_hash",
        "attempt",
        "name",
        "version_id",
        "is_metadata",
        "key_name",  # This is the final part of the path components and is generally of the form `a.b.c.d.metadata` or `a.b.c.d`
        "root_prefix",  # This is the root prefix of the path like `mf.checkpoints`
    ],
)

# Every checkpoint key will be prefixed with a `/checkpoint/` and will have a structure of the form:
# <root_prefix>/checkpoints/artifacts/<flow_name>/<step_name>/<scope>/<task_identifier>/<pathspec_hash>.<attempt>.<name>.<version_id>
# The root_prefix can be set at the decorator level and the /checkpoints/ is a fixed prefix that also helps distinguish between the
# different types of objects such as `checkpoints`/`models` etc.
CHECKPOINT_KEY_PATTERN = re.compile(
    r"^(?P<root_prefix>.*)/checkpoints/artifacts/(?P<flow_name>.*?)/(?P<step_name>.*?)/(?P<scope>.*?)/(?P<task_identifier>.*?)/(?P<key_name>.*?)$"
)

CHECKPOINT_ARTIFACT_MD_KEY_PATTERN = re.compile(
    r"^(?P<root_prefix>.*)/checkpoints/artifact_metadata/(?P<flow_name>.*?)/(?P<step_name>.*?)/(?P<scope>.*?)/(?P<task_identifier>.*?)/(?P<key_name>.*?)$"
)


CHECKPOINT_METADATA_KEY_PATTERN = re.compile(
    r"^(?P<root_prefix>.*)/checkpoints/metadata/(?P<flow_name>.*?)/(?P<runid>.*?)/(?P<step_name>.*?)/(?P<taskid>.*?)/(?P<key_name>.*?)$"
)


def __decompose_checkpoint_artifact_metadata_key(key):
    # Match the string with the pattern
    match = re.match(CHECKPOINT_ARTIFACT_MD_KEY_PATTERN, key)
    # Extract the values into a dictionary if there's a match
    if match:
        values = match.groupdict()
        return values

    raise KeyNotCompatibleWithObjectException(key, "checkpoint_artifact_metadata")


def __decompose_checkpoint_metadata_key(key):
    # Match the string with the pattern
    match = re.match(CHECKPOINT_METADATA_KEY_PATTERN, key)
    # Extract the values into a dictionary if there's a match
    if match:
        values = match.groupdict()
        return values

    raise KeyNotCompatibleWithObjectException(key, "checkpoint_metadata")


def __decompose_checkpoint_key(key):
    # Match the string with the pattern
    match = re.match(CHECKPOINT_KEY_PATTERN, key)
    # Extract the values into a dictionary if there's a match
    if match:
        values = match.groupdict()
        return values

    raise KeyNotCompatibleWithObjectException(key, "checkpoint")


def _decompose_artifact_key(key, data_object):
    root_prefix, flowname, stepname, scope, task_identifier, chckpt_id = (
        data_object["root_prefix"],
        data_object["flow_name"],
        data_object["step_name"],
        data_object["scope"],
        data_object["task_identifier"],
        data_object["key_name"],
    )

    if len(chckpt_id.split(".")) < 4:
        raise KeyNotCompatibleWithObjectException(
            key,
            "checkpoint",
            "key_name is not in the correct format. Expected 4 parts, got %s"
            % len(chckpt_id.split(".")),
        )

    is_metadata = chckpt_id.split(".")[-1] == "metadata"
    key_name = chckpt_id
    if is_metadata:
        key_name = ".".join(chckpt_id.split(".")[:-1])

    pathspec_hash, attempt, name, version_id = chckpt_id.split(".")[:4]
    return CheckpointsPathComponents(
        flow_name=flowname,
        step_name=stepname,
        scope=scope,
        task_identifier=task_identifier,
        pathspec_hash=pathspec_hash,
        attempt=attempt,
        name=name,
        version_id=version_id,
        is_metadata=is_metadata,
        key_name=key_name,
        root_prefix=root_prefix,
    )


def decompose_key_artifact_metadata_store(
    key,
) -> CheckpointsPathComponents:
    data_object = __decompose_checkpoint_artifact_metadata_key(key)
    return _decompose_artifact_key(key, data_object)


def decompose_key_artifact_store(
    key,
) -> CheckpointsPathComponents:
    """
    Convert Key into Path Components.
    PATH COMPONENTS: mf.checkpoints/artifacts/<flow_name>/<step_name>/<scope>/<task_identifier>/<pathspec_hash>.<attempt>.<name>.<version_id>

    """
    data_object = __decompose_checkpoint_key(key)
    return _decompose_artifact_key(key, data_object)


def decompose_key_metadata_store(
    key,
) -> CheckpointsPathComponents:
    """
    Convert Key into Path Components.
    PATH COMPONENTS: mf.checkpoints/artifacts/<flow_name>/<step_name>/<scope>/<task_identifier>/<pathspec_hash>.<attempt>.<name>.<version_id>

    """
    _data = __decompose_checkpoint_metadata_key(key)
    root_prefix, flowname, runid, stepname, taskid, chckpt_id = (
        _data["root_prefix"],
        _data["flow_name"],
        _data["runid"],
        _data["step_name"],
        _data["taskid"],
        _data["key_name"],
    )
    if len(chckpt_id.split(".")) < 3:
        raise KeyNotCompatibleWithObjectException(
            key,
            "checkpoint_metadata",
            "key_name is not in the correct format. Expected 3 parts, got %s"
            % len(chckpt_id.split(".")),
        )

    is_metadata = chckpt_id.split(".")[-1] == "metadata"
    key_name = chckpt_id
    if is_metadata:
        key_name = ".".join(chckpt_id.split(".")[:-1])

    attempt, name, version_id = chckpt_id.split(".")[:3]
    return CheckpointsPathComponents(
        flow_name=flowname,
        step_name=stepname,
        scope=None,
        task_identifier=None,
        pathspec_hash=pathspec_hash("/".join([flowname, runid, stepname, taskid])),
        attempt=attempt,
        name=name,
        version_id=version_id,
        is_metadata=is_metadata,
        key_name=key_name,
        root_prefix=root_prefix,
    )


class CheckpointDatastore(DatastoreInterface):

    """
    Consisits of 3 main components:
    - `artifact_store`: This is where the checkpoint artifacts are stored.
        - This key to the checkpoint in this store becomes the "key for the checkpoint"
    - `metadata_store`: This is where the metadata of the checkpoint is stored it based on the currently executing task (path structure resembles that of a metaflow pathspec).
        - This store helps retrieve information about all the checkpoints stored for a task during the execution.
    - `artifact_metadatastore`: This is similar to the metadata store but holds a pathstructure similar to the artifact store.
        - this store helps reverse lookup the Checkpoint metadata object from the checkpoint key.
    """

    ROOT_PREFIX = CHECKPOINTS_STORAGE_PREFIX

    artifact_store: ObjectStorage = None

    metadata_store: ObjectStorage = None

    artifact_metadatastore: ObjectStorage = None

    _NAME_ENTROPY = None

    pathspec = None

    @property
    def metadata_ready(self):
        return self.metadata_store is not None

    @property
    def artifact_ready(self):
        return self.artifact_store is not None

    def set_root_prefix(self, root_prefix):
        self.ROOT_PREFIX = root_prefix
        if self.metadata_store is not None:
            self.metadata_store.set_full_prefix(root_prefix)
        if self.artifact_store is not None:
            self.artifact_store.set_full_prefix(root_prefix)
        if self.artifact_metadatastore is not None:
            self.artifact_metadatastore.set_full_prefix(root_prefix)

    @classmethod
    def init_read_store(
        cls,
        storage_backend: DataStoreStorage,
        pathspec=None,
        checkpoint_key=None,
    ):
        """
        This will initialize the datastore for reading.

        - If there is only the pathspec that's provided then it can mean the user is doing a list operations
        - if only the checkpoint_key is provided then it can mean the user is trying to load a specific checkpoint
        """
        datastore = cls()
        if pathspec is not None:
            datastore.metadata_store = ObjectStorage(
                storage_backend,
                root_prefix=cls.ROOT_PREFIX,
                path_components=[
                    "checkpoints",
                    METADATA_STORE_NAME,
                    *pathspec.split("/"),
                ],
            )
            datastore.pathspec = pathspec

            datastore._NAME_ENTROPY = pathspec_hash(pathspec)
        elif checkpoint_key is not None:
            _key_decomp = cls.decompose_key(checkpoint_key)
            _path_components = [
                _key_decomp.flow_name,
                _key_decomp.step_name,
                _key_decomp.scope,
                _key_decomp.task_identifier,
            ]
            datastore.artifact_store = ObjectStorage(
                storage_backend,
                root_prefix=cls.ROOT_PREFIX,
                path_components=["checkpoints", ARTIFACT_STORE_NAME] + _path_components,
            )
            datastore.artifact_metadatastore = ObjectStorage(
                storage_backend,
                root_prefix=cls.ROOT_PREFIX,
                path_components=["checkpoints", ARTIFACT_METADATA_STORE_NAME]
                + _path_components,
            )
            datastore._NAME_ENTROPY = _key_decomp.pathspec_hash
            datastore.set_root_prefix(_key_decomp.root_prefix)
        else:
            raise DatastoreReadInitException(
                "pathspec or checkpoint_key must be provided"
            )

        return datastore

    def create_key_name(
        self,
        *args,
    ):
        return ".".join([str(a) for a in args])

    @classmethod
    def init_write_store(
        cls,
        storage_backend: DataStoreStorage,
        pathspec,
        scope,
        task_identifier,
    ):
        if any([pathspec is None, scope is None, task_identifier is None]):
            raise DatastoreWriteInitException(
                "pathspec, scope, task_identifier must be provided"
            )
        datastore = cls()
        flow_name, runid, step_name, taskid = pathspec.split("/")

        datastore.metadata_store = ObjectStorage(
            storage_backend,
            root_prefix=cls.ROOT_PREFIX,
            path_components=[
                "checkpoints",
                METADATA_STORE_NAME,
                flow_name,
                runid,
                step_name,
                taskid,
            ],
        )
        datastore.pathspec = pathspec

        datastore._NAME_ENTROPY = pathspec_hash(pathspec)
        path_components = [
            flow_name,
            step_name,
            scope,
            task_identifier,
        ]
        datastore.artifact_store = ObjectStorage(
            storage_backend,
            root_prefix=cls.ROOT_PREFIX,
            path_components=["checkpoints", ARTIFACT_STORE_NAME] + path_components,
        )

        datastore.artifact_metadatastore = ObjectStorage(
            storage_backend,
            root_prefix=cls.ROOT_PREFIX,
            path_components=["checkpoints", ARTIFACT_METADATA_STORE_NAME]
            + path_components,
        )
        return datastore

    def save(
        self,
        local_paths: Union[str, List[str]],
        attempt,
        version_id,
        name=DEFAULT_NAME,
        metadata={},
        set_latest=True,
    ) -> CheckpointArtifact:

        if not (self.artifact_ready and self.metadata_ready):
            raise DatastoreNotReadyException(
                "Checkpoints Datastore is not ready for write operations"
            )

        _key = self.create_key_name(
            self._NAME_ENTROPY,
            attempt,
            name,
            version_id,
            # AT Write TIME pathspec hash is AlWAYS RESOLVABLE
            # because `metadata_store` is ALWAYS SET
        )
        full_checkpoint_url, key_path, file_size = self.artifact_store._save_tarball(
            _key,
            local_paths,
        )
        _metadata = dict(
            size=file_size,
            pathspec=self.pathspec,
            pathspec_hash=self._NAME_ENTROPY,
            attempt=attempt,
            key=key_path,
            type=CheckpointArtifact.TYPE,
            url=full_checkpoint_url,
            name=name,
            created_on=datetime.now().isoformat(),
            metadata=safe_serialize(metadata),
            storage_format="tar",
            creation_context="task",
            version_id=version_id,
        )

        _art_key = self.create_key_name(
            self._NAME_ENTROPY,
            attempt,
            name,
            version_id,
            "metadata",
        )
        _md_key = self.create_key_name(
            attempt,
            name,
            version_id,
            "metadata",
        )
        for _key, store in zip(
            [_art_key, _md_key],
            [
                self.artifact_metadatastore,
                self.metadata_store,
            ],
        ):
            store._save_metadata(_key, _metadata)
            if set_latest:
                store._save_metadata("latest", _metadata)
        return CheckpointArtifact.from_dict(_metadata)

    @allow_safe
    def latest(self, current_task=True) -> CheckpointArtifact:
        if current_task:
            if not self.metadata_ready:
                raise DatastoreNotReadyException(
                    "Checkpoint store is not ready to read the latest checkpoint in current task"
                )
            _md = self.metadata_store._load_metadata("latest")
        else:
            if not self.artifact_ready:
                raise DatastoreNotReadyException(
                    "Checkpoint store is not ready to read the latest checkpoint"
                )
            _md = self.artifact_metadatastore._load_metadata("latest")

        return CheckpointArtifact.from_dict(_md)

    def load(
        self,
        local_path,
        version_id,
        attempt,
        name,
    ):
        if not self.artifact_ready:
            raise ValueError("Datastore is not ready to load")
        _key = self.create_key_name(
            self._NAME_ENTROPY,
            attempt,
            name,
            version_id,
        )
        return self.artifact_store._load_tarball(_key, local_path)

    def load_metadata(
        self,
        attempt,
        version_id,
        name=DEFAULT_NAME,
    ) -> dict:
        if self.metadata_ready:
            return self.metadata_store._load_metadata(
                self.create_key_name(
                    attempt,
                    name,
                    version_id,
                    "metadata",
                )
            )
        elif self.artifact_ready:
            return self.artifact_metadatastore._load_metadata(
                self.create_key_name(
                    self._NAME_ENTROPY,
                    attempt,
                    name,
                    version_id,
                    "metadata",
                )
            )
        raise DatastoreNotReadyException("Datastore is not ready to load metadata")

    def list(
        self,
        name=None,
        attempt=None,
        within_task=True,
    ):

        if not within_task:
            if not self.artifact_ready:
                raise DatastoreNotReadyException(
                    "Checkpoint datastore is not ready to list all checkpoints"
                )
            return _recover_checkpoints(
                self.artifact_metadatastore,
                key_decomposer=decompose_key_artifact_metadata_store,
                name=name,
                attempt=attempt,
            )

        if not self.metadata_ready:
            raise DatastoreNotReadyException(
                "Checkpoint datastore is not ready to list checkpoints within the task"
            )
        return _recover_checkpoints(
            self.metadata_store,
            key_decomposer=decompose_key_metadata_store,
            name=name,
            attempt=attempt,
        )

    @classmethod
    def decompose_key(cls, key) -> CheckpointsPathComponents:
        return decompose_key_artifact_store(key)


def _recover_checkpoints(
    datastore: ObjectStorage,
    key_decomposer: Callable[[str], CheckpointsPathComponents],
    name: Optional[str] = None,
    attempt: Optional[int] = None,
) -> Iterator[CheckpointArtifact]:
    def _validate_name(info: CheckpointsPathComponents):
        if name is not None and info.name != name:
            return False
        return True

    def _filter_based_on_attempts(info: CheckpointsPathComponents):
        if attempt is None:
            return True
        if type(attempt) == int:
            return str(info.attempt) == str(attempt)
        return True

    # `datastore.list_paths` will have very different outputs based on the type of datastore.
    # - for CheckpointMetadataStore it will list ALL checkpoints within the task.
    # - for CheckpointArtifactMetadataStore it will list ALL checkpoints within scope/the task-identifier.
    #   - Meaning that for retrieving checkpoints during retries/re-executions can be a lot faster.

    # If we want we can even list within only-scope by using the
    # CheckpointArtifactStore it will list ALL checkpoints within the "scope"
    # (this can be astoundingly large if things are running inside foreaches)

    for path_tup in datastore.list_paths([""]):
        try:
            obj_info = key_decomposer(
                path_tup.key,
            )
        except KeyNotCompatibleWithObjectException as e:
            # this means that we hit an object that might be a reference but not something we are looking for
            continue
        if not _filter_based_on_attempts(obj_info):
            continue
        if not _validate_name(obj_info):
            continue
        metadata = datastore._load_metadata(
            datastore.create_key_name(obj_info.key_name, "metadata")
        )
        yield CheckpointArtifact.from_dict(metadata)
