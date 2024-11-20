import os
from typing import Optional
from metaflow.datastore.datastore_storage import DataStoreStorage
from ..datastore.core import ObjectStorage, DatastoreInterface, STORAGE_FORMATS
from .exceptions import ModelException
from ..exceptions import KeyNotFoundError
from ..datastructures import ModelArtifact
from collections import namedtuple
from ..exceptions import KeyNotCompatibleWithObjectException
from datetime import datetime
import re

MODELS_PEFFIX = "mf.models"


ARTIFACT_STORE_NAME = "artifacts"

METADATA_STORE_NAME = "metadata"

ARTIFACT_METADATA_STORE_NAME = "artifact_metadata"


# Every model key will be prefixed with a `/models/` and will have a structure of the form:
# <root_prefix>/models/artifacts/<model-uuid>
# The root_prefix can be set at the decorator level and the /models/ is a fixed prefix that also helps distinguish between the
# different types of objects such as `checkpoints`/`models` etc.
MODELS_KEY_PATTERN = re.compile(
    r"^(?P<root_prefix>.*)/models/artifacts/(?P<model_id>.*?)$"
)

MODELS_METADATA_KEY_PATTERN = re.compile(
    r"^(?P<root_prefix>.*)/models/metadata/(?P<flow_name>.*?)/(?P<runid>.*?)/(?P<step_name>.*?)/(?P<taskid>.*?)/(?P<key_name>.*?)$"
)

ModelPathComponents = namedtuple(
    "ModelPathComponents",
    [
        "model_uuid",
        "root_prefix",
    ],
)


def __decompose_model_artifact_key(key):
    match = MODELS_KEY_PATTERN.match(key)
    if match:
        return match.groupdict()
    else:
        raise KeyNotCompatibleWithObjectException(key, "model", "Regex doesn't match")


def decompose_model_artifact_key(key):
    decomposed_key = __decompose_model_artifact_key(key)
    root_prefix, model_id = decomposed_key["root_prefix"], decomposed_key["model_id"]
    return ModelPathComponents(
        model_uuid=model_id,
        root_prefix=root_prefix,
    )


class ModelDatastore(DatastoreInterface):

    ROOT_PREFIX = MODELS_PEFFIX

    _READ_UUID = None

    MODE = None  # "READ" or "WRITE

    artifact_store: ObjectStorage = None
    metadata_store: ObjectStorage = None
    artifact_metadatastore: ObjectStorage = None  # We don't need this store for now

    def save(
        self,
        artifact: ModelArtifact,
        file_path,
        storage_format=STORAGE_FORMATS.TAR,
    ):
        save_func = None
        if storage_format == STORAGE_FORMATS.TAR:
            save_func = self.artifact_store._save_tarball
        elif storage_format == STORAGE_FORMATS.FILES:
            save_func = self.artifact_store._save_objects
        else:
            raise ModelException(
                "Incompatible storage format %s for `current.model.save`"
                % (storage_format)
            )

        full_object_url, key_path, file_size = save_func(artifact.uuid, file_path)
        _metadata = artifact.to_dict()
        _metadata.update(
            {
                "size": file_size,
                "url": full_object_url,
                "key": key_path,
                "created_on": datetime.now().isoformat(),
            }
        )
        self.save_metadata(
            attempt=artifact.attempt,
            model_id=artifact.uuid,
            metadata=_metadata,
        )
        return ModelArtifact.from_dict(_metadata)

    def load_metadata(
        self,
        model_id,
    ):
        return self.artifact_metadatastore._load_metadata(
            self.artifact_metadatastore.create_key_name(model_id, "metadata")
        )

    def save_metadata(
        self,
        attempt,
        model_id,
        metadata,
    ):
        self.metadata_store._save_metadata(
            self.metadata_store.create_key_name(attempt, model_id, "metadata"), metadata
        )
        self.artifact_metadatastore._save_metadata(
            self.artifact_metadatastore.create_key_name(model_id, "metadata"),
            metadata,
        )

    def load(
        self,
        model_id,
        path,
    ):
        try:
            metadata = self.artifact_metadatastore._load_metadata(
                self.artifact_metadatastore.create_key_name(model_id, "metadata")
            )
            storage_format = metadata.get("storage_format")
            if storage_format is None:
                raise ModelException(
                    "Malformed metadata for model with id %s." % model_id
                )
        except KeyNotFoundError:
            raise ModelException(
                "Metadata related to model with id %s not found." % model_id
            )
        load_func = None
        if storage_format == STORAGE_FORMATS.TAR:
            load_func = self.artifact_store._load_tarball
        elif storage_format == STORAGE_FORMATS.FILES:
            load_func = self.artifact_store._load_objects
        else:
            raise ModelException(
                "Unsupported storage format for model with id %s." % model_id
            )
        load_func(
            model_id,
            path,
        )

    def list(self, *args, **kwargs):
        # It will list only based on task since all models
        # are stored under one prefix.
        raise NotImplementedError

    @classmethod
    def init_read_store(
        cls,
        storage_backend: DataStoreStorage,
        pathspec: Optional[str] = None,
        attempt: Optional[str] = None,
        model_key=None,
        *args,
        **kwargs,
    ):
        datastore = cls()
        if pathspec is not None:
            if not all([pathspec, attempt]):
                raise ValueError("pathspec and attempt are required")

            datastore.metadata_store = ObjectStorage(
                storage_backend,
                root_prefix=cls.ROOT_PREFIX,
                path_components=["models", METADATA_STORE_NAME] + pathspec.split("/"),
            )
            datastore.artifact_store = ObjectStorage(
                storage_backend,
                root_prefix=cls.ROOT_PREFIX,
                path_components=["models", ARTIFACT_STORE_NAME],
            )
            datastore.artifact_metadatastore = ObjectStorage(
                storage_backend,
                root_prefix=cls.ROOT_PREFIX,
                path_components=["models", ARTIFACT_METADATA_STORE_NAME],
            )

        elif model_key is not None:
            model_path_decomp = decompose_model_artifact_key(model_key)
            datastore.artifact_store = ObjectStorage(
                storage_backend,
                root_prefix=model_path_decomp.root_prefix,
                path_components=["models", ARTIFACT_STORE_NAME],
            )
            datastore.artifact_metadatastore = ObjectStorage(
                storage_backend,
                root_prefix=model_path_decomp.root_prefix,
                path_components=["models", ARTIFACT_METADATA_STORE_NAME],
            )
            datastore._READ_UUID = model_path_decomp.model_uuid
        datastore.MODE = "READ"
        return datastore

    @classmethod
    def decompose_key(cls, key):
        return decompose_model_artifact_key(key)

    @classmethod
    def init_write_store(
        cls, storage_backend: DataStoreStorage, pathspec: str, attempt, *args, **kwargs
    ):
        datastore = cls()
        if any([pathspec is None, attempt is None]):
            raise ValueError("pathspec and attempt are required")

        datastore.metadata_store = ObjectStorage(
            storage_backend,
            root_prefix=cls.ROOT_PREFIX,
            path_components=["models", METADATA_STORE_NAME] + pathspec.split("/"),
        )
        datastore.artifact_store = ObjectStorage(
            storage_backend,
            root_prefix=cls.ROOT_PREFIX,
            path_components=["models", ARTIFACT_STORE_NAME],
        )
        datastore.artifact_metadatastore = ObjectStorage(
            storage_backend,
            root_prefix=cls.ROOT_PREFIX,
            path_components=["models", ARTIFACT_METADATA_STORE_NAME],
        )
        datastore.MODE = "WRITE"
        return datastore
