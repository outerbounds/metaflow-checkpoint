from metaflow.exception import MetaflowException
from metaflow.datastore.datastore_storage import DataStoreStorage
from metaflow.plugins.datastores.local_storage import LocalStorage
import os
from functools import partial
import sys

from typing import Iterator, List, Union, Tuple, Optional
import os
import json
from io import BytesIO
from collections import namedtuple
from datetime import datetime
import tempfile

from ..utils.tar_utils import create_tarball_on_disk, extract_tarball
from ..utils.general import safe_serialize
from .storage_injections import STORAGE_INJECTIONS
from ..exceptions import KeyNotFoundError
import json

# mimic a subset of the behavior of the Metaflow S3Object
DatastoreBlob = namedtuple("DatastoreBlob", "blob url path")
ListPathResult = namedtuple("ListPathResult", "full_url key")
COMPRESSION_METHOD = None


def warning_message(
    message, logger=None, ts=False, prefix="[@checkpoint][artifact-store]"
):
    msg = "%s %s" % (prefix, message)
    if logger:
        logger(msg, timestamp=ts, bad=True)
    else:
        print(msg, file=sys.stderr)


def allow_safe(func):
    def wrapper(*args, **kwargs):
        _safe = False
        if kwargs.pop("safe", False):
            _safe = True
        try:
            return func(*args, **kwargs)
        except KeyNotFoundError as e:
            if _safe:
                return None
            raise e
        except ValueError as e:
            if _safe:
                return None
            raise e

    return wrapper


class ObjectStorage(object):
    """
    `ObjectStorage` wraps around the DataStoreStorage object and provides the lower level
    storage APIs needed by the subsequent classes. This datastore's main function that
    distingushes it from the DataStoreStorage object is that it manages a generalizable
    path structure over different storage backends (s3, azure, gs, local etc.).

    This object will be used to create multiple "Logical" datastores for constructs like
    `Checkpoints`, `Models` etc.

    Usage
    -----
    ```
    storage = ObjectStorage(
        storage_backend,
        root_prefix = "mf.checkpoints",
        path_components = ["artifacts", "MyFlow", "step_a", "cd2312rd12d", "x5avasdhtsdfqw"]
    )
    ```
    """

    # Prefix that is created using the root_prefix and the path_components
    FULL_PREFIX = None

    def __init__(
        self,
        storage_backend: DataStoreStorage,
        root_prefix: str,  # `mf.checkpoints`
        path_components: List,  # artifacts/flow_name/step_name/scope/task_identifier
    ):
        self._backend: DataStoreStorage = storage_backend
        self._storage_root = self.resolve_root(self._backend.TYPE)
        self._path_components = path_components
        self.set_full_prefix(root_prefix)
        self._inject_methods_to_storage_backend()

    @property
    def path_components(self):
        return self._path_components

    def _inject_methods_to_storage_backend(
        self,
    ):
        """
        The DataStoreStorage object is a core Metaflow object and
        it doesn't contain some a method for saving file. This method
        injects the save_file method into the DataStoreStorage object
        based on the storage backend.
        """

        if self._backend.TYPE not in STORAGE_INJECTIONS:
            raise NotImplementedError(
                "Storage backend %s not supported with @checkpoint."
                % self._backend.TYPE
            )

        method = STORAGE_INJECTIONS[self._backend.TYPE]
        setattr(self._backend, "save_file", partial(method, self._backend))

    def set_full_prefix(self, root_prefix):
        self.FULL_PREFIX = os.path.join(root_prefix, "/".join(self._path_components))

    @staticmethod
    def resolve_root(storage_type):
        if storage_type == "s3":
            from metaflow.metaflow_config import DATASTORE_SYSROOT_S3

            return DATASTORE_SYSROOT_S3
        elif storage_type == "azure":
            from metaflow.metaflow_config import DATASTORE_SYSROOT_AZURE

            return DATASTORE_SYSROOT_AZURE
        elif storage_type == "gs":
            from metaflow.metaflow_config import DATASTORE_SYSROOT_GS

            return DATASTORE_SYSROOT_GS
        elif storage_type == "local":
            return LocalStorage.get_datastore_root_from_config(
                lambda x: x, create_on_absent=True
            )
        else:
            raise NotImplementedError(
                "Datastore is not support backend %s" % (storage_type)
            )

    def full_base_url(self, prefix=None):
        if prefix is None:
            return self._storage_root
        return os.path.join(self._storage_root, prefix)

    def create_key_name(self, *args):
        return ".".join([str(arg) for arg in args])

    @property
    def datastore_root(self):
        assert self.FULL_PREFIX is not None, "FULL_PREFIX is not set."
        path_prefix = self.FULL_PREFIX
        # The FULL_PREFIX property is used to construct the root based on the
        # storage type and will be used to construct the path to the
        # objects in the datastore during read/write operations.

        # If the storage type is s3, then the root is just the `FULL_PREFIX`
        # because the S3Storage object uses as S3 client which has the
        # root already set.
        if self._backend.TYPE != "s3":
            path_prefix = self.full_base_url(prefix=path_prefix)

        return self.FULL_PREFIX

    def resolve_key_relative_path(self, key):
        return os.path.join(self.FULL_PREFIX, key)

    def resolve_key_full_url(self, key):
        return os.path.join(self.full_base_url(), self.resolve_key_relative_path(key))

    def resolve_key_path(self, key):
        return os.path.join(self.datastore_root, key)

    # TODO: [CORE-CLEANUP]: Make this HYPER Efficient!
    def put(
        self, key: str, obj: Union[str, bytes], overwrite: bool = False
    ) -> str:  # Path to where it got stored.
        """
        TODO : [THIS IS TERRIBLY INEFFICIENT]
        """
        "Put a single object into the datastore's `key` index."
        _save_object = None
        if isinstance(obj, bytes):
            _save_object = BytesIO(obj)
        else:
            _save_object = BytesIO(obj.encode("utf-8"))
        _path = self.resolve_key_path(key)
        self._backend.save_bytes(
            [(_path, _save_object)],
            overwrite=overwrite,
        )
        return self.resolve_key_relative_path(key)

    def put_files(self, key_paths: List[Tuple[str, str]], overwrite=False):
        results = []
        for key, path in key_paths:
            _kp = self.resolve_key_path(key)
            self._backend.save_file(_kp, path, overwrite=overwrite)
            results.append(self.resolve_key_relative_path(key))
        return results

    # TODO: [CORE-CLEANUP]: Make this HYPER inefficient!
    def get(self, key) -> DatastoreBlob:
        """
        TODO : [THIS IS TERRIBLY INEFFICIENT]
        """
        "Get a single object residing in the datastore's `key` index."
        datastore_url = self.resolve_key_path(key)
        with self._backend.load_bytes([datastore_url]) as get_results:
            for key, path, meta in get_results:
                if path is not None:
                    with open(path, "rb") as f:
                        blob_bytes = f.read()
                        return DatastoreBlob(
                            blob=blob_bytes,
                            url=datastore_url,
                            path=path,
                            # text=blob_bytes.decode("utf-8"),
                        )
                else:
                    raise KeyNotFoundError(datastore_url)

    def get_file(self, key):
        datastore_url = self.resolve_key_path(key)
        return self._backend.load_bytes([datastore_url])

    def list_paths(self, keys) -> Iterator[ListPathResult]:
        "List all objects in the datastore's `keys` index."

        def _full_url_convert(lcr_path):
            if self._backend.TYPE == "s3":
                return os.path.join(self.full_base_url(), lcr_path)
            return lcr_path

        def _relative_url_convert(lcr_path):
            if self._backend.TYPE == "s3":
                return lcr_path
            np = lcr_path.replace(self.full_base_url(), "")
            if np.startswith("/"):
                return np[1:]
            return np

        keys = [self.resolve_key_path(key) for key in keys]
        for list_content_result in self._backend.list_content(keys):
            yield ListPathResult(
                full_url=_full_url_convert(list_content_result.path),
                key=_relative_url_convert(list_content_result.path),
            )

    def _save_tarball(
        self,
        key,
        local_paths: Union[str, List[str]],
    ):
        suffix = ".tar"
        with tempfile.NamedTemporaryFile(suffix=suffix) as temp_file:
            create_tarball_on_disk(
                local_paths,
                output_filename=temp_file.name,
                compression_method=None,
            )
            file_size = os.path.getsize(temp_file.name)
            _ = self.put_files([(key, temp_file.name)], overwrite=True)
            return (
                self.resolve_key_full_url(key),
                self.resolve_key_relative_path(key),
                file_size,
            )

    def _load_tarball(
        self,
        key,
        local_path,
    ):

        with self.get_file(key) as get_results:
            for key, path, meta in get_results:
                if path is None:
                    raise KeyNotFoundError(key)
                extract_tarball(path, local_path, compression_method=None)
        return local_path

    def _save_metadata(
        self,
        key,
        metadata,
    ):
        return self.put(key, json.dumps(safe_serialize(metadata)), overwrite=True)

    def _load_metadata(self, key):
        return json.loads(self.get(key).blob)


class DatastoreInterface:
    """
    This is the root abstraction used by any underlying datastores like Checkpoint/Model etc.
    to create the saving/loading mechanics using multiple ObjectStores.

    The inherited classes require the following implemented:
        - `ROOT_PREFIX` : The root prefix for the datastore such as `mf.checkpoints` or `mf.models`.
        - `init_read_store` : The method to initialize the read store; The inheriting class can compose together any number of `BaseDatastore` objects
        - `init_write_store` : The method to initialize the write store; The inheriting class can compose together any number of `BaseDatastore` objects
        - `save` : The method to save the artifact data.
        - `load` : The method to load the artifact data.
        - `save_metadata` : The method to save the metadata about the artifact.
        - `load_metadata` : The method to load the metadata about the artifact.
        - `list` : The method to list all the artifacts.
    """

    ROOT_PREFIX = None

    def set_root_prefix(self, root_prefix):
        """
        This function helps ensuring that the root prefix of the datastore
        and it's underlying ObjectStores can change trivially.
        """
        raise NotImplementedError

    # ! INHERIT AND OVERRIDE
    def save(self, *args, **kwargs):
        raise NotImplementedError

    # ! INHERIT AND OVERRIDE
    def load(self, *args, **kwargs):
        raise NotImplementedError

    # ! INHERIT AND OVERRIDE
    def save_metadata(self, *args, **kwargs):
        raise NotImplementedError

    # ! INHERIT AND OVERRIDE
    def load_metadata(self, *args, **kwargs):
        raise NotImplementedError

    # ! INHERIT AND OVERRIDE
    def list(self, *args, **kwargs):
        raise NotImplementedError

    # ! INHERIT AND OVERRIDE
    @classmethod
    def init_read_store(cls, storage_backend: DataStoreStorage, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def init_write_store(cls, storage_backend: DataStoreStorage, *args, **kwargs):
        raise NotImplementedError


def resolve_root(datastore_type):
    return ObjectStorage.resolve_root(datastore_type)
