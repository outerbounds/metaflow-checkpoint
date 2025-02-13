from metaflow.exception import MetaflowException
from metaflow.datastore.datastore_storage import DataStoreStorage
from metaflow.plugins.datastores.local_storage import LocalStorage
import os
from functools import partial
import sys
import shutil

from typing import Iterator, List, Union, Tuple, Optional
import os
import json
from pathlib import Path
from io import BytesIO
from collections import namedtuple
from datetime import datetime
import tempfile

from ..utils.tar_utils import create_tarball_on_disk, extract_tarball
from ..utils.general import safe_serialize
from .storage_injections import (
    STORAGE_INJECTIONS_SINGLE_FILE_SAVE,
    STORAGE_INJECTIONS_MULTIPLE_FILE_SAVE,
    STORAGE_INJECTIONS_LOAD_FILES,
)
from ..exceptions import KeyNotFoundError
import json

# mimic a subset of the behavior of the Metaflow S3Object
DatastoreBlob = namedtuple("DatastoreBlob", "blob url path")
ListPathResult = namedtuple("ListPathResult", "full_url key")
COMPRESSION_METHOD = None


class STORAGE_FORMATS:
    TAR = "tar"
    FILES = "files"


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


def _extract_paths_inside_directory(directory):
    """
    This function will extract all the file-paths inside a directory
    and returns the following:
        - A list of relative paths of the files inside the directory
        - The total size of the directory in bytes.
        - The absolute paths of the files inside the directory.
    """
    import os

    relative_paths = []
    absolute_paths = []
    total_size = 0

    directory = os.path.abspath(directory)

    for root, dirs, files in os.walk(directory):
        for file in files:
            abs_path = os.path.join(root, file)
            rel_path = os.path.relpath(abs_path, directory)
            relative_paths.append(rel_path)
            absolute_paths.append(abs_path)
            total_size += os.path.getsize(abs_path)

    return relative_paths, total_size, absolute_paths


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
        self._storage_root = self._backend.datastore_root
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
        needed_methods = [
            "save_file",
            "save_files",
            "load_files",
        ]

        if all([hasattr(self._backend, m) for m in needed_methods]):
            return

        if self._backend.TYPE not in STORAGE_INJECTIONS_SINGLE_FILE_SAVE:
            raise NotImplementedError(
                "Storage backend %s not supported with @checkpoint."
                % self._backend.TYPE
            )

        method = STORAGE_INJECTIONS_SINGLE_FILE_SAVE[self._backend.TYPE]
        setattr(self._backend, "save_file", partial(method, self._backend))

        if self._backend.TYPE not in STORAGE_INJECTIONS_MULTIPLE_FILE_SAVE:
            raise NotImplementedError(
                "Storage backend %s not supported with @checkpoint."
                % self._backend.TYPE
            )
        method = STORAGE_INJECTIONS_MULTIPLE_FILE_SAVE[self._backend.TYPE]
        setattr(self._backend, "save_files", partial(method, self._backend))

        if self._backend.TYPE not in STORAGE_INJECTIONS_LOAD_FILES:
            raise NotImplementedError(
                "Storage backend %s not supported with @checkpoint."
                % self._backend.TYPE
            )
        method = STORAGE_INJECTIONS_LOAD_FILES[self._backend.TYPE]
        setattr(self._backend, "load_files", partial(method, self._backend))

    def set_full_prefix(self, root_prefix):
        self.FULL_PREFIX = os.path.join(root_prefix, "/".join(self._path_components))

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
        if self._backend.TYPE != "s3" and self._backend.TYPE != "s3-compatible":
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

    def put_file(self, key: str, path: str, overwrite=False):
        _kp = self.resolve_key_path(key)
        self._backend.save_file(_kp, path, overwrite=overwrite)
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

    def list_paths(self, keys, recursive=False) -> Iterator[ListPathResult]:
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
        if not recursive:
            for list_content_result in self._backend.list_content(keys):
                yield ListPathResult(
                    full_url=_full_url_convert(list_content_result.path),
                    key=_relative_url_convert(list_content_result.path),
                )
        else:

            def _list_content_recursive(x_keys):
                _keys = []
                for list_content_result in self._backend.list_content(x_keys):
                    if list_content_result.is_file:
                        _keys.append(
                            ListPathResult(
                                full_url=_full_url_convert(list_content_result.path),
                                key=_relative_url_convert(list_content_result.path),
                            )
                        )
                    else:
                        _keys.extend(
                            _list_content_recursive([list_content_result.path])
                        )
                return _keys

            for x in _list_content_recursive(keys):
                yield x

    def _save_objects(
        self,
        key: str,
        local_path: str,
    ):
        # this function will just store all the objects as is to the datastore without taring them
        # up. This is useful when we are storing a single file or a directory that doesn't need to be
        # tarred up.
        object_size = None
        relative_paths_inside_key = []
        absolute_paths_of_files = []
        if os.path.isdir(local_path):
            (
                relative_paths_inside_key,
                object_size,
                absolute_paths_of_files,
            ) = _extract_paths_inside_directory(local_path)
        else:
            object_size = os.path.getsize(local_path)
            relative_paths_inside_key.append(
                os.path.basename(local_path),
            )
            absolute_paths_of_files.append(local_path)

        # at this point, since we already have a key, we need to ensure that we are placing everything
        # under the key. This is because the key will be used as the identifier and we will end up extracting
        # the contents of the object inside the key to the local path.
        key_paths = [
            (self.resolve_key_path(os.path.join(key, rel_path)), abs_path)
            for rel_path, abs_path in zip(
                relative_paths_inside_key, absolute_paths_of_files
            )
        ]
        self._backend.save_files(key_paths, overwrite=True)
        return (
            self.resolve_key_full_url(key),
            self.resolve_key_relative_path(key),
            object_size,
        )

    def _save_tarball(
        self,
        key,
        local_path: str,
    ):
        suffix = ".tar"
        with tempfile.NamedTemporaryFile(suffix=suffix) as temp_file:
            create_tarball_on_disk(
                local_path,
                output_filename=temp_file.name,
                compression_method=None,
            )
            file_size = os.path.getsize(temp_file.name)
            _ = self.put_file(key, temp_file.name, overwrite=True)
            return (
                self.resolve_key_full_url(key),
                self.resolve_key_relative_path(key),
                file_size,
            )

    def _load_objects(
        self,
        key,
        # The key here is ideally a key from the datastore that we want to load
        # This can be a checkpoint key or a model key.
        local_directory,
        # Its assumed here that the local path where everything is getting
        # extracted is a directory.
    ):
        list_path_results = list(self.list_paths([key], recursive=True))
        # print(list_path_results)
        keys = [p.key for p in list_path_results]
        # We directly call load bytes here because `self.get_file` will add the root of the datastore
        # to the path and we don't want that.
        with self._backend.load_files(keys) as get_results:
            for list_key, path, meta in get_results:
                if path is None:
                    continue

                path_within_dir = os.path.relpath(list_key, self.resolve_key_path(key))
                # We need to construct the right path over here based on the
                # where the key is present in the object.
                # Figure a relative path from the end of the key
                # to the actual file/directory within it.
                # We do this because we want the entire directory structure
                # to be preserved when we download the objects on local.
                move_to_path = os.path.join(local_directory, path_within_dir)
                Path(move_to_path).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(
                    path,
                    move_to_path,
                )

        return local_directory

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

    def __str__(self) -> str:
        return f"""
        ObjectStorage:
            - Backend: {self._backend.TYPE}
            - Path Components: {self._path_components}
            - Full Prefix: {self.FULL_PREFIX}
            - Datastore Root: {self._backend.datastore_root}

        """


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
