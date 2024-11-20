import os
import time
import tempfile
from typing import Dict, List, Optional, Tuple, Union

from ..datastore.core import STORAGE_FORMATS
from ..exceptions import (
    KeyNotFoundError,
    KeyNotCompatibleException,
    IncompatibleObjectTypeException,
)
from .model_storage import ModelDatastore
from .exceptions import LoadingException
from ..datastore.utils import safe_serialize
from ..utils.general import get_path_size, unit_convert, warning_message
from ..utils.identity_utils import MAX_HASH_LEN, safe_hash
from ..utils.serialization_handler import TarHandler, SERIALIZATION_HANDLERS
from ..datastructures import ModelArtifact, Factory, MetaflowDataArtifactReference
from uuid import uuid4
import shutil

OBJECT_MAX_SIZE_ALLOWED_FOR_ARTIFACT = unit_convert(3, "GB", "B")


def create_write_store(pathspec, attempt, storage_backend) -> ModelDatastore:
    return ModelDatastore.init_write_store(
        storage_backend=storage_backend,
        pathspec=pathspec,
        attempt=attempt,
    )


def create_read_store(
    storage_backend,
    model_key=None,
    pathspec=None,
    attempt=None,
) -> ModelDatastore:
    return ModelDatastore.init_read_store(
        storage_backend=storage_backend,
        pathspec=pathspec,
        model_key=model_key,
        attempt=attempt,
    )


def _get_id(label):
    return "_".join([str(label), uuid4().hex.replace("-", "")])


class LoadedModels:
    """
    A class that loads models from the datastore and stores them in a temporary directory.
    This class helps manage all the models loaded via `@model(load=...)` decorator and
    `current.model.load` method.

    It is exposed via the `current.model.loaded` property. It is a dictionary like object
    that stores the loaded models in a temporary directory. The keys of the dictionary are the
    artifact names and the values are the paths to the temporary directories where the models are stored.

    Usage:
    ------
    ```python
        @model(load=["model_key", "chckpt_key"])
        @step
        def mid_step(self):
            import os
            os.listdir(current.model.loaded["model_key"])
            os.listdir(current.model.loaded["chckpt_key"])
    ```
    """

    def __init__(
        self,
        storage_backend,
        flow,
        artifact_references: Union[
            List[str],
            List[Tuple[str, Union[str, None]]],
            str,
        ],  # List of artifact names or Dict of artifact names and their subsequent paths
        best_effort=False,
        temp_dir_root=None,
        mode="eager",  # eager / lazy (not supported yet as interface needs work.)
        logger=None,
        # lazy mode will only load the model when it is accessed and not present
        # eager mode will load all the models at the start
    ) -> None:
        self._storage_backend = storage_backend
        self._loaded_model_info = {}
        self._loaded_models = {}
        self._temp_directories = {}
        self._temp_dir_root = temp_dir_root
        self._best_effort = best_effort
        # self._flow = flow
        self._logger = logger
        self._artifact_loading_mode = mode
        # self._artifact_names = artifact_names
        self._init_loaded_models(
            flow,
            artifact_references,
        )

    def _warn(self, message):
        warning_message(message, logger=self._logger, ts=False, prefix="[@model]")

    @property
    def info(self):
        return self._loaded_model_info

    def _init_loaded_models(self, flow, artifact_references):
        _art_refs = []
        if type(artifact_references) == str:
            _art_refs = [(artifact_references, None)]
        if type(artifact_references) == list:
            # Its assumed over here that the artifact_references have been validly parse until now
            for art_ref in artifact_references:
                if type(art_ref) == str:
                    _art_refs.append((art_ref, None))
                elif type(art_ref) == tuple:
                    _art_refs.append(art_ref)

        _hydrated_artifacts = []
        for artifact_name, path in _art_refs:
            artifact = getattr(flow, artifact_name, None)
            if artifact is None:
                raise LoadingException(
                    f"Artifact {artifact_name} not found in flow. Please check if `self.{artifact_name}` is defined in the flow"
                )
            if (
                type(artifact) == str
            ):  # If its a string then it means its a key reference
                try:
                    _hydrated_artifact = Factory.load_metadata_from_key(
                        artifact, self._storage_backend
                    )
                except KeyNotFoundError:
                    raise LoadingException(
                        "Artifact %s not found in the datastore" % artifact_name
                    )
                except KeyNotCompatibleException:
                    raise LoadingException(
                        "Artifact %s not compatible with any of the supported objects"
                        % artifact_name
                    )
            elif (
                isinstance(artifact, MetaflowDataArtifactReference)
                or type(artifact) == dict
            ):
                try:
                    _hydrated_artifact = Factory.hydrate(artifact)
                except ValueError:
                    raise LoadingException(
                        "Artifact %s is not a valid artifact reference. Accepted types are `str`, `dict` or `MetaflowDataArtifactReference`"
                        % artifact_name
                    )
            else:
                raise LoadingException(
                    f"Artifact %s is not a valid type. Accepted types are `str`, `dict` or `MetaflowDataArtifactReference`"
                    % artifact_name
                )
            _hydrated_artifacts.append((_hydrated_artifact, artifact_name, path))

        for (
            _hydrated_artifact,
            artifact_name,
            path,
        ) in _hydrated_artifacts:
            self._loaded_model_info[artifact_name] = _hydrated_artifact.to_dict()
            self._warn(
                "Loading Artifact with name `%s` [type:%s] with key: %s"
                % (artifact_name, _hydrated_artifact.TYPE, _hydrated_artifact.key)
            )
            start_time = time.time()
            self._load_artifact(artifact_name, _hydrated_artifact, path)
            end_time = time.time()
            self._warn(
                "Loaded artifact `%s` in %s seconds"
                % (
                    artifact_name + "[type:%s]" % _hydrated_artifact.TYPE,
                    str(round(end_time - start_time, 2)),
                )
            )

    def _add_model(self, artifact, path=None) -> str:
        _hydrated_artifact = None
        if type(artifact) == str:  # If its a string then it means its a key reference
            try:
                _hydrated_artifact = Factory.load_metadata_from_key(
                    artifact, self._storage_backend
                )
            except KeyNotFoundError:
                raise LoadingException(
                    f"`reference` argument string given to `current.model.load` could not be found. Reference doesn't exists in the datastore"
                )
            except KeyNotCompatibleException:
                raise LoadingException(
                    f"`reference` argument string given to `current.model.load` is not compatible with the supported artifact types"
                )
        elif (
            isinstance(artifact, MetaflowDataArtifactReference)
            or type(artifact) == dict
        ):
            try:
                _hydrated_artifact = Factory.hydrate(artifact)
            except (ValueError, IncompatibleObjectTypeException):
                raise LoadingException(
                    f"`reference` argument given to `current.model.load` is not of a valid type. Accepted types are `str`, `dict` or `MetaflowDataArtifactReference`"
                )
        else:
            raise LoadingException(
                f"`reference` argument given to `current.model.load` is not of a valid type. Accepted types are `str`, `dict` or `MetaflowDataArtifactReference`"
            )
        # Since `loaded_model_info` is user facing, we keep the object-key as the key instead
        # of the hash of the key (like we do for the temp directories)
        self._loaded_model_info[_hydrated_artifact.key] = _hydrated_artifact.to_dict()

        art_key_name = safe_hash(_hydrated_artifact.key)[:6]
        try:
            start_time = time.time()
            self._load_artifact(art_key_name, _hydrated_artifact, path)
            end_time = time.time()
            self._warn(
                "Loaded artifact `%s` in %s seconds"
                % (
                    art_key_name + "[type:%s]" % _hydrated_artifact.TYPE,
                    str(round(end_time - start_time, 2)),
                )
            )
        except LoadingException:
            raise LoadingException(
                f"Artifact reference specified in {_hydrated_artifact.key} not found in the datastore"
            )
        return self._loaded_models[art_key_name]

    def _load_artifact(self, artifact_name, artifact, path):
        try:
            _hydrated_artifact = Factory.hydrate(artifact)
            if path is not None:
                os.makedirs(path, exist_ok=True)
                self._loaded_models[artifact_name] = path
            else:
                # Ensure that if a tempdir root path is provided and nothing
                # exists then we end up creating that path. This helps ensure
                # that rouge paths with arbirary Filesystems get created before
                # temp dirs exists.
                if self._temp_dir_root is not None:
                    if not os.path.exists(self._temp_dir_root):
                        os.makedirs(self._temp_dir_root, exist_ok=True)

                self._temp_directories[artifact_name] = tempfile.TemporaryDirectory(
                    dir=self._temp_dir_root, prefix=f"metaflow_models_{artifact_name}_"
                )
                self._loaded_models[artifact_name] = self._temp_directories[
                    artifact_name
                ].name
            Factory.load(
                _hydrated_artifact,
                self._loaded_models[artifact_name],
                self._storage_backend,
            )
        except KeyNotFoundError:
            if self._best_effort:
                if artifact_name in self._temp_directories:
                    self._temp_directories[artifact_name].cleanup()

                self._loaded_models[artifact_name] = None
            raise LoadingException(
                f"Artifact reference specified in {artifact_name} not found in the datastore"
            )

    def __getitem__(self, key):
        if key not in self._loaded_models:
            raise KeyError(f"Model {key} not found in loaded models")

        return self._loaded_models[key]

    def __contains__(self, key):  # Artifact name in loaded models
        return key in self._loaded_models

    def __iter__(self):
        return iter({k: v.name for k, v in self._loaded_models.items()})

    def __len__(self):
        return len(self._loaded_models)

    def cleanup(self, artifact_name):
        if artifact_name not in self._loaded_models:
            raise KeyError(f"Model {artifact_name} not found in loaded models")

        if artifact_name in self._temp_directories:
            self._temp_directories[artifact_name].cleanup()
            del self._temp_directories[artifact_name]
        else:
            model_path = self._loaded_models[artifact_name]
            if model_path is not None:
                shutil.rmtree(model_path)

        del self._loaded_models[artifact_name]
        del self._loaded_model_info[artifact_name]

    def _cleanup(self):
        for name, _tempdir in self._temp_directories.items():
            if _tempdir is not None:
                _tempdir.cleanup()


class ModelSerializer:

    _LOADED_MODELS = None

    _saved_models = []

    def __init__(self, pathspec, attempt, storage_backend) -> None:
        self._datastore = create_write_store(pathspec, attempt, storage_backend)
        self._storage_backend = storage_backend
        self._pathspec = pathspec
        self._attempt = attempt
        flowname, _, stepname, _ = self._pathspec.split("/")
        self._default_label = "_".join([flowname, stepname])

    def _set_loaded_models(self, loaded_models: LoadedModels):
        self._LOADED_MODELS = loaded_models

    @property
    def loaded(self) -> "LoadedModels":
        return self._LOADED_MODELS

    def _get_model_artifact(
        self, metadata, serializer, storage_format, model_uuid, label=None
    ):
        return ModelArtifact.create(
            pathspec=self._pathspec,
            attempt=self._attempt,
            model_uuid=model_uuid,
            metadata=safe_serialize({} if metadata is None else metadata),
            source="task",
            serializer=serializer,
            storage_format=storage_format,
            label=label,
        )

    def _serialize_to_datastore(
        self, path, serializer: str, storage_format: str, metadata=None, label=None
    ):
        if label is None:
            label = self._default_label
        model_uuid = _get_id(label=label)
        artifact = self._get_model_artifact(
            metadata=metadata,
            serializer=serializer,
            storage_format=storage_format,
            model_uuid=model_uuid,
        )
        return self._datastore.save(artifact, path, storage_format=storage_format)

    def save(
        self,
        path,
        label=None,
        metadata=None,
        storage_format=STORAGE_FORMATS.TAR,
    ):
        if storage_format not in [
            STORAGE_FORMATS.TAR,
            STORAGE_FORMATS.FILES,
        ]:
            raise ValueError(
                "Unsupported storage format. Expected one of `%s` got %s"
                % (
                    "or ".join([STORAGE_FORMATS.TAR, STORAGE_FORMATS.FILES]),
                    storage_format,
                )
            )

        model_artifact = self._serialize_to_datastore(
            path,
            serializer=storage_format,
            storage_format=storage_format,
            metadata=metadata,
            label=label,
        ).to_dict()
        self._saved_models.append(model_artifact)
        return model_artifact

    def load(
        self,
        reference: Union[str, MetaflowDataArtifactReference, dict],
        path: Optional[str] = None,
    ):
        """
        Load a model/checkpoint from the datastore to a temporary directory or a specified path.

        Returns:
        --------
        str : The path to the temporary directory where the model is loaded.
        """

        if reference is None:
            raise ValueError(
                "reference arguement to `current.model.load` cannot be None"
            )
        # TODO [POST-RELEASE] : Implement the model_id loading
        return self._LOADED_MODELS._add_model(reference, path)


def _load_model(
    storage_backend,
    model_key=None,
    path=None,
):
    # Needed by factory methods to load the model.
    model_store = create_read_store(storage_backend, model_key=model_key)
    model_store.load(
        model_store._READ_UUID,
        path,
    )


def _load_model_metadata(
    storage_backend,
    model_key=None,
):
    # Needed by factory methods to load the model.
    model_store = create_read_store(storage_backend, model_key=model_key)
    return model_store.load_metadata(
        model_store._READ_UUID,
    )
