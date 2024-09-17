import os
import tempfile
from typing import Dict, List, Optional, Tuple, Union


from ..exceptions import KeyNotFoundError
from .model_storage import ModelDatastore
from .exceptions import LoadingException
from ..datastore.utils import safe_serialize
from ..utils.general import get_path_size, unit_convert, warning_message
from ..utils.identity_utils import MAX_HASH_LEN
from ..utils.serialization_handler import TarHandler, SERIALIZATION_HANDLERS
from ..datastructures import ModelArtifact, Factory, MetaflowDataArtifactReference
from uuid import uuid4

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
            storage_backend,
            artifact_references,
        )

    def _warn(self, message):
        warning_message(message, logger=self._logger, ts=False, prefix="[@model]")

    @property
    def info(self):
        return self._loaded_model_info

    def _init_loaded_models(self, flow, storage_backend, artifact_references):
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
                raise LoadingException(f"Artifact {artifact_name} not found in flow")
            if (
                type(artifact) == str
            ):  # If its a string then it means its a key reference
                _hydrated_artifact = Factory.load_metadata_from_key(
                    artifact, storage_backend
                )
            elif (
                isinstance(artifact, MetaflowDataArtifactReference)
                or type(artifact) == dict
            ):
                _hydrated_artifact = Factory.hydrate(artifact)
            else:
                raise LoadingException(
                    f"Artifact {artifact_name} is not a valid artifact reference"
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
            self._load_artifact(
                artifact_name, _hydrated_artifact, storage_backend, path
            )

    def _load_artifact(self, artifact_name, artifact, storage_backend, path):
        try:
            _hydrated_artifact = Factory.hydrate(artifact)
            if path is not None:
                os.makedirs(path, exist_ok=True)
                self._loaded_models[artifact_name] = path
            else:
                self._temp_directories[artifact_name] = tempfile.TemporaryDirectory(
                    dir=self._temp_dir_root, prefix=f"metaflow_models_{artifact_name}_"
                )
                self._loaded_models[artifact_name] = self._temp_directories[
                    artifact_name
                ].name

            Factory.load(
                _hydrated_artifact,
                self._loaded_models[artifact_name],
                storage_backend,
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

    def _cleanup(self):
        for _tempdir in self._temp_directories.values():
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
    def loaded(self):
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
        return self._datastore.save(artifact, path)

    def save(
        self,
        path,
        label=None,
        metadata=None,
    ):
        model_artifact = self._serialize_to_datastore(
            path,
            TarHandler.TYPE,  # TODO [POST RELEASE]: clean this up to even support directories
            TarHandler.STORAGE_FORMAT,
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

        if reference is None:
            raise ValueError(
                "reference arguement to `current.model.load` cannot be None"
            )
        if path is None:
            raise ValueError("`current.model.load` requires a path to load the model")

        if reference is not None:
            if type(reference) == dict or isinstance(
                reference, MetaflowDataArtifactReference
            ):
                Factory.load(
                    Factory.hydrate(reference),
                    path,
                    self._storage_backend,
                )
            elif type(reference) == str:
                Factory.load_from_key(reference, path, self._storage_backend)
        # TODO [POST-RELEASE] : Implement the model_id loading


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
