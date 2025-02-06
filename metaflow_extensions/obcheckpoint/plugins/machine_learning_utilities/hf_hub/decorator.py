import hashlib
import time
import shutil
import json
from ..checkpoints.decorator import (
    CheckpointDecorator,
    CurrentCheckpointer,
    warning_message,
)
import sys
import os
import tempfile
from metaflow.metadata_provider import MetaDatum

HUGGINGFACE_HUB_ROOT_PREFIX = "mf.huggingface_hub"


def get_tqdm_class():
    from tqdm.std import tqdm as std_tqdm

    class TqdmExt(std_tqdm):
        def __init__(self, *args, **kwargs):
            kwargs["file"] = sys.stdout
            kwargs["desc"] = (
                "[@huggingface_hub][HF-Download]"
                if not kwargs.get("desc", None)
                else "[@huggingface_hub][HF-Download] " + kwargs["desc"]
            )
            kwargs["leave"] = False
            super().__init__(*args, **kwargs)

    return TqdmExt


def show_progress():
    import logging
    from tqdm.contrib.logging import tqdm_logging_redirect

    tqdm_logger = logging.getLogger("tqdm")
    tqdm_logger.setLevel(logging.INFO)
    return tqdm_logging_redirect()


def download_model_from_huggingface(**kwargs):
    import huggingface_hub
    from glob import glob
    import os

    try:
        kwargs.pop("tqdm_class", None)
        with show_progress():
            huggingface_hub.snapshot_download(**kwargs, tqdm_class=get_tqdm_class())
    except Exception as e:
        raise e


class HuggingfaceRegistry:
    """
    This object provides syntactic sugar
    over [huggingface_hub](https://github.com/huggingface/huggingface_hub)'s
    [snapshot_download](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/file_download#huggingface_hub.snapshot_download) function.
    The `current.huggingface_hub.snapshot_download` function downloads objects from huggingface hub and saves them to the Metaflow's datastore under the
    `<repo_type>/<repo_id>` name. The `repo_type` is by default `model` and can be overriden by passing the `repo_type` parameter to the `snapshot_download` function.
    """

    _checkpointer: CurrentCheckpointer = None
    _loaded_models = None

    def __init__(self, logger) -> None:
        self._logger = logger

    def _set_checkpointer(self, checkpointer: CurrentCheckpointer, temp_dir_root=None):
        self._checkpointer = checkpointer
        self._checkpointer._default_checkpointer._checkpointer.set_root_prefix(
            HUGGINGFACE_HUB_ROOT_PREFIX
        )
        self._loaded_models = HuggingfaceLoadedModels(
            checkpointer=self, logger=self._logger, temp_dir_root=temp_dir_root
        )

    @property
    def loaded(self):
        return self._loaded_models

    def _cache_name(self, name):
        return hashlib.md5(name.encode()).hexdigest()[:10]

    def _warn(self, message):
        warning_message(
            message,
            self._logger,
            prefix="[@huggingface_hub]",
        )

    def _load_or_cache_model(self, **kwargs) -> dict:
        from metaflow import current

        repo_name = kwargs["repo_id"]
        repo_type = kwargs.get("repo_type", "model")
        force_download = kwargs.get("force_download", False)
        chckpt_name = self._cache_name("{}/{}".format(repo_type, repo_name))
        chckpts = list(self._checkpointer.list(name=chckpt_name, full_namespace=True))
        if len(chckpts) > 0 and not force_download:
            return chckpts[0]

        _kwargs = kwargs.copy()
        _kwargs["local_dir"] = self._checkpointer.directory
        _kwargs["local_dir_use_symlinks"] = False
        start_time = time.time()
        self._warn(
            "Downloading %s from huggingface to path %s"
            % (repo_name, _kwargs["local_dir"]),
        )
        download_model_from_huggingface(**_kwargs)
        download_completion_time = time.time()
        download_time = str(round(download_completion_time - start_time, 2))
        self._warn(
            "Downloaded %s from huggingface in %s seconds. Saving checkpoint to datastore."
            % (repo_name, download_time),
        )
        chckpt_ref = self._checkpointer.save(
            name=chckpt_name,
            metadata={
                "repo_id": repo_name,
                "registry": "huggingface",
                "repo_type": repo_type,
            },
            # We set this statically to files here because
            # it will be a lot more performant than tar mode.
            storage_format="files",
        )
        _save_time = str(round(time.time() - download_completion_time, 2))
        self._warn(
            "huggingface checkpoint for %s saved to datastore in %s seconds"
            % (repo_name, _save_time),
        )
        # wipe the directory so that it's unpolluted for another function call.
        shutil.rmtree(self._checkpointer.directory)
        return chckpt_ref

    def snapshot_download(self, **kwargs) -> dict:
        """
        Downloads a model from huggingface hub and cache's it to the Metaflow's datastore.
        It passes down all the parameters to the `huggingface_hub.snapshot_download` function.

        Returns
        -------
        dict
            A reference to the artifact that was saved/retrieved from the Metaflow's datastore.
        """
        if "repo_id" not in kwargs:
            raise ValueError("repo_id is required for snapshot_download")
        return self._load_or_cache_model(**kwargs)


class HuggingfaceLoadedModels:
    """Manages loaded Hugging Face models and provides access to their local paths.

    This class is accessed through `current.huggingface_hub.loaded` and provides a dictionary-like
    interface to access the local paths of the huggingface repos specified in the `load` argument of the `@huggingface_hub` decorator.

    Usage:
    ------
    ```python
    # Basic loading and access
    @huggingface_hub(load=["mistralai/Mistral-7B-Instruct-v0.1"])
    @step
    def my_step(self):
        # Access the local path of a loaded model
        model_path = current.huggingface_hub.loaded["mistralai/Mistral-7B-Instruct-v0.1"]

        # Check if a model is loaded
        if "mistralai/Mistral-7B-Instruct-v0.1" in current.huggingface_hub.loaded:
            print("Model is loaded!")

    # Custom path and advanced loading
    @huggingface_hub(load=[
        ("mistralai/Mistral-7B-Instruct-v0.1", "/custom/path"),  # Specify custom path
        {
            "repo_id": "org/model-name",
            "force_download": True,  # Force fresh download
            "repo_type": "dataset"   # Load dataset instead of model
        }
    ])
    @step
    def another_step(self):
        # Models are available at specified paths
        pass
    ```
    """

    def __init__(
        self, checkpointer: "HuggingfaceRegistry", logger, temp_dir_root=None
    ) -> None:
        from metaflow import current

        self._namespace = current.namespace
        self._checkpointer = checkpointer
        self._logger = logger
        self._loaded_models = {}
        self._loaded_model_info = {}
        self._temp_directories = {}
        self._temp_dir_root = temp_dir_root

    def _warn(self, message):
        warning_message(
            message,
            self._logger,
            prefix="[@huggingface_hub]",
        )

    def _get_or_create_model_path(self, repo_id, chckpt_name, path=None):
        """Handle model path creation/retrieval logic"""
        if path is not None:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
            return path

        # Create temp directory for model
        if self._temp_dir_root is not None:
            if not os.path.exists(self._temp_dir_root):
                os.makedirs(self._temp_dir_root, exist_ok=True)

        self._temp_directories[repo_id] = tempfile.TemporaryDirectory(
            dir=self._temp_dir_root, prefix=f"metaflow_hf_{chckpt_name}_"
        )
        return self._temp_directories[repo_id].name

    def _download_and_cache_model(self, repo_id, repo_type, path, **kwargs):
        """Download model from HF Hub and cache in datastore"""
        chckpt_name = self._checkpointer._cache_name("{}/{}".format(repo_type, repo_id))
        self._warn(
            "Downloading %s from huggingface to path %s" % (repo_id, path),
        )

        # Download directly to specified path
        kwargs["local_dir"] = path
        kwargs["local_dir_use_symlinks"] = False
        download_model_from_huggingface(repo_id=repo_id, repo_type=repo_type, **kwargs)

        self._warn(
            "Cached %s in datastore for namespace %s" % (repo_id, self._namespace),
        )
        # Save/cache in datastore
        chckpt_ref = self._checkpointer._checkpointer.save(
            name=chckpt_name,
            metadata={
                "repo_id": repo_id,
                "registry": "huggingface",
                "repo_type": repo_type,
                "kwargs": kwargs,
            },
            path=path,
            storage_format="files",
        )
        return chckpt_ref

    def _load_from_datastore(self, chckpt_ref, path):
        """Load model from datastore to specified path"""
        self._warn(
            "Loading model from datastore to %s. Model being loaded: %s"
            % (path, chckpt_ref["key"])
        )
        self._checkpointer._checkpointer.load(chckpt_ref, path=path)

    def _load_model(self, repo_id, path=None, repo_type="model", **kwargs):
        """
        Load a model from either the datastore or Hugging Face Hub.

        Args:
            repo_id (str): The Hugging Face model repo ID
            path (str, optional): Specific path to load the model into
            repo_type (str, optional): Type of repo (model/dataset)
            **kwargs: Additional arguments passed to snapshot_download
        """
        chckpt_name = self._checkpointer._cache_name("{}/{}".format(repo_type, repo_id))
        chckpts = list(
            self._checkpointer._checkpointer.list(name=chckpt_name, full_namespace=True)
        )

        # Setup model path and load if needed
        model_path = self._get_or_create_model_path(repo_id, chckpt_name, path=path)
        # Get or download model reference
        if len(chckpts) == 0 or kwargs.get("force_download", False):
            self._warn(
                f"Model {repo_id} not found in datastore, downloading from HuggingFace Hub"
            )
            chckpt_ref = self._download_and_cache_model(
                repo_id, repo_type, model_path, **kwargs
            )
        else:  # This means that more than 1 checkpoint exists
            chckpt_ref = chckpts[0]
            self._load_from_datastore(chckpt_ref, model_path)

        # Update tracking
        self._loaded_models[repo_id] = model_path
        self._loaded_model_info[repo_id] = chckpt_ref
        return model_path

    def __getitem__(self, key):
        if key not in self._loaded_models:
            raise KeyError(f"Model {key} not found in loaded models")
        return self._loaded_models[key]

    def __contains__(self, key):
        return key in self._loaded_models

    @property
    def info(self):
        return self._loaded_model_info

    def cleanup(self):
        for tempdir in self._temp_directories.values():
            tempdir.cleanup()


class HuggingfaceHubDecorator(CheckpointDecorator):
    """
    Decorator that helps cache, version and store models/datasets from huggingface hub.

    Parameters
    ----------
    temp_dir_root : str, optional
        The root directory that will hold the temporary directory where objects will be downloaded.

    load: Union[List[str], List[Tuple[Dict, str]], List[Tuple[str, str]], List[Dict], None]
        The list of repos (models/datasets) to load.

        Loaded repos can be accessed via `current.huggingface_hub.loaded`. If load is set, then the following happens:

        - If repo (model/dataset) is not found in the datastore:
            - Downloads the repo from Hugging Face Hub to a temporary directory (or uses specified path) for local access
            - Stores it in Metaflow's datastore (s3/gcs/azure etc.) with a unique name based on repo_type/repo_id
                - All HF models loaded for a `@step` will be cached separately under flow/step/namespace.

        - If repo is found in the datastore:
            - Loads it directly from datastore to local path (can be temporary directory or specified path)


    MF Add To Current
    -----------------
    huggingface_hub -> metaflow_extensions.obcheckpoint.plugins.machine_learning_utilities.hf_hub.decorator.HuggingfaceRegistry
        The `@huggingface_hub` injects a `huggingface_hub` object into the `current` object. This object provides syntactic sugar
        over [huggingface_hub](https://github.com/huggingface/huggingface_hub)'s
        [snapshot_download](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/file_download#huggingface_hub.snapshot_download) function.
        The `current.huggingface_hub.snapshot_download` function downloads objects from huggingface hub and saves them to the Metaflow's datastore under the
        `<repo_type>/<repo_id>` name. The `repo_type` is by default `model` and can be overriden by passing the `repo_type` parameter to the `snapshot_download` function.


        Usage:
        ------

        **Usage: creating references of models from huggingface that may be loaded in downstream steps**
        ```python
            @huggingface_hub
            @step
            def pull_model_from_huggingface(self):
                # `current.huggingface_hub.snapshot_download` downloads the model from the Hugging Face Hub
                # and saves it in the backend storage based on the model's `repo_id`. If there exists a model
                # with the same `repo_id` in the backend storage, it will not download the model again. The return
                # value of the function is a reference to the model in the backend storage.
                # This reference can be used to load the model in the subsequent steps via `@model(load=["llama_model"])`

                self.model_id = "mistralai/Mistral-7B-Instruct-v0.1"
                self.llama_model = current.huggingface_hub.snapshot_download(
                    repo_id=self.model_id,
                    allow_patterns=["*.safetensors", "*.json", "tokenizer.*"],
                )
                self.next(self.train)
        ```

        **Usage: loading models directly from huggingface hub or from cache (from metaflow's datastore)**
        ```python
            @huggingface_hub(load=["mistralai/Mistral-7B-Instruct-v0.1"])
            @step
            def pull_model_from_huggingface(self):
                path_to_model = current.huggingface_hub.loaded["mistralai/Mistral-7B-Instruct-v0.1"]
        ```

        ```python
            @huggingface_hub(load=[("mistralai/Mistral-7B-Instruct-v0.1", "/my-directory"), ("myorg/mistral-lora, "/my-lora-directory")])
            @step
            def finetune_model(self):
                path_to_model = current.huggingface_hub.loaded["mistralai/Mistral-7B-Instruct-v0.1"]
                # path_to_model will be /my-directory
        ```

        ```python
            # Takes all the arguments passed to `snapshot_download`
            # except for `local_dir`
            @huggingface_hub(load=[
                {
                    "repo_id": "mistralai/Mistral-7B-Instruct-v0.1",
                },
                {
                    "repo_id": "myorg/mistral-lora",
                    "repo_type": "model",
                },
            ])
            @step
            def finetune_model(self):
                path_to_model = current.huggingface_hub.loaded["mistralai/Mistral-7B-Instruct-v0.1"]
                # path_to_model will be /my-directory
        ```
    """

    defaults = {
        "temp_dir_root": None,
        "load": None,  # Can be list of repo_ids or dicts with repo_id and other params
    }

    name = "huggingface_hub"

    def step_init(
        self, flow, graph, step_name, decorators, environment, flow_datastore, logger
    ):
        self._flow_datastore = flow_datastore
        self._logger = logger
        self._chkptr = None
        self._collector_thread = None

        self._registry = HuggingfaceRegistry(logger)

    def _resolve_settings(self):
        return {
            "load_policy": "none",
            "temp_dir_root": self.attributes.get("temp_dir_root"),
        }

    def task_pre_step(
        self,
        step_name,
        task_datastore,
        metadata,
        run_id,
        task_id,
        flow,
        graph,
        retry_count,
        max_user_code_retries,
        ubf_context,
        inputs,
    ):
        super().task_pre_step(
            step_name,
            task_datastore,
            metadata,
            run_id,
            task_id,
            flow,
            graph,
            retry_count,
            max_user_code_retries,
            ubf_context,
            inputs,
        )
        self._runid, self._step_name, self._task_id = run_id, step_name, task_id
        self._metadata_provider = metadata
        # Handle loading models if load argument is provided
        load_models = self.attributes.get("load")
        if load_models is not None:
            if not isinstance(load_models, list):
                raise ValueError(
                    f"Invalid load argument format: {load_models}. "
                    "Must be a list of strings, tuples, or dicts"
                )

            for model_spec in load_models:
                if isinstance(model_spec, str):
                    # List[str] - repo_ids
                    self._registry.loaded._load_model(model_spec)
                elif isinstance(model_spec, tuple):
                    # Validate tuple length
                    if len(model_spec) != 2:
                        raise ValueError(
                            f"Invalid tuple format: {model_spec}. "
                            "Must be (dict, path) or (repo_id, path)"
                        )
                    spec, path = model_spec

                    if isinstance(spec, dict):
                        # List[Tuple[Dict, str]] - (model config, path)
                        if "repo_id" not in spec:
                            raise ValueError(
                                f"Missing repo_id in dict specification: {spec}"
                            )
                        repo_id = spec.pop("repo_id")
                        self._registry.loaded._load_model(repo_id, path=path, **spec)
                    elif isinstance(spec, str):
                        # List[Tuple[str, str]] - (repo_id, path)
                        self._registry.loaded._load_model(spec, path=path)
                    else:
                        raise ValueError(
                            f"Invalid tuple first element: {spec}. "
                            "Must be dict or string"
                        )
                elif isinstance(model_spec, dict):
                    # List[Dict] - model configs
                    if "repo_id" not in model_spec:
                        raise ValueError(
                            f"Missing repo_id in dict specification: {model_spec}"
                        )
                    repo_id = model_spec.pop("repo_id")
                    self._registry.loaded._load_model(repo_id, **model_spec)
                else:
                    raise ValueError(
                        f"Invalid model specification format: {model_spec}. "
                        "Must be string, tuple (dict/str, path), or dict"
                    )

        self.loaded_models_data = self._registry.loaded.info
        model_keys = [
            model_ref["key"] for _, model_ref in self.loaded_models_data.items()
        ]
        # Register metadata about the models that are loaded so that we can
        # use it for book keeping in the future.
        if len(model_keys) > 0:
            self._metadata_provider.register_metadata(
                self._runid,
                self._step_name,
                self._task_id,
                [
                    MetaDatum(
                        "hf-loaded-models",
                        json.dumps({"keys": model_keys}),
                        "model-registry",
                        tags=[
                            "attempt_id:%s" % str(retry_count),
                        ],
                    )
                ],
            )

    def _setup_current(self):
        from metaflow import current

        self._registry._set_checkpointer(self._chkptr)
        current._update_env({"huggingface_hub": self._registry})

    def task_decorate(
        self, step_func, flow, graph, retry_count, max_user_code_retries, ubf_context
    ):
        return step_func

    def task_post_step(
        self, step_name, flow, graph, retry_count, max_user_code_retries
    ):
        self._chkptr.cleanup()

    def task_exception(
        self, exception, step_name, flow, graph, retry_count, max_user_code_retries
    ):
        self._chkptr.cleanup()
