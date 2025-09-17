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
    This object provides a thin, Metaflow-friendly layer over huggingface_hub's [snapshot_download](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/file_download#huggingface_hub.snapshot_download):

    - Snapshot references (persist-and-reuse): Use `current.huggingface_hub.snapshot_download(repo_id=..., ...)` to ensure a repo is available in the Metaflow datastore. If absent, it is downloaded once and saved; the call returns a reference dict you can store and load later (for example via `@model`).

    - On-demand local access (context manager):  Use `current.huggingface_hub.load(repo_id=..., [path=...], ...)` as a context manager to obtain a local filesystem path for immediate use. If the repo exists in the datastore, it is loaded from there; otherwise it is fetched from the Hugging Face Hub and then cached in the datastore. When `path` is omitted, a temporary directory is created and cleaned up automatically when the context exits. When `path` is provided, files are placed there and are not cleaned up by the context manager.

    Repos are cached in the datastore using the huggingface_hub.snapshot_download's arguments. The cache
    key may include: `repo_id`, `repo_type`, `revision`, `ignore_patterns`,
    and `allow_patterns` (see `cache_scope` for how keys are scoped).

    Examples
    --------
    ```python
    # Snapshot reference:
    ref = current.huggingface_hub.snapshot_download(
        repo_id="google-bert/bert-base-uncased",
        allow_patterns=["*.json"]
    )
    # Explicit Model Loading with Context manager:

    with current.huggingface_hub.load(
        repo_id="google-bert/bert-base-uncased",
        allow_patterns=["*.json"]
    ) as local_path:
        my_model = torch.load(os.path.join(local_path, "model.bin"))
    ```
    """

    _checkpointer: CurrentCheckpointer = None
    _loaded_models: "HuggingfaceLoadedModels"
    _cache_scope: str = "checkpoint"

    def __init__(self, logger) -> None:
        self._logger = logger

    def _override_based_on_cache_scope(
        self, checkpointer: CurrentCheckpointer, cache_scope
    ):
        overrides = []
        if cache_scope == "global":
            overrides = [
                "mf:internal",
                "huggingface-hub",
                "registry:global",
                "fully-global",
            ]
        elif cache_scope == "flow":
            overrides = [
                "mf:internal",
                "huggingface-hub",
                "registry:flow",
                checkpointer._flow_name,
            ]
        if len(overrides) > 0:
            checkpointer._default_checkpointer._checkpointer.override_path_components(
                path_components=overrides
            )
        checkpointer._default_checkpointer._checkpointer.set_root_prefix(
            HUGGINGFACE_HUB_ROOT_PREFIX
        )
        return checkpointer

    def _set_checkpointer(
        self, checkpointer: CurrentCheckpointer, temp_dir_root=None, cache_scope=None
    ):
        self._cache_scope = cache_scope
        self._checkpointer = self._override_based_on_cache_scope(
            checkpointer, cache_scope
        )
        self._loaded_models = HuggingfaceLoadedModels(
            checkpointer=self, logger=self._logger, temp_dir_root=temp_dir_root
        )

    @property
    def loaded(self) -> "HuggingfaceLoadedModels":
        """This property provides a dictionary-like interface to access the local paths of the huggingface repos specified in the `load` argument of the `@huggingface_hub` decorator."""
        return self._loaded_models

    def _scoped_name(self, repo_id, repo_type, hf_kwargs):
        # All possible parameters that need a new unique instance in the cache.
        keys_to_consider = [
            "revision",
            "ignore_patterns",
            "allow_patterns",
        ]
        final_keys = [repo_type, repo_id]

        if self._cache_scope != "checkpoint":
            # For non-'checkpoint' scopes ('flow' and 'global'), include selected Hugging Face
            # kwargs (revision, ignore_patterns, allow_patterns) in the cache key so that
            # materially different downloads resolve to distinct cache entries.
            #
            # For 'checkpoint' scope we intentionally keep the original cache key format
            # based only on repo_type/repo_id. Changing it to include the extra kwargs
            # would cause cache-busts and force re-downloads for existing users.
            # Not changing the cache name is acceptable because 'checkpoint'
            # scope is already highly granular (namespace/flow/step/foreach-index),
            # so omitting these kwargs does not harm correctness while
            # preserving backward compatibility.
            for k in keys_to_consider:
                if k in hf_kwargs and hf_kwargs[k] is not None:
                    final_keys.append(json.dumps(hf_kwargs[k], default=str))
        return self._cache_name("/".join(final_keys))

    def _cache_name(self, name):
        return hashlib.md5(name.encode()).hexdigest()[:10]

    def _warn(self, message):
        warning_message(
            message,
            self._logger,
            prefix="[@huggingface_hub]",
        )

    def _load_or_cache_model(self, **kwargs) -> dict:
        """
        This function will:
            1. return  a reference to the model it the datastore if found.
            2. otherwise it will download the model, save to datastore and return refernce to that checkpoint in the datastore.
        """
        from metaflow import current

        repo_name = kwargs["repo_id"]
        repo_type = kwargs.get("repo_type", "model")
        force_download = kwargs.get("force_download", False)
        chckpt_name = self._scoped_name(repo_name, repo_type, kwargs)
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
        Downloads a model from the Hugging Face Hub and caches it in the Metaflow datastore.
        It passes all parameters to the `huggingface_hub.snapshot_download` function.

        Returns
        -------
        dict
            A reference to the artifact saved to or retrieved from the Metaflow datastore.
        """
        if "repo_id" not in kwargs:
            raise ValueError("repo_id is required for snapshot_download")
        return self._load_or_cache_model(**kwargs)

    def load(self, repo_id=None, path=None, repo_type="model", **kwargs):
        """
        Context manager to load a Hugging Face repo (model/dataset) to a local path.

        - If `path` is provided, the repo is loaded there and the same path is yielded.
        - If `path` is not provided, a temporary directory is created, the repo is
          loaded there, the path is yielded, and the directory is cleaned up when
          the context exits.

        Parameters
        ----------
        repo_id : str, optional
            The Hugging Face repo ID. If omitted, must be provided via kwargs["repo_id"].
        path : str, optional
            Target directory to place files. If None, a temp directory is created.
        repo_type : str, optional
            Repo type (e.g., "model", "dataset"). Defaults to "model".
        **kwargs : Any
            Additional args forwarded to snapshot_download (e.g. force_download, revision,
            allow_patterns, ignore_patterns, etc.).

        Yields
        ------
        str
            Local filesystem path where the repo is available.
        """
        # Lazy import to avoid top-level dependency on contextlib
        from contextlib import contextmanager

        if repo_id is None:
            raise ValueError("repo_id is required for load()")
        kwargs_copy = kwargs.copy()
        repo_type = repo_type or "model"

        @contextmanager
        def _cm(resolved_repo_id, resolved_path, repo_type, kwargs_copy):
            created_tempdir = None
            try:
                if resolved_path is None:
                    # Build a deterministic prefix using the scoped cache name
                    chckpt_name = self._scoped_name(
                        resolved_repo_id, repo_type, {**kwargs_copy}
                    )
                    temp_dir_parent = self._loaded_models._temp_dir_root
                    created_tempdir = tempfile.TemporaryDirectory(
                        dir=temp_dir_parent, prefix=f"metaflow_hf_{chckpt_name}_"
                    )
                    target_path = created_tempdir.name
                else:
                    os.makedirs(resolved_path, exist_ok=True)
                    target_path = resolved_path

                model_path = self._loaded_models._load_model(
                    resolved_repo_id,
                    path=target_path,
                    repo_type=repo_type,
                    **kwargs_copy
                )
                yield model_path
            finally:
                if created_tempdir is not None:
                    created_tempdir.cleanup()

        return _cm(repo_id, path, repo_type, kwargs_copy)


class HuggingfaceLoadedModels:
    """Manages loaded HuggingFace models/datasets and provides access to their local paths.

    `current.huggingface_hub.loaded` provides a dictionary-like interface to access the local paths of the huggingface repos specified in the `load` argument of the `@huggingface_hub` decorator.

    Examples
    --------
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
        chckpt_name = self._checkpointer._scoped_name(repo_id, repo_type, kwargs)
        self._warn(
            "Downloading %s from huggingface to path %s" % (repo_id, path),
        )

        # Download directly to specified path
        kwargs["local_dir"] = path
        kwargs["local_dir_use_symlinks"] = False
        download_model_from_huggingface(repo_id=repo_id, repo_type=repo_type, **kwargs)

        self._warn(
            "Caching %s in the datastore" % (repo_id),
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

        self._warn(
            "Cached %s in the datastore with the key %s" % (repo_id, chckpt_ref["key"]),
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
        chckpt_name = self._checkpointer._scoped_name(repo_id, repo_type, kwargs)
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
        """
        Returns metadata information about all loaded models from Hugging Face Hub.
        This property provides access to the metadata of models that have been loaded
        via the `@huggingface_hub(load=...)` decorator. The metadata includes information
        such as model repository details, storage location, and any cached information
        from the datastore. Returns a dictionary where keys are model repository IDs and values are metadata
        dictionaries containing information about each loaded model.
        """
        return self._loaded_model_info

    def cleanup(self):
        for tempdir in self._temp_directories.values():
            tempdir.cleanup()


class HuggingfaceHubDecorator(CheckpointDecorator):
    """
    Decorator that helps cache, version, and store models/datasets from the Hugging Face Hub.

    Examples
    --------

    ```python
    # **Usage: creating references to models from the Hugging Face Hub that may be loaded in downstream steps**
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

    # **Usage: explicitly loading models at runtime from the Hugging Face Hub or from cache (from Metaflow's datastore)**
    @huggingface_hub
    @step
    def run_training(self):
        # Temporary directory (auto-cleaned on exit)
        with current.huggingface_hub.load(
            repo_id="google-bert/bert-base-uncased",
            allow_patterns=["*.bin"],
        ) as local_path:
            # Use files under local_path
            train_model(local_path)
            ...

    # **Usage: loading models directly from the Hugging Face Hub or from cache (from Metaflow's datastore)**

    @huggingface_hub(load=["mistralai/Mistral-7B-Instruct-v0.1"])
    @step
    def pull_model_from_huggingface(self):
        path_to_model = current.huggingface_hub.loaded["mistralai/Mistral-7B-Instruct-v0.1"]

    @huggingface_hub(load=[("mistralai/Mistral-7B-Instruct-v0.1", "/my-directory"), ("myorg/mistral-lora", "/my-lora-directory")])
    @step
    def finetune_model(self):
        path_to_model = current.huggingface_hub.loaded["mistralai/Mistral-7B-Instruct-v0.1"]
        # path_to_model will be /my-directory


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

    Parameters
    ----------
    temp_dir_root : str, optional
        The root directory that will hold the temporary directory where objects will be downloaded.

    cache_scope : str, optional
        The scope of the cache. Can be `checkpoint` / `flow` / `global`.
            - `checkpoint` (default): All repos are stored like objects saved by `@checkpoint`.
                i.e., the cached path is derived from the namespace, flow, step, and Metaflow foreach iteration.
                Any repo downloaded under this scope will only be retrieved from the cache when the step runs under the same namespace in the same flow (at the same foreach index).

            - `flow`: All repos are cached under the flow, regardless of namespace.
                i.e., the cached path is derived solely from the flow name.
                When to use this mode: (1) Multiple users are executing the same flow and want shared access to the repos cached by the decorator. (2) Multiple versions of a flow are deployed, all needing access to the same repos cached by the decorator.

            - `global`: All repos are cached under a globally static path.
                i.e., the base path of the cache is static and all repos are stored under it.
                When to use this mode:
                    - All repos from the Hugging Face Hub need to be shared by users across all flow executions.
            - Each caching scope comes with its own trade-offs:
                - `checkpoint`:
                    - Has explicit control over when caches are populated (controlled by the same flow that has the `@huggingface_hub` decorator) but ends up hitting the Hugging Face Hub more often if there are many users/namespaces/steps.
                    - Since objects are written on a `namespace/flow/step` basis, the blast radius of a bad checkpoint is limited to a particular flow in a namespace.
                - `flow`:
                    - Has less control over when caches are populated (can be written by any execution instance of a flow from any namespace) but results in more cache hits.
                    - The blast radius of a bad checkpoint is limited to all runs of a particular flow.
                    - It doesn't promote cache reuse across flows.
                - `global`:
                    - Has no control over when caches are populated (can be written by any flow execution) but has the highest cache hit rate.
                    - It promotes cache reuse across flows.
                    - The blast radius of a bad checkpoint spans every flow that could be using a particular repo.

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

        This object provides a thin, Metaflow-friendly layer over
        [huggingface_hub]'s `snapshot_download`:

        - Snapshot references (persist-and-reuse):
            Use `current.huggingface_hub.snapshot_download(repo_id=..., ...)` to
            ensure a repo is available in the Metaflow datastore. If absent, it is
            downloaded once and saved; the call returns a reference dict you can
            store and load later (for example via `@model`).

        - On-demand local access (context manager):
            Use `current.huggingface_hub.load(repo_id=..., [path=...], ...)` as a
            context manager to obtain a local filesystem path for immediate use.
            If the repo exists in the datastore, it is loaded from there;
            otherwise it is fetched from the Hugging Face Hub and then cached in
            the datastore. When `path` is omitted, a temporary directory is
            created and cleaned up automatically when the context exits. When
            `path` is provided, files are placed there and are not cleaned up by
            the context manager.

        Repos are cached in the datastore using the huggingface_hub.snapshot_download's arguments. The cache
        key may include: `repo_id`, `repo_type`, `revision`, `ignore_patterns`,
        and `allow_patterns` (see `cache_scope` for how keys are scoped).

        > Usage Styles
        ```python
        # Snapshot reference:
        ref = current.huggingface_hub.snapshot_download(
            repo_id="google-bert/bert-base-uncased",
            allow_patterns=["*.json"]
        )

        # Explicit Model Loading with Context manager:
        with current.huggingface_hub.load(
            repo_id="google-bert/bert-base-uncased",
            allow_patterns=["*.json"]
        ) as local_path:
            my_model = torch.load(os.path.join(local_path, "model.bin"))
        ```

        @@ Returns
        ----------
        HuggingfaceRegistry

    """

    defaults = {
        "temp_dir_root": None,
        "load": None,  # Can be list of repo_ids or dicts with repo_id and other params
        "cache_scope": "checkpoint",  # can be `checkpoint` / `flow` / `namespace` / `global`
    }

    name = "huggingface_hub"

    def step_init(
        self, flow, graph, step_name, decorators, environment, flow_datastore, logger
    ):
        self._flow_datastore = flow_datastore
        self._logger = logger
        self._chkptr = None
        self._collector_thread = None

        if self.attributes.get("cache_scope") not in ["checkpoint", "flow", "global"]:
            raise ValueError(
                f"Invalid cache_scope for @huggingface_hub: {self.attributes.get('cache_scope')}. "
                "Must be 'checkpoint', 'flow', or 'global'"
            )

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
        self._cache_scope = self.attributes.get("cache_scope")

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

        self._registry._set_checkpointer(
            self._chkptr,
            cache_scope=self.attributes.get("cache_scope"),
            temp_dir_root=self.attributes.get("temp_dir_root"),
        )
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
