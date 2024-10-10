import hashlib
import shutil
from ..checkpoints.decorator import (
    CheckpointDecorator,
    CurrentCheckpointer,
    warning_message,
)

HUGGINGFACE_HUB_ROOT_PREFIX = "mf.huggingface_hub"


def download_model_from_huggingface(**kwargs):
    import huggingface_hub
    from glob import glob
    import os

    try:
        huggingface_hub.snapshot_download(**kwargs)
        pass
    except Exception as e:
        raise e


class HuggingfaceRegistry:
    _checkpointer: CurrentCheckpointer = None

    def __init__(self, logger) -> None:
        self._logger = logger

    def _set_checkpointer(self, checkpointer: CurrentCheckpointer):
        self._checkpointer = checkpointer
        self._checkpointer._default_checkpointer._checkpointer.set_root_prefix(
            HUGGINGFACE_HUB_ROOT_PREFIX
        )

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
        chckpts = list(self._checkpointer.list(name=chckpt_name, within_task=False))
        if len(chckpts) > 0 and not force_download:
            return chckpts[0]

        _kwargs = kwargs.copy()
        _kwargs["local_dir"] = self._checkpointer.directory
        _kwargs["local_dir_use_symlinks"] = False
        self._warn(
            "Downloading %s from huggingface" % repo_name,
        )
        download_model_from_huggingface(**_kwargs)
        self._warn(
            "Downloaded %s from huggingface. Saving checkpoint" % repo_name,
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
        self._warn(
            "huggingface checkpoint for %s saved" % repo_name,
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


class HuggingfaceHubDecorator(CheckpointDecorator):
    """
    Decorator that helps cache, version and store models/datasets from huggingface hub.

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
    """

    defaults = {}

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
        return {"scope": "namespace", "load_policy": "none"}

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
