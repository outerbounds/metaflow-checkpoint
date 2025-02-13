# from functools import wraps

import os
from metaflow.metadata_provider.metadata import MetadataProvider
from .core import ModelSerializer, LoadedModels
from ..card_utils import CardDecoratorInjector
from .cards.model_card import ModelListRefresher, ModelsCollector
from ..datastore.task_utils import storage_backend_from_flow

# from .lineage import checkpoint_load_related_metadata, trace_lineage
from ..datastructures import ModelArtifact

# from .cards import CardDecoratorInjector, create_checkpoint_card, null_card
from metaflow.decorators import StepDecorator
from metaflow.metadata_provider import MetaDatum
from typing import List, Dict, Union, Tuple, Optional, Callable, TYPE_CHECKING
from functools import wraps, partial
import tempfile
from ..datastore.decorator import set_datastore_context
from metaflow.flowspec import INTERNAL_ARTIFACTS_SET
import json

if TYPE_CHECKING:
    import metaflow


def warning_message(message, logger=None, ts=False, prefix="[@model]"):
    msg = "%s %s" % (prefix, message)
    if logger:
        logger(msg, timestamp=ts, bad=True)


class ModelDecorator(StepDecorator):
    """
    Enables loading / saving of models within a step.


    Parameters
    ----------
    load : Union[List[str],str,List[Tuple[str,Union[str,None]]]], default: None
        Artifact name/s referencing the models/checkpoints to load. Artifact names refer to the names of the instance variables set to `self`.
        These artifact names give to `load` be reference objects or reference `key` string's from objects created by:
        - `current.checkpoint`
        - `current.model`
        - `current.huggingface_hub`

        If a list of tuples is provided, the first element is the artifact name and the second element is the path the artifact needs be unpacked on
        the local filesystem. If the second element is None, the artifact will be unpacked in the current working directory.
        If a string is provided, then the artifact corresponding to that name will be loaded in the current working directory.

    temp_dir_root : str, default: None
        The root directory under which `current.model.loaded` will store loaded models


    MF Add To Current
    -----------------
    model -> metaflow_extensions.obcheckpoint.plugins.machine_learning_utilities.modeling_utils.core.ModelSerializer
        The object used for loading / saving models.
        `current.model` exposes a `save` method to save models and a `load` method to load models.
        `current.model.loaded` exposes the paths to the models loaded via the `load` argument in the @model decorator
        or models loaded via `current.model.load`.

        Usage (Saving a model):
        -------

        ```
        @model
        @step
        def train(self):
            # current.model.save returns a dictionary reference to the model saved
            self.my_model = current.model.save(
                path_to_my_model,
                label="my_model",
                metadata={
                    "epochs": 10,
                    "batch-size": 32,
                    "learning-rate": 0.001,
                }
            )
            self.next(self.test)

        @model(load="my_model")
        @step
        def test(self):
            # `current.model.loaded` returns a dictionary of the loaded models
            # where the key is the name of the artifact and the value is the path to the model
            print(os.listdir(current.model.loaded["my_model"]))
            self.next(self.end)
        ```

        Usage (Loading models):
        -------

        ```
        @step
        def train(self):
            # current.model.load returns the path to the model loaded
            checkpoint_path = current.model.load(
                self.checkpoint_key,
            )
            model_path = current.model.load(
                self.model,
            )
            self.next(self.test)
        ```


        @@ Returns
        -------
        ModelSerializer
            The object used for loading / saving models.
    """

    name = "model"

    defaults = {
        "load": None,  # This can be a list of models to load from artifact references.
        "best_effort_load": False,  # If True, it will ignore missing artifacts and continue.
        "temp_dir_root": None,
    }

    METADATUM_TYPE = "model-registry"

    def step_init(
        self, flow, graph, step_name, decorators, environment, flow_datastore, logger
    ):
        self._flow = flow
        self._flow_datastore = flow_datastore
        self._logger = logger
        self._serializer = None
        self.deco_injector = CardDecoratorInjector()
        self.deco_injector.attach_card_decorator(
            flow,
            step_name,
            ModelListRefresher.CARD_ID,
            "blank",
            refresh_interval=2,
        )
        self._collector_thread = None
        # We add to INTERNAL_ARTIFACTS_SET here because the decorator adds internal artifacts to the
        # flow. Adding to INTERNAL_ARTIFACTS_SET avoids having any crashes when `merge_artifacts`
        # is called.
        INTERNAL_ARTIFACTS_SET.add("_task_stored_models")

    def _save_metadata_about_models(self, explicit_model_list: List[Dict]):
        _md_list = []
        self._flow._task_stored_models = explicit_model_list
        # We do `-20` to ensure that we only store the last 20 models.
        saved_models = []
        for idx, md in enumerate(explicit_model_list[-20:]):
            model_art = ModelArtifact.hydrate(md)
            saved_models.append(model_art.key)

        if len(saved_models) > 0:
            _md_list.append(
                MetaDatum(
                    field="saved-models",
                    value=json.dumps({"keys": saved_models}),
                    type=self.METADATUM_TYPE,
                    tags=[
                        "attempt_id:%s" % str(model_art.attempt),
                    ],
                )
            )

        self._metadata_provider.register_metadata(
            self._runid, self._step_name, self._task_id, _md_list
        )

    def task_pre_step(
        self,
        step_name,
        task_datastore,
        metadata: MetadataProvider,
        run_id,
        task_id,
        flow,
        graph,
        retry_count,
        max_user_code_retries,
        ubf_context,
        inputs,
    ):
        set_datastore_context(flow, metadata, run_id, step_name, task_id, retry_count)
        self._metadata_provider = metadata
        self._runid = run_id
        self._task_id = task_id
        self._step_name = step_name
        self._setup_current(flow, retry_count)

    def task_exception(
        self, exception, step_name, flow, graph, retry_count, max_user_code_retries
    ):
        if self._collector_thread is not None:
            self._collector_thread.stop()

    def task_post_step(
        self, step_name, flow, graph, retry_count, max_user_code_retries
    ):
        if self._serializer is None:
            return
        self._save_metadata_about_models(self._serializer._saved_models)
        if self._collector_thread:
            self._collector_thread.stop()

    def task_decorate(
        self, step_func, flow, graph, retry_count, max_user_code_retries, ubf_context
    ):
        from metaflow import current

        self._collector_thread = ModelsCollector(
            ModelListRefresher(current.model.loaded.info),
            interval=1,
        )
        # Wrap the step_func in a function that will write to current.card["checkpoint_info"] the lineage card
        # and then call the step_func.
        def _wrapped_step_func(_collector_thread, *args, **kwargs):
            _collector_thread.start()
            try:
                return step_func(*args, **kwargs)
            finally:
                _collector_thread.stop()

        return partial(_wrapped_step_func, self._collector_thread)

    def _setup_current(self, flow, retry_count):
        from metaflow import current

        storage_backend = storage_backend_from_flow(
            flow=flow,
        )

        loaded_models = LoadedModels(
            storage_backend=storage_backend,
            flow=flow,
            logger=self._logger,
            artifact_references=[]
            if self.attributes["load"] is None
            else self.attributes["load"],
            best_effort=self.attributes["best_effort_load"],
            temp_dir_root=self.attributes["temp_dir_root"],
        )

        self.loaded_models_data = loaded_models.info
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
                        field="loaded-models",
                        value=json.dumps({"keys": model_keys}),
                        type=self.METADATUM_TYPE,
                        tags=[
                            "attempt_id:%s" % str(retry_count),
                        ],
                    )
                ],
            )

        serializer = ModelSerializer(
            pathspec=current.pathspec,
            attempt=current.retry_count,
            storage_backend=storage_backend,
        )
        serializer._set_loaded_models(loaded_models)
        current._update_env(
            {
                "model": serializer,
            }
        )
        self._serializer = serializer
