# from functools import wraps

import os
from metaflow.metadata.metadata import MetadataProvider
from .core import ModelSerializer, LoadedModels
from ..card_utils import CardDecoratorInjector
from .cards.model_card import ModelListRefresher, ModelsCollector
from ..utils import flowspec_utils

# from .lineage import checkpoint_load_related_metadata, trace_lineage
from ..datastructures import ModelArtifact

# from .cards import CardDecoratorInjector, create_checkpoint_card, null_card
from metaflow.decorators import StepDecorator
from metaflow.metadata import MetaDatum
from typing import List, Dict, Union, Tuple, Optional, Callable, TYPE_CHECKING
from functools import wraps, partial
import tempfile
from metaflow.flowspec import INTERNAL_ARTIFACTS_SET
import json

if TYPE_CHECKING:
    import metaflow


def warning_message(message, logger=None, ts=False, prefix="[@model]"):
    msg = "%s %s" % (prefix, message)
    if logger:
        logger(msg, timestamp=ts, bad=True)


class ModelDecorator(StepDecorator, CardDecoratorInjector):

    name = "model"

    defaults = {
        "load": None,  # This can be a list of models to load from artifact references.
        "best_effort_load": False,  # If True, it will ignore missing artifacts and continue.
    }

    METADATUM_TYPE = "model-registry"

    def step_init(
        self, flow, graph, step_name, decorators, environment, flow_datastore, logger
    ):
        self._flow = flow
        self._flow_datastore = flow_datastore
        self._logger = logger
        self._serializer = None
        self.attach_card_decorator(
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
        for idx, md in enumerate(explicit_model_list[-20:]):
            model_art = ModelArtifact.hydrate(md)
            _md_list.append(
                MetaDatum(
                    field="model-key-%s" % idx,
                    value=model_art.key,
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
        self._metadata_provider = metadata
        self._runid = run_id
        self._task_id = task_id
        self._step_name = step_name
        self._setup_current(flow)

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

    def _setup_current(self, flow):
        from metaflow import current

        storage_backend = flowspec_utils.resolve_storage_backend(
            run=flow,
        )

        loaded_models = LoadedModels(
            storage_backend=storage_backend,
            flow=flow,
            logger=self._logger,
            artifact_references=[]
            if self.attributes["load"] is None
            else self.attributes["load"],
            best_effort=self.attributes["best_effort_load"],
        )

        self.loaded_models_data = loaded_models.info

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
