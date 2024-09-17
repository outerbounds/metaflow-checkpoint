# from functools import wraps
from datetime import datetime
import os

from .exceptions import CheckpointException
from ..utils import flowspec_utils
from ..card_utils import CardDecoratorInjector, AsyncPeriodicRefresher
from .cards.checkpoint_lister import CheckpointListRefresher, CheckpointsCollector
from .cards import create_checkpoint_card, null_card
from .lineage import checkpoint_load_related_metadata, trace_lineage
from .constructors import (
    DEFAULT_NAME,
    load_checkpoint,
    _instantiate_checkpoint,
)
from .core import (
    ScopeResolver,
    CheckpointLoadPolicy,
)

# from .lineage import checkpoint_load_related_metadata, trace_lineage
from ..datastructures import CheckpointArtifact
from .constants import (
    CHECKPOINT_TAG_PREFIX,
    CHECKPOINT_TASK_IDENTIFIER_ENV_VAR_NAME,
    CHECKPOINT_UID_ENV_VAR_NAME,
    TASK_CHECKPOINTS_ARTIFACT_NAME,
    TASK_LATEST_CHECKPOINT_ARTIFACT_NAME,
)

# from .cards import CardDecoratorInjector, create_checkpoint_card, null_card
from metaflow.decorators import StepDecorator, FlowDecorator
from metaflow.flowspec import INTERNAL_ARTIFACTS_SET
from .final_api import Checkpoint
from typing import List, Dict, Union, Tuple, Optional, Callable, TYPE_CHECKING
from functools import wraps, partial
import tempfile
import json

if TYPE_CHECKING:
    import metaflow


try:
    unicode
except NameError:
    unicode = str
    basestring = str


def _store_checkpoint_ref_as_data_artifact(
    flow,
    attempt,
    checkpointer: "CurrentCheckpointer",
):
    _chckpts = []
    # There are two code paths we will try out:
    # One code path will try and get the latest checkpoint since
    # Another one will try and retrive all of them. On reason we do this is because
    # `list` call to the datastore can return things out of order and since we are also filtering for a
    # max value in the list, we might end up with a checkpoint that is not the latest.

    # This GLITCH is that if a user, is running Flows concurrently and they are all updating the value of
    # `latest` then it can surely create a problem.
    latest_chckpt = (
        checkpointer._default_checkpointer._checkpointer._get_latest_checkpoint(
            safe=True
        )
    )

    for checkpoint in checkpointer._default_checkpointer.list(
        attempt=attempt, as_dict=False, within_task=True
    ):
        _chckpts.append(checkpoint)

    time_sorted = lambda z: sorted(
        z, key=lambda x: datetime.fromisoformat(x.created_on), reverse=True
    )
    _chckpts = time_sorted(_chckpts)
    # TASK_CHECKPOINTS_ARTIFACT_NAME is a list of dictionaries that will be stored in the datastore.
    # This is a variable hidden from the user but will be used to locate the checkpoints
    # in the datastore.
    setattr(
        flow,
        TASK_CHECKPOINTS_ARTIFACT_NAME,
        [c.to_dict() for c in _chckpts],
    )
    setattr(
        flow,
        TASK_LATEST_CHECKPOINT_ARTIFACT_NAME,
        None,
    )
    if latest_chckpt:
        setattr(
            flow,
            TASK_LATEST_CHECKPOINT_ARTIFACT_NAME,
            latest_chckpt.to_dict(),
        )


def warning_message(message, logger=None, ts=False, prefix="[@checkpoint]"):
    msg = "%s %s" % (prefix, message)
    if logger:
        logger(msg, timestamp=ts, bad=True)


class CurrentCheckpointer:
    @property
    def task_identifier(self):
        return self._task_identifier

    @property
    def directory(self):
        return self._temp_chckpt_dir.name

    @property
    def is_loaded(self):
        return self._loaded_checkpoint is not None

    @property
    def info(self):
        return self._loaded_checkpoint

    def __init__(
        self,
        flow,
        task_identifier,
        resolved_scope,
        logger,
        gang_scheduled_task=False,
        temp_dir_root=None,
    ) -> None:
        from metaflow import current

        self._temp_dir_root = temp_dir_root
        self._resolved_scope = resolved_scope
        self._logger = logger
        self._loaded_checkpoint = None
        self._temp_chckpt_dir = tempfile.TemporaryDirectory(
            prefix="metaflow_checkpoint_", dir=self._temp_dir_root
        )
        self._task_identifier = task_identifier
        self._default_checkpointer = _instantiate_checkpoint(
            Checkpoint(),
            flow=flow,
            task_identifier=task_identifier,
            scope=resolved_scope,
            gang_scheduled_task=gang_scheduled_task,
        )
        os.environ[CHECKPOINT_TASK_IDENTIFIER_ENV_VAR_NAME] = self._task_identifier
        os.environ[CHECKPOINT_UID_ENV_VAR_NAME] = str(
            self._default_checkpointer._checkpointer._checkpoint_uid
        )

    def _setup_task_first_load(self, load_policy, flow):
        checkpoint = None
        # TODO : Its assumed the `load_policy` is set correctly and validation has
        # happened before this point.
        if load_policy == "eager":
            checkpoint = CheckpointLoadPolicy.eager(
                self._default_checkpointer._checkpointer._checkpoint_datastore,
                flow,
            )
        elif load_policy == "fresh":
            checkpoint = CheckpointLoadPolicy.fresh(
                self._default_checkpointer._checkpointer._checkpoint_datastore,
                flow,
            )
        if checkpoint is None:
            return None

        warning_message(
            "Loading the following checkpoint:\n\t[pathspec] %s\n\t[key] %s\n\t[created on] %s\n\t[url] %s"
            % (
                checkpoint.pathspec,
                checkpoint.key,
                checkpoint.created_on,
                checkpoint.url,
            ),
            logger=self._logger,
            ts=False,
        )
        load_checkpoint(checkpoint=checkpoint, local_path=self.directory)
        self._loaded_checkpoint = checkpoint
        return checkpoint

    def save(
        self,
        path: Optional[Union[str, os.PathLike, List[str], List[os.PathLike]]] = None,
        name: Optional[str] = DEFAULT_NAME,
        metadata: Optional[Dict] = {},
        latest: bool = True,
    ):
        if path is None:
            path = self.directory
        return self._default_checkpointer.save(
            path_or_paths=path, name=name, metadata=metadata, latest=latest
        )

    def list(
        self,
        name: Optional[str] = None,
        task: Optional[Union[str, "metaflow.Task"]] = None,
        attempt: Optional[int] = None,
        within_task: bool = True
        # `within_task` is a hidden option. If set to False, it will list all checkpoints in "scope" (current namespace).
        # If true, it will list all checkpoints created within the task
    ):
        return self._default_checkpointer.list(
            name=name,
            task=task,
            attempt=attempt,
            as_dict=True,
            within_task=within_task,
        )

    def cleanup(self):
        self._temp_chckpt_dir.cleanup()

    def refresh_directory(self):
        self.cleanup()
        self._temp_chckpt_dir = tempfile.TemporaryDirectory(
            prefix="metaflow_checkpoint_", dir=self._temp_dir_root
        )

    def load(
        self,
        reference: Union[str, Dict, CheckpointArtifact],
        path: Optional[str] = None,
    ):
        """
        loads a checkpoint reference from the datastore. (resembles a read op)

        This can have two meanings:
            - If the path is provided, it will load the checkpoint in the provided path
            - If no path is providede, it will load the checkpoint in the default directory

        Parameters
        ----------

        `reference` :
            - can be a string, dict or a CheckpointArtifact object:
                - string: a string reference to the checkpoint
                - dict: a dictionary form of the CheckpointArtifact
                - CheckpointArtifact: a CheckpointArtifact object reference to the checkpoint
        """
        if path is None:
            self.refresh_directory()
            path = self.directory
        return Checkpoint().load(reference, path=path)


def merge_dicts_with_precedence(*args: dict) -> dict:
    """
    Merges multiple dictionaries, respecting the order of precedence.

    This function takes any number of dictionary arguments and merges them into a single dictionary.
    If the same key exists in multiple dictionaries, the value from the dictionary that appears
    last in the argument list takes precedence, except where the value is None, in which case
    the search continues in the earlier dictionaries for a non-None value.

    The operation is not recursive and will only consider top-level keys.

    Parameters:
    - args: A variable number of dictionary arguments. Each argument must be a dictionary.

    Returns:
    - dict: A single dictionary that results from merging the input dictionaries according to their order of precedence.

    Examples:
    - merge_dicts_with_precedence(defaults, attrs)
      Here, `defaults` is a dictionary of default values, and `attrs` contains override values.
      Any None values in `attrs` will result in values from `defaults` being used.

    - merge_dicts_with_precedence(defaults, global_config, attrs)
      In this scenario, `global_config` can override `defaults`, and `attrs` can override both
      `defaults` and `global_config`. The order of arguments defines the precedence.

    Note:
    The function behaves differently if the order of the arguments changes, reflecting the
    precedence of the values set based on their position in the argument list.
    """
    unique_set_of_keys = set()
    for dictionary in args:
        unique_set_of_keys.update(dictionary.keys())

    final_dict = {}
    for key in unique_set_of_keys:
        for dictionary in args:
            if key in dictionary and dictionary[key] is not None:
                final_dict[key] = dictionary[key]
    return final_dict


def _greater_than_one_set(*args):
    return len([a for a in args if a]) > 1


class CheckpointDecorator(StepDecorator, CardDecoratorInjector):
    """ """

    _task_identifier = None

    name = "checkpoint"

    defaults = {
        # `load_policy` defines the policy for the checkpoint loading during the execution of different runs.
        # It can be : ["eager", "none", "fresh"],
        "load_policy": None,  #
        "temp_dir_root": None,  # Root directory for the temporary checkpoint directory.
    }

    LOAD_POLCIES = [
        # Check the `CheckpointLoadPolicy` for more documentation on these modes.
        "eager",
        "none",
        "fresh",
    ]

    def _resolve_settings(self):
        return merge_dicts_with_precedence(
            {"load_policy": "fresh"},
            self.attributes,
        )

    def step_init(
        self, flow, graph, step_name, decorators, environment, flow_datastore, logger
    ):
        if (
            self.attributes["load_policy"] is not None
            and self.attributes["load_policy"] not in self.LOAD_POLCIES
        ):
            raise CheckpointException(
                "`load_policy` of %s is not supported. Supported policies are %s"
                % (self.attributes["load_policy"], ", ".join(self.LOAD_POLCIES))
            )

        # We add to INTERNAL_ARTIFACTS_SET here because the decorator adds internal artifacts to the
        # flow. Adding to INTERNAL_ARTIFACTS_SET avoids having any crashes when `merge_artifacts`
        # is called.
        INTERNAL_ARTIFACTS_SET.update(
            [TASK_CHECKPOINTS_ARTIFACT_NAME, TASK_LATEST_CHECKPOINT_ARTIFACT_NAME]
        )
        self._flow_datastore = flow_datastore
        self._logger = logger

        self.attach_card_decorator(
            flow,
            step_name,
            CheckpointListRefresher.CARD_ID,
            "blank",
            refresh_interval=10,
        )
        self._chkptr = None
        self._collector_thread = None

    def task_exception(
        self, exception, step_name, flow, graph, retry_count, max_user_code_retries
    ):
        if self._collector_thread is not None:
            self._collector_thread.run_update()
            self._collector_thread.stop()

        if self._chkptr is not None:
            _store_checkpoint_ref_as_data_artifact(flow, retry_count, self._chkptr)
            self._chkptr.cleanup()
            self._chkptr = None

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
        from metaflow import current

        settings = self._resolve_settings()
        load_policy = settings["load_policy"]
        self._load_policy = load_policy
        resolved_scope = self._resolve_scope()
        gang_scheduled_task = graph[step_name].parallel_step

        if gang_scheduled_task and not getattr(
            current.parallel, "control_task_id", None
        ):
            # gang_scheduled_task's need a `current.parallel.control_task_id` to be set.
            # This is needed so that the metadata_store of the CheckpointDatastore is set to
            # the control task and hence when the gang restarts for any load policy, we are able
            # to load the checkpoint that was written by the control task.
            # Ideally in distributed data parallel scenarios, it is important to have a control task
            # that is writing the checkpoints and the other workers are reading from it when any restart
            # happens.
            gang_scheduled_task = False
            warning_message(
                "The task is a gang scheduled task but the control task id is not set. Metaflow version needs to be upgrade",
                logger=self._logger,
                ts=False,
            )

        default_task_identifier = flowspec_utils.resolve_task_identifier(
            run=flow,
            gang_scheduled_task=gang_scheduled_task,
            gang_schedule_task_idf_index=0,
            # TODO : Make this a little more customizable in the future.
            # since the @parallel tasks can even be HPO style tasks instead
            # of gang scheduled tasks.
        )

        # TODO : Print a warning messages saying that the step is an @parallel
        # step and it will mean that the decorator is follow gang scheduling semantics
        # so all workers and control task will be writing to the same path (i.e. the same
        # task identifier). If users wish to write checkpoints that are across different
        # workers, they should ensure that the `name` is differently set in the `save`
        # method so that checkpoints don't get overwritten.
        self._loaded_checkpoint = self._setup_checkpointer(
            flow,
            default_task_identifier,
            resolved_scope,
            load_policy,
            gang_scheduled_task=gang_scheduled_task,
        )
        self._loaded_checkpoint_lineage = []
        if self._loaded_checkpoint is not None:
            entries = checkpoint_load_related_metadata(
                self._loaded_checkpoint, retry_count
            )
            metadata.register_metadata(run_id, step_name, task_id, entries)
            checkpoint_list = trace_lineage(flow, self._loaded_checkpoint)
            if checkpoint_list:
                self._loaded_checkpoint_lineage = checkpoint_list
        self._setup_current()

    def task_decorate(
        self, step_func, flow, graph, retry_count, max_user_code_retries, ubf_context
    ):
        # Wrap the step_func in a function that will write to current.card["checkpoint_info"] the lineage card
        # and then call the step_func.
        self._collector_thread = CheckpointsCollector(
            CheckpointListRefresher(
                self._loaded_checkpoint,
                self._loaded_checkpoint_lineage,
                self._load_policy,
            ),
            interval=5,
        )

        def _wrapped_step_func(_collector_thread, *args, **kwargs):

            _collector_thread.start()
            try:
                return step_func(*args, **kwargs)
            finally:
                _collector_thread.stop()

        return partial(_wrapped_step_func, self._collector_thread)

    def task_post_step(
        self, step_name, flow, graph, retry_count, max_user_code_retries
    ):
        if self._collector_thread is not None:
            self._collector_thread.run_update()
            self._collector_thread.stop()

        if self._chkptr is not None:
            _store_checkpoint_ref_as_data_artifact(flow, retry_count, self._chkptr)
            self._chkptr.cleanup()
            self._chkptr = None

    def _resolve_scope(self):
        from metaflow import Run, current

        filtered_tags = [
            t
            for t in Run("%s/%s" % (current.flow_name, current.run_id)).tags
            if CHECKPOINT_TAG_PREFIX in t
        ]
        if len(filtered_tags) > 0:
            warning_message(
                "The Run has tags set with the special '%s' prefix; The checkpoints for this task will be namespaced under %s"
                % (CHECKPOINT_TAG_PREFIX, filtered_tags[0]),
                logger=self._logger,
                ts=False,
            )
            return ScopeResolver.from_tags(filtered_tags)
        else:
            return ScopeResolver.from_namespace()

    def _setup_checkpointer(
        self,
        flow,
        default_task_identifier,
        resolved_scope,
        load_policy,
        gang_scheduled_task=False,
    ):
        self._chkptr = CurrentCheckpointer(
            flow=flow,
            task_identifier=default_task_identifier,
            resolved_scope=resolved_scope,
            logger=self._logger,
            gang_scheduled_task=gang_scheduled_task,
        )
        return self._chkptr._setup_task_first_load(load_policy, flow)

    def _setup_current(
        self,
    ):
        from metaflow import current

        current._update_env(
            {
                "checkpoint": self._chkptr,
            }
        )
