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
    _instantiate_checkpoint_for_writes,
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
    DEFAULT_STORAGE_FORMAT,
)
from ..datastore.decorator import set_datastore_context

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
        attempt=attempt, as_dict=False, full_namespace=False
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
        # Ensure that if a tempdir root path is provided and nothing
        # exists then we end up creating that path. This helps ensure
        # that rouge paths with arbirary Filesystems get created before
        # temp dirs exists.
        if temp_dir_root is not None:
            if not os.path.exists(temp_dir_root):
                os.makedirs(temp_dir_root, exist_ok=True)
        self._temp_chckpt_dir = tempfile.TemporaryDirectory(
            prefix="metaflow_checkpoint_", dir=self._temp_dir_root
        )
        self._task_identifier = task_identifier
        self._default_checkpointer = _instantiate_checkpoint_for_writes(
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
        path: Optional[Union[str, os.PathLike]] = None,
        name: Optional[str] = DEFAULT_NAME,
        metadata: Optional[Dict] = {},
        latest: bool = True,
        storage_format: str = DEFAULT_STORAGE_FORMAT,
    ):
        """
        Saves the checkpoint to the datastore.

        Parameters
        ----------
        path : Optional[Union[str, os.PathLike]], default: None
            The path to save the checkpoint. Accepts a file path or a directory path.
                - If a directory path is provided, all the contents within that directory will be saved.
                When a checkpoint is reloaded during task retries, `the current.checkpoint.directory` will
                contain the contents of this directory.
                - If a file path is provided, the file will be directly saved to the datastore (with the same filename).
                When the checkpoint is reloaded during task retries, the file with the same name will be available in the
                `current.checkpoint.directory`.
                - If no path is provided then the `current.checkpoint.directory` will be saved as the checkpoint.

        name : Optional[str], default: "mfchckpt"
            The name of the checkpoint.

        metadata : Optional[Dict], default: {}
            Any metadata that needs to be saved with the checkpoint.

        latest : bool, default: True
            If True, the checkpoint will be marked as the latest checkpoint.
            This helps determine if the checkpoint gets loaded when the task restarts.

        storage_format : str, default: files
            If `tar`, the contents of the directory will be tarred before saving to the datastore.
            If `files`, saves directory directly to the datastore.

        """
        if path is None:
            path = self.directory
        return self._default_checkpointer.save(
            path=path,
            name=name,
            metadata=metadata,
            latest=latest,
            storage_format=storage_format,
        )

    def list(
        self,
        name: Optional[str] = None,
        task: Optional[Union[str, "metaflow.Task"]] = None,
        attempt: Optional[int] = None,
        full_namespace: bool = False,  # If True, list all checkpoints in the full namespace
    ) -> List[Dict]:
        """
        lists the checkpoints in the current task or the specified task.

        When users call `list` without any arguments, it will list all the checkpoints in the currently executing
        task (this includes all attempts). If the `list` method is called without any arguments outside a Metaflow Task execution context,
        it will raise an exception. Users can also call `list` with `attempt` argument to list all checkpoints within a
        the specific attempt of the currently executing task.

        When a `task` argument is provided, the `list` method will return all the checkpoints
        for a task's latest attempt unless a specific attempt number is set in the `attempt` argument.
        If the `Task` object contains a `DataArtifact` with all the previous checkpoints, then the `list` method will return
        all the checkpoints from the data artifact. If for some reason the DataArtifact is not written, then the `list` method will
        return all checkpoints directly from the checkpoint's datastore.

        Usage:
        ------

        ```python
        current.checkpoint.list(name="best") # lists checkpoints in the current task with the name "best"
        current.checkpoint.list( # Identical as the above one but lists checkpoints from the specified task with the name "best"
            task="anotherflow/somerunid/somestep/sometask",
            name="best"
        )
        current.checkpoint.list() # lists **all** the checkpoints in the current task (including the ones from all attempts)
        ```

        Parameters
        ----------

        name : Optional[str], default: None
            Filter checkpoints by name.

        task : Optional[Union["metaflow.Task", str]], default: None
            The task to list checkpoints from. Can be either a `Task` object or a task pathspec string.
            If None, lists checkpoints for the current task.
            Raises an exception if task is not provided when called outside a Metaflow Task execution context.

        attempt : Optional[Union[int, str]], default: None
            Filter checkpoints by attempt.
            If `task` is not None and `attempt` is None, then it will load the task's latest attempt

        full_namespace : bool, default: False
            If True, lists checkpoints from the full namespace.
            Only allowed during a Metaflow Task execution context.
            Raises an exception if `full_namespace` is set to True when called outside a Metaflow Task execution context.

        Returns
        -------
        List[Dict]
        """
        return self._default_checkpointer.list(
            name=name,
            task=task,
            attempt=attempt,
            as_dict=True,
            full_namespace=full_namespace,
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
            - If no path is provided, it will load the checkpoint in the default directory

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


class CheckpointDecorator(StepDecorator):
    """
    Enables checkpointing for a step.


    Parameters
    ----------
    load_policy : str, default: "fresh"
        The policy for loading the checkpoint. The following policies are supported:
            - "eager": Loads the the latest available checkpoint within the namespace.
            With this mode, the latest checkpoint written by any previous task (can be even a different run) of the step
            will be loaded at the start of the task.
            - "none": Do not load any checkpoint
            - "fresh": Loads the lastest checkpoint created within the running Task.
            This mode helps loading checkpoints across various retry attempts of the same task.
            With this mode, no checkpoint will be loaded at the start of a task but any checkpoints
            created within the task will be loaded when the task is retries execution on failure.

    temp_dir_root : str, default: None
        The root directory under which `current.checkpoint.directory` will be created.


    MF Add To Current
    -----------------
    checkpoint -> metaflow_extensions.obcheckpoint.plugins.machine_learning_utilities.checkpoints.decorator.CurrentCheckpointer
        The `@checkpoint` decorator makes saving/loading checkpoints available through the `current.checkpoint`.
        The object exposes `save`/`load`/`list` methods for saving/loading checkpoints.

        You can check if a checkpoint is loaded by `current.checkpoint.is_loaded` and get the checkpoint information
        by using `current.checkpoint.info`. The `current.checkpoint.directory` returns the path to the checkpoint directory
        where the checkpoint maybe loaded or saved.

        Usage (Saving Checkpoints):
        -------
        ```
        @checkpoint
        @step
        def train(self):
            model = create_model(self.parameters, checkpoint_path = None)
            for i in range(self.epochs):
                # some training logic
                loss = model.train(self.dataset)
                if i % 10 == 0:
                    model.save(
                        current.checkpoint.directory,
                    )
                    # saves the contents of the `current.checkpoint.directory` as a checkpoint
                    # and returns a reference dictionary to the checkpoint saved in the datastore
                    self.latest_checkpoint = current.checkpoint.save(
                        name="epoch_checkpoint",
                        metadata={
                            "epoch": i,
                            "loss": loss,
                        }
                    )
        ```
        Usage (Using Loaded Checkpoints):
        -------
        ```
        @retry(times=3)
        @checkpoint
        @step
        def train(self):
            # Assume that the task has restarted and the previous attempt of the task
            # saved a checkpoint
            checkpoint_path = None
            if current.checkpoint.is_loaded: # Check if a checkpoint is loaded
                print("Loaded checkpoint from the previous attempt")
                checkpoint_path = current.checkpoint.directory

            model = create_model(self.parameters, checkpoint_path = checkpoint_path)
            for i in range(self.epochs):
                ...
        ```

        @@ Returns
        -------
        CurrentCheckpointer
            The object for handling checkpointing within a step.
    """

    _task_identifier = None

    name = "checkpoint"

    defaults = {
        # `load_policy` defines the policy for the checkpoint loading during the execution of different runs.
        # It can be : ["eager", "none", "fresh"],
        "load_policy": "fresh",  #
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
        self.deco_injector = CardDecoratorInjector()
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

        self.deco_injector.attach_card_decorator(
            flow,
            step_name,
            CheckpointListRefresher.CARD_ID,
            "blank",
            refresh_interval=2,
        )
        self._chkptr = None
        self._collector_thread = None

    def task_exception(
        self, exception, step_name, flow, graph, retry_count, max_user_code_retries
    ):
        if self._collector_thread is not None:
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

        set_datastore_context(flow, metadata, run_id, step_name, task_id, retry_count)
        settings = self._resolve_settings()
        load_policy = settings["load_policy"]
        self._load_policy = load_policy
        resolved_scope = self._resolve_scope()
        gang_scheduled_task = graph[step_name].parallel_step

        temp_dir_root = settings.get(
            "temp_dir_root",
        )

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
            # TODO [POST RELEASE]: Make this a little more customizable in the future.
            # since the @parallel tasks can even be HPO style tasks instead
            # of gang scheduled tasks.
        )

        if gang_scheduled_task:
            # A step with an @parallel will mean that the decorator is follow gang scheduling semantics
            # so all workers and control task will be writing to the same path (i.e. the same
            # task identifier). If users wish to write checkpoints that are across different
            # workers, they should ensure that the `name` is differently set in the `save`
            # method so that checkpoints don't get overwritten.
            warning_message(
                (
                    "The step has a @parallel decorator and so checkpoints will be treated as if"
                    "they are coming from the same task. The checkpoints will be written/loaded from the control task."
                    "All tasks within this step will write to the same path i.e. the path of the control task."
                ),
                logger=self._logger,
                ts=False,
            )
        self._loaded_checkpoint = self._setup_checkpointer(
            flow,
            default_task_identifier,
            resolved_scope,
            load_policy,
            gang_scheduled_task=gang_scheduled_task,
            temp_dir_root=temp_dir_root,
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
            interval=3,
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
        temp_dir_root=None,
    ):
        self._chkptr = CurrentCheckpointer(
            flow=flow,
            task_identifier=default_task_identifier,
            resolved_scope=resolved_scope,
            logger=self._logger,
            gang_scheduled_task=gang_scheduled_task,
            temp_dir_root=temp_dir_root,
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
