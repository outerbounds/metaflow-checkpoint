from typing import Iterable, Union, List, Dict, Any, Tuple, Optional, TYPE_CHECKING
import tempfile
from ..datastructures import CheckpointArtifact
from .constants import (
    CHECKPOINT_UID_ENV_VAR_NAME,
    DEFAULT_NAME,
    TASK_CHECKPOINTS_ARTIFACT_NAME,
    DEFAULT_STORAGE_FORMAT,
)
from .constructors import (
    _instantiate_checkpoint_for_writes,
    _instantiate_checkpointer_for_list,
    load_checkpoint,
)
from .exceptions import CheckpointNotAvailableException, CheckpointException

if TYPE_CHECKING:
    import metaflow
    from .core import Checkpointer


def _extract_task_object(
    task: Union["metaflow.Task", str], attempt: Optional[Union[int, str]] = None
):
    from metaflow import Task

    if attempt is not None:
        try:
            _attempt = int(attempt)
        except ValueError:
            raise ValueError("Attempt number must be an integer. Got: %s" % attempt)
    if isinstance(task, str):
        if attempt is not None:
            task = Task(task, attempt=_attempt, _namespace_check=False)
        task = Task(task, _namespace_check=False)
    elif isinstance(task, Task) and attempt is not None:
        task = Task(task.pathspec, attempt=_attempt, _namespace_check=False)
    return task


def _extract_checkpoints_from_task_object(
    task: "metaflow.Task",
):
    try:
        return task[TASK_CHECKPOINTS_ARTIFACT_NAME].data
    except NameError:
        raise CheckpointNotAvailableException(
            "Checkpoints were not recorded for the task"
        )
    except KeyError:
        raise CheckpointNotAvailableException(
            "Checkpoints were not recorded for the task"
        )


def _inside_task_context():
    """
    If we can instantiate a `Checkpoint` object to write, then we are ideally in Metaflow task "context".
    Meaning that we might be inside a Metaflow task process or in a process spawed by a metaflow task process
    """
    try:
        _instantiate_checkpoint_for_writes(Checkpoint())
        return True
    except ValueError:
        return False


class Checkpoint:

    _checkpointer: "Checkpointer" = None

    def __init__(self, temp_dir_root=None, init_dir=False):
        self._temp_dir_root = temp_dir_root
        self._checkpoint_dir = None
        if init_dir:
            self._checkpoint_dir = tempfile.TemporaryDirectory(dir=self._temp_dir_root)

    @property
    def directory(self):
        if self._checkpoint_dir is None:
            return None
        return self._checkpoint_dir.name

    def _set_checkpointer(self, checkpointer: "Checkpointer"):
        self._checkpointer = checkpointer

    def save(
        self,
        path=None,
        metadata=None,
        latest=True,
        name=DEFAULT_NAME,
        storage_format=DEFAULT_STORAGE_FORMAT,
    ) -> Dict:
        """
        Saves the checkpoint to the datastore

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
                - If no path is provided then the `Checkpoint.directory` will be saved as the checkpoint.

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
        if path is None and self.directory is None:
            raise ValueError(
                "`path` cannot be None when the Checkpoint object is not instantiated with a context manager. "
            )
        if path is None:
            path = self.directory
        if self._checkpointer is None:
            # If the `Checkpoint` object is being used by `CurrentCheckpointer` then we have already set the `_checkpointer`
            # attribute. If it is not being set by `CurrentCheckpointer` then the user might be calling it in an outside
            # process or within main process. So we try to instantiate it for writes.
            self = self._init_checkpoint_for_writes(self)

        if metadata is None:
            metadata = {}
        return self._checkpointer.save(
            path=path,
            name=name,
            metadata=metadata,
            latest=latest,
            storage_format=storage_format,
        ).to_dict()

    @classmethod
    def _init_checkpoint_for_writes(cls, self):
        try:
            self = _instantiate_checkpoint_for_writes(self)
        except ValueError as e:
            raise CheckpointException(
                (
                    "`Checkpoint.save` can only be called within a Metaflow Task execution. If you "
                    "are calling `Checkpoint.save` outside a Metaflow Task process, the @checkpoint decorator "
                    "is set on the @step calling this method. If the decorator is set, then ensure that you inherit "
                    "the environment variables from the Metaflow Task process or pass the `METAFLOW_CHECKPOINT_UID` "
                    "environment variable from the Metaflow Task process to your subprocess."
                )
            )
        return self

    def generate_key(self, name: str, version_id: int = None):
        if self._checkpointer is None:
            self = self._init_checkpoint_for_writes(self)
        return self._checkpointer.artifact_id(name, version_id)

    def __enter__(self):
        if self._checkpoint_dir is None:
            self._checkpoint_dir = tempfile.TemporaryDirectory(dir=self._temp_dir_root)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._checkpoint_dir.cleanup()
        self._checkpoint_dir = None

    def list(
        self,
        name: Optional[str] = None,
        task: Optional[Union["metaflow.Task", str]] = None,
        attempt: Optional[Union[int, str]] = None,
        full_namespace: bool = False,  # list within the full namespace
        as_dict: bool = True,  # This is not public!
    ) -> List[Union[Dict, CheckpointArtifact]]:

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

        Checkpoint().list(name="best") # lists checkpoints in the current task with the name "best"
        Checkpoint().list(task="anotherflow/somerunid/somestep/sometask", name="best") # Identical as the above one but
        Checkpoint().list() # lists **all** the checkpoints in the current task (including the ones from all attempts)

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

        def _sort_checkpoints_by_version(dicts: list[Union[Dict, CheckpointArtifact]]):
            def _get_version_id(x: Union[Dict, CheckpointArtifact]):
                if isinstance(x, CheckpointArtifact):
                    return x.version_id
                return x.get("version_id", -1)

            return sorted(
                dicts,
                key=_get_version_id,
                reverse=True,
            )

        # There are two potential code paths here:
        # 1. Checkpoint.list is being called outside the Task Process in a Notebook or a script to extract the
        # checkpoints. At this point, we can check if the checkpoints are stored in the `Task` object;
        # if that fails, we can check if checkpoints are written within the Metadata store. of the task. Both code paths
        # are valid when Metaflow is being called from a notebook or a script. We allow the metadata store access to
        # ensure that the checkpoints are not lost even the task completely crashed we were unable to log any data artifacts.
        # 2. Checkpoint.list is being called within a Task Process. In this case we can directly access the datastore if the
        # task object is not available. If the task object is available , we do the same things as specified in (1.)

        # The following table outlines the outcomes of the different code paths.

        # |  MF TASK CONTEXT | Task Object  | Outcome
        # | ---------------- | ------------ | ------------
        # | -----False------ | ----False--- | (C1) Raise Exception
        # | -----False------ | ----True---- | (C2) Search Task's DataArtifact and if not then search the Task's Metadata store
        # | -----True------- | ----False--- | (C3) instantiate the checkpointer and search it's metadata store
        # | -----True------- | ----True---- | (C4) Search Task's DataArtifact and if not then search the Task's Metadata store

        _task_object = None
        if task is not None:
            # This function will set the a `metaflow.Task` object with the right
            # attempt number if provided.
            _task_object = _extract_task_object(task, attempt)

        checkpoint_data_artifact = None
        if _task_object is not None:
            try:
                checkpoint_data_artifact = _extract_checkpoints_from_task_object(
                    _task_object
                )
            except CheckpointNotAvailableException:
                pass

        # If the checkpoint data artifact for the Task in question is available then we can directly return the list of
        # checkpoints from the task object.
        # Outcome : (C2) / (C4)
        if checkpoint_data_artifact is not None:
            # Since the data artifact will have everything in the dictionary format
            # we will check if the user wants CheckpointArtifacts or dicts.
            return _sort_checkpoints_by_version(
                [
                    CheckpointArtifact.hydrate(chckpt) if not as_dict else chckpt
                    for chckpt in checkpoint_data_artifact
                    if (name is None or CheckpointArtifact.hydrate(chckpt).name == name)
                ]
            )

        not_within_task_context = not _inside_task_context()
        # At this point if we are not even given a Task object and we are also not even insde a Metaflow's Task execution
        # then we can raise an error that users cannot call `Checkpoint.list` outside a Metaflow Task execution.
        # Outcome : (C1)
        if _task_object is None and not_within_task_context:
            raise ValueError(
                "Calling `Checkpoint.list` requires a `task` argument when its is called outside a Metaflow process."
            )

        if not_within_task_context and full_namespace:
            # This means the user is trying to access checkpoints in "scope" but they are not even inside a Metaflow Task execution.
            # context. Which is not allowed.
            raise ValueError(
                "Calling `Checkpoint.list` with `full_namespace=False` outside a Metaflow Task's execution context is not allowed."
            )

        # At this point if we have a task object then either that task's checkpoint data artifact
        # was not written. Then we can try and do a list from the task's Metadata store.
        # We instantiate a checkpointer and for that attempt then call the `list` method.
        # To do this check it doesn't matter if we are within a Metaflow Task execution context or not.
        # Outcome : (C2) / (C4)
        if _task_object is not None:
            # If there is an explicit task object provided by the user
            # Then we will list the checkpoints found in the task's latest attempt
            _checkpointer = _instantiate_checkpointer_for_list(_task_object)
            return _sort_checkpoints_by_version(
                [
                    chckpt.to_dict() if as_dict else chckpt
                    for chckpt in _checkpointer._list_checkpoints(
                        name=name,
                        attempt=_checkpointer._attempt,
                        within_task=not full_namespace,
                    )
                ]
            )

        # There is no task object clearly the user has called `Checkpoint.list`
        # within a Metaflow Task execution context (if it was not within task execution context
        # were not then we would have alreadyraised an exception). (outcome : (C1))
        # Hence we will check if there is a checkpointer or we will safely instantiate
        # a write checkpointer.
        # Outcome : (C3)
        _checkpointer = self._checkpointer
        if _checkpointer is None:
            self = self._init_checkpoint_for_writes(self)
            _checkpointer = self._checkpointer

        # Since at this point we know the user is calling `current.checkpoint.list`
        # without any `task`, that means the user is trying to list all checkpoints
        # within the current executing task.
        return _sort_checkpoints_by_version(
            [
                chckpt.to_dict() if as_dict else chckpt
                for chckpt in _checkpointer._list_checkpoints(
                    name=name,
                    attempt=attempt,  # Return all attempts if `attempt` is None
                    within_task=not full_namespace,
                )
            ]
        )

    def load(
        self,
        reference: Union[str, Dict, CheckpointArtifact],
        path: Optional[str] = None,
    ):
        """
        loads a checkpoint reference from the datastore. (resembles a read op)

        Parameters
        ----------

        `reference` :
            - can be a string, dict or a CheckpointArtifact object:
                - string: a string reference to the checkpoint (checkpoint key)
                - dict: a dictionary reference to the checkpoint
                - CheckpointArtifact: a CheckpointArtifact object reference to the checkpoint
        """
        if path is None and self.directory is None:
            raise ValueError(
                "`path` cannot be None when the Checkpoint object is not instantiated with a context manager. "
            )
        if path is None:
            path = self.directory

        load_checkpoint(
            checkpoint=reference,
            local_path=path,
        )
