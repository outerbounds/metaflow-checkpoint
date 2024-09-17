from typing import Iterable, Union, List, Dict, Any, Tuple, Optional, TYPE_CHECKING
import tempfile
from ..datastructures import CheckpointArtifact
from .constants import (
    CHECKPOINT_UID_ENV_VAR_NAME,
    DEFAULT_NAME,
    TASK_CHECKPOINTS_ARTIFACT_NAME,
)
from .constructors import (
    _instantiate_checkpoint,
    load_checkpoint,
)
from .exceptions import CheckpointNotAvailableException

if TYPE_CHECKING:
    import metaflow
    from .core import Checkpointer


def _extract_checkpoints_for_task(
    task: Union["metaflow.Task", str], attempt: Optional[Union[int, str]] = None
):
    from metaflow import Task

    if isinstance(task, str):
        if attempt is not None:
            try:
                _attempt = int(attempt)
            except ValueError:
                raise ValueError("Attempt number must be an integer. Got: %s" % attempt)
            task = Task(task, attempt=_attempt, _namespace_check=False)
        task = Task(task, _namespace_check=False)
    try:
        return task[TASK_CHECKPOINTS_ARTIFACT_NAME].data
    except NameError:
        raise CheckpointNotAvailableException(
            "Checkpoints were not recorded for the task"
        )


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

    def save(self, path_or_paths=None, metadata=None, latest=True, name=DEFAULT_NAME):
        """
        saves the checkpoint to the datastore (resemble's a create op)

        Parameters
        ----------

        - `path_or_paths`:
            - path/s to directory/ies
            - path/s to file/s
        - `name`:
            - name of the checkpoint
        - `metadata`:
            - metadata of the checkpoint
        """
        if path_or_paths is None and self.directory is None:
            raise ValueError(
                "`path` cannot be None when the Checkpoint object is not instantiated with a context manager. "
            )
        if path_or_paths is None:
            path_or_paths = self.directory
        if self._checkpointer is None:
            self = _instantiate_checkpoint(self)
        if metadata is None:
            metadata = {}
        return self._checkpointer.save(
            paths=path_or_paths, name=name, metadata=metadata, latest=latest
        ).to_dict()

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
        as_dict: bool = True,  # This is not public!
        within_task: bool = True,  # This is not public!
        # `within_task` is a hidden option. If set to False, it will list all checkpoints in "scope" (current namespace).
        # If true, it will list all checkpoints created within the task
    ) -> Iterable[Union[Dict, CheckpointArtifact]]:

        """
        lists the checkpoints in the datastore based on the Task.
        It will always be task scoped.

        Usage:
        ------

        ```python

        Checkpoint().list(name="best") # lists checkpoints in the current task with the name "best"
        Checkpoint().list(name="best", task="anotherflow/somerunid/somestep/sometask") # lists checkpoints in the another task with the name "best"
        Checkpoint().list(task="anotherflow/somerunid/somestep/sometask", name="best") # Identical as the above one but
        Checkpoint().list() # lists all the checkpoints in the current task

        ```

        Parameters
        ----------

        - `name`:
            - name of the checkpoint to filter for
        - `task`:
            - Task object outside the one that is currently set in the `Checkpoint` object; Can be a pathspec string.
        - `attempt`:
            - attempt number of the task (optional filter. If none, then lists all checkpoints from all attempts)
        """

        if self._checkpointer is None and task is None:
            raise ValueError(
                "Calling `Checkpoint.list` requires a `task` argument when its is called outside a Metaflow process."
            )
        _gen = None
        if task is not None:
            _gen = _extract_checkpoints_for_task(task, attempt)
            for chckpt in _gen:
                art = CheckpointArtifact.hydrate(chckpt)
                if name is not None and art.name != name:
                    continue
                if as_dict:
                    yield art.to_dict()
                else:
                    yield art
            return
        else:
            _gen = self._checkpointer._list_checkpoints(
                name=name, attempt=attempt, within_task=within_task
            )
        for checkpoint in _gen:
            if as_dict:
                yield checkpoint.to_dict()
            else:
                yield checkpoint

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
