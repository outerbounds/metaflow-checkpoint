import json
import os
import pickle
import tempfile
from typing import Iterable, Union, List, Dict, Any, Tuple, Optional, TYPE_CHECKING
from ..datastructures import CheckpointArtifact
from .checkpoint_storage import _search_checkpoints_in_metadata_store
from .constants import (
    CHECKPOINT_UID_ENV_VAR_NAME,
    DEFAULT_NAME,
    TASK_CHECKPOINTS_ARTIFACT_NAME,
    DEFAULT_STORAGE_FORMAT,
)
from .constructors import (
    _instantiate_checkpoint_for_writes,
    _instantiate_checkpointer_for_list,
    _instantiate_checkpointer_for_global_reads,
    load_checkpoint,
)
from .exceptions import CheckpointNotAvailableException, CheckpointException

if TYPE_CHECKING:
    import metaflow
    from .core import Checkpointer

# ---------------------------------------------------------------------------
# Implicit serialization helpers
# ---------------------------------------------------------------------------

# Manifest file written alongside serialized field files so load() knows what to restore.
IMPLICIT_MANIFEST_FILENAME = "__implicit_checkpoint__.json"

PICKLE_FORMAT = "pickle"
RAW_FORMAT = "raw"
_SUPPORTED_FORMATS = (PICKLE_FORMAT, RAW_FORMAT)


def _get_implicit_fields(flow, exclude=None, include=None) -> List[Tuple[str, Any]]:
    """Return list of (field_name, value) to checkpoint.

    ``include`` and ``exclude`` are mutually exclusive (matching the behaviour
    of Metaflow's ``merge_artifacts``).  If ``include`` is given, only those
    fields are returned and a ``ValueError`` is raised for any name not present
    on *flow*.  If ``exclude`` is given, all public fields except the listed
    names are returned.
    """
    if include and exclude:
        raise ValueError("`include` and `exclude` are mutually exclusive in @checkpoint")

    all_public = [
        (name, value)
        for name, value in flow.__dict__.items()
        if not name.startswith("_") and not callable(value)
    ]

    if include:
        include_set = set(include)
        available = {name for name, _ in all_public}
        missing = include_set - available
        if missing:
            raise ValueError(
                "Fields specified in `include` not found on self: %s"
                % sorted(missing)
            )
        return [(name, value) for name, value in all_public if name in include_set]

    exclude_set = set(exclude or [])
    return [(name, value) for name, value in all_public if name not in exclude_set]


def _serialize_value(value, fmt) -> bytes:
    """Serialize *value* to bytes using *fmt*."""
    if fmt == PICKLE_FORMAT:
        return pickle.dumps(value)
    elif fmt == RAW_FORMAT:
        if not isinstance(value, (bytes, bytearray)):
            raise TypeError(
                "Field with serialization format 'raw' must be bytes or bytearray, "
                "got %s. Use the default 'pickle' format for other types."
                % type(value).__name__
            )
        return bytes(value)
    else:
        raise ValueError(
            "Unsupported serialization format %r. Supported formats: %s"
            % (fmt, _SUPPORTED_FORMATS)
        )


def _deserialize_value(data, fmt) -> Any:
    """Deserialize bytes produced by ``_serialize_value`` back to a Python object."""
    if fmt == PICKLE_FORMAT:
        return pickle.loads(data)
    elif fmt == RAW_FORMAT:
        return data
    else:
        raise ValueError("Unsupported serialization format %r." % fmt)


# ---------------------------------------------------------------------------


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
        else:
            # else prevents Task(task) receiving a Task object instead of a string when attempt is set.
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

    def __init__(self, temp_dir_root=None):
        # init_dir removed: temp directories are now created per-operation in save()/load().
        self._temp_dir_root = temp_dir_root
        self._checkpoint_dir = None

    @property
    def directory(self) -> Optional[str]:
        """
        The directory where a checkpoint is loaded
        """
        if self._checkpoint_dir is None:
            return None
        return self._checkpoint_dir.name

    def _set_checkpointer(self, checkpointer: "Checkpointer"):
        self._checkpointer = checkpointer

    def save(
        self,
        flow,
        exclude=None,
        include=None,
        name=DEFAULT_NAME,
        metadata={},
        latest=True,
        storage_format=DEFAULT_STORAGE_FORMAT,
        temp_dir_root=None,
    ) -> CheckpointArtifact:
        """
        Serializes public attributes of *flow* into a checkpoint.

        Parameters
        ----------
        flow : FlowSpec
            The Metaflow step's ``self`` — source of attribute values.

        exclude : list of str, optional
            Attribute names to skip.  All other public non-underscore,
            non-callable attributes on ``flow.__dict__`` are checkpointed.
            Cannot be specified together with ``include``.

        include : list of str, optional
            Explicit list of attribute names to checkpoint.  Only these fields
            are saved; all others are ignored.  Raises ``ValueError`` if any
            name is not present on *flow* at save time.
            Cannot be specified together with ``exclude``.

        name : str, default: "mfchckpt"
            The name of the checkpoint.

        metadata : dict, default: {}
            User metadata to attach to the checkpoint.

        latest : bool, default: True
            Mark this checkpoint as the latest.

        storage_format : str, default: files
            If ``tar``, the checkpoint directory is tarred before uploading.
            If ``files``, files are uploaded directly.
        """
        if self._checkpointer is None:
            self = self._init_checkpoint_for_writes(self)

        field_items = _get_implicit_fields(flow, exclude=exclude, include=include)

        if not field_items:
            raise ValueError(
                "No fields found to checkpoint. Use `include=[...]` to specify "
                "which fields to save, `exclude=[...]` to skip specific fields, "
                "or set public attributes on self before calling current.checkpoint.save()."
            )

        field_manifest = {}

        with tempfile.TemporaryDirectory(prefix="mf_implicit_save_", dir=temp_dir_root) as tmp_dir:
            for field_name, value in field_items:
                data = _serialize_value(value, PICKLE_FORMAT)
                filename = field_name + ".pkl"
                with open(os.path.join(tmp_dir, filename), "wb") as f:
                    f.write(data)
                field_manifest[field_name] = {"format": PICKLE_FORMAT, "filename": filename}

            manifest_payload = {"version": 1, "fields": field_manifest}
            with open(os.path.join(tmp_dir, IMPLICIT_MANIFEST_FILENAME), "w") as f:
                json.dump(manifest_payload, f, indent=2)

            return self._checkpointer.save(
                path=tmp_dir,
                name=name,
                metadata=dict(metadata),
                internal_metadata=manifest_payload,
                latest=latest,
                storage_format=storage_format,
            )

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

        When users call `list` without any arguments, it will list all the checkpoints in the currently executing task (this includes all attempts). If the `list` method is called without any arguments outside a Metaflow Task execution context, it will raise an exception. Users can also call `list` with `attempt` argument to list all checkpoints within a the specific attempt of the currently executing task.

        When a `task` argument is provided, the `list` method will return all the checkpoints for a task's latest attempt unless a specific attempt number is set in the `attempt` argument. If the `Task` object contains a `DataArtifact` with all the previous checkpoints, then the `list` method will return all the checkpoints from the data artifact. If for some reason the DataArtifact is not written, then the `list` method will return all checkpoints directly from the checkpoint's datastore.

        Examples
        --------

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

        def _sort_checkpoints_by_version(dicts: List[Union[Dict, CheckpointArtifact]]):
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
        flow,
        temp_dir_root=None,
    ):
        """
        Loads a checkpoint and deserializes its fields back onto *flow*.

        Downloads the checkpoint identified by *reference* to a temporary
        directory, reads the ``__implicit_checkpoint__.json`` manifest, and
        calls ``setattr(flow, field_name, value)`` for each recorded field.

        Parameters
        ----------
        reference : str, dict, or CheckpointArtifact
            The checkpoint to load — a key string, artifact dict, or
            CheckpointArtifact object.

        flow : FlowSpec
            The Metaflow step's ``self`` — destination for deserialized values.
        """
        with tempfile.TemporaryDirectory(prefix="mf_implicit_load_", dir=temp_dir_root) as tmp_dir:
            load_checkpoint(checkpoint=reference, local_path=tmp_dir)

            manifest_path = os.path.join(tmp_dir, IMPLICIT_MANIFEST_FILENAME)
            if not os.path.exists(manifest_path):
                raise CheckpointException(
                    "Checkpoint does not contain an implicit manifest (%s). "
                    "This checkpoint was not saved in implicit mode."
                    % IMPLICIT_MANIFEST_FILENAME
                )

            with open(manifest_path, "r") as f:
                manifest_payload = json.load(f)

            fields_info = manifest_payload.get("fields", {})
            for field_name, field_info in fields_info.items():
                fmt = field_info["format"]
                filename = field_info["filename"]
                filepath = os.path.join(tmp_dir, filename)
                if not os.path.exists(filepath):
                    raise CheckpointException(
                        "Field file %r missing from checkpoint directory." % filename
                    )
                with open(filepath, "rb") as f:
                    data = f.read()
                value = _deserialize_value(data, fmt)
                setattr(flow, field_name, value)

    @staticmethod
    def inspect(
        reference: Union[str, Dict, CheckpointArtifact],
    ) -> Optional[Dict]:
        """
        Returns the implicit checkpoint manifest without downloading checkpoint files.

        Reads field names, formats, and filenames from the stored metadata record
        rather than downloading the checkpoint, making this a fast metadata-only
        operation suitable for inspecting large checkpoints.

        Parameters
        ----------
        reference : str, dict, or CheckpointArtifact
            The checkpoint to inspect — a key string, artifact dict, or
            CheckpointArtifact object.

        Returns
        -------
        dict or None
            The implicit manifest (``{"version": 1, "fields": {...}}``), or
            ``None`` if the checkpoint was not saved in implicit mode.
        """
        if isinstance(reference, CheckpointArtifact):
            return reference.implicit_manifest
        if isinstance(reference, dict):
            return CheckpointArtifact.hydrate(reference).implicit_manifest
        if isinstance(reference, str):
            art = CheckpointArtifact._load_metadata_from_key(reference, None)
            return art.implicit_manifest
        raise ValueError(
            "reference must be a CheckpointArtifact, dict, or key string, got %r"
            % type(reference)
        )

    def _search(
        self,
        pathspec: Optional[str] = None,
        root_prefix: Optional[str] = None,
        as_dict: bool = True,
    ):
        """
        __Experimental__ : Interface may break until this comment is removed.

        Search for checkpoints across tasks, steps, runs, or flows.

        This method provides a global search capability for checkpoints stored in the
        metadata store. Unlike the `list()` method which is scoped to the current task,
        `_search()` can query checkpoints across the entire checkpoint namespace. This method
        also directly relies on the checkpoint storage and completely bypasses the metaflow client
        API to retrieve objects directly present in the checkpoint storage.

        This method is designed for book-keeping over utility driven usage ( like needed
        during metaflow runtime ).

        The search scope is controlled by the `pathspec` parameter, which supports
        hierarchical filtering at different levels of granularity.

        Parameters
        ----------
        pathspec : str, optional
            Hierarchical filter to scope the search. Supports the following patterns:
            - `None` (default) - Search all checkpoints globally across all flows (slow)
            - `"flow_name"` - All checkpoints for a specific flow
            - `"flow_name/run_id"` - All checkpoints for a specific run
            - `"flow_name/run_id/step_name"` - All checkpoints for a specific step
            - `"flow_name/run_id/step_name/task_id"` - All checkpoints for a specific task

        root_prefix : str, optional
            Override the default checkpoint storage root prefix. This allows searching
            in alternate storage locations (e.g., HuggingFace Hub cache).
            If None, uses the default checkpoint storage prefix.
            For searching huggingface models stored by @huggingface_hub set the value
            to `mf.huggingface_hub`

        as_dict : bool, default True
            If True, yields checkpoint artifacts as dictionaries. If False, yields
            CheckpointArtifact objects. Dictionary format is useful for serialization,
            storage, or when you need a simple key-value representation.

        Yields
        ------
        dict or CheckpointArtifact
            If `as_dict=True` (default), yields dictionaries with checkpoint metadata.
            If `as_dict=False`, yields CheckpointArtifact objects.

            Each result contains metadata including:
            - key: Storage key for the checkpoint
            - pathspec: Metaflow pathspec (flow/run/step/task)
            - attempt: Task attempt number
            - version_id: Unique version identifier
            - name: Checkpoint name
            - created_on: Timestamp of creation
            - type: 'checkpoint'

        Raises
        ------
        ValueError
            If pathspec has more than 4 components (flow/run/step/task).

        Examples
        --------
        **Search all checkpoints globally:**

        ```python
        from metaflow_checkpoint import Checkpoint

        # Find all checkpoints across all flows (as dictionaries by default)
        # Note: Can be slow if you have many checkpoints
        for checkpoint_dict in Checkpoint._search():
            print(f"Found: {checkpoint_dict['key']}")
        ```

        **Search checkpoints for based on a flow:**

        ```python
        # Find all checkpoints for 'MyTrainingFlow'
        for checkpoint in Checkpoint._search(pathspec="MyTrainingFlow"):
            print(f"Flow checkpoint: {checkpoint['name']} (attempt {checkpoint['attempt']})")

        # Find all checkpoints for a specific run
        for checkpoint in Checkpoint._search(pathspec="MyTrainingFlow/1234"):
            print(f"Run checkpoint: {checkpoint['key']}")

        # Find all checkpoints saved in the 'train' step (as objects)
        for checkpoint in Checkpoint._search(pathspec="MyTrainingFlow/1234/train", as_dict=False):
            print(f"Step checkpoint: {checkpoint.version_id}")

        # Find all checkpoints for a specific task (across all attempts)
        for checkpoint in Checkpoint._search(pathspec="MyTrainingFlow/1234/train/5678"):
            print(f"Task checkpoint (attempt {checkpoint['attempt']}): {checkpoint['key']}")
        ```

        **Search in alternate storage (e.g., HuggingFace Hub):**

        ```python
        # Search HuggingFace models for the training flow
        for model in Checkpoint._search(pathspec="MyTrainingFlow", root_prefix="mf.huggingface_hub"):
            print(f"HF model: {model['key']}")

        # Export checkpoints to JSON for external tools
        import json
        checkpoints = list(Checkpoint._search(pathspec="MyTrainingFlow/1234"))
        with open('checkpoints.json', 'w') as f:
            json.dump(checkpoints, f, indent=2)
        ```
        """
        checkpointer = _instantiate_checkpointer_for_global_reads()
        if root_prefix is not None:
            checkpointer.set_root_prefix(root_prefix)

        for obj in _search_checkpoints_in_metadata_store(
            checkpointer._checkpoint_datastore.metadata_store, pathspec=pathspec
        ):
            if as_dict:
                yield obj.to_dict()
            else:
                yield obj
