from hashlib import sha256
import json
from typing import TYPE_CHECKING
from metaflow.exception import MetaflowException
from ..exceptions import TODOException
from .constants import MAX_HASH_LEN


if TYPE_CHECKING:
    import metaflow


class AttemptEnum:
    LATEST = "latest"
    PREVIOUS = "previous"
    FIRST = "first"

    @classmethod
    def values(cls):
        return [cls.LATEST, cls.PREVIOUS, cls.FIRST]


class ResumeTaskNotFoundError(MetaflowException):
    pass


def pathspec_hash(pathspec: str):
    return safe_hash(pathspec)[:8]


def safe_hash(value):
    """Safely hash arbitrary values and a created a sha for that object, including those that are not natively hashable."""
    try:
        # Try to hash the value directly, works for immutable objects
        return sha256(hash(value)).hexdigest()
    except TypeError:
        # If direct hashing fails, serialize the object and hash the serialized string
        # This is slower but will work for un-hashable objects like lists and dicts
        try:
            serialized = json.dumps(value, sort_keys=True).encode()
            return sha256(serialized).hexdigest()
        except TypeError as e:
            # Handle the case where the object cannot be serialized
            raise ValueError(f"Value of type {type(value)} is not safely hashable: {e}")


def _resolve_hash_of_current_foreach_stack(run: "metaflow.FlowSpec"):
    """
    There are several nuances to this problem.
    1. The `stack_of_values` can have duplicates.
        - Duplicates can be a problem because the
    """
    if run.index is None:
        return None

    stack = run.foreach_stack()
    if stack is None:
        return None

    stack_of_values = [x[2] for x in stack]
    # create a hash of all the inputs in the stack.
    hashed_inputs = "".join([safe_hash(x) for x in stack_of_values])

    foreach_task_identifier = sha256((hashed_inputs).encode()).hexdigest()

    return foreach_task_identifier


def _resolve_hash_for_particular_parallel_index(
    run: "metaflow.FlowSpec",
    index,
):
    if run.index is None:
        return None
    stack = run.foreach_stack()
    if stack is None:
        return None
    stack_of_values = [x[2] for x in stack]
    stack_of_values.pop()
    stack_of_values.append(index)
    # create a hash of all the inputs in the stack.
    hashed_inputs = "".join([safe_hash(x) for x in stack_of_values])

    parallel_task_identifier = sha256((hashed_inputs).encode()).hexdigest()

    return parallel_task_identifier


class TaskIdentifier:
    @classmethod
    def from_flowspec_origin_run(cls, run: "metaflow.FlowSpec"):
        from metaflow import current

        if not current.is_running_flow:
            raise TODOException("TODO: Set error about Checkpointer(run=self).")

        raise NotImplementedError

    @classmethod
    def for_parallel_task_index(cls, run: "metaflow.FlowSpec", index: int):
        """
        This class is meant to mint a task-identifier for a parallel task based on the
        index of the task in the gang.
        """
        base_task_identifier = _resolve_hash_for_particular_parallel_index(run, index)
        if base_task_identifier is None:
            raise TODOException(
                "TODO: Unable to resolve the base task idenfitier for the parallel task."
            )
        return base_task_identifier[:MAX_HASH_LEN]

    @classmethod
    def from_flowspec(cls, run: "metaflow.FlowSpec"):
        from metaflow import current

        if not current.is_running_flow:
            raise TODOException("TODO: Set error when flow is not running .")

        base_task_identifier = None
        # Only resolving identifier for foreach value first.
        # If it is not present, then we will resolve the identifier
        # based on the `task-id` of the previous task.
        base_task_identifier = _resolve_hash_of_current_foreach_stack(run)

        attempt = current.retry_count
        if base_task_identifier is None:
            # If it is not a foreach task, the the identifier can be
            # set a static value (like hash of the stepname) since it will
            # make list prefixing a lot simpler and the logic of understanding
            # base identity a lot simpler. Otherwise we will have
            # to distinguish between how we resolve the identifier for
            # foreach and non-foreach tasks in different usecases.
            step_name = current.step_name
            base_task_identifier = safe_hash(step_name)

        # TODO: Notify the user of the following cases, when Metaflow can __might__ have a state clash.:
        # - base_task_identifier from foreach stack has the same hash as hash of the step_name.
        # - the base_task_identifier is not computable for the foreach stack because the values
        #   are not computable.

        return base_task_identifier[:MAX_HASH_LEN]

    def from_task(cls, task: "metaflow.Task", use_origin=True):

        return safe_hash(task.parent.id)[:MAX_HASH_LEN]
