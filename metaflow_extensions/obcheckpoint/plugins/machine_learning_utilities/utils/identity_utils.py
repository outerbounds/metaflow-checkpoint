from hashlib import sha256
import json
from typing import TYPE_CHECKING
from metaflow.exception import MetaflowException
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


class IdentityException(MetaflowException):
    pass


class UnhashableValueException(MetaflowException):
    def __init__(self, type):
        self._type = type
        super().__init__(f"Value of type %s is not safely hashable." % type)


class FlowNotRunningException(MetaflowException):
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
            raise UnhashableValueException(str(type(value)))


def _resolve_hash_of_current_foreach_stack(stack):

    stack_of_values = [x[2] for x in stack]

    # We add a value to the stack which is a deterministic value that is random enough that clashes become unlikely.
    # Otherwise if the foreach value can contain a value that is same as the stepname, then the hash of the foreach
    # value and the stepname will be the same causing a potential state clash
    stack_of_values.append(safe_hash("foreach-task-entropy"))
    # create a hash of all the inputs in the stack.
    hashed_inputs = "".join([safe_hash(x) for x in stack_of_values])

    foreach_task_identifier = sha256((hashed_inputs).encode()).hexdigest()

    return foreach_task_identifier


def _resolve_hash_for_particular_parallel_index(
    run: "metaflow.FlowSpec",
    index,
):
    # It is assumed here that the function is only called for a run
    # with a parallel task.
    stack = run.foreach_stack()
    # to derive the task identifier of the gang scheduled parallel jobs
    # we need to ensure that all tasks within the gang get the same task identifer.
    # This value is based on the index of the task provided to this function. (for most default cases it will be 0)
    # We achieve this be poping the index of the foreach stack and replacing it with the index of the task.
    stack_of_values = [x[2] for x in stack]
    stack_of_values.pop()
    # When we replace the index of the foreach stack, we can also potentially hit a situation where the task
    # is converted from the parallel task to a regular foreach task. At this point we need to ensure that the
    # task identifier is not the same as the parallel task. To ensure this, we add a value to the stack which
    # is a deterministic value that is random enough that clashes become unlikely.
    stack_of_values.extend([index, safe_hash("parallel-task-entropy")])

    # create a hash of all the inputs in the stack.
    hashed_inputs = "".join([safe_hash(x) for x in stack_of_values])

    parallel_task_identifier = sha256((hashed_inputs).encode()).hexdigest()

    return parallel_task_identifier


class TaskIdentifier:
    """
    Any change to this class's core logic can create SEVERE backwards compatibility issues
    since this class helps derive the task identifier for the checkpoints.

    IDEALLY, the identifier construction logic of this file should be kept as is.
    """

    @classmethod
    def for_parallel_task_index(cls, run: "metaflow.FlowSpec", index: int):
        """
        This class is meant to mint a task-identifier for a parallel task based on the
        index of the task in the gang.
        """
        base_task_identifier = _resolve_hash_for_particular_parallel_index(run, index)
        return base_task_identifier[:MAX_HASH_LEN]

    @classmethod
    def from_flowspec(cls, run: "metaflow.FlowSpec"):
        from metaflow import current

        if not current.is_running_flow:
            raise FlowNotRunningException

        base_task_identifier = None
        # Only resolving identifier for foreach value first.
        # If it is not present, then we will resolve the identifier
        # based on the stepname.
        if run.index is not None:
            stack = run.foreach_stack()
            base_task_identifier = _resolve_hash_of_current_foreach_stack(stack)
        else:
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
