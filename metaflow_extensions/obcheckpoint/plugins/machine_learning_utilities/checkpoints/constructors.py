from typing import List, Dict, Union, Tuple, Optional, TYPE_CHECKING
from .core import (
    Checkpointer,
    WriteResolver,
    ReadResolver,
)
from ..datastructures import CheckpointArtifact
from .constants import CHECKPOINT_UID_ENV_VAR_NAME, DEFAULT_NAME
import os

if TYPE_CHECKING:
    import metaflow
    from .final_api import Checkpoint


def _instantiate_checkpoint(
    checkpoint_object: "Checkpoint",
    flow: Optional["metaflow.FlowSpec"] = None,  # User Facing
    task_identifier: Optional[str] = None,  # Advanced options
    scope: Optional[str] = None,  # Advanced options
    gang_scheduled_task: bool = False,
):
    if flow is not None:
        datastore, resolver_info = WriteResolver.from_run(
            flow,
            scope=scope,
            task_identifier=task_identifier,
            gang_scheduled_task=gang_scheduled_task,
        )
    elif CHECKPOINT_UID_ENV_VAR_NAME in os.environ:
        datastore, resolver_info = WriteResolver.from_environment()
    else:
        raise ValueError(
            "Creating a `Checkpoint` object requires either a `FlowSpec` instance or a `METAFLOW_CHECKPOINT_UID` environment variable."
        )

    # Technically at this point `resolver_info` contains all the information to create
    # any new checkpointer capable of writing.
    if any(
        a is None for a in resolver_info
    ):  # Ideally we should NOT be hitting this line at ALL!
        raise ValueError(
            "Missing enough information to instantiate a Checkpoint object."
        )
    chkpt = Checkpointer(
        datastore=datastore,
        attempt=resolver_info.attempt,
    )
    chkpt._checkpoint_uid = WriteResolver.construct_checkpoint_id(resolver_info)
    checkpoint_object._set_checkpointer(chkpt)
    return checkpoint_object


# Technically this method should ensure that the Checkpointer also takes the
# attempts as a part of the `load` method.
# READ METHOD
def load_checkpoint(  # Return load status
    checkpoint: Union[CheckpointArtifact, dict, str],  # User Facing
    local_path: str,  # User Facing
):
    # It's assumed that everything is already correctly passsed
    if isinstance(checkpoint, CheckpointArtifact) or isinstance(checkpoint, dict):
        _load_checkpoint_from_reference(checkpoint, local_path)
    elif isinstance(checkpoint, str):
        _load_checkpoint_from_key(checkpoint, local_path)
    else:
        raise ValueError(
            "Invalid checkpoint object. Expected CheckpointArtifact, dict or str. Got: %s"
            % type(checkpoint)
        )


def _load_checkpoint_from_reference(
    checkpoint: Union[CheckpointArtifact, dict],  # User Facing
    local_path: str,  # User Facing
):
    # this will already set the version on the checkpointer.
    _checkpointer = Checkpointer._from_checkpoint(checkpoint)
    _name = CheckpointArtifact.hydrate(checkpoint).name
    version = _checkpointer.current_version
    _checkpointer._load_checkpoint(local_path, version_id=version, name=_name)


def _load_checkpoint_from_key(
    checkpoint_key: str,
    local_path: str,
):
    _checkpointer = Checkpointer._from_key(checkpoint_key)
    key_decomp = _checkpointer._checkpoint_datastore.decompose_key(checkpoint_key)
    _checkpointer._load_checkpoint(
        local_path, version_id=key_decomp.version_id, name=key_decomp.name
    )


def _coalesce(*args):
    for arg in args:
        if arg is not None:
            return arg
    return None
