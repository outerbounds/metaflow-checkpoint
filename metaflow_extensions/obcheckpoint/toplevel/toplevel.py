__mf_extensions__ = "obcheckpoint"

from ..plugins.machine_learning_utilities.checkpoints.final_api import Checkpoint
from ..plugins.machine_learning_utilities.datastructures import load_model
from ..plugins.machine_learning_utilities.datastore.context import artifact_store_from

try:
    # Look up version using importlib.metadata
    from importlib.metadata import version

    __version__ = version("metaflow_checkpoint")
except:
    try:
        from .version import package_version as _version

        __version__ = _version
    except:
        # this happens on remote environments since the job package
        # does not have a version
        __version__ = None
