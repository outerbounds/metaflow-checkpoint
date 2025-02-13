__mf_extensions__ = "obcheckpoint"

from ..plugins.machine_learning_utilities.checkpoints.final_api import Checkpoint
from ..plugins.machine_learning_utilities.datastructures import load_model
from ..plugins.machine_learning_utilities.datastore.context import artifact_store_from

try:
    import pkg_resources

    __version__ = pkg_resources.get_distribution("metaflow-checkpoint").version
except:
    # this happens on remote environments since the job package
    # does not have a version
    __version__ = None
