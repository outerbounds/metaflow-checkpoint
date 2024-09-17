__mf_extensions__ = "obcheckpoint"

import pkg_resources
from ..plugins.machine_learning_utilities.checkpoints import (
    final_api as checkpoint_utils,
)
from ..plugins.machine_learning_utilities.datastructures import load_model

try:
    __version__ = pkg_resources.get_distribution("metaflow-checkpoint").version
except:
    # this happens on remote environments since the job package
    # does not have a version
    __version__ = None
