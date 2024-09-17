STEP_DECORATORS_DESC = [
    (
        "checkpoint",
        ".machine_learning_utilities.checkpoints.decorator.CheckpointDecorator",
    ),
    (
        "model",
        ".machine_learning_utilities.modeling_utils.decorator.ModelDecorator",
    ),
    (
        "huggingface_hub",
        ".machine_learning_utilities.hf_hub.decorator.HuggingfaceHubDecorator",
    ),
    (
        "disk_profiler",
        ".machine_learning_utilities.checkpoints.cards.diskspace_usage_card.DiskUsageProfilerDecorator",
    ),
]

###
# CONFIGURE: Similar to datatools, you can make visible under metaflow.plugins.* other
#            submodules not referenced in this file
###
__mf_promote_submodules__ = []
