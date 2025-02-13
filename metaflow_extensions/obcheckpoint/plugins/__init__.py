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
]
FLOW_DECORATORS_DESC = [
    (
        "with_artifact_store",
        ".machine_learning_utilities.datastore.decorator.ArtifactStoreFlowDecorator",
    )
]
DATASTORES_DESC = [
    (
        "s3-compatible",
        ".datastores.s3_compat.S3CompatibleStorage",
    )
]
###
# CONFIGURE: Similar to datatools, you can make visible under metaflow.plugins.* other
#            submodules not referenced in this file
###
__mf_promote_submodules__ = []
