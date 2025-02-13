from metaflow.decorators import FlowDecorator, StepDecorator
from metaflow.metadata_provider import MetaDatum
from metaflow.exception import MetaflowException


class ArtifactStoreFlowDecorator(FlowDecorator):
    """
    Allows setting external datastores to save data for the
    `@checkpoint`/`@model`/`@huggingface_hub` decorators.

    This decorator is useful when users wish to save data to a different datastore
    than what is configured in Metaflow. This can be for variety of reasons:

    1. Data security: The objects needs to be stored in a bucket (object storage) that is not accessible by other flows.
    2. Data Locality: The location where the task is executing is not located in the same region as the datastore.
        - Example: Metaflow datastore lives in US East, but the task is executing in Finland datacenters.
    3. Data Lifecycle Policies: The objects need to be archived / managed separately from the Metaflow managed objects.
        - Example: Flow is training very large models that need to be stored separately and will be deleted more aggressively than the Metaflow managed objects.

    Usage:
    ----------

    - Using a custom IAM role to access the datastore.

        ```python
        @with_artifact_store(
            type="s3",
            config=lambda: {
                "root": "s3://my-bucket-foo/path/to/root",
                "role_arn": ROLE,
            },
        )
        class MyFlow(FlowSpec):

            @checkpoint
            @step
            def start(self):
                with open("my_file.txt", "w") as f:
                    f.write("Hello, World!")
                self.external_bucket_checkpoint = current.checkpoint.save("my_file.txt")
                self.next(self.end)

        ```

    - Using credentials to access the s3-compatible datastore.

        ```python
        @with_artifact_store(
            type="s3",
            config=lambda: {
                "root": "s3://my-bucket-foo/path/to/root",
                "client_params": {
                    "aws_access_key_id": os.environ.get("MY_CUSTOM_ACCESS_KEY"),
                    "aws_secret_access_key": os.environ.get("MY_CUSTOM_SECRET_KEY"),
                },
            },
        )
        class MyFlow(FlowSpec):

            @checkpoint
            @step
            def start(self):
                with open("my_file.txt", "w") as f:
                    f.write("Hello, World!")
                self.external_bucket_checkpoint = current.checkpoint.save("my_file.txt")
                self.next(self.end)

        ```

    - Accessing objects stored in external datastores after task execution.

        ```python
        run = Run("CheckpointsTestsFlow/8992")
        with artifact_store_from(run=run, config={
            "client_params": {
                "aws_access_key_id": os.environ.get("MY_CUSTOM_ACCESS_KEY"),
                "aws_secret_access_key": os.environ.get("MY_CUSTOM_SECRET_KEY"),
            },
        }):
            with Checkpoint() as cp:
                latest = cp.list(
                    task=run["start"].task
                )[0]
                print(latest)
                cp.load(
                    latest,
                    "test-checkpoints"
                )

        task = Task("TorchTuneFlow/8484/train/53673")
        with artifact_store_from(run=run, config={
            "client_params": {
                "aws_access_key_id": os.environ.get("MY_CUSTOM_ACCESS_KEY"),
                "aws_secret_access_key": os.environ.get("MY_CUSTOM_SECRET_KEY"),
            },
        }):
            load_model(
                task.data.model_ref,
                "test-models"
            )
        ```
    Parameters:
    ----------

    type: str
        The type of the datastore. Can be one of 's3', 'gcs', 'azure' or any other supported metaflow Datastore.

    config: dict or Callable
        Dictionary of configuration options for the datastore. The following keys are required:
        - root: The root path in the datastore where the data will be saved. (needs to be in the format expected by the datastore)
            - example: 's3://bucket-name/path/to/root'
            - example: 'gs://bucket-name/path/to/root'
            - example: 'https://myblockacc.blob.core.windows.net/metaflow/'
        - role_arn (optional): AWS IAM role to access s3 bucket (only when `type` is 's3')
        - session_vars (optional): AWS session variables to access s3 bucket (only when `type` is 's3')
        - client_params (optional): AWS client parameters to access s3 bucket (only when `type` is 's3')

    """

    name = "with_artifact_store"
    defaults = {
        "type": None,  # can be one of 's3', 'gcs', 'azure'
        "config": None,  # Dictionary of configuration options
    }

    def _export_context(self):
        _flw_deco = self
        _config = _flw_deco.attributes["config"]
        # Check if the config is a callable lambda
        if callable(_flw_deco.attributes["config"]):
            _config = _flw_deco.attributes["config"]()
        return {"type": _flw_deco.attributes["type"], "config": _config}

    def flow_init(
        self, flow, graph, environment, flow_datastore, metadata, logger, echo, options
    ):
        from .context import datastore_context

        cannot_be_none = ["type", "config"]
        for k in cannot_be_none:
            if self.attributes[k] is None:
                raise MetaflowException(
                    f"@{self.name} cannot have {k} be set to `None`."
                )
        _current_context = self._export_context()
        datastore_context.flow_init_context(_current_context)
        task_md = datastore_context.to_task_metadata()
        metadata.add_sticky_tags(
            sys_tags=["%s:%s" % (k, v) for k, v in task_md.items()]
        )


def set_datastore_context(flow, metadata, run_id, step_name, task_id, retry_count):
    from .context import datastore_context

    _flw_deco = None
    for decorators in flow._flow_decorators.values():
        for dec in decorators:
            if isinstance(dec, ArtifactStoreFlowDecorator):
                _flw_deco = dec
        else:
            continue

    if _flw_deco is None:
        # print("No datastore context found to set")
        return None

    _current_context = _flw_deco._export_context()
    datastore_context.flow_init_context(_current_context)

    task_md = datastore_context.to_task_metadata()
    entries = [
        MetaDatum(
            field=k,
            value=v,
            type="mf-checkpoint-artifact-store",
            tags=[
                "attempt_id:%s" % str(retry_count),
            ],
        )
        for k, v in task_md.items()
    ]
    metadata.register_metadata(run_id, step_name, task_id, entries)
