# Metaflow Checkpoint

Imagine running a machine learning training job or any data processing task that takes hours or even days to complete. In such scenarios, you don't want failures or collaboration complexities to force you to start over and lose all the progress made. **This is where Metaflow's new decorators—`@checkpoint`, `@model`, and `@huggingface_hub`—come into play.** These decorators are specifically designed to address these challenges by simplifying checkpointing, model management, and efficient loading of external models, **ensuring that your long-running jobs can be resumed seamlessly after a failure and that models and checkpoints are properly versioned in multi-user environments.**

This repository introduces three new decorators for [Metaflow](https://metaflow.org) that address these challenges:

- **`@checkpoint`**: Simplifies saving and reloading checkpoints within your Metaflow flows.
- **`@huggingface_hub`**: Enables efficient loading and caching of large models from Hugging Face Hub.
- **`@model`**: Allows for easy saving and loading of models created during your Metaflow flows.

Examples for these decorators can be found in [this repository](https://github.com/outerbounds/metaflow-checkpoint-examples/tree/master). 

## Features

### `@checkpoint` Decorator

The `@checkpoint` decorator alleviates the pain points associated with saving and reloading the state of your program (a Metaflow `@step`) in Metaflow flows. It also handles version control in multi-user settings by isolating checkpoints per user and run. Whether it's a checkpoint created by a machine learning model or intermediate data required in case of crashes, this decorator simplifies state management and failure recovery.

- **Checkpointing**: Save the state of your `@step` at designated points.
- **Seamless Recovery**: Restart your job from the last checkpoint upon retries without any manual intervention.
- **User Isolation**: Checkpoints are managed per user to prevent overwriting in collaborative environments.
- **Ease of Use**: Minimal code changes required to implement checkpointing.

### `@huggingface_hub` Decorator

The `@huggingface_hub` decorator allows you to load large models from Hugging Face Hub and cache them for increased performance benefits. It also ensures that models are versioned and managed appropriately in multi-user environments.

- **Efficient Model Loading**: Load models on-the-fly from Hugging Face Hub.
- **Caching Mechanism**: Cache models locally to avoid redundant downloads.
- **Version Control**: Manages different versions of models to prevent conflicts.
- **Integration with Metaflow**: Easily incorporate models across your Metaflow flows.

### `@model` Decorator

The `@model` decorator provides a trivial way to save and load models/checkpoints created as part of your Metaflow flow. 

- **Simplified Model Loading**: Automatically load models based on references and identifiers created by decorators such as `@model`/`@checkpoint`/`@huggingface_hub`. 
- **Model Identity**: Associates a uniquie identity to models so that there is clear distinction between different versions making it easy to track their lineage. 


