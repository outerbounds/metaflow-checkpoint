@checkpoint Decorator for Metaflow
Problem Statement
Metaflow users working with @checkpoint face a few key challenges:

Manual checkpointing / automatic retries

There’s quite a bit of boilerplate code just to get retries due to the generic support of any type of artifact. This means that anything saved must have its own custom saving and loading. A certain schema set could instead offer a more seamless way to save and load generic artifacts.

External registry integrations / extended persistence

The current @checkpoint decorator applies to singular runs and retries. However, there are times when checkpoints will come from past runs, thus requiring some integrations with external model registries (e.g. Huggingface hub). Saving of checkpoints would also be useful if saved to model registries for any downstream processing (e.g. evaluating each checkpoint).

Integration with distributed computing

Flows that use distributed computing often have an odd paradigm where a training script launcher is added onto Metaflow’s current object. The launcher is akin to a CLI tool (e.g. torchrun) in that complex objects cannot be passed to it - only strings and other primitive values. Current usage of @checkpoint relies on access to Metaflow’s current object for other attributes (e.g. current.checkpoint.save()) which may get lost when executing distributed training.
Goals & Non-Goals
Goals
Checkpoint usage is simplified and easy to integrate with automatic retries
Checkpointing is possible with distributed workflows
Checkpoints can be persisted and retrieved past a step
Non-Goals


User Experience: In-step checkpointing
Before: Current @checkpoint

import os
import random
from metaflow import FlowSpec, current, step, retry, checkpoint

class CheckpointCounterFlow(FlowSpec):
    @step
    def start(self):
        self.counter = 0
        self.next(self.flaky_count)

    @checkpoint
    @retry
    @step
    def flaky_count(self):
        cp_path = os.path.join(current.checkpoint.directory, "counter")

        def _save_counter():
            print(f"Checkpointing counter value {self.counter}")
            with open(cp_path, "w") as f:
                f.write(str(self.counter))
            self.latest_checkpoint = current.checkpoint.save()

        def _load_counter():
            if current.checkpoint.is_loaded:
                with open(cp_path) as f:
                    self.counter = int(f.read())
                print(f"Checkpoint loaded!")

        _load_counter()
        print("Counter is now", self.counter)

        while self.counter < 10:
            self.counter += 1
            if self.counter % 2 == 0:
                _save_counter()

            if random.random() < 0.2:
                raise Exception("Bad luck! Try again!")

        self.next(self.end)

    @step
    def end(self):
        print("Final counter", self.counter)

After: With “explicit” checkpoint artifacts
Notes:
“Explicit” here refers to the user explicitly telling Metaflow what objects to checkpoint
Eliminates need for users to deal with flimsy file paths compared to current

Open questions:
Is this still too much overhead?

class HfRegistryCheckpointFlow(FlowSpec):
    @step
    def start(self):
        self.next(self.train)


    @retry
    @huggingface(models=["meta-llama/Llama-2-7b"])
    @checkpoint(schema={
        "meta-llama/Llama-2-7b": {
            ".model": "torchscript",
            ".state_dict": "pt_state_dict",
            "epoch": "txt",
            "other1": "bytes",  # generic bytes type
            "other2": CheckpointType(),  # could let people define their own cp type
        }
    })
    @step
    def train(self):
        # use huggingface decorator directly
        latest_model_path = current.huggingface.models["meta-llama/Llama-2-7b"]


        # but if checkpoint decorator exists, then you can bypass the paths above
        llama_cp = current.checkpoints["meta-llama/Llama-2-7b"]
        
        # load checkpoint artifacts - loaded using the defined schema
        if llama_cp.is_loaded:
            loaded_cp = llama_cp.load()
            model = loaded_cp["model"]
            state_dict = loaded_cp["state_dict"]
            epoch = loaded_cp["epoch"]
            model.load_state_dict(state_dict)
        else:
            model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
            state_dict = model.state_dict()
            epoch = 0


        for epoch in range(epoch, 10): # simulate training epochs
            # train the model
            model.train()
            state_dict = model.state_dict()


            # save checkpoint artifacts - saved using the defined schema
            llama_cp.save({
                "model": model,
                "state_dict": state_dict,
                "epoch": epoch + 1,
            })


        self.next(self.end)


    @step
    def end(self):
        pass
After: With “implicit” checkpoint artifacts
Notes: 
“Implicit” here refers to Metaflow checkpointing everything on “self” without any user directive to do so.
Even more hands-off than “explicit” checkpoint artifacts but introduces more points of failure from Metaflow

Open questions
How do we deal with non-pickleable objects?
Let user define serialization, exclude list, etc

class ImplicitCheckpointFlow(FlowSpec):
    @step
    def start(self):
        self.next(self.train_cp)

    @huggingface(models=["meta-llama/Llama-2-7b"])
    @checkpoint(
         fields=["model", "epoch"]
         serialization_config={"model": "raw"},
    )
    @step
    def train_cp(self):
        # if checkpoint is loaded, load it, otherwise user defines defaults
        if current.checkpoint.is_loaded:
            current.checkpoint.load()
        else:
            self.model = current.huggingface.models["meta-llama/Llama-2-7b"]
            self.epoch = 0
            current.checkpoint.save()

        for epoch in range(epoch, 10):
            # train the model
            self.model.train()
            self.epoch = epoch

            # automatically checkpoint everything that exists on self
            current.checkpoint.save()

        self.next(self.end)

    @step
    def end(self):
        pass


