"""
Example: crash-safe PyTorch training with @checkpoint.

Demonstrates the core patterns for using @checkpoint in a real training loop:

  - @retry + @checkpoint(load_policy="fresh") makes the step resumable after
    any interruption (OOM, preemption, spot-instance eviction, etc.)
  - On retry, the latest checkpoint is detected automatically and
    current.checkpoint.load() restores model weights, optimizer state, and
    the current epoch onto self — training continues from where it stopped
  - include=[...] limits checkpointing to training-state fields only,
    avoiding accidentally pickling large intermediate data
  - Named checkpoints ("latest" every N epochs, "best" on val-loss improvement)
    let downstream steps or external tooling retrieve a specific model version

Run:
    python flows/example_pytorch_training.py run
    python flows/example_pytorch_training.py run --epochs 50 --lr 1e-2
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from metaflow import FlowSpec, Parameter, card, checkpoint, current, image, retry, step, gpu, T4

from checkpoint_card import emit_checkpoint_dir_card


class CheckpointedPyTorchTrainingFlow(FlowSpec):
    """
    Trains a small MLP on a synthetic regression task with checkpointing.

    The training step is decorated with @retry + @checkpoint so that any
    interruption is transparently recovered: on the next attempt the model
    and optimizer are restored from the most recent checkpoint and the loop
    continues from the saved epoch rather than epoch 0.
    """

    epochs = Parameter("epochs", default=5, type=int, help="Total training epochs")
    lr = Parameter("lr", default=1e-3, type=float, help="Adam learning rate")
    batch_size = Parameter("batch_size", default=128, type=int, help="Mini-batch size")
    save_every = Parameter(
        "save_every", default=1, type=int, help="Save a 'latest' checkpoint every N epochs"
    )

    @step
    def start(self):
        self.next(self.train)

    @retry(times=2)
    @checkpoint(
        load_policy="fresh",
        # Only checkpoint the fields that define training state.
        # Flow-level artifacts (e.g. final_* set after the loop) are excluded.
        include=["epoch", "best_loss", "model_state", "optimizer_state"],
    )
    @card(id="checkpoint_dir")
    @image(
        base_image="pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime",
        python_packages=["torchvision"],
    )
    @gpu(flavor=T4.size_1x)
    @step
    def train(self):
        """
        Main training loop.

        On the first attempt:
          - model and optimizer are initialised from scratch
          - best_loss is set to inf

        On retry (after any failure):
          - current.checkpoint.is_loaded will be True
          - current.checkpoint.load() restores self.epoch, self.model_state,
            self.optimizer_state, and self.best_loss from the latest checkpoint
          - the loop resumes from self.epoch + 1
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        N_FEATURES = 16

        # --- Model ---
        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(N_FEATURES, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                )

            def forward(self, x):
                return self.net(x)

        # --- Dataset ---
        torch.manual_seed(42)
        n = 2000
        X = torch.randn(n, N_FEATURES)
        w = torch.randn(N_FEATURES, 1) * 0.5
        y = X @ w + 0.05 * torch.randn(n, 1)
        split = int(0.8 * n)
        train_ds = TensorDataset(X[:split], y[:split])
        val_ds = TensorDataset(X[split:], y[split:])

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        model = MLP()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        start_epoch = 0

        # ------------------------------------------------------------------
        # Resume from checkpoint when retrying after a failure
        # ------------------------------------------------------------------
        if current.checkpoint.is_loaded:
            print("[@checkpoint] Resuming from checkpoint …")
            current.checkpoint.load()
            model.load_state_dict(self.model_state)
            optimizer.load_state_dict(self.optimizer_state)
            start_epoch = self.epoch + 1
            print(
                "  resumed at epoch %d  (best val loss so far: %.4f)"
                % (start_epoch, self.best_loss)
            )
        else:
            self.best_loss = float("inf")

        # ------------------------------------------------------------------
        # Training loop
        # ------------------------------------------------------------------
        train_loss = val_loss = 0.0

        for epoch in range(start_epoch, self.epochs):

            # --- train ---
            model.train()
            train_loss = 0.0
            for X, y in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # --- validate ---
            model.eval()
            with torch.no_grad():
                val_loss = (
                    sum(criterion(model(X), y).item() for X, y in val_loader)
                    / len(val_loader)
                )

            print(
                "epoch %3d/%d  train=%.4f  val=%.4f"
                % (epoch + 1, self.epochs, train_loss, val_loss)
            )

            # Snapshot training state so the checkpoint captures the latest values
            self.epoch = epoch
            self.model_state = model.state_dict()
            self.optimizer_state = optimizer.state_dict()

            # --- periodic checkpoint ---
            if (epoch + 1) % self.save_every == 0:
                current.checkpoint.save(
                    name="latest",
                    metadata={"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss},
                )
                print("  → latest checkpoint saved (epoch %d)" % epoch)

            # --- best-model checkpoint ---
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                current.checkpoint.save(
                    name="best",
                    metadata={"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss},
                )
                print("  → best checkpoint saved (val_loss=%.4f)" % val_loss)

        self.final_train_loss = train_loss
        self.final_val_loss = val_loss
        emit_checkpoint_dir_card()
        self.next(self.end)

    @step
    def end(self):
        print("\n=== Training complete ===")
        print("  final train loss : %.4f" % self.final_train_loss)
        print("  final val loss   : %.4f" % self.final_val_loss)
        print("  best val loss    : %.4f" % self.best_loss)


if __name__ == "__main__":
    CheckpointedPyTorchTrainingFlow()
