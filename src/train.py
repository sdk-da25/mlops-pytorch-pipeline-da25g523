"""Training entrypoint for the CIFAR-10 classifier.

Reads all model/hyperparameter/path settings from
``configs/training_config.yaml``. Progress is logged two ways: as one JSON
object per line to stdout (no other output is printed, so stdout stays
JSON-lines parseable end to end), and as TensorBoard scalars under
``<checkpoint_dir>/tensorboard`` for interactive inspection (``tensorboard
--logdir <checkpoint_dir>/tensorboard``).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import get_dataloaders
from model import get_model

CONTAINER_CONFIG_PATH = "/app/configs/training_config.yaml"
LOCAL_CONFIG_PATH = "configs/training_config.yaml"


def load_config() -> dict:
    config_path = CONTAINER_CONFIG_PATH if Path(CONTAINER_CONFIG_PATH).exists() else LOCAL_CONFIG_PATH
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    """Run one epoch over ``loader``. Pass ``optimizer=None`` to evaluate."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_train):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            total += inputs.size(0)

    return total_loss / total, correct / total


def main() -> None:
    config = load_config()

    model_config = config["model"]
    training_config = config["training"]
    data_config = config["data"]
    output_config = config["output"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_dataloaders(
        data_dir=data_config["data_dir"], batch_size=training_config["batch_size"]
    )
    model = get_model(model_config["architecture"], model_config["num_classes"]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config["learning_rate"])

    checkpoint_dir = Path(output_config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / output_config["model_name"]

    writer = SummaryWriter(log_dir=str(checkpoint_dir / "tensorboard"))

    patience = training_config["early_stopping_patience"]
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, training_config["epochs"] + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, None, device)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        writer.add_scalar("Accuracy/val", val_accuracy, epoch)

        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 4),
                    "train_accuracy": round(train_accuracy, 4),
                    "val_loss": round(val_loss, 4),
                    "val_accuracy": round(val_accuracy, 4),
                }
            ),
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                },
                checkpoint_path,
            )
            print(
                json.dumps(
                    {
                        "event": "checkpoint_saved",
                        "epoch": epoch,
                        "val_loss": round(val_loss, 4),
                        "path": str(checkpoint_path),
                    }
                ),
                flush=True,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    json.dumps(
                        {
                            "event": "early_stopping",
                            "epoch": epoch,
                            "best_val_loss": round(best_val_loss, 4),
                        }
                    ),
                    flush=True,
                )
                break

    writer.close()
    print(json.dumps({"event": "training_complete", "best_val_loss": round(best_val_loss, 4)}), flush=True)


if __name__ == "__main__":
    main()
