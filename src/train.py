import json
import os
from datetime import datetime, timezone

import torch
from torch.utils.data import DataLoader

from model import ANN
from data import get_datasets

def get_model_version():
    version_file = "VERSION"

    if not os.path.exists(version_file):
        raise FileNotFoundError(
            "VERSION file not found"
        )

    with open(
        version_file,
        "r",
        encoding="utf-8"
    ) as f:
        version = f.read().strip()

    if not version:
        raise ValueError(
            "VERSION file is empty"
        )

    return version

def train():
    train_dataset, _ = get_datasets()

    batch_size = 64
    learning_rate = 0.001
    epochs = 5

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    model = ANN()

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    for epoch in range(epochs):

        model.train()

        total_loss = 0

        for images, labels in train_loader:

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss: {average_loss:.4f}"
        )

    # ============================================================
    # Save model
    # ============================================================

    os.makedirs("models", exist_ok=True)

    model_path = "models/ann_mnist.pth"

    torch.save(
        model.state_dict(),
        model_path
    )

    print(f"Model saved to: {model_path}")

    # ============================================================
    # Create model metadata
    # ============================================================

    metadata = {
        "model_name": "ANN-MNIST",

        "model_version": get_model_version(),

        "git_commit": os.getenv(
            "GITHUB_SHA",
            "local"
        ),

        "training_date": datetime.now(
            timezone.utc
        ).isoformat(),

        "epochs": epochs,

        "learning_rate": learning_rate,

        "batch_size": batch_size,

        "test_loss": None,

        "test_accuracy": None
    }

    metadata_path = "models/model_metadata.json"

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    print(
        f"Metadata saved to: {metadata_path}"
    )


if __name__ == "__main__":
    train()