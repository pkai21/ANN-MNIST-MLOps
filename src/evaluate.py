import json
import os
from datetime import datetime, timezone

import torch
from torch.utils.data import DataLoader

from model import ANN
from data import get_datasets


def evaluate():

    _, test_dataset = get_datasets()

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False
    )

    model = ANN()

    model.load_state_dict(
        torch.load(
            "models/ann_mnist.pth",
            map_location="cpu"
        )
    )

    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():

        for images, labels in test_loader:

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    accuracy = correct / total

    average_loss = (
        total_loss / len(test_loader)
    )

    print("=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    print(
        f"Test Loss     : {average_loss:.4f}"
    )
    print(
        f"Test Accuracy : {accuracy:.4f}"
    )
    print(
        f"Accuracy (%)  : {accuracy * 100:.2f}%"
    )
    print("=" * 50)

    # ============================================================
    # Update model metadata
    # ============================================================

    metadata_path = "models/model_metadata.json"

    if os.path.exists(metadata_path):

        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as f:

            metadata = json.load(f)

    else:

        metadata = {
            "model_name": "ANN-MNIST"
        }

    metadata["test_loss"] = average_loss

    metadata["test_accuracy"] = accuracy

    metadata["evaluation_date"] = datetime.now(
        timezone.utc
    ).isoformat()

    metadata["git_commit"] = os.getenv(
        "GITHUB_SHA",
        metadata.get(
            "git_commit",
            "local"
        )
    )

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
        f"Metadata updated: {metadata_path}"
    )

    return accuracy


if __name__ == "__main__":
    evaluate()