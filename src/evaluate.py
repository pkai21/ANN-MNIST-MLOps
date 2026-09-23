import json
import os
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

    correct = 0
    total = 0

    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0

    with torch.no_grad():

        for images, labels in test_loader:

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

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

    print(f"Test Loss     : {average_loss:.4f}")
    print(f"Test Accuracy : {accuracy:.4f}")
    print(f"Accuracy (%)  : {accuracy * 100:.2f}%")

    print("=" * 50)

    metadata = {
        "model": "ANN-MNIST",
        "version": os.getenv("GITHUB_SHA", "local"),
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100,
        "test_loss": average_loss,
        "threshold": 0.95,
        "epochs": 5,
        "learning_rate": 0.001
    }

    os.makedirs("models", exist_ok=True)

    with open("models/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("Model metadata saved to models/model_metadata.json")

    return accuracy


if __name__ == "__main__":
    accuracy = evaluate()

    threshold = 0.95

    if accuracy < threshold:
        raise RuntimeError(
            f"Model accuracy {accuracy:.4f} "
            f"is below required threshold {threshold:.4f}"
        )

    print(
        f"Model quality check passed: "
        f"{accuracy:.4f} >= {threshold:.4f}"
    )