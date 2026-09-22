import torch
from torch.utils.data import DataLoader

from model import ANN
from data import get_datasets


def evaluate():

    # =========================
    # 1. Load test dataset
    # =========================
    _, test_dataset = get_datasets()

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False
    )

    # =========================
    # 2. Create model
    # =========================
    model = ANN()

    # =========================
    # 3. Load trained weights
    # =========================
    model.load_state_dict(
        torch.load(
            "models/ann_mnist.pth",
            map_location="cpu"
        )
    )

    # Chuyển model sang chế độ evaluation
    model.eval()

    # =========================
    # 4. Evaluate
    # =========================
    correct = 0
    total = 0

    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0

    with torch.no_grad():

        for images, labels in test_loader:

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            # Lấy class có xác suất/logit cao nhất
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    # =========================
    # 5. Calculate metrics
    # =========================
    accuracy = correct / total

    average_loss = total_loss / len(test_loader)

    print("=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)

    print(f"Test Loss     : {average_loss:.4f}")
    print(f"Test Accuracy : {accuracy:.4f}")
    print(f"Accuracy (%)  : {accuracy * 100:.2f}%")

    print("=" * 50)

    return accuracy


if __name__ == "__main__":
    evaluate()