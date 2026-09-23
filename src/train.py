import torch
import os
from torch.utils.data import DataLoader
from model import ANN
from data import get_datasets


def train():
    train_dataset, _ = get_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    model = ANN()

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 5

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

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss: {total_loss / len(train_loader):.4f}"
        )

    # Tạo thư mục models nếu chưa tồn tại
    os.makedirs("models", exist_ok=True)

    # Lưu model
    torch.save(
        model.state_dict(),
        "models/ann_mnist.pth"
    )

    print("Model saved to models/ann_mnist.pth")


if __name__ == "__main__":
    train()