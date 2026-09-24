import requests
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor


API_URL = "http://127.0.0.1:8000"


def test_health():
    response = requests.get(
        f"{API_URL}/health"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_prediction():
    dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    image, label = dataset[0]

    pixels = image.flatten().tolist()

    response = requests.post(
        f"{API_URL}/predict",
        json={
            "pixels": pixels
        }
    )

    assert response.status_code == 200

    data = response.json()

    prediction = data["prediction"]
    confidence = data["confidence"]

    assert 0 <= prediction <= 9
    assert 0 <= confidence <= 1
    assert prediction == int(label)