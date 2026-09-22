from torchvision import datasets
from torchvision.transforms import ToTensor


def get_datasets():

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    return train_dataset, test_dataset