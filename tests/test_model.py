import torch

from src.model import ANN


def test_model_output_shape():

    model = ANN()

    x = torch.randn(4, 1, 28, 28)

    output = model(x)

    assert output.shape == (4, 10)