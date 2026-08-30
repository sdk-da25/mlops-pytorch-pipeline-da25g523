"""Unit tests for the CIFAR-10 model factory.

These tests only exercise ``model.get_model`` and never touch
``dataset.py``, so no CIFAR-10 download is triggered. ``pyproject.toml``
puts ``src`` on the pytest path.
"""
import pytest
import torch

from model import get_model


@pytest.mark.parametrize("architecture", ["resnet18", "simplecnn"])
def test_get_model_output_shape(architecture: str) -> None:
    model = get_model(architecture, num_classes=10)
    inputs = torch.randn(2, 3, 32, 32)

    outputs = model(inputs)

    assert outputs.shape == (2, 10)


def test_get_model_unknown_architecture_raises() -> None:
    with pytest.raises(ValueError):
        get_model("not-a-real-architecture", num_classes=10)
