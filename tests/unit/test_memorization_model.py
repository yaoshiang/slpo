"""Unit tests for the memorization model."""

import pytest
import torch
from tests.utils.memorization_model import MemorizationModel


@pytest.fixture
def weights():
    retval = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            [
                [10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0],
                [16.0, 17.0, 18.0],
            ],
        ]
    )
    return retval


@pytest.fixture
def model(weights):
    """Fixture to create a simple model for testing."""
    torch.manual_seed(42)  # Set seed for reproducibility
    retval = MemorizationModel(num_examples=2, seq_len=3, vocab_size=3)

    retval.weights.data = weights
    return retval
    # Set the weights to a known value for reproducibility.


def test_bad_example_idx(model):
    """Test that an invalid example index raises an error."""
    with pytest.raises(ValueError) as excinfo:
        model.forward(-1)
    assert "Invalid example index: -1, num_examples=2" in str(excinfo.value)


def test_bad_token_ids(model):
    """Test that an invalid token_ids length raises an error."""
    with pytest.raises(ValueError) as excinfo:
        model.joint_prob(0, [0, 1, 2, 3])
    assert "Invalid token_ids length: 4, seq_len=3" in str(excinfo.value)


def test_forward(weights, model):
    # Arrange
    values = model(0)
    expected = weights[0]

    # Act
    result = model(0)

    # Assert
    assert torch.allclose(result, expected)


def test_joint_prob(weights, model):
    # Arrange

    token_ids = [2, 1, 0]
    lps = torch.log_softmax(weights[0], axis=-1)

    first = lps[0, 2]
    second = lps[1, 1]
    third = lps[2, 0]

    expected = torch.exp(first) * torch.exp(second) * torch.exp(third)

    # Act
    result = model.joint_prob(0, token_ids)

    # Assert
    assert torch.allclose(expected, result)


def test_joint_log_prob(weights, model):
    # Arrange

    token_ids = [2, 0, 2]
    lps = torch.log_softmax(weights[0], axis=-1)

    first = lps[0, 2]
    second = lps[1, 0]
    third = lps[2, 2]

    expected = first + second + third

    # Act
    result = model.joint_log_prob(0, token_ids)

    # Assert
    assert torch.allclose(expected, result)


def test_joint_log_prob_short(weights, model):
    # Arrange

    token_ids = [2, 0]
    lps = torch.log_softmax(weights[0], axis=-1)

    first = lps[0, 2]
    second = lps[1, 0]

    expected = first + second

    # Act
    result = model.joint_log_prob(0, token_ids)

    # Assert
    assert torch.allclose(expected, result)
