"""Unit tests for the memorization model."""

import pytest
import torch
from test_utils.memorization_model import MemorizationModel


@pytest.fixture
def logits():
    retval = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0],
                [2.0, 4.0, 6.0],
                [3.0, 9.0, 12.0],
            ],
            [
                [3.0, 2.0, 1.0],
                [0.0, -0.5, -1.0],
                [-1.0, -1.25, -1.5],
            ],
        ]
    )
    return retval


@pytest.fixture
def model(logits):
    """Fixture to create a simple model for testing."""
    retval = MemorizationModel(batch_size=2, seq_len=3, vocab_size=3)
    retval.logits.data = logits
    return retval
    # Set the logits to a known value for reproducibility.


def test_bad_input(model):
    """Test that an invalid input on forward raises an error."""
    with pytest.raises(ValueError) as excinfo:
        model.forward(torch.tensor([1, -1]))
    assert "All values of x should be 1" in str(excinfo.value), excinfo.value


def test_bad_token_ids(model):
    """Test that an invalid token_ids length raises an error."""
    with pytest.raises(ValueError) as excinfo:
        model.joint_prob(0, [0, 1, 2, 3])
    assert "Invalid token_ids length: 4, seq_len=3" in str(excinfo.value), (
        excinfo.value
    )


def test_forward(logits, model):
    # Arrange
    expected = torch.log_softmax(logits, -1)

    # Act
    result = torch.log_softmax(model(torch.tensor([1, 1])), -1)

    # Assert
    torch.testing.assert_close(result, expected)


def test_joint_prob(logits, model):
    # Arrange
    token_ids = [2, 1, 0]
    lps = torch.log_softmax(logits[0], axis=-1)

    first = lps[0, 2]
    second = lps[1, 1]
    third = lps[2, 0]

    expected = torch.exp(first) * torch.exp(second) * torch.exp(third)

    # Act
    result = model.joint_prob(0, token_ids)

    # Assert
    torch.testing.assert_close(result, expected)


def test_joint_log_prob(logits, model):
    # Arrange

    token_ids = [2, 0, 2]
    lps = torch.log_softmax(logits[0], axis=-1)

    first = lps[0, 2]
    second = lps[1, 0]
    third = lps[2, 2]

    expected = first + second + third

    # Act
    result = model.joint_log_prob(0, token_ids)

    # Assert
    torch.testing.assert_close(result, expected)


def test_joint_log_prob_short(logits, model):
    # Arrange

    token_ids = [2, 0]
    lps = torch.log_softmax(logits[0], axis=-1)

    first = lps[0, 2]
    second = lps[1, 0]

    expected = first + second

    # Act
    result = model.joint_log_prob(0, token_ids)

    # Assert
    torch.testing.assert_close(result, expected)
