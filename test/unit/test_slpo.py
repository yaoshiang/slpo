import pytest
import torch
from test_utils.memorization_model import MemorizationModel

from slpo.slpo import _compute_logprob_y_bar_y


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
    retval = MemorizationModel(num_examples=2, seq_len=3, vocab_size=3)
    retval.logits.data = logits
    return retval
    # Set the logits to a known value for reproducibility.


def test_compute_log_p_and_log_p_not_y():
    # Arrange
    log_probs = torch.stack(
        [
            torch.log(torch.tensor([0.1, 0.2, 0.7])),
            torch.log(torch.tensor([0.2, 0.3, 0.5])),
            torch.log(torch.tensor([0.3, 0.4, 0.3])),
        ]
    )
    y_tokens = torch.tensor(
        [
            1,
            2,
            0,
        ]
    )
    expected_log_p_y = torch.log(torch.tensor(0.2 * 0.5 * 0.3))
    expected_log_p_not_y = torch.log(1.0 - torch.exp(expected_log_p_y))

    # Act
    log_p_y, log_p_not_y = _compute_logprob_y_bar_y(log_probs, y_tokens)

    # Assert
    assert torch.isclose(log_p_y, expected_log_p_y, atol=1e-6), (
        f"Expected log_p_y={expected_log_p_y}, got {log_p_y}"
    )

    assert torch.isclose(log_p_not_y, expected_log_p_not_y, atol=1e-6), (
        f"Expected log_p_not_y={expected_log_p_not_y}, got {log_p_not_y}"
    )

    # Ensure log_p_y is always greater than log_p_not_y (valid probability distribution)
    assert torch.isclose(
        torch.tensor(1.0),
        torch.exp(log_p_y) + torch.exp(log_p_not_y),
        atol=1e-6,
    ), (
        f"log_p_y ({torch.exp(log_p_y)}) + log_p_not_y ({torch.exp(log_p_not_y)}) should ~= 100%."
    )


# Test that logsoftmax can handle -inf values.
def test_functional_inf():
    # Arrange
    x = torch.tensor([[0.0, 0.0, -float("inf")]])
    y_expected = torch.tensor(
        [
            [
                torch.log(torch.tensor(0.5)),
                torch.log(torch.tensor(0.5)),
                -float("inf"),
            ]
        ]
    )

    # Act
    y_true = torch.nn.functional.log_softmax(x, dim=-1)

    # Assert
    assert torch.allclose(y_expected, y_true, atol=1e-6)


# Test that y_w and y_l will be close to our target value after optimization.
# def test_slpo_training(model):
#     opt = torch.optim.AdamW(model.parameters(), lr=0.1)


#     for epoch in range(100):
