import torch

from slpo.slpo import _compute_logprob_y_bar_y, _logdiffexp, slpo_loss


def test_logdiffexp_corners():
    # Arrange
    t1 = torch.log(torch.tensor([0.0, 1.0, 0.1, 1.0]))
    t2 = torch.log(torch.tensor([0.0, 1.0, 0.1, 0.0]))

    expected = torch.tensor([float("-inf"), float("-inf"), float("-inf"), 0.0])

    # Act
    result = _logdiffexp(t1, t2)

    # Assert
    assert torch.allclose(expected, result, atol=1e-6)


def test_logdiffexp():
    # Arrange
    t1 = torch.log(torch.tensor([0.10, 0.45, 0.9]))
    t2 = torch.log(torch.tensor([0.05, 0.21, 0.0]))

    expected = torch.log(torch.exp(t1) - torch.exp(t2))

    # Act
    result = _logdiffexp(t1, t2)

    # Assert
    assert torch.allclose(expected, result, atol=1e-6)


def test_compute_logprob_y_bar_y_minimal():
    # Arrange
    log_probs = torch.stack(
        [
            torch.log_softmax(torch.tensor([0.0, 0.0]), -1),
            torch.log_softmax(torch.tensor([0.0, 0.0]), -1),
        ]
    )
    y_tokens = torch.tensor([0, 1])
    expected_log_p_y = torch.log(torch.tensor(0.25))
    expected_log_p_not_y = torch.log(torch.tensor(0.75))

    # Act
    log_p_y, log_p_not_y = _compute_logprob_y_bar_y(log_probs, y_tokens)

    # Assert
    assert torch.isclose(log_p_y, expected_log_p_y, atol=1e-6), (
        f"Expected log_p_y={expected_log_p_y}, got {log_p_y}"
    )

    assert torch.isclose(log_p_not_y, expected_log_p_not_y, atol=1e-6), (
        f"Expected log_p_not_y={expected_log_p_not_y}, got {log_p_not_y}"
    )

    # Ensure log_p_y + log_p_not_y ~= 100% (valid probability distribution)
    assert torch.isclose(
        torch.tensor(1.0),
        torch.exp(log_p_y) + torch.exp(log_p_not_y),
        atol=1e-6,
    ), (
        f"log_p_y ({torch.exp(log_p_y)}) + log_p_not_y ({torch.exp(log_p_not_y)}) should ~= 100%."
    )


def test_compute_logprob_y_bar_y():
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

    # Ensure log_p_y + log_p_not_y ~= 100% (valid probability distribution)
    assert torch.isclose(
        torch.tensor(1.0),
        torch.exp(log_p_y) + torch.exp(log_p_not_y),
        atol=1e-6,
    ), (
        f"log_p_y ({torch.exp(log_p_y)}) + log_p_not_y ({torch.exp(log_p_not_y)}) should ~= 100%."
    )


def test_functional_inf():
    """Test that logsoftmax can handle -inf values."""
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


def test_slpo_grads_loser():
    # Arrange
    output = torch.ones((2, 3), requires_grad=True)
    target = {
        "pi_ref_w": torch.tensor(0.03),
        "pi_ref_l": torch.tensor(0.07),
        "y": torch.tensor([1, 0]),
        "winner": torch.tensor(False, dtype=torch.bool),
    }

    # Act
    loss = slpo_loss(output, target)
    loss.backward()

    # Assert

    # These are the y_l tokens, so the gradients should be positive (less weight on loser).
    assert output.grad[0, 1] > 0.0, f"output.grad={output.grad}"
    assert output.grad[1, 0] > 0.0, f"output.grad={output.grad}"

    # These are the \overline{y_l} tokens, so the gradients should be negative.
    assert output.grad[0, 0] < 0.0, f"output.grad={output.grad}"
    assert output.grad[0, 2] < 0.0, f"output.grad={output.grad}"
    assert output.grad[1, 1] < 0.0, f"output.grad={output.grad}"
    assert output.grad[1, 2] < 0.0, f"output.grad={output.grad}"


def test_slpo_grads_winner():
    # Arrange
    output = torch.ones((2, 3), requires_grad=True)
    target = {
        "pi_ref_w": torch.tensor(0.1),
        "pi_ref_l": torch.tensor(0.1),
        "y": torch.tensor([1, 0]),
        "winner": torch.tensor(True, dtype=torch.bool),
    }

    # Act
    loss = slpo_loss(output, target)
    loss.backward()

    # Assert

    # These are the y_w tokens, so the grads should be neg (more weight on winner).
    assert output.grad[0, 1] < 0.0, f"output.grad={output.grad}"
    assert output.grad[1, 0] < 0.0, f"output.grad={output.grad}"

    # These are the \overline{y_w} tokens, so the grads should be positive (less weight on winner).
    assert output.grad[0, 0] > 0.0, f"output.grad={output.grad}"
    assert output.grad[0, 2] > 0.0, f"output.grad={output.grad}"
    assert output.grad[1, 1] > 0.0, f"output.grad={output.grad}"
    assert output.grad[1, 2] > 0.0, f"output.grad={output.grad}"
