import pytest
import torch

from slpo.slpo import (
  _compute_logprob_y_bar_y,
  _logdiffexp,
  _logsumexp,
  slpo_loss,
)


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


def test_logsumexp():
  # Arrange
  t1 = torch.log_softmax(torch.randn(3), -1)
  t2 = torch.log_softmax(torch.randn(3), -1)

  expected = torch.logsumexp(torch.stack([t1, t2], dim=0), 0)

  # Act
  result = _logsumexp(t1, t2)

  # Assert
  assert torch.allclose(expected, result, atol=1e-6)


def test_logprob_y_bar_y_float32_dtype_warns():
  # Arrange
  log_probs = torch.stack(
    [
      torch.log_softmax(torch.tensor([0.0, 0.0]), -1),
      torch.log_softmax(torch.tensor([0.0, 0.0]), -1),
    ]
  )
  y_tokens = torch.tensor([0, 1])

  # Assert
  with pytest.warns() as e:
    # Act
    _compute_logprob_y_bar_y(log_probs, y_tokens)


def test_logprob_y_bar_y_double_dtype_returned():
  # Arrange
  log_probs = torch.stack(
    [
      torch.log_softmax(torch.tensor([0.0, 0.0], dtype=torch.float64), -1),
      torch.log_softmax(torch.tensor([0.0, 0.0], dtype=torch.float64), -1),
    ]
  )
  y_tokens = torch.tensor([0, 1])

  # Act
  logprob_y_bar, logprob_bar_y = _compute_logprob_y_bar_y(log_probs, y_tokens)

  # Assert
  assert logprob_y_bar.dtype == torch.float64, (
    f"Expected dtype=torch.float64, got {logprob_y_bar.dtype}"
  )
  assert logprob_bar_y.dtype == torch.float64, (
    f"Expected dtype=torch.float64, got {logprob_bar_y.dtype}"
  )


def test_compute_logprob_y_bar_y_minimal():
  # Arrange
  log_probs = torch.stack(
    [
      torch.log_softmax(torch.tensor([0.0, 0.0], dtype=torch.float64), -1),
      torch.log_softmax(torch.tensor([0.0, 0.0], dtype=torch.float64), -1),
    ]
  )
  y_tokens = torch.tensor([0, 1])
  expected_log_p_y = torch.log(torch.tensor(0.25, dtype=torch.float64))
  expected_log_p_not_y = torch.log(torch.tensor(0.75, dtype=torch.float64))

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
    torch.tensor(1.0, dtype=log_p_y.dtype),
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
  ).to(torch.float64)
  y_tokens = torch.tensor(
    [
      1,
      2,
      0,
    ]
  )
  expected_log_p_y = torch.log(
    torch.tensor(0.2 * 0.5 * 0.3, dtype=torch.float64)
  )
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
    torch.tensor(1.0, dtype=log_p_y.dtype),
    torch.exp(log_p_y) + torch.exp(log_p_not_y),
    atol=1e-6,
  ), (
    f"log_p_y ({torch.exp(log_p_y)}) + log_p_not_y ({torch.exp(log_p_not_y)}) should ~= 100%."
  )


def test_slpo_grads_loser():
  # Arrange
  output = torch.ones((2, 3), requires_grad=True)
  target = {
    "logprob_ref_w": torch.log(torch.tensor(0.03, dtype=torch.float64)),
    "logprob_ref_l": torch.log(torch.tensor(0.07, dtype=torch.float64)),
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


def test_slpo_grads_winner_and_large_prob_warning():
  # Arrange
  output = torch.ones(
    (2, 3), requires_grad=True
  )  # Starting joint prob of any seq is 9%.
  target = {
    "logprob_ref_w": torch.tensor(0.1).double().log(),
    "logprob_ref_l": torch.tensor(0.1).double().log(),
    "y": torch.tensor([1, 0]),
    "winner": torch.tensor(True, dtype=torch.bool),
  }

  # Act
  with pytest.warns(RuntimeWarning, match=r"Expected exp.+"):
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
