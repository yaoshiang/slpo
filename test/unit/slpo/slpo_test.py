import pytest
import torch

from slpo.slpo import (
  _logdiffexp,
  _logsumexp,
  slpo_loss_single,
  y_ybar,
  y_ybar_single,
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


def test_y_ybar_single_on_simple_input():
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
  log_p_y, log_p_not_y = y_ybar_single(log_probs, y_tokens)

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


def test_y_ybar_single_nontrivial():
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
  log_p_y, log_p_not_y = y_ybar_single(log_probs, y_tokens)

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


def create_log_probs(
  batch_size: int,
  vocab_size: int,
  seq_len: int,
  dtype: torch.dtype,
  device: torch.device | None = None,
) -> torch.Tensor:
  """Create a tensor of log probabilities that are nearly uniform.

  The "nearly uniform" avoids weird outliers.

  Args:
      vocab_size (int): The size of the vocabulary.
      seq_len (int): The length of the sequence.

  Returns:
      torch.Tensor: A tensor of shape (seq_len, vocab_size) containing log probabilities.
  """
  if device is None:
    device = torch.get_default_device()

  logits = torch.randn(
    batch_size, seq_len, vocab_size, dtype=dtype, device=device
  )
  logits = torch.clamp(logits, min=-1.0, max=1.0)  # Avoid extreme values
  log_probs = torch.log_softmax(logits, dim=-1)
  return log_probs


@pytest.mark.parametrize(
  "batch_size,vocab_size,seq_len",
  [
    (2, 3, 3),
    (4, 5, 7),
    (8, 128, 128_000),
  ],
)
def test_y_ybar_on_complex_batched(
  batch_size: int, vocab_size: int, seq_len: int
):
  # Arrange
  log_probs = create_log_probs(batch_size, vocab_size, seq_len, torch.float64)
  y_tokens = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))

  # Act
  logp_y, logp_ybar = y_ybar(log_probs, y_tokens)

  # Assert
  for i in range(batch_size):
    expected_logp_y, expected_logp_ybar = y_ybar_single(
      log_probs[i], y_tokens[i]
    )

    torch.testing.assert_close(logp_y[i], expected_logp_y)
    torch.testing.assert_close(logp_ybar[i], expected_logp_ybar)


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
  loss = slpo_loss_single(output, target)
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
