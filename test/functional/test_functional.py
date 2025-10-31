"""Series of tests of PyTorch behavior.

These are called functional tests rather than unit tests.
"""

import math

import torch
import torch.nn.functional as F


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


def test_functional_kldiv():
  # Arrange
  logp = torch.log(torch.tensor([1e-5, 1.0 - 1e-5]))
  logq = torch.log(torch.tensor([1.1e-5, 1.0 - 1.1e-5]))

  # Act
  kl = F.kl_div(logq, logp, reduction="none", log_target=True)

  # Assert
  assert kl.size() == (2,)
  assert torch.allclose(kl, torch.tensor([-1e-6, +1e-6]), atol=2e-7, rtol=0.8)


# Expected grad: tensor([ 1.0000e-07, -5.9605e-08], grad_fn=<SubBackward0>)
# Actual grad: tensor([ 1.0000e-07, -5.9605e-08])
def test_functional_kldiv_grad():
  # Arrange: probabilities close to zero and one, p and q very close
  p = torch.tensor(
    [0.000001, 0.999999], dtype=torch.float32, requires_grad=False
  )
  logits_q = torch.log(torch.tensor([0.0000011, 0.999989]))
  logits_q.requires_grad = True

  # Compute log-probabilities
  logprob_q = F.log_softmax(logits_q, dim=-1)

  # Act: KL divergence expects target probabilities (when log_target=False)
  kl = F.kl_div(logprob_q, p, reduction="sum", log_target=False)
  kl.backward()

  # Analytical gradient: grad w.r.t logits is (q - p)
  prob_q = logprob_q.exp()
  expected_grad = prob_q - p

  # Assert
  print(f"Expected grad: {expected_grad}")
  print(f"Actual grad: {logits_q.grad}")
  assert torch.allclose(logits_q.grad, expected_grad, atol=1e-6, rtol=0.01), (
    f"Grad mismatch: {logits_q.grad=} vs {expected_grad=}"
  )
  assert not torch.any(logits_q.grad.isnan()), (
    f"NaN gradient detected.  \nExpected grad: {expected_grad} \nActual grad: {logits_q.grad}"
  )
  assert not torch.any(logits_q.grad.isinf()), (
    f"Inf gradient detected.  \nExpected grad: {expected_grad} \nActual grad: {logits_q.grad}"
  )
  assert not torch.any(logits_q.grad.isposinf()), (
    f"+Inf gradient detected.  \nExpected grad: {expected_grad} \nActual grad: {logits_q.grad}"
  )
  assert not torch.any(logits_q.grad.isneginf()), (
    f"-Inf gradient detected \nExpected grad: {expected_grad} \nActual grad: {logits_q.grad}"
  )
  assert not torch.any(logits_q.grad == 0.0), (
    f"zero gradient detected. \nExpected grad: {expected_grad} \nActual grad: {logits_q.grad}"
  )


# Expected grad: [9.999999999999991e-12, -1.000000082740371e-11]
# Actual grad: tensor([ 1.0000e-11, -1.0000e-11], dtype=torch.float64)
def test_functional_kldiv_grad_with_logs():
  # Arrange
  p = [1e-10, 1.0 - 1e-10]  # Python has infinite precision.
  q = [1.1e-10, 1.0 - 1.1e-10]  # Python has infinite precision.

  log_p = [math.log(p[0]), math.log(p[1])]
  log_q = [math.log(q[0]), math.log(q[1])]

  logit_p = torch.tensor(log_p, requires_grad=False, dtype=torch.double)
  logit_q = torch.tensor(log_q, requires_grad=True, dtype=torch.double)

  logprob_p = torch.log_softmax(logit_p, dim=-1)
  logprob_q = torch.log_softmax(logit_q, dim=-1)

  # Act
  # KL divergence expects target probabilities (when log_target=False)
  kl = F.kl_div(logprob_q, logprob_p, reduction="sum", log_target=True)
  kl.backward()

  # Analytical gradient: grad w.r.t logits is (q - p)
  expected_grad = [q[0] - p[0], q[1] - p[1]]

  # Assert
  print(f"Expected grad: {expected_grad}")
  print(f"Actual grad: {logit_q.grad}")
  assert math.isclose(
    logit_q.grad[0], expected_grad[0], rel_tol=0.01, abs_tol=1e-6
  ), f"Grad mismatch: {logit_q.grad=} vs {expected_grad=}"
  assert math.isclose(
    logit_q.grad[1], expected_grad[1], rel_tol=0.01, abs_tol=1e-6
  ), f"Grad mismatch: {logit_q.grad=} vs {expected_grad=}"

  assert not torch.any(logit_q.grad.isnan()), (
    f"NaN gradient detected.  \nExpected grad: {expected_grad} \nActual grad: {logit_q.grad}"
  )
  assert not torch.any(logit_q.grad.isinf()), (
    f"Inf gradient detected.  \nExpected grad: {expected_grad} \nActual grad: {logit_q.grad}"
  )
  assert not torch.any(logit_q.grad.isposinf()), (
    f"+Inf gradient detected.  \nExpected grad: {expected_grad} \nActual grad: {logit_q.grad}"
  )
  assert not torch.any(logit_q.grad.isneginf()), (
    f"-Inf gradient detected \nExpected grad: {expected_grad} \nActual grad: {logit_q.grad}"
  )
  assert not torch.any(logit_q.grad == 0.0), (
    f"zero gradient detected. \nExpected grad: {expected_grad} \nActual grad: {logit_q.grad}"
  )
