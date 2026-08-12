import math

import pytest
import torch

from slpo.slpo_adapter import _get_batch_logps

torch.set_printoptions(precision=17)


@pytest.mark.parametrize(
  "B,S,V",
  (
    (1, 2, 2),  # The minimal case
    (1, 8, 16),  # Arbitrary n.
    (1, 2048, 128_000),  # Checking numerical stability at long sequences.
    (64, 8, 16),  # Batch size > 1
  ),
)
def test_get_batch_logps_without_masking(B, S, V):
  logits = torch.zeros(B * S * V, dtype=torch.float64).reshape(B, S, V)
  labels = torch.randint(low=0, high=V, size=(B, S), dtype=torch.long)
  # Since the model is autoregressive, we get S-1 predictions.
  expected_logp_y = torch.tensor([-math.log(V ** (S - 1))], dtype=torch.float64)

  expected_logp_y_bar = torch.log1p(-torch.exp(expected_logp_y))

  expected_logp_y = expected_logp_y.tile(B)
  expected_logp_y_bar = expected_logp_y_bar.tile(B)
  # Act
  logp_y, logp_y_bar, _ = _get_batch_logps(logits, labels)

  # Assert
  torch.testing.assert_close(
    logp_y,
    expected_logp_y,
    msg=(
      f"{expected_logp_y=}\n{logp_y=}\n"
      f"{torch.exp(expected_logp_y)=}, {torch.exp(logp_y)=}"
    ),
  )

  torch.testing.assert_close(
    logp_y_bar,
    expected_logp_y_bar,
    msg=(
      f"{expected_logp_y_bar=}\n{logp_y_bar=}\n"
      f"{torch.exp(expected_logp_y_bar)=}, {torch.exp(logp_y_bar)=}"
    ),
  )

  # Ensure logp_y + logp_y_bar ~= 100% (valid probability distribution)
  torch.testing.assert_close(
    torch.ones_like(logp_y),
    torch.exp(logp_y) + torch.exp(logp_y_bar),
  )


def test_get_batch_logps_with_masking():
  # Arrange
  B, S, V = 4, 8, 16
  logits = torch.zeros(B * S * V, dtype=torch.float64).reshape(B, S, V)
  labels = torch.randint(low=0, high=V, size=(B, S), dtype=torch.long)

  # Masking by setting labels to -100
  # Batch 0: 0 masked (all valid) -> 7 predictions
  # Batch 1: 1 masked -> 6 predictions. Mask 1 token.
  labels[1, 1] = -100
  # Batch 2: 2 masked -> 5 predictions. Mask 2 tokens.
  labels[2, [1, 2]] = -100
  # Batch 3: 3 masked -> 4 predictions. Mask 3 tokens.
  labels[3, [1, 3, 5]] = -100  # Not contiguous, to mimic multi-turn conv.

  valid_counts = torch.tensor([7, 6, 5, 4], dtype=torch.float64)

  expected_logp_y = -valid_counts * math.log(V)
  expected_logp_y_bar = torch.log1p(-torch.exp(expected_logp_y))

  # Act
  logp_y, logp_y_bar, _ = _get_batch_logps(logits, labels)

  # Assert
  torch.testing.assert_close(
    logp_y,
    expected_logp_y,
    msg=f"{expected_logp_y=}\n{logp_y=}",
  )

  torch.testing.assert_close(
    logp_y_bar,
    expected_logp_y_bar,
    msg=f"{expected_logp_y_bar=}\n{logp_y_bar=}",
  )

  torch.testing.assert_close(
    torch.ones_like(logp_y),
    torch.exp(logp_y) + torch.exp(logp_y_bar),
  )


def test_get_batch_logps_non_uniform():
  # Arrange
  probs = torch.tensor(
    [
      [
        (0.1, 0.1, 0.8),
        (0.2, 0.1, 0.7),
        (0.3, 0.1, 0.6),
        (0.4, 0.1, 0.5),
      ]
    ]
  ).to(torch.float64)
  logps = torch.log(probs)
  logits = logps + 1.5  # logits are shift invariant.
  labels = torch.tensor([[1, 2, 0, 1]], dtype=torch.int64)

  expected_logp_y = logps[:, [0, 1, 2], [2, 0, 1]].sum(-1)
  expected_logp_ybar = torch.log1p(-torch.exp(expected_logp_y))

  # Act
  logp_y, logp_ybar, _ = _get_batch_logps(logits, labels)

  # Assert
  torch.testing.assert_close(
    logp_y,
    expected_logp_y,
    msg=f"{expected_logp_y=}\n{logp_y=}\n{torch.exp(expected_logp_y)=}\n{torch.exp(logp_y)=}",
  )

  torch.testing.assert_close(
    logp_ybar,
    expected_logp_ybar,
    msg=f"Expected logp_ybar={expected_logp_ybar}, got {logp_ybar}",
  )

  torch.testing.assert_close(
    torch.exp(logp_y) + torch.exp(logp_ybar),
    torch.ones_like(logp_y),
    msg=f"{torch.exp(logp_y)=} + {torch.exp(logp_ybar)=} should ~= 100%.",
  )
