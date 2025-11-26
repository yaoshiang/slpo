import math

from matplotlib.pyplot import step

import fixtures
import pytest
import torch

from slpo.slpo import _get_batch_logps, _logdiffexp, _logsumexp


def test_logdiffexp_corners():
  # Arrange
  t1 = torch.log(torch.tensor([[0.0, 1.0, 0.1, 1.0]]))
  t2 = torch.log(torch.tensor([[0.0, 1.0, 0.1, 0.0]]))

  expected = torch.tensor([[float("-inf"), float("-inf"), float("-inf"), 0.0]])

  # Act
  result = _logdiffexp(t1, t2)
  print(f"{t1=}\n{t2=}\n{result=}")

  # Assert
  torch.testing.assert_close(expected, result)


def test_logdiffexp():
  # Arrange
  t1 = torch.log(torch.tensor([[0.10, 0.45, 0.9]]))
  t2 = torch.log(torch.tensor([[0.05, 0.21, 0.0]]))

  expected = torch.log(torch.exp(t1) - torch.exp(t2))

  # Act
  result = _logdiffexp(t1, t2)

  # Assert
  torch.testing.assert_close(expected, result)


def test_logsumexp():
  # Arrange
  t1 = torch.log_softmax(torch.randn(1, 3), -1)
  t2 = torch.log_softmax(torch.randn(1, 3), -1)

  expected = torch.logsumexp(torch.stack([t1, t2], dim=0), 0)

  # Act
  result = _logsumexp(t1, t2)

  # Assert
  torch.testing.assert_close(expected, result)


@pytest.mark.parametrize(
  "B,S,V",
  (
    (1, 2, 2),  # The minimal case
    (1, 8, 16),  # Arbitrary n.
    (1, 2028, 128_000),  # Checking numerical stability at long sequences.
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
  logp_y, logp_y_bar = _get_batch_logps(logits, labels)
  print(f"{logp_y=}\n{logp_y_bar=}")

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

  expected_logp_y = torch.log(
    torch.tensor([1 / V ** (S - 1)], dtype=torch.float64)
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
  labels[3, [1, 3, 5]] = -100  # Not continugous, to mimic multi-turn conv.

  # Expected values
  # Since the model is autoregressive, we get S-1 predictions initially.
  # Masking reduces this count.
  valid_counts = torch.tensor([7, 6, 5, 4], dtype=torch.float64)

  expected_logp_y = -valid_counts * math.log(V)
  expected_logp_y_bar = torch.log1p(-torch.exp(expected_logp_y))

  # Act
  logp_y, logp_y_bar = _get_batch_logps(logits, labels)

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

  # Ensure logp_y + logp_y_bar ~= 100% (valid probability distribution)
  torch.testing.assert_close(
    torch.ones_like(logp_y),
    torch.exp(logp_y) + torch.exp(logp_y_bar),
  )


def test_get_batch_logps_non_uniform():
  # Arrange
  B, S, V = 1, 3, 4
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

  # Setup expected values
  expected_logp_y = logps[:, [0, 1, 2], [2, 0, 1]].sum(-1)
  expected_logp_ybar = torch.log1p(-torch.exp(expected_logp_y))

  # Act
  logp_y, logp_ybar = _get_batch_logps(logits, labels)

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

  # Ensure log_p_y + log_p_not_y ~= 100% (valid probability distribution)

  torch.testing.assert_close(
    torch.exp(logp_y) + torch.exp(logp_ybar),
    torch.ones_like(logp_y),
    msg=f"{torch.exp(logp_y)=} + {torch.exp(logp_ybar)=} should ~= 100%.",
  )


def test_slpo_loss_on_singleton_case():
  # Arrange: dataset size
  B, S, V = 1, 7, 11
  
  # Arrange: reference logprobs
  ref_w = 0.22
  ref_l = 0.11
  ref_wbar = 1.0 - ref_w
  ref_lbar = 1.0 - ref_l
  logp_w = torch.log(torch.tensor(ref_w))
  logp_l = torch.log(torch.tensor(ref_l))
  logp_wbar = torch.log(torch.tensor(ref_wbar))
  logp_lbar = torch.log(torch.tensor(ref_lbar))

  # Arrange: target. Shape B, S. 
  # Make sure they don't overlap so that logp_w and logp_l can be 
  # "predicted" pefectly by the model. 
  w = torch.tensor([[1, 2, 3, 4, 5, 6, 7]], dtype=torch.int64)
  l = torch.tensor([[8, 9, 10, 0, 1, 2, 3]], dtype=torch.int64)
  
  # Arrange student
  model = fixtures.Memo(B, S, V)

  # Arrange training
  optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

  # Act
  for step in range(100):
    x = torch.randn((B, S, V))
    model.train()
    y = model(x)
    = _get_batch_logps(y, torch.zeros((B, S), dtype=torch.int32))
    

    
    loss = compute_slpo_loss_batch(model(input), target)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    optimizer.param_groups[0]["lr"] = 0.1 / step

def test_slpo_batch_size_two():
  # Arrange
  model = fixtures.Memo(
    batch_size=2, seq_len=2, vocab_size=3
  )  # Prime numbers for ease of testing.

  optimizer = torch.optim.SGD(
    model.parameters(), lr=0.0
  )  # LR scheduled in loop.
  targets = [
    {
      "logprob_ref_w": torch.tensor(0.01).double().log(),
      "logprob_ref_l": torch.tensor(0.01).double().log(),
      "y": torch.tensor([0, 1], dtype=torch.int32),
      "winner": torch.tensor(True, dtype=torch.bool),
    },
    {
      "logprob_ref_w": torch.tensor(0.03).double().log(),
      "logprob_ref_l": torch.tensor(0.07).double().log(),
      "y": torch.tensor([2, 1], dtype=torch.int32),
      "winner": torch.tensor(False, dtype=torch.bool),
    },
  ]
  dataset = DictDataset(torch.ones((2,)).double(), targets)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

  # Act
  for epoch in range(100):
    optimizer.param_groups[0]["lr"] = 100.0 / (epoch + 1.0)
    for batch in dataloader:
      input, target = batch
      model.train()
      optimizer.zero_grad()
      loss = compute_slpo_loss_batch(model(input), target)
      loss.backward()
      optimizer.step()

      logprob_0 = model.logprob(0, targets[0]["y"]).double()
      logprob_1 = model.logprob(1, targets[1]["y"]).double()

  # Assert
  torch.testing.assert_close(
    logprob_0,
    torch.tensor(0.02).double().log(),
    atol=0.01,
    rtol=0.1,
  )
  torch.testing.assert_close(
    logprob_1.exp(),
    torch.tensor(0.0).double(),
    atol=0.01,
    rtol=0.1,
  )  # This is the expected value for the losing sequence.


@pytest.mark.parametrize("seq_length", [2, 5, 128])
@pytest.mark.parametrize("vocab_size", [2, 7, 1000])
def test_slpo_batch_many_negatives(seq_length, vocab_size):
  # Arrange
  model = fixtures.Memo(batch_size=2, seq_len=seq_length, vocab_size=vocab_size)

  optimizer = torch.optim.SGD(
    model.parameters(), lr=1.0
  )  # High learning rate - we are testing blasting down the loser to zero.
  targets = [
    {
      "logprob_ref_w": torch.tensor(0.02).double().log(),
      "logprob_ref_l": torch.tensor(0.05).double().log(),
      "y": torch.arange(seq_length) % vocab_size,
      "winner": torch.tensor(True, dtype=torch.bool),
    },
    {
      "logprob_ref_w": torch.tensor(0.03, dtype=torch.float64).log(),
      "logprob_ref_l": torch.tensor(0.07, dtype=torch.float64).log(),
      "y": torch.arange(seq_length) % vocab_size,
      "winner": torch.tensor(False, dtype=torch.bool),
    },
  ]
  dataset = DictDataset(torch.ones((2,)), targets)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

  ref_joint_prob_1 = model.logprob(1, targets[1]["y"])

  # Act
  for epoch in range(100):
    for batch in dataloader:
      input, target = batch
      optimizer.zero_grad()
      loss = compute_slpo_loss_batch(model(input), target)
      loss.backward(retain_graph=True)
      optimizer.step()

  # Assert
  joint_prob_1 = model.logprob(1, targets[1]["y"])

  assert joint_prob_1 < ref_joint_prob_1 or ref_joint_prob_1 < -100, (
    f"{joint_prob_1=}, {ref_joint_prob_1=}"
  )


@pytest.mark.parametrize("seq_length", [5, 128])
@pytest.mark.parametrize("vocab_size", [7, 1000])
def test_slpo_batch_many_positives(seq_length, vocab_size):
  # Arrange
  model = fixtures.Memo(batch_size=2, seq_len=seq_length, vocab_size=vocab_size)

  optimizer = torch.optim.SGD(
    model.parameters(), lr=1.0
  )  # High learning rate - we are testing blasting down the loser to zero.
  targets = [
    {
      "logprob_ref_w": torch.tensor(0.002).double().log(),
      "logprob_ref_l": torch.tensor(0.005).double().log(),
      "y": torch.arange(seq_length) % vocab_size,
      "winner": torch.tensor(True, dtype=torch.bool),
    },
    {
      "logprob_ref_w": torch.tensor(0.003).double().log(),
      "logprob_ref_l": torch.tensor(0.007).double().log(),
      "y": torch.arange(seq_length) % vocab_size,
      "winner": torch.tensor(False, dtype=torch.bool),
    },
  ]
  dataset = DictDataset(torch.ones((2,)), targets)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

  ref_joint_prob_0 = model.logprob(0, targets[1]["y"])

  # Act
  for epoch in range(100):
    for batch in dataloader:
      input, target = batch
      optimizer.zero_grad()
      loss = compute_slpo_loss_batch(model(input), target)
      loss.backward(retain_graph=True)
      optimizer.step()

  # Assert
  joint_prob_0 = model.logprob(0, targets[1]["y"])

  assert joint_prob_0 > ref_joint_prob_0 or ref_joint_prob_0 > -0.00001, (
    f"{joint_prob_0=}, {ref_joint_prob_0=}"
  )
