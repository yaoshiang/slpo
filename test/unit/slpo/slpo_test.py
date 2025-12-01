import copy
import math

import fixtures
import pytest
import torch

from slpo import slpo
from slpo.slpo import _get_batch_logps, _logdiffexp, _logsumexp

torch.set_printoptions(precision=17)


def format(tensor: torch.Tensor) -> str:
  value = tensor.item()
  precision = 12
  chunk_size = 4
  formatted_value = f"{value:.{precision}f}"
  parts = formatted_value.split(".")
  if len(parts) != 2:
    return formatted_value  # Return as is if no decimal point
  whole, decimal = parts
  chunked_decimal = " ".join(
    [decimal[i : i + chunk_size] for i in range(0, len(decimal), chunk_size)]
  )
  return f"{whole}.{chunked_decimal}"


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
  logp_y, logp_y_bar = _get_batch_logps(logits, labels)

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


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
def test_calc_targets(seed, alpha):
  torch.manual_seed(seed)
  # Arrange
  B = 1
  # Ensure that exp(reference_chosen_logps) + exp(reference_rejected_logps) <= 1
  # We can do this by generating random numbers that sum to <= 1.
  p_total = torch.rand(B).to(torch.float64)
  split = torch.rand(B).to(torch.float64)
  p_chosen = p_total * split
  p_rejected = p_total * (1 - split)

  reference_chosen_logps = p_chosen.log()
  reference_rejected_logps = p_rejected.log()

  # Act
  w_w, w_l, w_w_bar, w_l_bar = slpo.calc_targets(
    alpha, reference_chosen_logps, reference_rejected_logps
  )

  print(
    f"\n{seed=}, {alpha=}\n"
    f"ref_prob_w = {format(reference_chosen_logps.exp())}\n"
    f"ref_prob_l = {format(reference_rejected_logps.exp())}\n"
    f"       w_w = {format(w_w.exp())} (log: {w_w.item():.4f})\n"
    f"       w_l = {format(w_l.exp())} (log: {w_l.item():.4f})\n"
    f"   w_w_bar = {format(w_w_bar.exp())} (log: {w_w_bar.item():.4f})\n"
    f"   w_l_bar = {format(w_l_bar.exp())} (log: {w_l_bar.item():.4f})\n"
  )

  # Assert
  # Check that w_w and w_w_bar sum to 1 in probability space
  torch.testing.assert_close(
    torch.exp(w_w) + torch.exp(w_w_bar),
    torch.ones_like(w_w),
    msg=f"{torch.exp(w_w)=} + {torch.exp(w_w_bar)=} should ~= 100%.",
  )

  # Check that w_l and w_l_bar sum to 1 in probability space
  torch.testing.assert_close(
    torch.exp(w_l) + torch.exp(w_l_bar),
    torch.ones_like(w_l),
    msg=f"{torch.exp(w_l)=} + {torch.exp(w_l_bar)=} should ~= 100%.",
  )

  # Check specific values
  expected_w_w = torch.log(
    torch.exp(reference_chosen_logps)
    + alpha * torch.exp(reference_rejected_logps)
  )
  torch.testing.assert_close(w_w, expected_w_w)

  expected_w_l = torch.log((1 - alpha) * torch.exp(reference_rejected_logps))
  torch.testing.assert_close(w_l, expected_w_l)


@pytest.mark.parametrize("seed", [101, 102, 103])
def test_apply_t(seed):
  # Arrange
  temperature = 100
  logp1 = (torch.rand(1) / 2.0).log()
  logp2 = slpo._logdiffexp(torch.zeros_like(logp1), logp1)

  # Expected
  expected_scaled_logp1 = logp1 / temperature - slpo._logsumexp(
    logp1 / temperature, logp2 / temperature
  )
  expected_scaled_logp2 = slpo._logdiffexp(
    torch.zeros_like(expected_scaled_logp1), expected_scaled_logp1
  )
  # Act
  scaled_logp1, scaled_logp2 = slpo.apply_t(logp1, logp2, temperature)

  print(
    f"{temperature=}\n"
    f"logp1.exp()={format(logp1)}\n"
    f"logp2.exp()={format(logp2)}\n"
    f"expected_scaled_logp1={format(expected_scaled_logp1)}\n"
    f"expected_scaled_logp2={format(expected_scaled_logp2)}\n"
    f"scaled_logp1={format(scaled_logp1)}\n"
    f"scaled_logp2={format(scaled_logp2)}"
  )

  # Sanity check
  assert logp1 < -0.6931471805599453, "logp1 should be less than log(0.5)."
  assert logp2 > -0.6931471805599453, "logp2 should be greater than log(0.5)."

  # Assert that both logprobs got closer to -0.6931471805599453 (log(0.5))
  assert scaled_logp1 > logp1, (
    "logp1 did not increase after temperature scaling."
  )
  assert scaled_logp2 < logp2, (
    "logp2 did not decrease after temperature scaling."
  )


@pytest.mark.parametrize(
  "B,S,V",
  (
    (1, 2, 2),  # The minimal case
    (1, 8, 16),  # Arbitrary n.
    (64, 8, 16),  # Batch size > 1
    (1, 2048, 128_000),  # Checking numerical stability at long sequences.
  ),
)
def test_slpo_on_logps(B, S, V):
  # Arrange
  torch.set_default_dtype(torch.double)
  p_w = torch.rand(1) * 0.001
  p_l = torch.rand(1) * 0.001
  p_wbar = 1.0 - p_w
  p_lbar = 1.0 - p_l
  p_w_ref = torch.rand(1) * 0.01
  p_l_ref = torch.rand(1) * 0.01
  alpha = 0.1

  logp_w = torch.log(p_w)
  logp_l = torch.log(p_l)
  logp_wbar = torch.log(p_wbar)
  logp_lbar = torch.log(p_lbar)
  logp_w_ref = torch.log(p_w_ref)
  logp_l_ref = torch.log(p_l_ref)

  # Act
  loss, metric1, metric2 = slpo.slpo_loss(
    logp_w,
    logp_l,
    logp_wbar,
    logp_lbar,
    logp_w_ref,
    logp_l_ref,
    alpha=alpha,
    t=float(S),
  )

  # Scale all the logps.
  scaled_logp_w, scaled_logp_wbar = slpo.apply_t(logp_w, logp_wbar, float(S))
  scaled_logp_l, scaled_logp_lbar = slpo.apply_t(logp_l, logp_lbar, float(S))

  # Compute complements for reference
  logp_w_ref_bar = slpo._logdiffexp(torch.zeros_like(logp_w_ref), logp_w_ref)
  logp_l_ref_bar = slpo._logdiffexp(torch.zeros_like(logp_l_ref), logp_l_ref)

  scaled_logp_w_ref, _ = slpo.apply_t(logp_w_ref, logp_w_ref_bar, float(S))
  scaled_logp_l_ref, _ = slpo.apply_t(logp_l_ref, logp_l_ref_bar, float(S))

  # Recompute the probs.
  p_w = scaled_logp_w.exp()
  p_wbar = scaled_logp_wbar.exp()
  p_l = scaled_logp_l.exp()
  p_lbar = scaled_logp_lbar.exp()
  p_w_ref = scaled_logp_w_ref.exp()
  p_l_ref = scaled_logp_l_ref.exp()

  w_w = p_w_ref + alpha * p_l_ref
  w_wbar = 1.0 - w_w
  w_l = (1 - alpha) * p_l_ref
  w_lbar = 1.0 - w_l

  # Expected
  # KL = sum(p * (log p - log q))
  # We have 4 components per batch item.
  # loss is mean over (B * 4) elements.
  log_w_w = torch.log(w_w)
  log_w_wbar = torch.log(w_wbar)
  log_w_l = torch.log(w_l)
  log_w_lbar = torch.log(w_lbar)

  term1 = w_w * (log_w_w - scaled_logp_w)
  term2 = w_wbar * (log_w_wbar - scaled_logp_wbar)
  term3 = w_l * (log_w_l - scaled_logp_l)
  term4 = w_lbar * (log_w_lbar - scaled_logp_lbar)

  expected = (term1 + term2 + term3 + term4) / 4

  # Assert
  torch.testing.assert_close(
    loss,
    expected,
    rtol=0.01,
    atol=0.0,
    msg=f"{expected=}\n{loss=}",
  )


@pytest.mark.parametrize("seed", [101, 102, 103])
@pytest.mark.parametrize("alpha", [0.0, 0.1, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("B,S,V", [(1, 8, 16)])  # (1, 2048, 128_000)))
def test_slpo_trains_model(seed, alpha, B, S, V):
  # Arrange
  torch.manual_seed(seed)
  ref_model = fixtures.Memo(B, S, V, 2)
  model = copy.deepcopy(ref_model)

  # DPO dataset concepts:
  # We have a batch with chosen and rejected sequences.
  # In this synthetic test, we generate them randomly.
  # We also simulate the prompt masking by setting the first half of the labels to -100.
  prompt_len = S // 2
  response_len = S - prompt_len

  prompt_tokens = torch.randint(
    low=0, high=V, size=(B, prompt_len), dtype=torch.long
  )
  chosen_response = torch.randint(
    low=0, high=V, size=(B, response_len), dtype=torch.long
  )
  rejected_response = torch.randint(
    low=0, high=V, size=(B, response_len), dtype=torch.long
  )

  # Construct labels
  # For labels, prompt part is -100
  prompt_labels = torch.full((B, prompt_len), -100, dtype=torch.long)

  chosen_labels = torch.cat([prompt_labels, chosen_response], dim=1)
  rejected_labels = torch.cat([prompt_labels, rejected_response], dim=1)

  # Construct input_ids (for completeness, though Memo ignores them)
  # Input ids should have the actual prompt tokens
  chosen_input_ids = torch.cat([prompt_tokens, chosen_response], dim=1)
  rejected_input_ids = torch.cat([prompt_tokens, rejected_response], dim=1)

  # Construct a batch that mimics a DPO PreferenceDataset batch
  batch = {
    "chosen_labels": chosen_labels,
    "rejected_labels": rejected_labels,
    # Memo model ignores input_ids, but we provide them for completeness of the interface
    "chosen_input_ids": chosen_input_ids,
    "rejected_input_ids": rejected_input_ids,
    "prompt_input_ids": prompt_tokens,
  }

  # DataLoader yields batches
  loader = [batch]

  # No momentum - no resonance.
  optim = torch.optim.SGD(model.parameters(), lr=1.0)  # Fixed LR

  for epoch in range(500):
    # optim.param_groups[0]["lr"] = 0.1 / (epoch + 1.0)
    for idx, batch in enumerate(loader):
      # Unpack batch
      chosen_labels = batch["chosen_labels"]
      rejected_labels = batch["rejected_labels"]
      chosen_input_ids = batch["chosen_input_ids"]
      rejected_input_ids = batch["rejected_input_ids"]

      optim.zero_grad()

      # Forward pass for chosen and rejected
      def concat_func(batch):
        return {
          "concatenated_input_ids": torch.cat(
            [batch["chosen_input_ids"], batch["rejected_input_ids"]], dim=0
          ),
          "concatenated_labels": torch.cat(
            [batch["chosen_labels"], batch["rejected_labels"]], dim=0
          ),
          "concatenated_attention_mask": torch.ones_like(
            torch.cat(
              [batch["chosen_input_ids"], batch["rejected_input_ids"]], dim=0
            )
          ),
        }

      logp_w, logp_l, logp_wbar, logp_lbar = slpo.concatenated_forward(
        model, batch, concat_func
      )

      with torch.inference_mode():
        (
          ref_logp_w,
          ref_logp_l,
          _,
          _,
        ) = slpo.concatenated_forward(ref_model, batch, concat_func)

      loss, _, _ = slpo.slpo_loss(
        logp_w,
        logp_l,
        logp_wbar,
        logp_lbar,
        ref_logp_w,
        ref_logp_l,
        alpha,
        t=float(S),
      )

      loss.backward()
      optim.step()

      print(
        f"{epoch=}, {idx=}, loss={format(loss)}:\n"
        f"    prob_w = {format(logp_w.exp())}\n"
        f"    prob_l = {format(logp_l.exp())}\n"
      )

      if epoch == 0 and idx == 0:
        initial_loss = loss.detach()
        initial_logp_w = logp_w.exp().detach()
        initial_logp_l = logp_l.exp().detach()
        initial_logp_wbar = logp_wbar.exp().detach()
        initial_logp_lbar = logp_lbar.exp().detach()

      if torch.isnan(loss):
        raise ValueError("Loss is NaN")

  final_loss = loss.detach()
  final_logp_w = logp_w.exp().detach()
  final_logp_l = logp_l.exp().detach()
  final_logp_wbar = logp_wbar.exp().detach()
  final_logp_lbar = logp_lbar.exp().detach()

  # Verify that the model converged to the target distribution
  # We must verify this in the SCALED space, because slpo_loss optimizes in the scaled space.

  # Scale final model outputs
  scaled_logp_w, scaled_logp_wbar = slpo.apply_t(logp_w, logp_wbar, float(S))
  scaled_logp_l, scaled_logp_lbar = slpo.apply_t(logp_l, logp_lbar, float(S))

  # Scale reference outputs
  # Need complements for reference
  ref_logp_w_bar = slpo._logdiffexp(torch.zeros_like(ref_logp_w), ref_logp_w)
  ref_logp_l_bar = slpo._logdiffexp(torch.zeros_like(ref_logp_l), ref_logp_l)

  scaled_ref_logp_w, _ = slpo.apply_t(ref_logp_w, ref_logp_w_bar, float(S))
  scaled_ref_logp_l, _ = slpo.apply_t(ref_logp_l, ref_logp_l_bar, float(S))

  target_w, target_l, target_wbar, target_lbar = slpo.calc_targets(
    alpha, scaled_ref_logp_w, scaled_ref_logp_l
  )

  print(
    f"INITIAL:loss={format(initial_loss)}\n"
    f"ref_prob_w = {format(ref_logp_w.exp())}\n"
    f"ref_prob_l = {format(ref_logp_l.exp())}\n"
    f"       w_w = {format(target_w.exp())}\n"
    f"       w_l = {format(target_l.exp())}\n"
    f"   w_w_bar = {format(target_wbar.exp())}\n"
    f"   w_l_bar = {format(target_lbar.exp())}\n"
    f"    prob_w = {format(final_logp_w)}\n"
    f"    prob_l = {format(final_logp_l)}\n"
    f" prob_wbar = {format(final_logp_wbar)}\n"
    f" prob_lbar = {format(final_logp_lbar)}\n"
  )
  print(
    f"FINAL: loss={format(final_loss)}\n"
    f"ref_prob_w = {format(ref_logp_w.exp())}\n"
    f"ref_prob_l = {format(ref_logp_l.exp())}\n"
    f"       w_w = {format(target_w.exp())}\n"
    f"       w_l = {format(target_l.exp())}\n"
    f"   w_w_bar = {format(target_wbar.exp())}\n"
    f"   w_l_bar = {format(target_lbar.exp())}\n"
    f"    prob_w = {format(final_logp_w)}\n"
    f"    prob_l = {format(final_logp_l)}\n"
    f" prob_wbar = {format(final_logp_wbar)}\n"
    f" prob_lbar = {format(final_logp_lbar)}\n"
  )

  # 90% of the way there is good enough.
  # Note: We compare scaled_logp_w with target_w

  # Initial scaled logp w (approximate for atol calculation)
  initial_scaled_logp_w, _ = slpo.apply_t(
    initial_logp_w.log(), initial_logp_wbar.log(), float(S)
  )
  initial_scaled_logp_l, _ = slpo.apply_t(
    initial_logp_l.log(), initial_logp_lbar.log(), float(S)
  )

  atol = 0.1 * (initial_scaled_logp_w - target_w).abs().item()
  torch.testing.assert_close(
    scaled_logp_w,
    target_w,
    atol=atol,
    rtol=0.0,
    msg="Chosen prob did not converge to target",
  )
  atol = 0.1 * (initial_scaled_logp_l - target_l).abs().item()
  torch.testing.assert_close(
    scaled_logp_l,
    target_l,
    atol=atol,
    rtol=0.0,
    msg="Rejected prob did not converge to target",
  )

  torch.testing.assert_close(
    slpo._logsumexp(logp_l, logp_lbar), torch.zeros_like(logp_l)
  )

  # Verify that loss decreased
  if alpha == 0:
    assert final_loss == initial_loss, "Loss should not change when alpha is 0."
  else:
    assert final_loss < initial_loss, "Loss did not decrease during training."
