import copy
import math
from typing import Final

import fixtures
import pytest
import torch

from slpo import slpo
from slpo.slpo import _get_batch_logps, log_comp

torch.set_printoptions(precision=17)


def format(tensor: torch.Tensor) -> str:
  """Assumes tensor is a log probability scalar tensor."""
  value = tensor.exp().item()
  precision = 16
  chunk_size = 4
  formatted_value = f"{value:.{precision}f}"
  parts = formatted_value.split(".")
  if len(parts) != 2:
    return formatted_value  # Return as is if no decimal point
  whole, decimal = parts
  chunked_decimal = " ".join(
    [decimal[i : i + chunk_size] for i in range(0, len(decimal), chunk_size)]
  )
  return f"{whole}.{chunked_decimal}     logp:{tensor.item():.20e}"


def test_log_comp_corners():
  # Arrange
  x = torch.log(torch.tensor([[0.0, 1.0]]))

  expected = torch.tensor([[0.0, float("-inf")]])

  # Act
  result = log_comp(x)
  print(f"{x=}\n{result=}")

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
def test_calc_targets_temp_eq_one(seed, alpha):
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
    alpha, 1.0, reference_chosen_logps, reference_rejected_logps
  )

  print(
    f"\n{seed=}, {alpha=}\n"
    f"ref_prob_w = {format(reference_chosen_logps)}\n"
    f"       w_w = {format(w_w)}\n"
    f"   w_w_bar = {format(w_w_bar)}\n"
    f"ref_prob_l = {format(reference_rejected_logps)}\n"
    f"       w_l = {format(w_l)}\n"
    f"   w_l_bar = {format(w_l_bar)}\n"
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
  expected_w_w = torch.logaddexp(
    reference_chosen_logps, reference_rejected_logps + math.log(alpha)
  )
  torch.testing.assert_close(w_w, expected_w_w)

  expected_w_l = reference_rejected_logps + math.log(1 - alpha)
  torch.testing.assert_close(w_l, expected_w_l)


def test_calc_targets_non_unit_temp():
  """Test temperature with a single hand calculated value."""
  # Arrange
  w = torch.tensor(0.1, dtype=torch.float64)
  l = torch.tensor(0.2, dtype=torch.float64)
  wbar = torch.tensor(0.9, dtype=torch.float64)
  lbar = torch.tensor(0.8, dtype=torch.float64)

  alpha = 0.3
  t: Final = 2.0

  # Act
  target_w_lp, target_l_lp, target_wbar_lp, target_lbar_lp = slpo.calc_targets(
    alpha, 2.0, w.log(), l.log()
  )

  # Assert
  # Check that w_w and w_w_bar sum to 1 in probability space
  torch.testing.assert_close(
    torch.exp(target_w_lp) + torch.exp(target_wbar_lp),
    torch.ones_like(target_w_lp),
    msg=f"{torch.exp(target_w_lp)=} + {torch.exp(target_wbar_lp)=} should ~= 100%.",
  )

  # Check that w_l and w_l_bar sum to 1 in probability space
  torch.testing.assert_close(
    torch.exp(target_l_lp) + torch.exp(target_lbar_lp),
    torch.ones_like(target_l_lp),
    msg=f"{torch.exp(target_l_lp)=} + {torch.exp(target_lbar_lp)=} should ~= 100%.",
  )

  # Hand calculated expected values
  mass = l * alpha
  expected_w_lp, expected_wbar_lp = torch.nn.functional.log_softmax(
    torch.tensor([(w + mass).log() / t, (wbar - mass).log() / t])
  )

  torch.testing.assert_close(target_w_lp, expected_w_lp)
  torch.testing.assert_close(target_wbar_lp, expected_wbar_lp)

  expected_l_lp, expected_lbar_lp = torch.nn.functional.log_softmax(
    torch.tensor([(l - mass).log() / t, (lbar + mass).log() / t])
  )
  torch.testing.assert_close(target_l_lp, expected_l_lp)
  torch.testing.assert_close(target_lbar_lp, expected_lbar_lp)


@pytest.mark.parametrize("alpha", [0.1, 0.9])
def test_calc_targets_low_logprobs(alpha):
  # Arrange
  reference_chosen_logps = torch.tensor([-1_234_567.0], dtype=torch.float64)
  reference_rejected_logps = torch.tensor([-1_234_567.0], dtype=torch.float64)

  # Act
  target_w, target_l, target_wbar, target_lbar = slpo.calc_targets(
    alpha, 1.0, reference_chosen_logps, reference_rejected_logps
  )

  print(
    f"\n{alpha=}\n"
    f"  ref_prob_w = {format(reference_chosen_logps)}\n"
    f"  ref_prob_l = {format(reference_rejected_logps)}\n"
    f"         w_w = {format(target_w)}\n"
    f"         w_l = {format(target_l)}\n"
    f"target_w_bar = {format(target_wbar)}\n"
    f"target_l_bar = {format(target_lbar)}\n"
  )

  # Check that target_w is bigger and target_l is smaller than ref.
  assert target_w.item() > reference_chosen_logps.item()
  assert target_l.item() < reference_rejected_logps.item()


@pytest.mark.parametrize("seed", [101, 102, 103])
def test_apply_t(seed):
  # Arrange
  temperature = 100
  logp1 = (torch.rand(1) / 2.0).log()
  logp2 = log_comp(logp1)

  # Expected
  expected_scaled_logp1 = logp1 / temperature - torch.logaddexp(
    logp1 / temperature, logp2 / temperature
  )
  expected_scaled_logp2 = log_comp(expected_scaled_logp1)

  # Act
  scaled_logp1, scaled_logp2 = slpo.apply_t(logp1, logp2, temperature)

  print(
    f"{temperature=}\n"
    f"logp1={format(logp1)}\n"
    f"logp2={format(logp2)}\n"
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
  p_w = torch.rand(1) * 0.000_001
  p_l = torch.rand(1) * 0.000_001
  p_wbar = 1.0 - p_w
  p_lbar = 1.0 - p_l
  p_w_ref = torch.rand(1) * 0.000_001
  p_l_ref = torch.rand(1) * 0.000_001
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

  # Calculate targets in original space
  target_w_log, target_l_log, target_w_bar_log, target_l_bar_log = (
    slpo.calc_targets(alpha, float(S), logp_w_ref, logp_l_ref)
  )

  # Scale all the logps.
  scaled_logp_w, scaled_logp_wbar = slpo.apply_t(logp_w, logp_wbar, float(S))
  scaled_logp_l, scaled_logp_lbar = slpo.apply_t(logp_l, logp_lbar, float(S))

  # Expected
  # KL = sum(p * (log p - log q))
  # We have 4 components per batch item.
  # loss is mean over (B * 4) elements.

  term1 = target_w_log.exp() * (target_w_log - scaled_logp_w)
  term2 = target_w_bar_log.exp() * (target_w_bar_log - scaled_logp_wbar)
  term3 = target_l_log.exp() * (target_l_log - scaled_logp_l)
  term4 = target_l_bar_log.exp() * (target_l_bar_log - scaled_logp_lbar)

  expected = (term1 + term2 + term3 + term4) / 4

  assert expected != 0.0, "Expected loss is zero, test is invalid."
  print(f"{expected=}\n{loss=}\n{term1=}\n{term2=}\n{term3=}\n{term4=}")

  # Assert
  torch.testing.assert_close(
    loss,
    expected,
    rtol=0.01,
    atol=0.0,
    msg=f"{expected=}\n{loss=}",
  )


@pytest.mark.parametrize("seed", [101, 102])
@pytest.mark.parametrize("alpha", [0.0, 0.1, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("B,S,V", [(1, 16, 1024)])
def test_slpo_trains_model(seed, alpha, B, S, V):
  # Arrange model
  torch.manual_seed(seed)
  ref_model = fixtures.Memo(B, S, V, 2)
  model = copy.deepcopy(ref_model)

  # Arrange data
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

  # Setup training loop
  epochs = 100
  optim = torch.optim.Adam(model.parameters(), lr=0.1)
  lr_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, 50)
  for epoch in range(epochs):
    lr_sched.step(epoch)
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
            [chosen_input_ids, rejected_input_ids], dim=0
          ),
          "concatenated_labels": torch.cat(
            [chosen_labels, rejected_labels], dim=0
          ),
          "concatenated_attention_mask": torch.ones_like(
            torch.cat([chosen_input_ids, rejected_input_ids], dim=0)
          ),
        }

      logp_w, logp_l, logp_wbar, logp_lbar = slpo.concatenated_forward(
        model, batch, concat_func
      )

      with torch.inference_mode():
        ref_logp_w, ref_logp_l, _, _ = slpo.concatenated_forward(
          ref_model, batch, concat_func
        )

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

      if epoch == 0 and idx == 0:
        initial_loss = loss.detach()
        initial_logp_w = logp_w.detach()
        initial_logp_l = logp_l.detach()

      if torch.isnan(loss):
        raise ValueError("Loss is NaN")

      loss.backward()
      optim.step()

  final_loss = loss.detach()
  final_logp_w = logp_w.detach()
  final_logp_l = logp_l.detach()

  # Verify that the model converged to the target distribution
  # w_w et al were calculated on temperature adjusted logprobs. But that is
  # an internal implementation detail. So recreate those weights without
  # temperature.
  target_logp_w, target_logp_l, target_logp_wbar, target_logp_lbar = (
    slpo.calc_targets(alpha, 1.0, ref_logp_w, ref_logp_l)
  )

  print(
    f"INITIAL:loss={initial_loss.item()}\n"
    f"   ref_logp_w = {format(ref_logp_w)}\n"
    f"target_logp_w = {format(target_logp_w)}\n"
    f"       logp_w = {format(final_logp_w)}\n"
    f"   ref_logp_l = {format(ref_logp_l)}\n"
    f"target_logp_l = {format(target_logp_l)}\n"
    f"       logp_l = {format(final_logp_l)}\n"
  )
  print(
    f"FINAL: loss={final_loss.item()}\n"
    f"   ref_logp_w = {format(ref_logp_w)}\n"
    f"target_logp_w = {format(target_logp_w)}\n"
    f"       logp_w = {format(final_logp_w)}\n"
    f"   ref_logp_l = {format(ref_logp_l)}\n"
    f"target_logp_l = {format(target_logp_l)}\n"
    f"       logp_l = {format(final_logp_l)}\n"
  )

  torch.testing.assert_close(
    torch.logaddexp(logp_w, logp_wbar), torch.zeros_like(logp_w)
  )
  torch.testing.assert_close(
    torch.logaddexp(logp_l, logp_lbar), torch.zeros_like(logp_l)
  )

  # 90% of the way there is good enough.
  atol = 0.1 * (initial_logp_w - target_logp_w).abs().item()
  torch.testing.assert_close(
    final_logp_w,
    target_logp_w,
    atol=atol,
    rtol=0.0,
    msg=f"winner logp did not converge to target\n{final_logp_w=}\n{target_logp_w=}",
  )
  target_is_neg_inf = torch.isneginf(target_logp_l)
  safe_target_logp_l = torch.where(
    target_is_neg_inf, torch.finfo(target_logp_l.dtype).min, target_logp_l
  )
  atol = 0.1 * (initial_logp_l - target_logp_l).abs().item()
  torch.testing.assert_close(
    final_logp_l,
    safe_target_logp_l,
    atol=atol,
    rtol=0.0,
    msg=f"loser logp did not converge to target\n{final_logp_l=}\n{target_logp_l=}",
  )

  # Verify that loss decreased
  if alpha == 0:
    assert final_loss == initial_loss, "Loss should not change when alpha is 0."
  else:
    assert final_loss < initial_loss, "Loss did not decrease during training."


@pytest.mark.skip(reason="Long running test")
def test_slpo_trains_bert():
  # Randomly initialized memo model has extreme logprobs,
  # beyond the representable values for mass shifts. Realistic logprobs
  # are in the range of -200 in the slpo repo.
  # So sometimes there is no shift possible onto the larger, chosen
  # distribution, and that's probably okay.
  #   INITIAL:loss=4.3064072628665436e-08
  #    ref_logp_w = 0.0000 0000 0000 0000     logp:-2.75977505984509480186e+03
  # target_logp_w = 0.0000 0000 0000 0000     logp:-2.75977505984509389236e+03
  #        logp_w = 0.0000 0000 0000 0000     logp:-2.75977520037452131874e+03
  #    ref_logp_l = 0.0000 0000 0000 0000     logp:-2.78747098005245788954e+03
  # target_logp_l = 0.0000 0000 0000 0000     logp:-2.79207615023844618918e+03
  #        logp_l = 0.0000 0000 0000 0000     logp:-2.79202189913104894003e+03

  # FINAL: loss=5.958916587453505e-12
  #    ref_logp_w = 0.0000 0000 0000 0000     logp:-2.75977505984509480186e+03
  # target_logp_w = 0.0000 0000 0000 0000     logp:-2.75977505984509389236e+03
  #        logp_w = 0.0000 0000 0000 0000     logp:-2.75977520037452131874e+03
  #    ref_logp_l = 0.0000 0000 0000 0000     logp:-2.78747098005245788954e+03
  # target_logp_l = 0.0000 0000 0000 0000     logp:-2.79207615023844618918e+03
  #        logp_l = 0.0000 0000 0000 0000     logp:-2.79202189913104894003e+03
  # test_slpo_trains_model(seed=101, alpha=0.99, B=1, S=512, V=30_522)

  test_slpo_trains_model(seed=102, alpha=0.99, B=1, S=512, V=30_522)


@pytest.mark.skip(reason="Long running test")
def test_slpo_trains_llama3():
  test_slpo_trains_model(seed=102, alpha=0.99, B=1, S=2048, V=128_000)
