import copy
import math

import fixtures
import pytest
import torch
import torch.nn.functional as F

from slpo import slpo, slpo_adapter
from slpo.slpo import log_comp

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


# ── log_comp ──────────────────────────────────────────────────────────────────


def test_log_comp_edge_cases():
  """Test log_comp with values at 0% and 100 probability."""
  # Arrange
  x = torch.log(torch.tensor([[0.0, 1.0]]))

  expected = torch.tensor([[0.0, float("-inf")]])

  # Act
  result = log_comp(x)
  print(f"{x=}\n{result=}")

  # Assert
  torch.testing.assert_close(expected, result)


# ── calc_targets ──────────────────────────────────────────────────────────────


def test_calc_targets_concrete():
  # Arrange: p_w=0.3, p_l=0.2, alpha=0.5
  #   target_w    = 0.3 + 0.5*0.2 = 0.4  → wbar = 0.6
  #   target_l    = (1-0.5)*0.2   = 0.1  → lbar = 0.9
  #   w_c  = (log(0.4) - log(0.6)) / 2
  #   wbar_c = -w_c
  #   l_c  = (log(0.1) - log(0.9)) / 2
  #   lbar_c = -l_c
  logp_w = torch.log(torch.tensor([0.3], dtype=torch.float64))
  logp_l = torch.log(torch.tensor([0.2], dtype=torch.float64))

  # Act
  target_w_c, target_l_c, target_wbar_c, target_lbar_c, _ = slpo.calc_targets(
    0.5, logp_w, logp_l
  )

  # Assert
  expected_w_c = (math.log(0.4) - math.log(0.6)) / 2
  expected_l_c = (math.log(0.1) - math.log(0.9)) / 2
  torch.testing.assert_close(
    target_w_c, torch.tensor([expected_w_c], dtype=torch.float64)
  )
  torch.testing.assert_close(
    target_l_c, torch.tensor([expected_l_c], dtype=torch.float64)
  )
  torch.testing.assert_close(
    target_wbar_c, torch.tensor([-expected_w_c], dtype=torch.float64)
  )
  torch.testing.assert_close(
    target_lbar_c, torch.tensor([-expected_l_c], dtype=torch.float64)
  )


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
def test_calc_targets(seed, alpha):
  torch.manual_seed(seed)
  # Arrange
  B = 1

  w = torch.rand(B).to(torch.float64) / 2.0
  l = torch.rand(B).to(torch.float64) / 2.0

  # Act
  w_c_lp, l_c_lp, wbar_c_lp, lbar_c_lp, _ = slpo.calc_targets(
    alpha, w.log(), l.log()
  )

  # Assert: outputs must be antisymmetric (centered logits sum to zero)
  torch.testing.assert_close(w_c_lp + wbar_c_lp, torch.zeros_like(w_c_lp))
  torch.testing.assert_close(l_c_lp + lbar_c_lp, torch.zeros_like(l_c_lp))

  # Recover probabilities via softmax over each centered-logit pair.
  probs_w = torch.softmax(torch.stack([w_c_lp, wbar_c_lp], dim=-1), dim=-1)
  probs_l = torch.softmax(torch.stack([l_c_lp, lbar_c_lp], dim=-1), dim=-1)

  torch.testing.assert_close(probs_w[..., 0], w + alpha * l)
  torch.testing.assert_close(probs_w[..., 1], 1 - w - alpha * l)
  torch.testing.assert_close(probs_l[..., 0], (1 - alpha) * l)
  torch.testing.assert_close(probs_l[..., 1], 1 - (1 - alpha) * l)


@pytest.mark.parametrize("alpha", [0.1, 0.9])
def test_calc_targets_low_logprobs(alpha):
  # Arrange: both sequences have negligibly small probability
  ref_w = torch.tensor([-1_234_567.0], dtype=torch.float64)
  ref_l = torch.tensor([-1_234_567.0], dtype=torch.float64)

  # Act
  w_c, l_c, wbar_c, lbar_c, _ = slpo.calc_targets(alpha, ref_w, ref_l)

  # Outputs are centered logits — antisymmetric
  torch.testing.assert_close(w_c + wbar_c, torch.zeros_like(w_c))
  torch.testing.assert_close(l_c + lbar_c, torch.zeros_like(l_c))

  # Both sequences are tiny, so both centered logits should be very negative
  assert torch.all(w_c < 0)
  assert torch.all(l_c < w_c), (
    f"Loser should have more negative logit: {l_c=}, {w_c=}"
  )


# ── slpo_loss (Huber on centered logprobs) ────────────────────────────────────


@pytest.mark.parametrize(
  "B,S,V",
  (
    (1, 2, 2),
    (1, 8, 16),
    (64, 8, 16),
    (1, 2048, 128_000),
  ),
)
def test_slpo_on_logps(B, S, V):
  """slpo_loss should equal Huber of centered (chosen, rejected) logprob pairs."""
  prev_dtype = torch.get_default_dtype()
  torch.set_default_dtype(torch.double)
  try:
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
    (loss, *_) = slpo.slpo_loss(
      logp_w,
      logp_l,
      logp_wbar,
      logp_lbar,
      logp_w_ref,
      logp_l_ref,
      alpha=alpha,
    )

    # Expected: max of the 4 Huber components per example.
    # Since loss_wbar == loss_w and loss_lbar == loss_l (by centering symmetry),
    # this reduces to max(loss_w, loss_l).
    target_w, target_l, target_wbar, target_lbar, _ = slpo.calc_targets(
      alpha, logp_w_ref, logp_l_ref
    )

    w_c = (logp_w - logp_wbar) / 2.0
    wbar_c = (logp_wbar - logp_w) / 2.0
    l_c = (logp_l - logp_lbar) / 2.0
    lbar_c = (logp_lbar - logp_l) / 2.0

    loss_w = F.huber_loss(w_c, target_w, reduction="none", delta=1.0)
    loss_l = F.huber_loss(l_c, target_l, reduction="none", delta=1.0)
    loss_wbar = F.huber_loss(wbar_c, target_wbar, reduction="none", delta=1.0)
    loss_lbar = F.huber_loss(lbar_c, target_lbar, reduction="none", delta=1.0)
    expected_loss = (
      torch.stack([loss_w, loss_l, loss_wbar, loss_lbar], dim=0)
      .max(dim=0)
      .values
    )

    torch.testing.assert_close(loss, expected_loss, rtol=1e-10, atol=0.0)
  finally:
    torch.set_default_dtype(prev_dtype)


@pytest.mark.parametrize("seed", [101, 102])
@pytest.mark.parametrize("alpha", [0.01, 0.1, 0.5, 0.9, 0.99])
@pytest.mark.parametrize("B,S,V", [(1, 16, 1024)])
def test_slpo_trains_model(seed, alpha, B, S, V):
  # Arrange model
  torch.manual_seed(seed)
  ref_model = fixtures.Memo(B, S, V, 2)
  model = copy.deepcopy(ref_model)

  # Arrange data
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

  prompt_labels = torch.full((B, prompt_len), -100, dtype=torch.long)
  chosen_labels = torch.cat([prompt_labels, chosen_response], dim=1)
  rejected_labels = torch.cat([prompt_labels, rejected_response], dim=1)

  chosen_input_ids = torch.cat([prompt_tokens, chosen_response], dim=1)
  rejected_input_ids = torch.cat([prompt_tokens, rejected_response], dim=1)

  batch = {
    "chosen_labels": chosen_labels,
    "rejected_labels": rejected_labels,
    "chosen_input_ids": chosen_input_ids,
    "rejected_input_ids": rejected_input_ids,
    "prompt_input_ids": prompt_tokens,
  }

  loader = [batch]

  epochs = 100
  optim = torch.optim.Adam(model.parameters(), lr=0.1)
  lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

  for epoch in range(epochs):
    lr_sched.step(epoch)
    for idx, batch in enumerate(loader):
      chosen_labels = batch["chosen_labels"]
      rejected_labels = batch["rejected_labels"]
      chosen_input_ids = batch["chosen_input_ids"]
      rejected_input_ids = batch["rejected_input_ids"]

      optim.zero_grad()

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

      logp_w, logp_l, logp_wbar, logp_lbar, chosen_lengths, rejected_lengths = slpo_adapter.concatenated_forward(
        model, batch, concat_func
      )

      with torch.inference_mode():
        ref_logp_w, ref_logp_l, _, _, _, _ = slpo_adapter.concatenated_forward(
          ref_model, batch, concat_func
        )

      (loss, *_) = slpo.slpo_loss(
        logp_w,
        logp_l,
        logp_wbar,
        logp_lbar,
        ref_logp_w,
        ref_logp_l,
        alpha,
        chosen_lengths=chosen_lengths,
        rejected_lengths=rejected_lengths,
      )

      if epoch == 0 and idx == 0:
        initial_loss = loss.mean().detach()
        initial_w_c = ((logp_w - logp_wbar) / 2.0).detach()

      if torch.isnan(loss).any():
        raise ValueError("Loss is NaN")

      loss.mean().backward()
      optim.step()

  final_loss = loss.mean().detach()
  final_logp_w = logp_w.detach()
  final_logp_l = logp_l.detach()
  final_logp_wbar = logp_wbar.detach()
  final_logp_lbar = logp_lbar.detach()

  # The loss drives the centered difference (w - l)/2 toward (target_w - target_l)/2.
  target_logp_w, target_logp_l, target_logp_wbar, target_logp_lbar, _ = (
    slpo.calc_targets(alpha, ref_logp_w, ref_logp_l)
  )

  final_w_c = (final_logp_w - final_logp_l) / 2.0
  target_w_c = (target_logp_w - target_logp_l) / 2.0

  print(
    f"INITIAL: loss={initial_loss.item()}, w_c={initial_w_c.item():.6f}, target_w_c={target_w_c.item():.6f}\n"
    f"FINAL:   loss={final_loss.item()}, w_c={final_w_c.item():.6f}, target_w_c={target_w_c.item():.6f}\n"
    f"  ref_logp_w={ref_logp_w.item():.4f}, ref_logp_l={ref_logp_l.item():.4f}\n"
    f"  target_logp_w={target_logp_w.item():.4f}, target_logp_l={target_logp_l.item():.4f}\n"
    f"  final_logp_w={final_logp_w.item():.4f}, final_logp_l={final_logp_l.item():.4f}\n"
  )

  torch.testing.assert_close(
    torch.logaddexp(final_logp_w, final_logp_wbar),
    torch.zeros_like(final_logp_w),
  )
  torch.testing.assert_close(
    torch.logaddexp(final_logp_l, final_logp_lbar),
    torch.zeros_like(final_logp_l),
  )

  initial_err = (initial_w_c - target_w_c).abs()
  final_err = (final_w_c - target_w_c).abs()
  assert final_err < initial_err, (
    f"Centered winner-loser gap did not improve\n{initial_err=}\n{final_err=}"
  )
  assert final_loss < initial_loss, "Loss did not decrease during training."


# @pytest.mark.skip(reason="Long running test")
def test_slpo_trains_bert():
  test_slpo_trains_model(seed=102, alpha=0.99, B=1, S=512, V=30_522)


# @pytest.mark.skip(reason="Long running test")
def test_slpo_trains_llama3():
  test_slpo_trains_model(seed=102, alpha=0.99, B=1, S=2048, V=128_000)
