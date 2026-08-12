"""Defines the SLPO loss function (Huber on centered logprobs)."""

from typing import TypeAlias

import torch
import torch.nn.functional as F


def check_numerics(
  tensor: torch.Tensor,
  name: str,
) -> bool:
  """Check a tensor for NaN or Inf values and print a diagnostic summary.

  Args:
    tensor: The tensor to check.
    name: A label for the tensor, printed in the diagnostic message.

  Returns:
    True if the tensor is clean, False if it contains NaN or Inf.

  Raises:
    ValueError: if the tensor contains NaN or Inf or -Inf.
  """
  n_nan = torch.isnan(tensor).sum().item()
  n_inf = torch.isinf(tensor).sum().item()

  if n_nan == 0 and n_inf == 0:
    return True

  n_total = tensor.numel()
  finite = tensor[torch.isfinite(tensor)]
  stats = (
    f"min={finite.min().item():.4g}, max={finite.max().item():.4g}"
    if finite.numel() > 0
    else "no finite values"
  )
  msg = (
    f"check_numerics [{name}]: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
    f"nan={n_nan}/{n_total} inf={n_inf}/{n_total} ({stats})"
  )
  print(msg)

  raise ValueError(msg)


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


def log_comp(log_p: torch.Tensor) -> torch.Tensor:
  """Calculate log(1 - exp(log_p)) in a numerically stable way."""
  cutoff = torch.log(torch.tensor(0.5, device=log_p.device, dtype=log_p.dtype))
  return torch.where(
    log_p <= cutoff,
    torch.log1p(-torch.exp(log_p)),
    torch.log(-torch.expm1(log_p)),
  )


def check_t_dim_ne_zero(logps: torch.Tensor) -> None:
  """Check that the S dimension of logps is not zero.

  Args:
    logps: shape (..., S, V) = log softmax outputs for batch of sequences,
      where B is batch size, S is timestep (sequence length), and V is vocab size.

  Raises:
    ValueError: if S == 0 (empty sequences not supported).
  """
  if logps.shape[-2] == 0:
    raise ValueError("Empty sequences are not supported.")


def check_logps_are_prob_dist(logps: torch.Tensor) -> None:
  """Check that the logps are a probability distribution.

  That is, the lse(logprobs) along the vocab dimension equal 0.0
  (sum of probs of tokens equals 100%, at all time steps).

  Args:
    logps: shape (..., T, V)

  Raises:
    ValueError: if LSE(logps, dim=V) is not equal to zero at all time steps.
  """
  logps_t = torch.logsumexp(logps, dim=-1)  # Shape (..., T)
  torch.testing.assert_close(
    logps_t, torch.zeros_like(logps_t), msg=f"{logps_t=}\n{logps=}"
  )


# _get_batch_logps and concatenated_forward live in slpo_adapter.py.


def slpo_loss_check_batch_size(
  model_chosen_logps: torch.Tensor,
  model_rejected_logps: torch.Tensor,
  model_chosen_logps_comp: torch.Tensor,
  model_rejected_logps_comp: torch.Tensor,
  reference_chosen_logps: torch.Tensor,
  reference_rejected_logps: torch.Tensor,
  alpha: float,
) -> None:
  """Helper to validate inputs to slpo_loss.

  Raises:
    ValueError: if any of the input tensors do not have the expected shape."""
  batch_size = model_chosen_logps.size(0)

  if model_chosen_logps.size() != (batch_size,):
    raise ValueError(
      f"Expected model_chosen_logps shape {(batch_size,)}, got {model_chosen_logps.size()}"
    )

  if model_rejected_logps.size() != (batch_size,):
    raise ValueError(
      f"Expected model_rejected_logps shape {(batch_size,)}, got {model_rejected_logps.size()}"
    )

  if model_chosen_logps_comp.size() != (batch_size,):
    raise ValueError(
      f"Expected model_chosen_logps_comp shape {(batch_size,)}, got {model_chosen_logps_comp.size()}"
    )

  if model_rejected_logps_comp.size() != (batch_size,):
    raise ValueError(
      f"Expected model_rejected_logps_comp shape {(batch_size,)}, got {model_rejected_logps_comp.size()}"
    )

  if reference_chosen_logps.size() != (batch_size,):
    raise ValueError(
      f"Expected reference_chosen_logps shape {(batch_size,)}, got {reference_chosen_logps.size()}"
    )

  if reference_rejected_logps.size() != (batch_size,):
    raise ValueError(
      f"Expected reference_rejected_logps shape {(batch_size,)}, got {reference_rejected_logps.size()}"
    )


def calc_targets(
  alpha,
  w_ref_logprobs,
  l_ref_logprobs,
):
  """Calculate target logits.

  Args:
    alpha: What percentage of the probability mass of the rejected sequence
      to assign to the chosen sequence. [0.0, 1.0), where 0.999 is the max
      acceptable value.
    w_ref_logprobs: Log probabilities of the chosen sequences under the reference.
      Shape: (batch_size,)
    l_ref_logprobs: Log probabilities of the rejected sequences under the reference.
      Shape: (batch_size,)

  Returns:
    target_w_logits: Target logit for the chosen sequences. Shape: (batch_size,)
    target_l_logits: Target logit for the rejected sequences. Shape: (batch_size,)
    target_wbar_logits: Target complement logit for the chosen sequences. Shape: (batch_size,)
    target_lbar_logits: Target complement logit for the rejected sequences. Shape: (batch_size,)
  """
  if alpha > 0.999:
    raise ValueError(
      f"alpha must be <= 0.999 to avoid -inf targets, got {alpha}"
    )

  # Local notation: all values are logprobs. w and l refer to all of the
  # following: ref, target, and intermediate values.

  device = w_ref_logprobs.device

  # Make w,l log probabilities.
  w = w_ref_logprobs.double()
  l = l_ref_logprobs.double()
  del w_ref_logprobs, l_ref_logprobs

  # Setup alpha in log space.
  a = torch.tensor(alpha, device=device, dtype=torch.float64).log()
  a_comp = torch.tensor(1.0 - alpha, device=device, dtype=torch.float64).log()

  # Shift mass from l to w.
  w = torch.logaddexp(w, l + a)
  l = l + a_comp

  # l can never exceed 100% because we only subtracted mass from it.
  assert torch.all(l <= 0.0), (
    f"logp of rejected sequence exceeds 100%: {l[l > 0.0]}"
  )

  # w can exceed 0 when the sum of response-only conditional log-probs
  # violates the probability axioms (e.g. short/high-confidence responses).
  # Mark those rows invalid and clamp to just below 0 to prevent NaN in
  # log_comp. Their losses will be zeroed out by the caller.
  valid = w <= 0.0
  _eps = torch.finfo(torch.float64).eps
  w = torch.clamp(w, max=-_eps)

  _f64 = torch.finfo(torch.float64)
  l = torch.clamp(l, min=_f64.min)

  # Calculate complement log probabilities
  wbar = log_comp(w)
  lbar = log_comp(l)

  # Check for total mass equals 100%.
  check_logps_are_prob_dist(torch.stack([w, wbar], dim=-1))
  check_logps_are_prob_dist(torch.stack([l, lbar], dim=-1))

  # Block grads.
  w = w.detach()
  l = l.detach()
  wbar = wbar.detach()
  lbar = lbar.detach()
  valid = valid.detach()

  return w, l, wbar, lbar, valid


torch_tensor: TypeAlias = torch.Tensor


# Although the DPO signature uses the token "policy", SLPO's entire goal
# is to eliminate RL concepts, so we use the term "model" here.
def slpo_loss(
  model_chosen_logps: torch.Tensor,
  model_rejected_logps: torch.Tensor,
  model_chosen_logps_comp: torch.Tensor,
  model_rejected_logps_comp: torch.Tensor,
  reference_chosen_logps: torch.Tensor,
  reference_rejected_logps: torch.Tensor,
  alpha: float,
  chosen_lengths: torch.Tensor | None = None,
  rejected_lengths: torch.Tensor | None = None,
) -> tuple:
  """Compute the SLPO loss for a batch of sequences.

  Centering: for each pair (chosen, rejected), we subtract the pair mean so
  that the two values are equal and opposite. This is done independently for
  the (w, l) pair and the (wbar, lbar) pair, for both student and teacher.
  Huber loss (delta=1.0) is then applied between centered student and teacher.

  Args:
    model_chosen_logps: Log probabilities of the chosen sequences under the model.
      Shape: (batch_size,)
    model_rejected_logps: Log probabilities of the rejected sequences under the model.
      Shape: (batch_size,)
    model_chosen_logps_comp: Complement log probabilities of the chosen sequences
      under the model. Shape: (batch_size,)
    model_rejected_logps_comp: Complement log probabilities of the rejected sequences
      under the model. Shape: (batch_size,)
    reference_chosen_logps: Log probabilities of the chosen sequences under the reference.
      Shape: (batch_size,)
    reference_rejected_logps: Log probabilities of the rejected sequences under the reference.
      Shape: (batch_size,)
    alpha: What percentage of the probability mass of the rejected sequence
      to assign to the chosen sequence. Should be "far" from 0.0 and 1.0
    chosen_lengths: Number of non-masked tokens in each chosen sequence.
      Shape: (batch_size,). When provided, logodds are normalized per token
      before computing the loss, preventing long sequences from dominating.
    rejected_lengths: Number of non-masked tokens in each rejected sequence.
      Shape: (batch_size,). Must be provided if chosen_lengths is provided.

  Returns:
    loss: Per-example loss (max of the 4 candidate Huber losses per example). Shape: (batch_size,).
    chosen_rewards: Reward values for the chosen sequences. Shape: (batch_size,)
    rejected_rewards: Reward values for the rejected sequences. Shape: (batch_size,)
    w_c: Centered model log-odds for chosen (= (logp_w - logp_wbar) / 2). Shape: (batch_size,)
    wbar_c: Centered model log-odds for chosen complement (= -w_c). Shape: (batch_size,)
    l_c: Centered model log-odds for rejected (= (logp_l - logp_lbar) / 2). Shape: (batch_size,)
    lbar_c: Centered model log-odds for rejected complement (= -l_c). Shape: (batch_size,)
    target_w: Target centered log-odds for chosen. Shape: (batch_size,)
    target_l: Target centered log-odds for rejected. Shape: (batch_size,)
    target_wbar: Target centered log-odds for chosen complement (= -target_w). Shape: (batch_size,)
    target_lbar: Target centered log-odds for rejected complement (= -target_l). Shape: (batch_size,)
    loss_w: Candidate Huber loss on chosen component (only active when it wins the max). Shape: (batch_size,)
    loss_l: Candidate Huber loss on rejected component (only active when it wins the max). Shape: (batch_size,)
    loss_wbar: Candidate Huber loss on chosen complement (only active when it wins the max). Shape: (batch_size,)
    loss_lbar: Candidate Huber loss on rejected complement (only active when it wins the max). Shape: (batch_size,)
  """
  # Cast to fp64 and rename. w, l, wbar, lbar are log probabilities.
  w = model_chosen_logps.to(torch.float64)
  l = model_rejected_logps.to(torch.float64)
  wbar = model_chosen_logps_comp.to(torch.float64)
  lbar = model_rejected_logps_comp.to(torch.float64)
  ref_w = reference_chosen_logps.to(torch.float64)
  ref_l = reference_rejected_logps.to(torch.float64)

  del model_chosen_logps
  del model_rejected_logps
  del model_chosen_logps_comp
  del model_rejected_logps_comp
  del reference_chosen_logps
  del reference_rejected_logps

  slpo_loss_check_batch_size(
    w,
    l,
    wbar,
    lbar,
    ref_w,
    ref_l,
    alpha,
  )

  check_numerics(w, "w")
  check_numerics(l, "l")
  check_numerics(wbar, "wbar")
  check_numerics(lbar, "lbar")
  check_numerics(ref_w, "ref_w")
  check_numerics(ref_l, "ref_l")

  chosen_rewards = (w - ref_w).detach()
  rejected_rewards = (l - ref_l).detach()

  # Calculate targets in log probability space.
  # valid is False for rows where exp(ref_w) + alpha*exp(ref_l) > 1,
  # which violates the probability axioms. Those rows get zero loss.
  target_w, target_l, target_wbar, target_lbar, valid = calc_targets(
    alpha, ref_w, ref_l
  )

  check_numerics(target_w, "target_w")
  check_numerics(target_l, "target_l")
  check_numerics(target_wbar, "target_wbar")
  check_numerics(target_lbar, "target_lbar")

  # Per-token normalization: divide logodds by sequence length so that long
  # sequences don't produce catastrophically large regression targets.
  # When chosen is much longer than rejected, target_w ≈ ref_l (dominated by
  # the shorter rejected sequence's probability), creating an unreachable
  # target of magnitude ~(len_chosen - len_rejected) logodds. Normalizing
  # by sequence length makes all gaps comparable across sequence lengths.
  if chosen_lengths is not None:
    n_w = chosen_lengths.to(torch.float64)
    n_l = rejected_lengths.to(torch.float64)
    model_w_logodds = (w - wbar) / n_w
    model_l_logodds = (l - lbar) / n_l
    target_w_logodds = (target_w - target_wbar) / n_w
    target_l_logodds = (target_l - target_lbar) / n_l
  else:
    model_w_logodds = w - wbar
    model_l_logodds = l - lbar
    target_w_logodds = target_w - target_wbar
    target_l_logodds = target_l - target_lbar

  loss_w = (
    F.huber_loss(model_w_logodds, target_w_logodds, reduction="none", delta=1.0)
    * valid
  )
  loss_l = (
    F.huber_loss(model_l_logodds, target_l_logodds, reduction="none", delta=1.0)
    * valid
  )

  # As this loss has been problematic, there are various diagnostic
  # options that build up towards the final loss.

  # A) As a diagnostic, Huber between w (model chosen logp) and target_w.
  # This directly regresses the chosen log-probability toward its target,
  # ignoring the complement. Confirmed working: model_w tracked target_w cleanly.
  # losses = F.huber_loss(w, target_w, reduction="none", delta=1.0) * valid

  # B) As a diagnostic, only apply the loss_w. This should push model_w_logodds
  # from ~-125 toward ~-81 (target_w_logodds).
  # As of May 9, 2026, this approach worked very well on a single batch
  # of data.
  # https://wandb.ai/yaoshiang/direct-preference-optimization/9gn0541u
  #
  # As of May 11, 2026, this approach also works on 4 batches of 16 rows.
  # test_slpo_16batch_pinned
  #
  # Sweep: lr_sweep5 (RMSprop + Huber, loss_w only), June 2026. Hypothesis:
  # loss_l interferes with loss_w (sweeps 2-4 all show gap_w worsening regardless
  # of optimizer/loss). Removing loss_l isolates the gap_w convergence signal.
  # Result: no improvement — gap_w still worsens. Interference hypothesis wrong.
  # losses = loss_w

  # C) Use the sum/mean of the two losses. Both W and L are needed: loss_l alone
  # creates "mush" (rejected ≈ chosen in the rehearsal dataset, so suppressing l
  # also suppresses w). The goal is the GAP (model_w_logodds - model_l_logodds)
  # widening over training, even if both absolute values decrease (DPO
  # displacement). long_run1 (June 2026): full dataset, multi-epoch, lr=1e-7.
  # Result: gap_l converged (gap_l → -0.26), gap_w completely stuck (-41.65).
  # Both logprobs dropped ~1.0 over 3 epochs; margin unchanged (-6.67 → -6.72).
  # Root cause: target_w is unreachable (asks P(chosen) > SFT ceiling).
  # losses = loss_w + loss_l

  # If w or l is dominating, use max to focus on the highest loss.
  # stacked = torch.stack([loss_w, loss_l], dim=-1)  # (B, 2)
  # losses = stacked.max(dim=-1).values  # (B,)

  # D) MSE instead of Huber for both w and l. Global gradient clipping already
  # provides the large-gradient robustness that Huber was designed for. With MSE,
  # gradient ∝ gap magnitude, so loss_w gets a ~66x gradient advantage over
  # loss_l when gap_w≈-33 and gap_l≈0.5. This should prevent loss_l from dragging
  # model_w down. As gap_w closes toward zero, the advantage diminishes naturally,
  # giving loss_l proportionally more influence (automatic curriculum).
  # Sweep: lr_sweep3 (RMSprop), May 2026. Result: identical convergence to Huber
  # (RMSprop normalizes away the magnitude advantage), plus CUDA crashes from
  # gradient explosion on outlier batches (gap_w=-98 → grad_norm=707k).
  # Sweep: lr_sweep4 (AdamW), June 2026. Result: same as RMSprop+Huber.
  # loss_w_d = (
  #   F.mse_loss(model_w_logodds, target_w_logodds, reduction="none") * valid
  # )
  # loss_l_d = (
  #   F.mse_loss(model_l_logodds, target_l_logodds, reduction="none") * valid
  # )
  # losses = loss_w_d + loss_l_d

  # E) SLPO-diff: shifted sigmoid on the margin. Δt (derived from α mass-transfer)
  # plays the role of DPO's log-ratios — it anchors where the margin should stop.
  # Loss = 0 when Δω = Δt exactly; gradient pulls back if overshooting.
  # Unlike DPO's KL penalty (soft, exponentially fading), Δt is a calibrated
  # restraint from a concrete probabilistic interpretation (how much mass α
  # transfers). Closest prior: SimPO (-log σ(Δω - γ)) but with γ = Δt computed
  # from reference via α rather than hand-tuned.
  # slpo_diff_run1 (July 2026).
  losses = (
    -F.logsigmoid(model_w_logodds - model_l_logodds - (target_w_logodds - target_l_logodds))
    * valid
  )

  # Zero out rows where the reference logprobs violated probability axioms.
  losses = losses * valid

  # --BEGIN DIAGNOSTIC CODE--
  n_invalid = (~valid).sum().item()
  if n_invalid > 0:
    print(
      f"slpo_loss: {n_invalid}/{valid.numel()} rows invalid "
      f"(exp(ref_w)+alpha*exp(ref_l)>1); their losses are zeroed."
    )
  # --END DIAGNOSTIC CODE--

  return (
    losses,
    chosen_rewards,
    rejected_rewards,
    w,
    wbar,
    l,
    lbar,
    target_w,
    target_l,
    target_wbar,
    target_lbar,
    loss_w,
    loss_l,
    valid,
  )
