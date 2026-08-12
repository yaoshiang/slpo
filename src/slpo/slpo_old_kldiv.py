"""Defines the SLPO loss function."""

from functools import partial
from typing import Tuple, TypeAlias

import torch


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


# _get_batch_logps and concatenated_forward have been moved to slpo_adapter.py.


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


def calc_targets(alpha, w_ref_logps, l_ref_logps):
  """Calculate logprob of w_w, w_l, w_bar_w, w_bar_l.

  Args:
    alpha: What percentage of the probability mass of the rejected sequence
      to assign to the chosen sequence. [0.0, 1.0]
    w_ref_logps: Log probabilities of the chosen sequences under the reference.
      Shape: (batch_size,)
    l_ref_logps: Log probabilities of the rejected sequences under the reference.
      Shape: (batch_size,)

  Returns:
    target_w_logps: Target log probabilities for the chosen sequences. Shape: (batch_size,)
    target_l_logps: Target log probabilities for the rejected sequences. Shape: (batch_size,)
    target_wbar_logps: Target complement log probabilities for the chosen sequences. Shape: (batch_size,)
    target_lbar_logps: Target complement log probabilities for the rejected sequences. Shape: (batch_size,)
  """
  # Local notation: all values are logprobs. w and l refer to all of the
  # following: ref, target, and intermediate values.

  device = w_ref_logps.device

  w = w_ref_logps
  l = l_ref_logps

  # Setup alpha in log space.
  a = torch.tensor(alpha, device=device, dtype=torch.float64).log()
  a_comp = torch.tensor(1.0 - alpha, device=device, dtype=torch.float64).log()

  # Shift probability mass of probs.
  w = torch.logaddexp(w, l + a)
  l = l + a_comp

  # Clamp to max 0 to avoid numerical issues if we exceed prob 1 (which causes NaNs in log_comp)
  w = torch.clamp(w, max=0.0)
  l = torch.clamp(l, max=0.0)

  wbar = log_comp(w)  # high chance of underflow to zero. Probably ok.
  lbar = log_comp(l)  # high chance of underflow to zero. Probably ok.

  # Block grads.
  w = w.detach()
  l = l.detach()
  wbar = wbar.detach()
  lbar = lbar.detach()

  return w, l, wbar, lbar


# def _safe_kl_div(
#   input: torch.Tensor,
#   target: torch.Tensor,
# ) -> torch.Tensor:
#   """Compute MSE on log-probs as an approximation to KL.

#   Args:
#     input: Input log probabilities. Shape: (batch_size, 2)
#     target: Target log probabilities. Shape: (batch_size, 2)

#   Returns:
#     Pointwise kl div. Shape: (batch_size, 2)
#   """
#   target_is_inf = torch.isinf(target)
#   safe_target = target.clone()
#   safe_target[target_is_inf] = input[target_is_inf].detach()

#   return torch.nn.functional.mse_loss(input, safe_target, reduction="none")


def kl_div(
  input: torch.Tensor,
  target: torch.Tensor,
) -> torch.Tensor:
  """Pointwise KL divergence, treating zero-probability target classes as 0 loss.

  Zero-probability target classes (target = -inf) contribute 0 to the KL sum
  by definition. torch.kl_div produces NaN for those entries (0 * -inf), so
  we mask them out before and after.
  """
  mask = torch.isinf(target)
  safe_target = torch.where(mask, torch.zeros_like(target), target)
  raw = torch.nn.functional.kl_div(
    input, safe_target, log_target=True, reduction="none"
  )
  return torch.where(mask, torch.zeros_like(raw), raw)


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
) -> Tuple[
  torch_tensor,
  torch_tensor,
  torch_tensor,
  torch_tensor,
  torch_tensor,
  torch_tensor,
  torch_tensor,
  torch_tensor,
  torch_tensor,
  torch_tensor,
  torch_tensor,
  torch_tensor,
  torch_tensor,
]:
  """Compute the SLPO loss for a batch of sequences.

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

  Returns:
    loss: Mean KL divergence loss across w and l components. Shape: (batch_size,)
    loss_w: KL divergence loss for chosen (w) sequences. Shape: (batch_size,)
    loss_l: KL divergence loss for rejected (l) sequences. Shape: (batch_size,)
    chosen_rewards: Reward values for the chosen sequences. Shape: (batch_size,)
    rejected_rewards: Reward values for the rejected sequences. Shape: (batch_size,)
    target_w: Target log probabilities for chosen sequences. Shape: (batch_size,)
    target_l: Target log probabilities for rejected sequences. Shape: (batch_size,)
    target_wbar: Target complement log probabilities for chosen sequences. Shape: (batch_size,)
    target_lbar: Target complement log probabilities for rejected sequences. Shape: (batch_size,)
    w: Model log probabilities for chosen sequences. Shape: (batch_size,)
    l: Model log probabilities for rejected sequences. Shape: (batch_size,)
    wbar: Model complement log probabilities for chosen sequences. Shape: (batch_size,)
    lbar: Model complement log probabilities for rejected sequences. Shape: (batch_size,)
  """
  # Cast to fp64 and rename. We always stay in logprob space.
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

  chosen_rewards = (w - ref_w).detach()
  rejected_rewards = (l - ref_l).detach()

  # Calculate targets in logprob space.
  target_w, target_l, target_wbar, target_lbar = calc_targets(
    alpha, ref_w, ref_l
  )

  stk1 = partial(torch.stack, dim=1)

  check_numerics(w, "w")
  check_numerics(l, "l")
  check_numerics(wbar, "wbar")
  check_numerics(lbar, "lbar")
  # target_* values can legitimately be -inf (log of zero probability) when
  # alpha = 1; skip check_numerics for them.

  loss_w = kl_div(stk1([w, wbar]), stk1([target_w, target_wbar])).mean(-1)
  loss_l = kl_div(stk1([l, lbar]), stk1([target_l, target_lbar])).mean(-1)

  check_numerics(loss_w, "loss_w")
  check_numerics(loss_l, "loss_l")

  loss = (loss_w + loss_l) / 2.0

  check_numerics(loss, "loss")

  return (
    loss,
    loss_w,
    loss_l,
    chosen_rewards,
    rejected_rewards,
    target_w,
    target_l,
    target_wbar,
    target_lbar,
    w,
    l,
    wbar,
    lbar,
  )
