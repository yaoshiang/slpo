"""Defines the SLPO loss function.

_y_ybar: implements tree walking trick to compute logp_y and logp_ybar.

slpo_loss: computes the SLPO loss for a single sequence.
    The implementation is fairly simple since the tree walking trick
    is implemented in y_ybar.

SLPO: a PyTorch loss module that computes the SLPO loss for a batch of
    sequences.
"""

import warnings
from typing import Tuple

import torch
from torch.nn import functional as F


def _logdiffexp(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
  """Helper for log diff exp on two tensors.

  This function computes log(exp(t1) - exp(t2)) in a stable way.

  If a corresponding element of t1 and t2 are both -inf, the result is set to
  -inf. This matches the linear space result of exp(-inf) - exp(-inf) = 0.

  Args:
    t1: First tensor.
    t2: Second tensor.

  Returns:
    A tensor containing log(exp(t1) - exp(t2)).
  """
  assert t1.size() == t2.size(), f"{t1.size()} != {t2.size()}"
  if torch.any(t1 < t2):
    raise ValueError("t1 must be greater than t2.")

  retval = t1 + torch.log1p(-torch.exp(t2 - t1))

  # Look for -inf in t1 and t2 and manually patch the result to -inf.
  retval[(t1 == float("-inf")) & (t2 == float("-inf"))] = float("-inf")

  return retval


def _logsumexp(t1: torch.Tensor, t2: torch.Tensor):
  """Helper for logsumexp on two tensors.

  This function computes log(exp(t1) + exp(t2)) in a stable way
  using the log-sum-exp trick.

  Args:
    t1: First tensor.
    t2: Second tensor, with the same shape as t1.

  Returns:
    A tensor containing log(exp(t1) + exp(t2)), same shape as t1 and t2.
  """
  assert t1.size() == t2.size(), f"{t1.size()} != {t2.size()}"

  retval = torch.logsumexp(torch.stack([t1, t2], dim=0), dim=0)

  return retval


def check_t_dim_ne_zero(logps: torch.Tensor) -> None:
  """Check that the T dimension of logps is not zero.

  Args:
    logps: shape (..., T, V) = log softmax outputs for batch of sequences,
      where B is batch size, T is timestep (sequence length), and V is vocab size.

  Raises:
    ValueError: if T == 0 (empty sequences not supported).
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
    logps_t, torch.zeros_like(logps_t), rtol=1e-2, atol=1e-5
  )


def y_ybar_single(
  logps: torch.Tensor, y_tokens: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Compute logp of y and not y (aka y_bar) sequences.

  This uses numerical stability and tree-walking tricks.
  compute ybar tractably and n

  For ease of reasoning, this is a non-batched implementation. A batched
  implementation follows and can be tested against this function.

  Args:
    logps: shape (T, V) = log softmax outputs for a single sequence,
      where T is timestep (eg sequence length, not tokens), and V is vocab size.
      Given numerical stability issues, this must be a tf.float64.
    y_tokens: shape (T,) = sequence of token-IDs.

  Returns:
    logp_y = log probability of exactly matching y_tokens.
    logp_ybar = log probability of any sequence that differs
      from y_tokens in at least one position.

  Raises:
    ValueError: if logps do not sum to 1.0 at any timestep.
    ValueError: if T == 0 (empty sequences not supported).
  """
  check_logps_are_prob_dist(logps)
  check_t_dim_ne_zero(logps)

  T, _ = logps.shape  # T = seq len, _ = vocab size
  device = logps.device
  dtype = logps.dtype

  # 1) Extract log probability of the chosen tokens
  logp_y_t = logps[torch.arange(T), y_tokens]  # Shape (T)
  logp_y = logp_y_t.sum()  # Scalar: shape ()

  # 2) Compute logp of the ybar sequences. Basically, we want to calculate:
  #      1.0 x prob(ybar_t1)
  #    x 1.0 x prob(y_t1, ybar_t2, ...)
  #    x 1.0 x prob(y_t1, y_t2, ybar_t3, ...)
  #    x 1.0 x prob(y_t1, y_t2, y_t3, ybar_t4, ...)
  #    x ...
  #
  # 2a) For all time steps t, calculate prob that all preceding tokens are in y.
  #     That is, prob(y_t1, ..., y_t(T-1)).
  #     We will bolt on ybar_T later.
  #     For the special case of t=0, we have logp_y_until_0 = 0:
  #     there are no preceding tokens, so the probability is 1.0 (log(1)=0).
  #     We throw away the final time step since we only need up to t=T-1.
  zeros = torch.zeros(1, device=device, dtype=dtype)
  logp_y_until_t = torch.cat([zeros, logp_y_t[:-1]], dim=-1)
  logp_y_joint_until_t = logp_y_until_t.cumsum(dim=-1)

  # 2b) Compute sum of the ybar_t in log space.
  #     Calculate the sum of all possible vocabs (100%), minus the y_t.
  logp_t = torch.logsumexp(logps, dim=-1)  # Shape (T,)
  logp_y_bar_t = _logdiffexp(logp_t, logp_y_t)  # Shape (T,)

  # 2c) Fuse the first half (y_t1..y_t(n-1)) and second half (ybar_tn).
  #     This is a joint prob, so we multiply the probs (sum the logp).
  logp_not_y_through_t = logp_y_joint_until_t + logp_y_bar_t  # Shape (T,)
  del logp_y_joint_until_t
  del logp_y_until_t

  # 2d) We have the logp for every possible sequence length. Sum them.
  logp_not_y = torch.logsumexp(logp_not_y_through_t, dim=-1)  # Scalar

  return logp_y, logp_not_y


def y_ybar(
  logps: torch.Tensor, y_tokens: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Batched version of y_y_bar_single.

  This uses numerical stability and tree-walking tricks to
  compute not-y tractably for batches of sequences.

  Args:
    logps: shape (B, T, V) = log softmax outputs for batch of sequences,
      where B is batch size, T is timestep (sequence length), and V is vocab size.
      Given numerical stability issues, this should be torch.float64.
    y_tokens: shape (B, T) = sequences of token-IDs for the batch.

  Returns:
    logp_y = log probability of exactly matching y_tokens for each sequence. Shape (B,)
    logp_not_y = log probability of any sequence that differs
      from y_tokens in at least one position for each sequence. Shape (B,)
  """
  check_logps_are_prob_dist(logps)
  check_t_dim_ne_zero(logps)

  B, T, V = logps.shape  # B = batch size, T = seq len, V = vocab size
  device = logps.device
  dtype = logps.dtype

  # 1) Extract log probability of the chosen tokens
  logp_y_t = torch.gather(logps, dim=-1, index=y_tokens.unsqueeze(-1))  # BT1
  logp_y_t = logp_y_t.squeeze(-1)  # BT
  logp_y = logp_y_t.sum(dim=-1)  # B

  # 2) Compute logp of the ybar sequences. Basically, we want to calculate:
  #      1.0 x prob(ybar_t1)
  #    x 1.0 x prob(y_t1, ybar_t2, ...)
  #    x 1.0 x prob(y_t1, y_t2, ybar_t3, ...)
  #    x 1.0 x prob(y_t1, y_t2, y_t3, ybar_t4, ...)
  #    x ...
  #
  # 2a) For all time steps t, calculate prob that all preceding tokens are in y.
  #     That is, prob(y_t1, ..., y_t(T-1)).
  #     We will bolt on ybar_T later.
  #     For the special case of t=0, we have logp_y_until_0 = 0:
  #     there are no preceding tokens, so the probability is 1.0 (log(1)=0).
  #     We throw away the final time step since we only need up to t=T-1.
  zeros = torch.zeros(B, 1, device=device, dtype=dtype)
  logp_y_until_t = torch.cat([zeros, logp_y_t[..., :-1]], dim=-1)
  logp_y_joint_until_t = logp_y_until_t.cumsum(dim=-1)

  # 2b) Compute sum of the ybar_t in log space.
  #     Calculate the sum of all possible vocabs (100%), minus the y_t.
  logp_t = torch.logsumexp(logps, dim=-1)  # Shape (..., T)
  logp_y_bar_t = _logdiffexp(logp_t, logp_y_t)  # Shape (..., T)

  # 2c) Multiply the first half p(y_t1..n-1) and second half p(y_bar_tn) for all
  #     sequence locations (addition in log space).
  logp_not_y_through_t = logp_y_joint_until_t + logp_y_bar_t  # Shape (..., T,)

  # 2d) Sum over all possible locations where the first y_bar token is at time step t.
  logp_not_y = torch.logsumexp(logp_not_y_through_t, dim=-1)  # Shape (B,)

  return logp_y, logp_not_y


def _slpo_loss_single_checks_helper(input, target):
  """Helper to validate inputs to slpo_loss"""
  N = target["y"].size(0)

  if input.dim() != 2:
    raise ValueError(f"Expected 2D input, got {input.dim()}")

  if not N <= input.size(0):
    raise ValueError(
      f"Expected target length <= input length, got "
      f"{input.size(0)} and {target['y'].size(0)}"
    )

  if not torch.all(target["logp_ref_w"] < 0.0):
    raise ValueError("Expected logp_ref_w to be negative.")

  if not torch.all(target["logp_ref_l"] < 0.0):
    raise ValueError("Expected logp_ref_l to be negative.")

  if not input.requires_grad:
    raise ValueError("Input logps must require gradients.")

  if torch.exp(target["logp_ref_w"]) > 0.1:
    warnings.warn(
      f"Expected exp(logp_ref_w) to be very close to zero, got {torch.exp(target['logp_ref_w'])}",
      RuntimeWarning,
    )

  if torch.exp(target["logp_ref_l"]) > 0.1:
    warnings.warn(
      f"Expected exp(log_prob_ref_l) to be very close to zero, got {torch.exp(target['logp_ref_l'])}",
      RuntimeWarning,
    )


def slpo_loss_single(input: torch.Tensor, target: dict) -> torch.Tensor:
  r"""Calculates the SLPO loss for a single sequence:
      w_w * log p_theta(y_w)
      + (1 - w_w) * log p_theta(\overline{y_w})
      + 0 * log p_theta(y_l)
      + 1 * log p_theta(\overline{y_l})

  Args:
      input: Float tensor of shape (S, V).
             Raw logits from the LM for each time-step s and vocab token v.
      target: A dict with entries:
          - 'logp_ref_w': scalar tensor, float. Ref logp for winning/chosen seq.
          - 'logp_ref_l': scalar tensor, float. Ref logp for losing/rejected seq.
          - 'y': shape (S), tensor, int. The winning/losing token IDs.
          - 'winner': scalar, tensor, bool. True means winner/chosen, False means
              loser/rejected.

  Returns:
      A scalar Tensor.

  Warnings:
      If the logp_ref_w or logp_ref_l are not less than -5 (i.e. 0.0067
      probability), a warning is raised. This is because the expected
      logp_ref_w and logp_ref_l are very close to zero.
  """
  _slpo_loss_single_checks_helper(input, target)

  N = target["y"].size(0)

  input = input.to(torch.float64)

  # Normalize
  logps = F.log_softmax(input, dim=-1)  # shape (S, V)

  # Get logp_theta(y) and logp_theta(\overline{y})
  logp_y, logp_bar_y = y_ybar_single(logps, target["y"])
  logp_p = torch.log_softmax(torch.stack([logp_y, logp_bar_y]) / N, -1)

  if target["winner"].item():
    # loss = -(w_w * logp_y + (1 - w_w) * logp_bar_y), rooted.
    logp_w_w = _logsumexp(
      target["logp_ref_w"].double(), target["logp_ref_l"].double()
    )
    logp_1_minus_w_w = torch.log(-torch.expm1(logp_w_w))
    pre_rooted_logp_q = torch.stack([logp_w_w, logp_1_minus_w_w])
    logp_q = torch.log_softmax(pre_rooted_logp_q / N, -1)

    loss = F.kl_div(
      logp_p,
      logp_q,
      log_target=True,
      reduction="batchmean",
    )

    assert logp_1_minus_w_w != 0.0, "logp_1m_w_w should not be zero."
    assert (
      torch.logsumexp(torch.stack([logp_w_w, logp_1_minus_w_w]), -1) > -0.000001
    ), f"{logp_w_w}, {logp_1_minus_w_w}"

  else:
    # loss = -(0.0 * logp_y + 1.0 * logp_bar_y), rooted.
    logp_q = torch.tensor([-torch.inf, 0.0]).double()  # logps.
    loss = F.kl_div(logp_p, logp_q, log_target=True, reduction="batchmean")

  return loss


def slpo_loss_checks_helper(
  policy_chosen_logps: torch.Tensor,
  policy_rejected_logps: torch.Tensor,
  reference_chosen_logps: torch.Tensor,
  reference_rejected_logps: torch.Tensor,
) -> None:
  """Helper to validate inputs to slpo_loss"""
  batch_size = policy_chosen_logps.size(0)

  if policy_chosen_logps.size() != (batch_size,):
    raise ValueError(
      f"Expected policy_chosen_logps shape {(batch_size,)}, got {policy_chosen_logps.size()}"
    )

  if policy_rejected_logps.size() != (batch_size,):
    raise ValueError(
      f"Expected policy_rejected_logps shape {(batch_size,)}, got {policy_rejected_logps.size()}"
    )

  if reference_chosen_logps.size() != (batch_size,):
    raise ValueError(
      f"Expected reference_chosen_logps shape {(batch_size,)}, got {reference_chosen_logps.size()}"
    )

  if reference_rejected_logps.size() != (batch_size,):
    raise ValueError(
      f"Expected reference_rejected_logps shape {(batch_size,)}, got {reference_rejected_logps.size()}"
    )


def slpo_loss(*args, **kwargs):
  return NotImplementedError("Placeholder for slpo_loss function.")
