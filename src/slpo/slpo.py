"""Defines the SLPO loss function.

_compute_logprob_y_bar_y: implements the tree walking trick to
    efficiently compute log_bar_y.

slpo_loss: computes the SLPO loss for a single sequence.
    The implementation is fairly simple since the tree walking trick
    is implemented in _compute_logprob_y_bar_y.

SLPO: a PyTorch loss module that computes the SLPO loss for a batch of
    sequences.
"""

import warnings
from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss


def _logdiffexp(t1, t2):
  """Helper for log diff exp on two tensors.

  This function computes log(exp(t1) - exp(t2)) in a stable way.
  """
  assert t1.size() == t2.size(), f"{t1.size()} != {t2.size()}"
  if torch.any(t1 < t2):
    raise ValueError("t1 must be greater than t2.")

  retval = t1 + torch.log1p(-torch.exp(t2 - t1))

  # Look for -inf in t1 and t2 and manually patch the result to -inf.
  retval[(t1 == float("-inf")) & (t2 == float("-inf"))] = float("-inf")

  return retval


def _logsumexp(t1, t2):
  """Helper for log sum exp on two tensors.

  This function computes log(exp(t1) + exp(t2)) in a stable way
  using the log-sum-exp trick.
  """
  assert t1.size() == t2.size(), f"{t1.size()} != {t2.size()}"

  retval = torch.logsumexp(torch.stack([t1, t2]), dim=0)

  return retval


def y_ybar_single(
  logprobs: torch.Tensor, y_tokens: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Compute log probability of y and not y (aka y_bar) sequences.

  This uses numerical stability and tree-walking tricks to
  compute y_bar tractably.

  For ease of reasoning, this is a non-batched implementation. A batched
  implementation follows and can be tested against this function.

  Args:
    logprobs: shape (T, V) = log softmax outputs for a single sequence,
      where T is timestep (eg sequence length, not tokens), and V is vocab size.
      Given numerical stability issues, this must be a tf.float64.
    y_tokens: shape (T,) = sequence of token-IDs.

  Returns:
    logprob_y = log probability of exactly matching y_tokens.
    logprob_not_y = log probability of any sequence that differs
      from y_tokens in at least one position.
  """
  T, _ = logprobs.shape  # T = sequence length, _ = vocab size
  device = logprobs.device
  dtype = logprobs.dtype

  # Logprobs should sum to 1.0 (logsumexp to 0.0) for each time step t.
  logprob_t = torch.logsumexp(logprobs, dim=-1)  # Shape (T,)

  if not torch.allclose(logprob_t, torch.zeros_like(logprob_t), atol=1e-3):
    raise ValueError(f"Expected logprobs to sum to 1.0: {logprob_t=}.")
  del logprob_t

  # Edge case: Handle empty sequences
  if T == 0:
    raise ValueError("Empty sequences are not supported.")

  # 1) Extract log probability of the chosen tokens
  logprob_y_t = logprobs[torch.arange(T), y_tokens]
  logprob_y = logprob_y_t.sum()

  # 2) Compute logp of the bar-y sequences. Basically, we want to calculate:
  #    prob(y_bar_t1)
  #    x prob(y_t1, y_bar_t2, ...)
  #    x prob(y_t1, y_t2, y_bar_t3, ...)
  #    x prob(y_t1, y_t2, y_t3, y_bar_t4, ...)
  #    x ...
  #
  # 2a) First, calculate the odds of the first tokens starting with y. The base
  #     case starts with "100%" (log(1)=0) to make the math work. The indexes
  #     are shifted, since for a sequence of length n, we would have n-1 y tokens and 1 bar_y token.
  logprob_y_until_t = torch.cat(
    [
      torch.tensor([0.0], device=device, dtype=dtype),
      torch.cumsum(logprob_y_t[:-1], dim=-1),
    ]
  )  # Shape (T,)

  # 2b) Compute sum of the y_bar_t in log space.
  #     Calculate the sum of all possible vocabs (100%), minus the y_t.
  logprob_t = torch.logsumexp(logprobs, dim=-1)  # Shape (T,)
  logprob_y_bar_t = _logdiffexp(logprob_t, logprob_y_t)  # Shape (T,)

  # 2c) Fuse the first half (y_t1..n-1) and second half (y_bar_tn) for all sequence locations.
  #     This is a joint prob, so we multiply the probs (sum in log space).
  logprob_not_y_through_t = logprob_y_until_t + logprob_y_bar_t  # Shape (T,)

  # 2d) Sum over all possible locations where the first y_bar token is at time step t.
  logprob_not_y = torch.logsumexp(logprob_not_y_through_t, dim=-1)  # Scalar

  assert (
    torch.logsumexp(torch.stack([logprob_y, logprob_not_y]), -1) > -0.000001
  ), (
    f"Expected logsumexp(logprob_y, logprob_not_y) to be almost 0.0, got \n"
    f"{torch.logsumexp(torch.stack([logprob_y, logprob_not_y]), -1)}, \n"
    f"{logprob_y},  \n{logprob_not_y}"
  )

  assert logprob_y.dtype == logprob_not_y.dtype == dtype

  return logprob_y, logprob_not_y


def y_ybar(
  logprobs: torch.Tensor, y_tokens: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Batched version of y_y_bar_single.

  This uses numerical stability and tree-walking tricks to
  compute not-y tractably for batches of sequences.

  Args:
    logprobs: shape (B, T, V) = log softmax outputs for batch of sequences,
      where B is batch size, T is timestep (sequence length), and V is vocab size.
      Given numerical stability issues, this should be torch.float64.
    y_tokens: shape (B, T) = sequences of token-IDs for the batch.

  Returns:
    logprob_y = log probability of exactly matching y_tokens for each sequence. Shape (B,)
    logprob_not_y = log probability of any sequence that differs
      from y_tokens in at least one position for each sequence. Shape (B,)
  """
  B, T, V = (
    logprobs.shape
  )  # B = batch size, T = sequence length, V = vocab size
  device = logprobs.device
  dtype = logprobs.dtype

  # Logprobs should sum to 1.0 (logsumexp to 0.0) for each time step t.
  logprob_t = torch.logsumexp(logprobs, dim=-1)  # Shape (B, T)

  if not torch.allclose(logprob_t, torch.zeros_like(logprob_t), atol=1e-3):
    raise ValueError(f"Expected logprobs to sum to 1.0: {logprob_t=}.")
  del logprob_t

  # Edge case: Handle empty sequences
  if T == 0:
    raise ValueError("Empty sequences are not supported.")

  # 1) Extract log probability of the chosen tokens
  batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(B, T)
  time_indices = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
  logprob_y_t = logprobs[batch_indices, time_indices, y_tokens]  # Shape (B, T)
  logprob_y = logprob_y_t.sum(dim=-1)  # Shape (B,)

  # 2) Compute logp of the bar-y sequences. Basically, we want to calculate:
  #    prob(y_bar_t1)
  #    x prob(y_t1, y_bar_t2, ...)
  #    x prob(y_t1, y_t2, y_bar_t3, ...)
  #    x prob(y_t1, y_t2, y_t3, y_bar_t4, ...)
  #    x ...
  #
  # 2a) First, calculate the odds of the first tokens starting with y. The base
  #     case starts with "100%" (log(1)=0) to make the math work. The indexes
  #     are shifted, since for a sequence of length n, we would have n-1 y tokens and 1 bar_y token.
  zeros = torch.zeros(B, 1, device=device, dtype=dtype)  # (B,1)
  logprob_y_until_t = torch.cat(
    [
      zeros,
      torch.cumsum(logprob_y_t[:, :-1], dim=-1),  # (B, T-1)
    ],
    dim=-1,
  )  # Shape (B, T)

  # 2b) Compute sum of the y_bar_t in log space.
  #     Calculate the sum of all possible vocabs (100%), minus the y_t.
  logprob_t = torch.logsumexp(logprobs, dim=-1)  # Shape (B, T)
  logprob_y_bar_t = _logdiffexp(logprob_t, logprob_y_t)  # Shape (B, T)

  # 2c) Fuse the first half (y_t1..n-1) and second half (y_bar_tn) for all sequence locations.
  #     This is a joint prob, so we multiply the probs (sum in log space).
  logprob_not_y_through_t = logprob_y_until_t + logprob_y_bar_t  # Shape (B, T)

  # 2d) Sum over all possible locations where the first y_bar token is at time step t.
  logprob_not_y = torch.logsumexp(logprob_not_y_through_t, dim=-1)  # Shape (B,)

  # Verify probability conservation for each sequence in the batch
  total_logprobs = torch.logsumexp(
    torch.stack([logprob_y, logprob_not_y]), dim=0
  )
  assert torch.all(total_logprobs > -0.000001), (
    f"Expected logsumexp(logprob_y, logprob_not_y) to be almost 0.0 for all sequences, got \n"
    f"{total_logprobs}, \n"
    f"logprob_y: {logprob_y}, \n"
    f"logprob_not_y: {logprob_not_y}"
  )

  assert logprob_y.dtype == logprob_not_y.dtype == dtype

  return logprob_y, logprob_not_y


def _slpo_loss_checks_helper(input, target):
  """Helper to validate inputs to slpo_loss"""
  N = target["y"].size(0)

  if input.dim() != 2:
    raise ValueError(f"Expected 2D input, got {input.dim()}")

  if not N <= input.size(0):
    raise ValueError(
      f"Expected target length <= input length, got "
      f"{input.size(0)} and {target['y'].size(0)}"
    )

  if not torch.all(target["logprob_ref_w"] < 0.0):
    raise ValueError("Expected logprob_ref_w to be negative.")

  if not torch.all(target["logprob_ref_l"] < 0.0):
    raise ValueError("Expected logprob_ref_l to be negative.")

  if torch.exp(target["logprob_ref_w"]) > 0.1:
    warnings.warn(
      f"Expected exp(logprob_ref_w) to be very close to zero, got {torch.exp(target['logprob_ref_w'])}",
      RuntimeWarning,
    )

  if torch.exp(target["logprob_ref_l"]) > 0.1:
    warnings.warn(
      f"Expected exp(log_prob_ref_l) to be very close to zero, got {torch.exp(target['logprob_ref_l'])}",
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
          - 'logprob_ref_w': scalar tensor, float. Ref logprob for winning/chosen seq.
          - 'logprob_ref_l': scalar tensor, float. Ref logprob for losing/rejected seq.
          - 'y': shape (S), tensor, int. The winning/losing token IDs.
          - 'winner': scalar, tensor, bool. True means winner/chosen, False means
              loser/rejected.

  Returns:
      A scalar Tensor.

  Warnings:
      If the logprob_ref_w or logprob_ref_l are not less than -5 (i.e. 0.0067
      probability), a warning is raised. This is because the expected
      logprob_ref_w and logprob_ref_l are very close to zero.
  """
  _slpo_loss_checks_helper(input, target)

  N = target["y"].size(0)

  # Convert to log-probs
  input = input.to(torch.float64)
  logprobs = F.log_softmax(input, dim=-1)  # shape (S, V)
  assert logprobs.requires_grad, "Input should require grad."

  # Get logprob_theta(y) and logprob_theta(\overline{y})
  logprob_y, logprob_bar_y = y_ybar_single(logprobs, target["y"])
  logprob_p = torch.log_softmax(torch.stack([logprob_y, logprob_bar_y]) / N, -1)

  if target["winner"].item():
    # loss = -(w_w * logprob_y + (1 - w_w) * logprob_bar_y), rooted.
    logprob_w_w = _logsumexp(
      target["logprob_ref_w"].double(), target["logprob_ref_l"].double()
    )
    logprob_1_minus_w_w = torch.log(-torch.expm1(logprob_w_w))
    pre_rooted_logprob_q = torch.stack([logprob_w_w, logprob_1_minus_w_w])
    logprob_q = torch.log_softmax(pre_rooted_logprob_q / N, -1)

    loss = F.kl_div(
      logprob_p,
      logprob_q,
      log_target=True,
      reduction="batchmean",
    )

    assert logprob_1_minus_w_w != 0.0, "logprob_1m_w_w should not be zero."
    assert (
      torch.logsumexp(torch.stack([logprob_w_w, logprob_1_minus_w_w]), -1)
      > -0.000001
    ), f"{logprob_w_w}, {logprob_1_minus_w_w}"

  else:
    # loss = -(0.0 * logprob_y + 1.0 * logprob_bar_y), rooted.
    logprob_q = torch.tensor([-torch.inf, 0.0]).double()  # logprobs.
    loss = F.kl_div(
      logprob_p, logprob_q, log_target=True, reduction="batchmean"
    )

  return loss


class SLPO(_Loss):
  """SLPO loss function."""

  __constants__ = ["reduction"]

  def __init__(self, reduction: str = "mean"):
    """
    Args:
        reduction (str): Reduction method ('mean', 'sum', 'none').
    """
    super().__init__(reduction=reduction)

  def forward(
    self, input: Tensor, target: Dict[str, List[List[Tensor]]]
  ) -> Tensor:
    """
    Computes the SLPO loss for a batch.

    Args:
        input (Tensor): Shape (N, S_max, V) - Raw logits from the model.
        target (dict): Dictionary of lists for per-row processing. The
            dict should have the same keys as the `slpo_loss` function:
            - 'pi_ref_w': Tensors, shape (N).
            - 'pi_ref_l': Tensors, shape (N).
            - 'y': List of List of tensors, shape (N, S).
            - 'winner': List of List of tensors, shape (N).

    Returns:
        - Scalar tensor if reduction="mean" or "sum".
        - Tensor of shape (N,) if reduction="none".
    """

    # batch_losses = []
    # for row_idx, x in enumerate(input):
    #   target_slice = {key: value[row_idx] for key, value in target.items()}
    #   batch_losses.append(slpo_loss(x, target_slice))
    # batch_losses = torch.stack(batch_losses)

    batch_losses = torch.tensor([0.0])

    if self.reduction == "mean":
      return batch_losses.mean()
    elif self.reduction == "sum":
      return batch_losses.sum()
    elif self.reduction == "none":
      return batch_losses  # Return per-row losses
    else:
      raise ValueError(f"Invalid reduction mode: {self.reduction}")
