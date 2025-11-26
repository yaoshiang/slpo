"""Defines the SLPO loss function."""

from typing import Callable, Dict, List, Tuple, Union

import torch


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
    logps_t, torch.zeros_like(logps_t), msg=f"{logps_t=}\n{logps=}"
  )


def _get_batch_logps(
  logits: torch.FloatTensor | torch.Tensor,
  labels: torch.LongTensor | torch.Tensor,
  average_log_prob: bool = False,
) -> (
  Tuple[torch.FloatTensor, torch.FloatTensor]
  | Tuple[torch.Tensor, torch.Tensor]
):
  """Compute the log probabilities of the given labels under the given logits.

  logits will be cast to float64 for numerical stability and the
  return values will also be float64.

  Args:
      logits: Logits of the model (unnormalized).
        Shape: (batch_size, sequence_length, vocab_size)
      labels: Labels for which to compute the log probabilities.
        Label tokens with a value of -100 are ignored.
        Shape: (batch_size, sequence_length)
      average_log_prob: If True, return the average log probability per
      (non-masked) token. Otherwise, return the sum of the log probabilities
      of the (non-masked) tokens.

  Returns:
      A pair of tensors of shape (batch_size,) containing the average/sum
      log probabilities of the given labels under the given logits, as
      well as the complement log probabilities.
  """
  # This section is from DPO repo.
  assert logits.shape[:-1] == labels.shape

  dtype = torch.float64

  logits = logits.to(dtype)
  labels = labels[:, 1:].clone()
  logits = logits[:, :-1, :]
  loss_mask = labels != -100

  # dummy token; we'll ignore the losses on these tokens later
  labels[labels == -100] = 0

  per_token_logps = torch.gather(
    logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
  ).squeeze(2)

  if average_log_prob:
    logp_y = (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
  else:
    logp_y = (per_token_logps * loss_mask).sum(-1)

  # This is new code using the tree-walking trick to calculcate the complement
  per_token_logps_masked = per_token_logps * loss_mask
  logps = logits.log_softmax(-1)  # Shape (B, T, V)
  check_logps_are_prob_dist(logps)
  check_t_dim_ne_zero(logps)

  B, T, V = logps.shape  # B = batch size, T = seq len, V = vocab size
  device = logps.device

  # Compute logp of the complements.
  #      1.0 x prob(ybar_t1)
  #    x 1.0 x prob(y_t1, ybar_t2, ...)
  #    x 1.0 x prob(y_t1, y_t2, ybar_t3, ...)
  #    x 1.0 x prob(y_t1, y_t2, y_t3, ybar_t4, ...)
  #    x ...
  #
  #     For all time steps T, calculate prob that all preceding tokens are in y.
  #     That is, prob(y_t1, ..., y_t(T-1)).
  #     We will bolt on ybar_T later.
  #     For the special case of t=0, we have logp_prefix = 0:
  #     there are no preceding tokens, so the probability is 1.0 (log(1)=0).
  #     We throw away the final time step since we only need up to t=T-1.
  zeros = torch.zeros(B, 1, device=device, dtype=dtype)
  per_token_logps_masked_shifted = torch.cat(
    [zeros, per_token_logps_masked[..., :-1]], dim=-1
  )
  prefix_logps = per_token_logps_masked_shifted.cumsum(dim=-1)

  # 2b) Compute sum of the ybar_t in log space.
  #     First, sum of all possible vocabs (always 100%).
  #     Second, minus the y_t to get ybar_t.
  per_step_logp = torch.logsumexp(logps, dim=-1)  # Shape (..., T)
  per_token_not_logps = _logdiffexp(
    per_step_logp, per_token_logps_masked
  )  # Shape (..., T)

  # 2c) Multiply the prob of the prefix (chosens) and final rejected token,
  #     for all sequence locations.
  per_sequence_logp_ybar = prefix_logps + per_token_not_logps  # Shape (..., T,)

  # 2d) Sum over all sequences.
  logp_ybar = torch.logsumexp(per_sequence_logp_ybar, dim=-1)  # Shape (B,)

  return logp_y, logp_ybar


def concatenated_forward(
  model: torch.nn.Module,
  batch: Dict[str, Union[List, torch.LongTensor]],
  concat_func: Callable,
) -> (
  Tuple[
    torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor
  ]
  | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
  """Based on DPO / trainers.py :: BasicTrainer :: concatenated_forward.

  Args:
    model: The model to compute log probabilities from.
    batch: A batch dictionary containing chosen and rejected sequences.
    concat_func: A function that concatenates the chosen and rejected
      sequences into a single batch for processing by the model.

  Returns:
    A tuple of four tensors:
      - chosen_logps: Log probabilities of the chosen sequences.
      - rejected_logps: Log probabilities of the rejected sequences.
      - chosen_logps_comp: Complement log probabilities of the chosen sequences.
      - rejected_logps_comp: Complement log probabilities of the rejected sequences.

  """
  concatenated_batch = concat_func(batch)
  all_logits = model(
    concatenated_batch["concatenated_input_ids"],
    attention_mask=concatenated_batch["concatenated_attention_mask"],
  ).logits.to(torch.float32)
  all_logps, all_logp_complements = _get_batch_logps(
    all_logits,
    concatenated_batch["concatenated_labels"],
    average_log_prob=False,
  )
  chosen_logps = all_logps[0][: batch["chosen_input_ids"].shape[0]]
  rejected_logps = all_logps[0][batch["chosen_input_ids"].shape[0] :]
  chosen_logps_comp = all_logps[1][: batch["chosen_input_ids"].shape[0]]
  rejected_logps_comp = all_logps[1][batch["chosen_input_ids"].shape[0] :]

  return chosen_logps, rejected_logps, chosen_logps_comp, rejected_logps_comp


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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    Unreduced loss. Shape: (batch_size,)
    chosen_rewards: Reward values for the chosen sequences. Shape: (batch_size,)
    rejected_rewards: Reward values for the rejected sequences. Shape: (batch_size,)
  """
  slpo_loss_check_batch_size(
    model_chosen_logps,
    model_rejected_logps,
    model_chosen_logps_comp,
    model_rejected_logps_comp,
    reference_chosen_logps,
    reference_rejected_logps,
    alpha,
  )

  device = model_chosen_logps.device

  # Cast to fp64
  model_chosen_logps = model_chosen_logps.to(torch.float64)
  model_rejected_logps = model_rejected_logps.to(torch.float64)
  model_chosen_logps_comp = model_chosen_logps_comp.to(torch.float64)
  model_rejected_logps_comp = model_rejected_logps_comp.to(torch.float64)
  reference_chosen_logps = reference_chosen_logps.to(torch.float64)
  reference_rejected_logps = reference_rejected_logps.to(torch.float64)

  # Calculcate w_w, w_l, w_bar_w, w_bar_l in a numerically stable way
  log_alpha = torch.tensor(alpha, device=device, dtype=torch.float64).log()
  log_1m_alpha = torch.tensor(
    1.0 - alpha, device=device, dtype=torch.float64
  ).log()

  w_w = _logsumexp(reference_chosen_logps, log_alpha + reference_rejected_logps)
  w_l = log_1m_alpha + reference_rejected_logps
  w_w_bar = _logdiffexp(torch.zeros_like(w_w), w_w)
  w_l_bar = _logdiffexp(torch.zeros_like(w_l), w_l)

  w_w = w_w.detach()
  w_l = w_l.detach()
  w_w_bar = w_w_bar.detach()
  w_l_bar = w_l_bar.detach()

  input = torch.stack(
    [
      model_chosen_logps,
      model_rejected_logps,
      model_chosen_logps_comp,
      model_rejected_logps_comp,
    ],
    dim=1,
  )  # Shape (B, 4)
  target = torch.stack(
    [
      w_w,
      w_l,
      w_w_bar,
      w_l_bar,
    ],
    dim=1,
  )  # Shape (B, 4)

  # CE(P,Q) = D_KL(P || Q) + CE(P)
  # input is P, target is Q in KL-divergence D_KL(P || Q)
  kl = torch.nn.functional.kl_div(
    input=input, target=target, log_target=True, reduction="none"
  ).sum(-1)
  # Calculate sum(w * log(w)) safely.
  # kl_div(input=0, target=w) = w * (log(w) - 0) = w * log(w)
  entropy = torch.nn.functional.kl_div(
    input=torch.zeros_like(target),
    target=target,
    log_target=True,
    reduction="none",
  ).sum(-1)
  loss = kl - entropy

  chosen_rewards = model_chosen_logps - model_chosen_logps_comp
  rejected_rewards = model_rejected_logps - model_rejected_logps_comp
  return loss, chosen_rewards, rejected_rewards
