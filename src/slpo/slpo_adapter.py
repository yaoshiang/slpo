"""DPO-trainer-specific adapter: converts batches and logits to SLPO inputs.

This module bridges the DPO training framework's batch format to the
framework-agnostic loss functions in slpo.py.
"""

from typing import Callable, List, Mapping, Tuple, Union

import torch

from slpo.slpo import check_logps_are_prob_dist, check_t_dim_ne_zero


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

  seq_lens = loss_mask.sum(-1)  # Shape: (batch_size,), count of non-masked tokens

  if average_log_prob:
    logp_y = (per_token_logps * loss_mask).sum(-1) / seq_lens
  else:
    logp_y = (per_token_logps * loss_mask).sum(-1)

  # This is new code using the tree-walking trick to calculate the complement

  # Setup.
  logps = logits.log_softmax(-1)  # Shape (B, S, V)
  check_logps_are_prob_dist(logps)
  check_t_dim_ne_zero(logps)

  B, S, V = logps.shape  # B = batch size, S = seq len, V = vocab size
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
  # per_token_logps comes from the calculation of y above. Multiplying
  # by loss_mask zeroes out the masked tokens (e.g. the masked tokens are
  # treated as if predicted with 100% certainty).
  per_token_logps_masked = per_token_logps * loss_mask
  per_token_logps_masked_shifted = torch.cat(
    [zeros, per_token_logps_masked[..., :-1]], dim=-1
  )
  prefix_logps = per_token_logps_masked_shifted.cumsum(dim=-1)

  # 2b) Compute sum of the ybar_t in log space.
  #     We could gather all values except y... but it's probably
  #     more efficient to "mask" each chosen token with -inf to make it
  #     not part of the logsumexp op.
  logps_clone = logps.clone()
  logps_clone.scatter_(2, labels.unsqueeze(2), float("-inf"))
  postfix_logps = torch.logsumexp(logps_clone, dim=-1)  # Shape (..., S)

  # 2c) If the final token is masked, then this sequence is not part of ybar:
  #     a bunch of y_t followed by a masked token is in the set y, not ybar.
  #     Make it disappear by setting the final logp to -inf - that will poison
  #     prefix and postfix to -inf, and when we logsumexp over all possible
  #     sequences, this sequence will not contribute. functional tests
  #     verify torch.logsumexp treats exp(-inf) = 0.
  postfix_logps = torch.where(
    loss_mask,
    postfix_logps,
    torch.tensor(float("-inf"), device=device, dtype=dtype),
  )

  # 2c) Sum the two parts: the starting y tokens, and the final y_bar token.
  per_sequence_logp_ybar = prefix_logps + postfix_logps  # Shape (..., S)

  # 2d) Sum over all sequences.
  logp_ybar = torch.logsumexp(per_sequence_logp_ybar, dim=-1)  # Shape (B,)

  return logp_y, logp_ybar, seq_lens


def concatenated_forward(
  model: torch.nn.Module,
  batch: Mapping[str, Union[List, torch.LongTensor, torch.Tensor]],
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
    A tuple of six tensors:
      - chosen_logps: Log probabilities of the chosen sequences.
      - rejected_logps: Log probabilities of the rejected sequences.
      - chosen_logps_comp: Complement log probabilities of the chosen sequences.
      - rejected_logps_comp: Complement log probabilities of the rejected sequences.
      - chosen_seq_lens: Number of non-masked tokens in each chosen sequence. Shape: (batch_size,)
      - rejected_seq_lens: Number of non-masked tokens in each rejected sequence. Shape: (batch_size,)

  """
  concatenated_batch = concat_func(batch)
  all_logits = model(
    concatenated_batch["concatenated_input_ids"],
    attention_mask=concatenated_batch["concatenated_attention_mask"],
  ).logits.to(torch.float32)
  all_logps, all_logp_complements, all_seq_lens = _get_batch_logps(
    all_logits,
    concatenated_batch["concatenated_labels"],
    average_log_prob=False,
  )
  chosen_logps = all_logps[: batch["chosen_input_ids"].shape[0]]
  rejected_logps = all_logps[batch["chosen_input_ids"].shape[0] :]
  chosen_logps_comp = all_logp_complements[: batch["chosen_input_ids"].shape[0]]
  rejected_logps_comp = all_logp_complements[
    batch["chosen_input_ids"].shape[0] :
  ]
  chosen_seq_lens = all_seq_lens[: batch["chosen_input_ids"].shape[0]]
  rejected_seq_lens = all_seq_lens[batch["chosen_input_ids"].shape[0] :]

  return chosen_logps, rejected_logps, chosen_logps_comp, rejected_logps_comp, chosen_seq_lens, rejected_seq_lens
