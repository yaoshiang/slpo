"""Defines the SLPO loss function.

_compute_logprob_y_bar_y: implements the tree walking trick to
    efficiently compute log_bar_y.

slpo_loss: computes the SLPO loss for a single sequence.
    The implementation is fairly simple since the tree walking trick
    is implemented in _compute_logprob_y_bar_y.

SLPO: a PyTorch loss module that computes the SLPO loss for a batch of
    sequences.
"""

from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss


def _compute_logprob_y_bar_y(
    logprobs: torch.Tensor, y_tokens: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given:
      logprobs: shape (T, V) = log softmax outputs for a single sequence.
      y_tokens: shape (T,) = sequence of token-IDs.

    Returns:
      (log_p_y, log_p_not_y):

      Where:
        log_p_y = log probability of exactly matching y_tokens.
        log_p_not_y = log probability of any sequence that differs
                      from y_tokens in at least one position.
    """
    T, _ = logprobs.shape  # T = sequence length, _ = vocab size

    # Logprobs should sum to log(1.0) = 0.0 for each time step t.
    logprob_t = torch.logsumexp(logprobs, dim=-1)
    assert torch.allclose(logprob_t, torch.zeros_like(logprob_t), atol=1e-3)

    # Edge case: Handle empty sequences
    if T == 0:
        raise ValueError("Empty sequences are not supported.")

    # 1) Extract log probability of the chosen tokens
    logprob_y_t = logprobs[torch.arange(T), y_tokens]
    logprob_y = logprob_y_t.sum()  # Log of joint prob of y seq over steps t.

    # 2) Compute sigma sums for "first mismatch" trick
    logprob_sigma_y_lt_t_shifted = torch.cat(
        [
            torch.tensor(
                [0.0], device=logprobs.device, dtype=logprobs.dtype
            ),  # Initial zero for sigma sum
            torch.cumsum(logprob_y_t, dim=0),  # Shape (T)
        ]
    )  # Shape (T+1)
    logprob_sigma_y_lt_t = logprob_sigma_y_lt_t_shifted[:-1]  # Shape (T,)
    del logprob_sigma_y_lt_t_shifted  # Free memory

    assert logprob_sigma_y_lt_t.size() == (T,)

    # 3) Compute log probability of any incorrect token at each step
    # Better to not use torch.log1p(-torch.exp(logprob_y_t)).
    # Uncertain if the gradients will be correct with that shortcut
    # given log_softmax layer. Instead, clone and mask y_t to -inf.
    # logsumexp will handle -inf values correctly.
    logprob_masked_y_t = logprobs.clone()
    logprob_masked_y_t[torch.arange(T), y_tokens] = -float("inf")
    logprob_bar_y_t = torch.logsumexp(logprob_masked_y_t, dim=-1)

    # 5) Compute the mismatch log probability
    logprob_not_y = torch.logsumexp(
        logprob_sigma_y_lt_t + logprob_bar_y_t, dim=0
    )

    return logprob_y, logprob_not_y


def slpo_loss(input: torch.Tensor, target: dict) -> torch.Tensor:
    r"""Calculates the SLPO loss for a single sequence:
        w_w * log p_theta(y_w)
        + (1 - w_w) * log p_theta(\overline{y_w})
        + 0 * log p_theta(y_l)
        + 1 * log p_theta(\overline{y_l})

    Args:
        input: Float tensor of shape (S, V).
               Raw logits from the LM for each time-step s and vocab token v.
        target: A dict with entries:
            - 'pi_ref_w': scalar tensor, float. Ref prob for winning/chosen seq.
            - 'pi_ref_l': scalar tensor, float. Ref prob for losing/rejected seq.
            - 'y': shape (S), tensor, int. The winning/losing token IDs.
            - 'winner': scalar, tensor, bool. True means winner/chosen, False means
                loser/rejected.

    Returns:
        A scalar Tensor.
    """
    # Checks
    assert input.dim() == 2, f"Expected 2D input, got {input.dim()}"
    assert input.size(0) == target["y"].size(0), (
        f"Expected input and target to have the same sequence length, got "
        f"{input.size(0)} and {target['y'].size(0)}"
    )

    # Convert to log-probs
    logprobs = F.log_softmax(input, dim=-1)  # shape (S, V)

    if target["winner"]:
        w_w = target["pi_ref_w"] + target["pi_ref_l"]

        # Get p_theta(y_w) and p_theta(\overline{y_w})
        log_p_y, log_p_bar_y = _compute_logprob_y_bar_y(logprobs, target["y"])
        loss = -(w_w * log_p_y + (1.0 - w_w) * log_p_bar_y)
    else:
        # Get p_theta(y_l) and p_theta(\overline{y_l})
        log_p_y, log_p_bar_y = _compute_logprob_y_bar_y(logprobs, target["y"])
        loss = -(0.0 * log_p_y + 1.0 * log_p_bar_y)

    return loss


class SLPO(_Loss):
    """SLPO loss function.

    # TODO: Vectorize. This implementation loops over rows and packed sequences.
    # TODO: Support packed sequences. For now, each sequence is padded.
    """

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

        batch_losses = []
        for row_idx, x in enumerate(input):
            target_slice = {
                key: value[row_idx] for key, value in target.items()
            }
            batch_losses.append(slpo_loss(x, target_slice))
        batch_losses = torch.stack(batch_losses)

        if self.reduction == "mean":
            return batch_losses.mean()
        elif self.reduction == "sum":
            return batch_losses.sum()
        elif self.reduction == "none":
            return batch_losses  # Return per-row losses
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")
