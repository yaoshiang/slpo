"""Defines the SLPO loss function."""

import torch

from torch import Tensor

from torch.nn import functional as F
from torch.nn.modules.loss import _Loss


class SLPO(_Loss):
    """The SLPO loss function.
    
    This loss function is used to align LLMs.
    
     Args:
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`.
    
    Shape:
      - Input: :math:`(N, V, T)` where :math:`N` is the batch size, 
      :math:`V` is the vocabulary size, and :math:`T` is the sequence length.
      - Target: Dictionary of 'pi_ref_w' holding :math:`(N)` of the winning sequence's reference probability,
        'pi_ref_l' holding :math:`(N)` of the losing sequence's reference probability,
        'winner' holding :math`(B, T)` representing the winning sequence, and
        'loser' holding :math`(B, T)` representing the losing sequence. 

    """
    __constants__ = ["reduction"]

    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return slpo_loss(input, target)
    
import torch
import torch.nn.functional as F

def slpo_loss_hard_no_loop(input: torch.Tensor, target: dict) -> torch.Tensor:
    r"""
    Calculates the "hard" SLPO loss:
      - E_n [
          w_w * log p_theta(y_w) 
        + (1 - w_w) * log p_theta(\overline{y_w})
        + 0 * log p_theta(y_l)
        + 1 * log p_theta(\overline{y_l})
      ]

    Args:
        input: Float tensor of shape (N, T, V) or (N, V, T).
               Raw logits from the LM for each batch element n,
               each time-step t, each vocab token v.
        target: A dict with entries:
            - 'pi_ref_w': shape (N,). The reference prob for the winning sequence.
            - 'pi_ref_l': shape (N,). The reference prob for the losing sequence.
            - 'winner':   shape (N, T). The winning token IDs for each batch example.
            - 'loser':    shape (N, T). The losing token IDs for each batch example.

    Returns:
        A scalar Tensor (mean over the batch).
    """
    # If input is (N, V, T), we transpose to (N, T, V) so that dim=-1 is vocabulary.
    if input.shape[1] != target['winner'].shape[1]:
        # Likely means shape is (N, V, T). We'll transpose so we have (N, T, V).
        input = input.transpose(1, 2)
    # Now input is (N, T, V).

    # Convert to log-probs
    log_probs = F.log_softmax(input, dim=-1)  # shape (N, T, V)

    # For convenience, define the "weight" for the winning path:
    #   w_w = pi_ref_w + pi_ref_l
    # We'll broadcast to shape (N,) automatically.
    w_w = target['pi_ref_w'] + target['pi_ref_l']

    # We'll get p_theta(y_w) and p_theta(\overline{y_w}) 
    log_p_yw, log_p_not_yw = _compute_log_p_and_log_p_not_y(
        log_probs, target['winner']
    )
    # Similarly for y_l
    log_p_yl, log_p_not_yl = _compute_log_p_and_log_p_not_y(
        log_probs, target['loser']
    )

    # Hard-SLPO objective, per example n:
    #  - [ w_w * log_p_yw
    #      + (1 - w_w) * log_p_not_yw
    #      + 1 * log_p_not_yl
    #      + 0 * log_p_yl
    #    ]
    # => negative sign outside
    loss_per_example = -(
        w_w * log_p_yw
        + (1.0 - w_w) * log_p_not_yw
        + 1.0 * log_p_not_yl
    )

    return loss_per_example.mean()


def _compute_log_p_and_log_p_not_y(log_probs: torch.Tensor,
                                   y_tokens: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    Given:
      log_probs: shape (N, T, V) = log softmax outputs
      y_tokens: shape (N, T) = each row is a sequence of token-IDs
    Returns:
      (log_p_y, log_p_not_y): each is shape (N,)

      Where log_p_y = log probability of exactly matching y_tokens,
            log_p_not_y = log probability of any sequence that differs 
                          from y_tokens in at least one position.
    """
    # 1) Gather the log-prob of the chosen token at each step => shape (N,T)
    #    sum across T => log p(y).
    #    But we also need partial prefix sums for the "first mismatch" trick.
    # 
    # We'll gather along dim=-1, which is the vocab dimension:
    #   y_tokens[n,t] is in [0..V-1].
    # We must reshape y_tokens to (N,T,1) to gather from (N,T,V).
    chosen_logp = log_probs.gather(dim=-1, index=y_tokens.unsqueeze(-1)).squeeze(-1)  # (N,T)
    
    # log_p_y: sum over the T dimension (each example independently)
    log_p_y = chosen_logp.sum(dim=1)  # => shape (N,)

    # 2) For the complement, we do the prefix trick:
    #
    #    prefix[t] = sum_{k=1..t} log p(y_k)
    #    mismatch[t] = prefix[t-1] + log( sum_{v != y_t} p(v_t) )
    #    log_p_not_y = logsumexp_{t=1..T} mismatch[t].
    # 
    # We'll build a prefix array of shape (N, T+1):
    prefix = torch.zeros(log_probs.size(0), log_probs.size(1) + 1,
                         device=log_probs.device)
    # prefix[:, t] = sum_{k=0..(t-1)} chosen_logp[:, k], if we do 0-based indexing
    prefix[:, 1:] = torch.cumsum(chosen_logp, dim=1)

    # log_all_t = logsumexp of all tokens at step t => shape (N,T)
    log_all_t = torch.logsumexp(log_probs, dim=-1)  # (N,T)

    # log_excl_t = log( sum_{v != y_t} p(v_t) ) = log_all_t + log(1 - exp(log_p(y_t) - log_all_t))
    # => must do a stable "1 - e^{x}" approach. We'll define a helper:
    def _safe_log_diff_exp(log_a, log_b, eps=1e-10):
        # compute log( exp(log_a) - exp(log_b) ), 
        # with a clamp to avoid negative or zero inside the log.
        # We'll do log_a as the larger one. Here we want:
        #   log( sum_{v != y_t} p(v_t) ) 
        # = log_all_t + log(1 - exp(chosen_logp[t] - log_all_t)).
        diff = log_b - log_a   # chosen_logp[t] - log_all_t
        # clamp exp(diff) to avoid going above 1 or below 0
        e_diff = torch.clamp(torch.exp(diff), max=1.0 - eps)
        out = torch.log1p(- e_diff)  # log(1 - e^(diff))
        return out

    # Now do that for each t:
    # log_excl_t = log_all_t + log(1 - exp( chosen_logp[t] - log_all_t ))
    # shape => (N,T)
    log_excl_t = log_all_t + _safe_log_diff_exp(log_all_t, chosen_logp)

    # mismatch[t] = prefix[t-1] + log_excl_t at step t-1
    # We can shift them by 1 index:
    prefix_tminus1 = prefix[:, :-1]       # shape (N,T)
    log_excl_tminus1 = log_excl_t         # shape (N,T)

    mismatch = prefix_tminus1 + log_excl_tminus1  # shape (N,T)

    log_p_not_y = torch.logsumexp(mismatch, dim=1)  # shape (N,)

    return log_p_y, log_p_not_y
