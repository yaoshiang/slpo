import types

import torch


class Memo(torch.nn.Module):
  """This model can memorize outputs.

  This is a perfectly parameterized model: it can output
  exactly the desired output. In other words, for each example, if
  there are n outputs, this model has n parameters. This means
  that this model never generalizes - it only memorizes.

  The input is expected to to be BSV, and the output will also be BSV.

  The last output is expected to be ignored.

  This model should be trained with the same input and target data on each step;
  the input data is ignored and the model is conditioned to output the target
  target data.

  Args:
      batch_size: Batch size.
      seq_len: Sequence length.
      vocab_size: Vocabulary size.
      tile_factor: Output will be tiled by this factor.
  """

  def __init__(self, batch_size, seq_len, vocab_size, tile_factor):
    super().__init__()

    self.logits = torch.nn.Parameter(
      torch.randn((batch_size, seq_len, vocab_size))
    )
    self.tile_factor = tile_factor

  def forward(self, x: torch.Tensor, **kwargs) -> types.SimpleNamespace:
    """Forward method of the model.

    Args:
        x: tensor of shape (BSV). Ignored.

    Returns:
        Object with .logits attribute containing tensor of shape (BSV)."""
    logits = self.logits
    if self.tile_factor > 1:
      assert x.shape[0] == self.tile_factor * logits.shape[0]
      logits = logits.repeat(self.tile_factor, 1, 1)

    return types.SimpleNamespace(logits=logits + (x * 0).sum())
