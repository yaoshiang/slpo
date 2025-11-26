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
  """

  def __init__(self, batch_size, seq_len, vocab_size):
    super().__init__()

    self.logits = torch.nn.Parameter(
      torch.randn((batch_size, seq_len, vocab_size))
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward method of the model.

    Args:
        x: tensor of shape (BSV). Ignored.

    Returns:
        Tensor of shape (BSV), the logits."""
    return (x * 0 + 1).sum() * self.logits
