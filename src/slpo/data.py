"""Data utils for SLPO andeach model's format."""

import torch


class GPT2Dataset(torch.utils.data.Dataset):
  """Dataset appropriate for a GPT2 model and SLPO loss function.

  Args:
      filename:
  """

  def __init__(self, filename: str):
    pass
