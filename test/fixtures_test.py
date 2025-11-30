"""Unit tests for the memorization model."""

import pytest
import torch
from fixtures import Memo


@pytest.fixture
def logits():
  retval = torch.tensor(
    [
      [
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [3.0, 9.0, 12.0],
      ],
      [
        [3.0, 2.0, 1.0],
        [0.0, -0.5, -1.0],
        [-1.0, -1.25, -1.5],
      ],
    ]
  )
  return retval


@pytest.fixture
def model(logits):
  """Fixture to create a simple model for testing."""
  retval = Memo(batch_size=2, seq_len=3, vocab_size=3)
  retval.logits = torch.nn.Parameter(logits)
  return retval
  # Set the logits to a known value for reproducibility.


def test_memo_forward(model, logits):
  """Test the forward method of the Memo model."""
  # Arrange
  B, S, V = logits.shape
  dummy_input1 = torch.randint(0, V, (B, S), dtype=torch.long)
  dummy_input2 = torch.randint(0, V, (B, S), dtype=torch.long)

  # Act Get the output from the model.
  output1 = model(dummy_input1)
  output2 = model(dummy_input2)

  # Assert that the outputs are the same regardless of input.

  torch.testing.assert_close(output1, output2)


def test_memo_trainable(model, logits):
  """Test that the Memo model's parameters are trainable."""
  # Arrange
  B, S, V = logits.shape
  optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
  dummy_input = torch.randint(0, V, (B, S), dtype=torch.long)
  target_output = torch.randint(0, V, (B, S), dtype=torch.long)
  criterion = torch.nn.CrossEntropyLoss()

  # Act
  for epoch in range(100):
    optimizer.zero_grad()
    model.train()
    output = model(dummy_input)
    loss = criterion(output.permute(0, 2, 1), target_output)
    loss.backward()
    optimizer.step()
    if epoch == 0:
      initial_output = output
      initial_loss = loss.item()

  final_loss = loss.item()
  # Assert that the loss has decreased.
  print(f"{initial_loss=}, {final_loss=}")
  assert final_loss < initial_loss, "Loss did not decrease during training."

  print(f"INITIAL: {initial_output.argmax(dim=-1)=}\n{target_output=}")
  print(f"FINAL: {output.argmax(dim=-1)=}\n{target_output=}")
  torch.testing.assert_close(output.argmax(dim=-1), target_output)
