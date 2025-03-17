import torch

from slpo import slpo
from test_utils import memorization_model


class DictDataset(
    torch.utils.data.Dataset[tuple[int, dict[str, torch.Tensor]]]
):
    def __init__(self, input, target):
        self.input = input
        self.target = target

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return self.input[idx], self.target[idx]


def test_slpo_batch_size_two():
    # Arrange
    model = memorization_model.MemorizationModel(
        batch_size=2, seq_len=2, vocab_size=3
    )  # Prime numbers for ease of testing.

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    criterion = slpo.SLPO()
    target = [
        {
            "pi_ref_w": torch.tensor(0.02),
            "pi_ref_l": torch.tensor(0.05),
            "y": torch.tensor([0, 1], dtype=torch.int32),
            "winner": torch.tensor(True, dtype=torch.bool),
        },
        {
            "pi_ref_w": torch.tensor(0.03),
            "pi_ref_l": torch.tensor(0.07),
            "y": torch.tensor([2, 1], dtype=torch.int32),
            "winner": torch.tensor(False, dtype=torch.bool),
        },
    ]
    dataset = DictDataset(torch.ones((2,)), target)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

    # Act
    for epoch in range(100):
        for batch in dataloader:
            input, target = batch
            optimizer.zero_grad()
            loss = criterion(model(input), target)
            loss.backward()
            optimizer.step()

    # Assert
    joint_prob_0 = model.joint_prob(0, target["y"][0])
    joint_prob_1 = model.joint_prob(1, target["y"][1])

    torch.testing.assert_close(
        joint_prob_0, torch.tensor(0.07), atol=0.01, rtol=0.1
    )  # y_ref_w + y_ref_l: 0.02 + 0.05
    torch.testing.assert_close(
        joint_prob_1, torch.tensor(0.0), atol=0.01, rtol=0.1
    )  # This is the expected value for the losing sequence.
