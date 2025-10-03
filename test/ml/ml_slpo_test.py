import pytest
import torch
from test_utils import memorization_model

from slpo import slpo


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

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0) # LR scheduled in loop.
    criterion = slpo.SLPO()
    targets = [
        {
            "logprob_ref_w": torch.tensor(0.01).double().log(),
            "logprob_ref_l": torch.tensor(0.01).double().log(),
            "y": torch.tensor([0, 1], dtype=torch.int32),
            "winner": torch.tensor(True, dtype=torch.bool),
        },
        {
            "logprob_ref_w": torch.tensor(0.03).double().log(),
            "logprob_ref_l": torch.tensor(0.07).double().log(),
            "y": torch.tensor([2, 1], dtype=torch.int32),
            "winner": torch.tensor(False, dtype=torch.bool),
        },
    ]
    dataset = DictDataset(torch.ones((2,)).double(), targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

    # Act
    for epoch in range(100):
        optimizer.param_groups[0]["lr"] = 100. / (epoch + 1.)
        for batch in dataloader:
            input, target = batch
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(input), target)
            loss.backward()
            optimizer.step()

            logprob_0 = model.logprob(0, targets[0]["y"]).double()
            logprob_1 = model.logprob(1, targets[1]["y"]).double()

    # Assert
    torch.testing.assert_close(
        logprob_0,
        torch.tensor(0.02).double().log(),
        atol=0.01,
        rtol=0.1,
    )
    torch.testing.assert_close(
        logprob_1.exp(),
        torch.tensor(0.0).double(),
        atol=0.01,
        rtol=0.1,
    )  # This is the expected value for the losing sequence.


@pytest.mark.parametrize("seq_length", [2, 5, 128])
@pytest.mark.parametrize("vocab_size", [2, 7, 1000])
def test_slpo_batch_many_negatives(seq_length, vocab_size):
    # Arrange
    model = memorization_model.MemorizationModel(
        batch_size=2, seq_len=seq_length, vocab_size=vocab_size
    )

    optimizer = torch.optim.SGD(
        model.parameters(), lr=1.0
    )  # High learning rate - we are testing blasting down the loser to zero.
    criterion = slpo.SLPO()
    targets = [
        {
            "logprob_ref_w": torch.tensor(0.02).double().log(),
            "logprob_ref_l": torch.tensor(0.05).double().log(),
            "y": torch.arange(seq_length) % vocab_size,
            "winner": torch.tensor(True, dtype=torch.bool),
        },
        {
            "logprob_ref_w": torch.tensor(0.03, dtype=torch.float64).log(),
            "logprob_ref_l": torch.tensor(0.07, dtype=torch.float64).log(),
            "y": torch.arange(seq_length) % vocab_size,
            "winner": torch.tensor(False, dtype=torch.bool),
        },
    ]
    dataset = DictDataset(torch.ones((2,)), targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

    ref_joint_prob_1 = model.logprob(1, targets[1]["y"])

    # Act
    for epoch in range(100):
        for batch in dataloader:
            input, target = batch
            optimizer.zero_grad()
            loss = criterion(model(input), target)
            loss.backward(retain_graph=True)
            optimizer.step()

    # Assert
    joint_prob_1 = model.logprob(1, targets[1]["y"])

    assert joint_prob_1 < ref_joint_prob_1 or ref_joint_prob_1 < -100, (
        f"{joint_prob_1=}, {ref_joint_prob_1=}"
    )


@pytest.mark.parametrize("seq_length", [5, 128])
@pytest.mark.parametrize("vocab_size", [7, 1000])
def test_slpo_batch_many_positives(seq_length, vocab_size):
    # Arrange
    model = memorization_model.MemorizationModel(
        batch_size=2, seq_len=seq_length, vocab_size=vocab_size
    )

    optimizer = torch.optim.SGD(
        model.parameters(), lr=1.0
    )  # High learning rate - we are testing blasting down the loser to zero.
    criterion = slpo.SLPO()
    targets = [
        {
            "logprob_ref_w": torch.tensor(0.002).double().log(),
            "logprob_ref_l": torch.tensor(0.005).double().log(),
            "y": torch.arange(seq_length) % vocab_size,
            "winner": torch.tensor(True, dtype=torch.bool),
        },
        {
            "logprob_ref_w": torch.tensor(0.003).double().log(),
            "logprob_ref_l": torch.tensor(0.007).double().log(),
            "y": torch.arange(seq_length) % vocab_size,
            "winner": torch.tensor(False, dtype=torch.bool),
        },
    ]
    dataset = DictDataset(torch.ones((2,)), targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

    ref_joint_prob_0 = model.logprob(0, targets[1]["y"])

    # Act
    for epoch in range(100):
        for batch in dataloader:
            input, target = batch
            optimizer.zero_grad()
            loss = criterion(model(input), target)
            loss.backward(retain_graph=True)
            optimizer.step()

    # Assert
    joint_prob_0 = model.logprob(0, targets[1]["y"])

    assert joint_prob_0 > ref_joint_prob_0 or ref_joint_prob_0 > -0.00001, (
        f"{joint_prob_0=}, {ref_joint_prob_0=}"
    )
