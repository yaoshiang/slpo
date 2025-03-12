import torch


class MemorizationModel(torch.nn.Module):
    def __init__(self, num_examples, seq_len, vocab_size):
        super(MemorizationModel, self).__init__()

        self.num_examples = num_examples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        self.weights = torch.nn.Parameter(
            torch.randn((num_examples, seq_len, vocab_size))
        )

    def forward(self, x):
        return torch.nn.functional.log_softmax(self.weights[x], dim=-1)

    def prob(self, example_idx, token_ids):
        return torch.exp(self.weights[example_idx, token_ids]).prod(axis=-1)

    def log_prob(self, example_idx, token_ids):
        return torch.log(self.prob(example_idx, token_ids)).sum()
