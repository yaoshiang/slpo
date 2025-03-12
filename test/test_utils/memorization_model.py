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

    def forward(self, example_idx):
        if not 0 <= example_idx < self.num_examples:
            raise ValueError(
                f"Invalid example index: {example_idx}, "
                f"num_examples={self.num_examples}"
            )
        weights = self.weights[example_idx]
        return torch.nn.functional.log_softmax(weights, dim=-1)

    def joint_prob(self, example_idx, token_ids):
        if not 0 <= example_idx < self.num_examples:
            raise ValueError(
                f"Invalid example index: {example_idx}, "
                f"num_examples={self.num_examples}"
            )

        if not 0 <= len(token_ids) <= self.seq_len:
            raise ValueError(
                f"Invalid token_ids length: {len(token_ids)}, "
                f"seq_len={self.seq_len}"
            )

        # Generate the logprobs.
        log_probs = self.forward(example_idx)
        assert log_probs.size() == (self.seq_len, self.vocab_size)

        # Gather just the token_ids we care about.
        selected_log_probs = log_probs[range(len(token_ids)), token_ids]
        assert selected_log_probs.size() == (len(token_ids),)

        # Exponentiate to get the probabilities.
        selected_probs = torch.exp(selected_log_probs)
        assert selected_probs.size() == (len(token_ids),)

        # Multiply the probabilities together to get the joint probability.
        joint_prob = selected_probs.prod()
        assert joint_prob.size() == ()

        return joint_prob

    def joint_log_prob(self, example_idx, token_ids):
        if not 0 <= example_idx < self.num_examples:
            raise ValueError(
                f"Invalid example index: {example_idx}, "
                f"num_examples={self.num_examples}"
            )

        if not 0 <= len(token_ids) <= self.seq_len:
            raise ValueError(
                f"Invalid token_ids length: {len(token_ids)}, "
                f"seq_len={self.seq_len}"
            )

        # Generate the logprobs.
        log_probs = self.forward(example_idx)
        assert log_probs.size() == (self.seq_len, self.vocab_size)

        # Gather just the token_ids we care about.
        selected_log_probs = log_probs[range(len(token_ids)), token_ids]
        assert selected_log_probs.size() == (len(token_ids),)

        # Sum the log probabilities to get the joint log probability.
        joint_log_prob = selected_log_probs.sum()
        assert joint_log_prob.size() == ()

        return joint_log_prob
