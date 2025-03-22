import torch


class MemorizationModel(torch.nn.Module):
    """This model can memorize outputs for a fixed number of examples.

    This is a perfectly parameterized model, meaning, it can output
    exactly the desired output. In other words, for each example, if
    there are n outputs, this model has n parameters. This means
    that this model never generalizes - it only memorizes."""

    def __init__(self, batch_size, seq_len, vocab_size):
        super(MemorizationModel, self).__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        self.logits = torch.nn.Parameter(
            torch.randn((batch_size, seq_len, vocab_size))
        )

    def forward(self, x):
        """Forward method of the model.

        Args:
            x: tensor of shape (N), all values should be ones.
                N should match the num_examples arg
                passed to the constructor.
        """
        if x.size() != (self.batch_size,):
            raise ValueError(
                f"Invalid input shape: {x.size()}, "
                f"expected {(self.batch_size,)}"
            )

        if not torch.all(x == 1):
            raise ValueError("All values of x should be 1")

        retval = torch.einsum("n, nsv -> nsv", x, self.logits)
        assert retval.size() == (self.batch_size, self.seq_len, self.vocab_size)
        return retval

    def logprob(self, example_idx, token_ids):
        if not 0 <= example_idx < self.batch_size:
            raise ValueError(
                f"Invalid example index: {example_idx}, "
                f"num_examples={self.batch_size}"
            )

        if not 0 <= len(token_ids) <= self.seq_len:
            raise ValueError(
                f"Invalid token_ids length: {len(token_ids)}, "
                f"seq_len={self.seq_len}"
            )

        # Generate the logprobs.
        logits = self.forward(torch.ones((self.batch_size,)))[example_idx].double()
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        assert logprobs.size() == (self.seq_len, self.vocab_size)

        # Gather just the token_ids we care about.
        selected_logprobs = logprobs[range(len(token_ids)), token_ids]
        assert selected_logprobs.size() == (len(token_ids),)

        # Sum the log probabilities to get the joint log probability.
        joint_log_prob = selected_logprobs.sum()
        assert joint_log_prob.size() == ()

        return joint_log_prob
