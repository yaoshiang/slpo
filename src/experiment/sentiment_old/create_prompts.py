"""Creates prompts between 2 and 8 tokens from the Stanford Sentiment Treebank (SST) dataset."""

import torch

import datasets
import transformers

def create_prompts() -> torch.utils.data.Dataset:
    """Creates prompts between 2 and 8 tokens from the Stanford Sentiment Treebank (SST) dataset.

    Returns:
        datasets.Dataset: A dataset containing the prompts.
    """

    # Download the Stanford Sentiment Treebank (SST) dataset
    sst = datasets.load_dataset("stanfordnlp/sst", split="train")

    # Get phi tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained("microsoft/phi-2")

    # Tokenize all sequences
    tokenized_sequences = [tokenizer.encode(sequence) for sequence in sst["sentence"]]

    # Create prompts of length between 2 and 8 tokens
    prompts = []
    for sequence in tokenized_sequences:
        for i in range(2, 8):
            if len(sequence) >= i:
                prompts.append(tokenized_sequences[:i])

    # Turn prompts into a dataset
    prompts_dataset = datasets.Dataset.from_generator(prompts)

    return prompts_dataset
