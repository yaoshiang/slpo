"""Loads IMDB dataset, prepares it for training."""

import random

import torch
import datasets
import transformers

def get_finetuning_dataset():
    """Returns the IMDB unsupervised dataset.

    This dataset is used to fine-tune GPT-2.
    """
    return datasets.load_dataset("stanfordnlp/imdb", split="unsupervised")

def _get_prefixes(tokenizer, ds:torch.utils.data.Dataset):
    ds = ds.filter(lambda x: len(x["text"].split()) <= 8)
    ds = ds.map(lambda x: tokenizer(x["text"], truncation=True, max_length=8, padding="max_length"), batched=True)
    ds = ds.map(lambda x: {"prefix": " ".join(x["input_ids"][:random.randint(2, 8)])}, batched=True)

def get_train_prefixes(tokenizer):
    """Generates 2-8 token prefixes from train."""
    ds = datasets.load_dataset("stanfordnlp/imdb", split="train")
    return _get_prefixes(tokenizer, ds)

def get_test_prefixes(tokenizer):
    """Generates 2-8 token prefixes from test."""
    ds = datasets.load_dataset("stanfordnlp/imdb", split="test")
    return _get_prefixes(tokenizer, ds)

def get_model():
    """Returns a pre-trained GPT2-large model."""
    return transformers.AutoModelForCausalLM.from_pretrained("gpt2-large", torch_dtype=torch.float16)

