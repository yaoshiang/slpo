"""Loads IMDB dataset, prepares it for training."""

import random

import torch
import datasets
import transformers

def get_finetuning_dataset():
    """Returns the IMDB unsupervised dataset.

    This dataset is used to fine-tune GPT-2.

    Returns:
        tokenized dataset: The IMDB dataset split into an unsupervised set.
    """
    ds = datasets.load_dataset("stanfordnlp/imdb", split="unsupervised")
    ds.map()

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

def sft_gpt():
    """Returns a pre-trained GPT2-large model with SFT."""
    model = get_model()
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2-large")
    tokenizer.pad_token = tokenizer.eos_token

    ds = get_finetuning_dataset()
    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=8
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    lr_scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=len(dataloader) * 1,  # 1 epochs
    )

    model.train()

    for epoch in range(1):  # 1 epoch
        for batch in dataloader:
            inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
