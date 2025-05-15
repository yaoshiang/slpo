"""Reproduces DPO's controlled sentiment experiment.

```text
The prompts are prefixes from the IMDB dataset of length 2-8 tokens. We use the pre-trained senti-
ment classifier siebert/sentiment-roberta-large-english as a ground-truth reward model
and gpt2-large as a base model. We use these larger models as we found the default ones to
generate low-quality text and rewards to be somewhat inaccurate. We first use supervised fine-tuning
on a subset of the IMDB data for 1 epoch. We then use this model to sample 4 completions for 25000
prefixes and create 6 preference pairs for each prefix using the ground-truth reward model. The RLHF
reward model is initialized from the gpt2-large model and trained for 3 epochs on the preference
datasets, and we take the checkpoint with the highest validation set accuracy. The “TRL” run uses
the hyper-parameters in the TRL library. Our implementation uses larger batch samples of 1024 per
PPO step.
```
"""

import logging
import sys
from typing import Final

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import hydra 
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

logger = logging.getLogger(__name__)

def _parse_args(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg))
    return float(cfg["lr"]), int(cfg["batch_size"])

@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    """Supervised fine-tuning of GPT2 on IMDB for one epoch."""
    ## Manage configs
    lr, batch_size = _parse_args(cfg)
    del cfg

    ## Prepare Devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Load Model
    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2-large")
    model.to(device)
    CONTEXT_LENGTH:Final[int] = model.config.n_positions
    logger.info("Context Length of the model: %s", CONTEXT_LENGTH)

    ## Load Data
    dataset = load_dataset("imdb", split="unsupervised")
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=CONTEXT_LENGTH, )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])


    ## Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)


    def train_step(model, batch, optimizer, lr_scheduler):
        model.train()

        breakpoint()
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)

        loss = outputs.loss

        loss.backward()
        optimizer.step()

        optimizer.zero_grad()

        lr_scheduler.step()
        return loss

    dataloader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)
    for batch in tqdm(dataloader):
        loss = train_step(model, batch, optimizer, lr_scheduler)
        logger.info(f"Loss: {loss.item()}")



if __name__ == "__main__":
    sys.exit(main())