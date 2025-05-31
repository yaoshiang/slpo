"""Loads IMDB dataset, prepares it for training."""

import logging
import sys

import torch
import transformers
from tqdm import tqdm

from . import data
from . import model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _train_step(net, inputs, optimizer):
    optimizer.zero_grad()
    outputs = net(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    return loss


def sft_gpt(path: str = "gpt2-sft") -> None:
    """Runs SFT on a pre-trained GPT2-large model."""
    logger.info("Loading model and tokenizer...")
    net, tokenizer = model.get_model_and_tokenizer()
    net.to("cuda")
    net.gradient_checkpointing_enable()
    # net.compile()

    tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading dataset...")
    ds = data.get_finetuning_dataset()
    dataloader = torch.utils.data.DataLoader(ds, batch_size=8)

    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-10, weight_decay=0.001)
    lr_scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=len(dataloader) * 1,
    )

    for epoch in range(1, 2):
        net.train()
        pbar = tqdm(dataloader, disable=not sys.stdout.isatty())
        for batch in pbar:
            inputs = tokenizer(
                batch["text"], return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(net.device) for k, v in inputs.items()}
            loss = _train_step(net, inputs, optimizer)
            lr_scheduler.step()

            pbar.set_description(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    logger.info("Saving model to %s", path)
    torch.save({"model_state_dict": net.state_dict()})


def main() -> int:
    sft_gpt()
    return 0


if __name__ == "__main__":
    sys.exit(main())
