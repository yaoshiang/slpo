"""GPT2 model for sentiment analysis."""

import transformers
import torch

def get_model_and_tokenizer() -> tuple:
    """Returns a pre-trained GPT2-large model."""
    model = transformers.AutoModelForCausalLM.from_pretrained(
        "gpt2-large", torch_dtype=torch.float16
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2-large")

    return (model, tokenizer)

def load_model(path: str = "gpt2-sft") -> tuple:
    """Loads a pre-trained GPT2-large model from the specified path."""
    model = transformers.AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(path)

    return (model, tokenizer)