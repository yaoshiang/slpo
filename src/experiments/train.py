# Some of this code is from unsloth and trl examples
# originally licensed Apache (compatible with GPLv3)
from unsloth import FastLanguageModel  # noqa F401

import logging
import random
import socket
from datetime import datetime

import pandas as pd
import torch
import yaml
from datasets import load_dataset
from transformers import HfArgumentParser
from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    ScriptArguments,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000

logging.basicConfig(
    format="%(levelname)s:%(asctime)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

def print_as_yaml(o):
    print(yaml.dump(o))


def load_model(training_args, model_args):
    lora_rank = model_args.lora_r

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_args.model_name_or_path,
        # fast_inference=True, # VLLM_USE_V1=1 is not supported with --quantization bitsandbytes.
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.6,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # Remove QKVO if out of memory
        lora_alpha=lora_rank,
        use_gradient_checkpointing="unsloth",  # Enable long context finetuning
        random_state=3407,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    return tokenizer, model

def start_record_memory_history() -> None:
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not recording memory history")
        return

    logger.info("Starting snapshot record_memory_history")
    torch.cuda.memory._record_memory_history(
        max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
    )

def stop_record_memory_history() -> None:
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not recording memory history")
        return

    logger.info("Stopping snapshot record_memory_history")
    torch.cuda.memory._record_memory_history(enabled=None)

def export_memory_snapshot(path) -> None:
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not exporting memory snapshot")
        return

    # Prefix for file names.
    host_name = socket.gethostname()
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    file_prefix = f"{path}/{host_name}_{timestamp}"

    try:
        logger.info(f"Saving snapshot to local file: {file_prefix}.pickle")
        torch.cuda.memory._dump_snapshot(f"{file_prefix}.pickle")
    except Exception as e:
        logger.error(f"Failed to capture memory snapshot {e}")
        return
from torch.utils.data import DataLoader


def show_trainer_examples(trainer):
    # Based on DPOTrainer's `generate_during_eval` option, which reports only to wandb and comet.

    datasets = trainer.eval_dataset if isinstance(trainer.eval_dataset, dict) else {'eval': trainer.eval_dataset}
    for ds_name, dataset in datasets.items():
        print(f"{ds_name=}")
        print(f"{dataset=}")
        dataloader = trainer.get_eval_dataloader(dataset)
        print(f"{dataloader.dataset=}")

        batch_size = 1
        dataloader_params = {
            "batch_size": batch_size,
            "shuffle": False,
        }

        # prepare dataloader
        dataloader_raw = trainer.accelerator.prepare(DataLoader(dataset, **dataloader_params))
        print(f"{dataloader_raw.dataset=}")

        num_samples = len(dataloader.dataset)
        print(f"{num_samples=}")
        random_indices = random.sample(range(num_samples), k=1)

        # Use dataloader.dataset.select to get the random batch without iterating over the DataLoader
        random_batch_dataset = dataloader.dataset.select(random_indices)
        print(f"\n\n{random_batch_dataset=}\n\n")
        random_batch = trainer.data_collator(random_batch_dataset)
        print(f"{random_batch=}\n\n")
        random_batch = trainer._prepare_inputs(random_batch)
        print(f"prepared {random_batch=}\n\n")

        policy_output_decoded, ref_output_decoded = trainer.generate_from_model_and_ref(trainer.model, random_batch)

        table = pd.DataFrame(
            columns=["Prompt", "Policy", "Ref Model"],
            data=[
                [prompt, pol[len(prompt) :], ref[len(prompt) :]]
                for prompt, pol, ref in zip(
                    random_batch_dataset["prompt"], policy_output_decoded, ref_output_decoded
                )
            ],
        )

        print(table)

def profile_training(trainer, steps):

    # Start recording memory snapshot history
    start_record_memory_history()

    for _ in range(steps):
        batch_samples, _ = trainer.get_batch_samples(iter(trainer.get_train_dataloader()), 1)
        inputs = batch_samples[0]
        trainer.training_step(trainer.model, inputs)

    # Create the memory snapshot file
    export_memory_snapshot(trainer._get_output_dir(None))

    # Stop recording memory snapshot history
    stop_record_memory_history()

def dataset_conf():
    return {
        "train": {"path": "Anthropic/hh-rlhf", "dataset_split": "train", "sample": None},
        "eval": {"path": "Anthropic/hh-rlhf", "dataset_split": "test", "sample": 300},
        "forget_ultra": {"path": "trl-lib/ultrafeedback_binarized", "dataset_split": "train", "sample": 300},
    }

def load_and_sample_datasets(ds_config, seed=42):
    ds_path = ds_config["path"]
    ds_name = ds_config.get("name", None)
    split = ds_config.get("dataset_split", "train")
    sample_size = ds_config.get("sample")

    # Load dataset
    ds = load_dataset(ds_path, name=ds_name, split=split)

    # Sample if specified
    if sample_size and sample_size < len(ds):
        ds = ds.shuffle().select(range(sample_size))
    return ds

def load_data(datasets):
    train_dataset = load_and_sample_datasets(datasets["train"])
    eval_datasets = {nick: load_and_sample_datasets(spec) for (nick, spec) in datasets.items() if nick != "train"}
    return train_dataset, eval_datasets

def load_and_train(dataset_specs, training_args, model_args, verbose=True, profile=False, show_examples=False):
    if verbose:
        print_as_yaml(training_args)
        print_as_yaml(dataset_specs)
        print_as_yaml(model_args)
    logger.info("Start loading model...")
    tokenizer, model = load_model(training_args, model_args)
    logger.info("... done. Start loading datasets...")
    training_dataset, eval_datasets = load_data(dataset_specs)
    logger.info("... done. Start defining trainer...")

    trainer = DPOTrainer(
        model,
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=(
            eval_datasets if training_args.eval_strategy != "no" else None
        ),
        processing_class=tokenizer,
    )
    logger.info("... done.")

    if profile:
        profile_training(trainer, 5)
    if show_examples:
        show_trainer_examples(trainer)
        print("examples go here")
    # train the model
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
    return trainer


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, DPOConfig, ModelConfig))
    script_args, training_args, model_args = (
        parser.parse_args_into_dataclasses()
    )

    load_and_train(script_args, training_args, model_args)
