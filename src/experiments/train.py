# Some of this code is from unsloth and trl examples
# originally licensed Apache (compatible with GPLv3)

from unsloth import FastLanguageModel
import torch
from transformers import HfArgumentParser
from datasets import load_dataset
import yaml
import logging
import socket
from datetime import datetime, timedelta


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


def load(script_args, training_args, model_args):
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

    dataset = load_dataset(
        script_args.dataset_name, name=script_args.dataset_config
    )
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    return tokenizer, model, dataset

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


def load_and_train(script_args, training_args, model_args, verbose=True, profile=False):
    if verbose:
        print_as_yaml(training_args)
        print_as_yaml(script_args)
        print_as_yaml(model_args)

    tokenizer, model, dataset = load(script_args, training_args, model_args)

    trainer = DPOTrainer(
        model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split]
        if training_args.eval_strategy != "no"
        else None,
        processing_class=tokenizer,
    )

    if profile:
        profile_training(trainer, 5)

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
