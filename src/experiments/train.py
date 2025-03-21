# Some of this code is from unsloth and trl examples
# originally licensed Apache (compatible with GPLv3)

from unsloth import FastLanguageModel
from transformers import HfArgumentParser
from datasets import load_dataset
import yaml

from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    ScriptArguments,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE


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


def load_and_train(script_args, training_args, model_args, verbose=True):
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

    # train and save the model
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
