import unsloth  # noqa F401
import tensorboard  # noqa F401

from pathlib import Path
from datetime import datetime as dt
from trl import (
    DPOConfig,
    ModelConfig,
)
from train import load_and_train, dataset_conf
import modal

# model_name = "unsloth/Qwen2.5-0.5B-unsloth-bnb-4bit"
model_name = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
model_name = "unsloth/mistral-7b-instruct-v0.3"

# create a Volume, or retrieve it if it exists
dt_str = dt.now().replace(microsecond=0).isoformat().replace(":", "-")
volume = modal.Volume.from_name("model-weights-vol", create_if_missing=True)
checkpoints_vol_name = f"training-output-vol-{dt_str}"
checkpoints_volume = modal.Volume.from_name(
    checkpoints_vol_name, create_if_missing=True
)

MODEL_DIR = Path("/models")
CHECKPOINTS_DIR = Path("/output_dir")

image = (
    modal.Image.debian_slim(python_version="3.12")
    # .apt_install("git")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system 'bitsandbytes>=0.45.3' 'datasets>=3.3.2' 'peft>=0.14.0' 'setuptools>=76.1.0' "
        "'torch>=2.6.0' 'torchao>=0.9.0' 'transformers==4.49.0' 'trl[quantization]>=0.15.2' 'unsloth>=2025.3.15' "
    )
    .run_commands(
        "uv pip install --system huggingface_hub[hf_transfer] 'tensorboard>=2.19.0'",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # and enable it
    .add_local_python_source(
        "train",
        copy=True,
    )
)

app = modal.App("po-example", image=image)


@app.function(
    volumes={
        MODEL_DIR: volume,
    },  # "mount" the Volume, sharing it with your function
    image=image,  # only download dependencies needed here
    gpu="any",
)
def download_model(
    repo_id: str = model_name,
    revision: str = None,  # include a revision to prevent surprises!
):
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, local_dir=MODEL_DIR / repo_id)
    print(f"Model downloaded to {MODEL_DIR / repo_id}")


def print_track():
    print(f"to track run:\nmodal volume get --force {checkpoints_vol_name} /runs .")

timeout = 5 * 3600
@app.function(
    timeout=timeout,
    gpu="a100",
    volumes={
        MODEL_DIR: volume,
        CHECKPOINTS_DIR: checkpoints_volume,
    },  # "mount" the Volume, sharing it with your function
)
def train_on_modal():
    print_track()
    load_and_train(
        dataset_conf(),
        DPOConfig(
            #max_steps=3,
            report_to="all",
            save_steps=500,
            eval_steps=500,
            eval_strategy="steps",
            eval_on_start=True,
            generate_during_eval=False,
            do_eval=True,
            per_device_train_batch_size=8,
            num_train_epochs=2,
            output_dir=CHECKPOINTS_DIR,
            per_device_eval_batch_size=16,
            #skip_memory_metrics=False,
            learning_rate = 4e-6,
            logging_steps = 1,
            optim = "paged_adamw_8bit",
            weight_decay = 0.0,
            lr_scheduler_type = "cosine",
        ),
        ModelConfig(
            model_name_or_path=model_name,
            lora_r=64,
        ),
        show_examples=True,
    )
    print_track()
