"""Dump chosen/rejected logprobs from a checkpoint to CSV for analysis.

Usage:
    python dump_logprobs.py \
        --archive .cache/yaoshiang/pythia28_sft_anthropic_HH__2025-10-21_16-48-21_233387/step-479232/policy.pt \
        --model EleutherAI/pythia-2.8b \
        --split train \
        --output logprobs.csv
"""

import argparse
import csv
import sys

import torch
import tqdm
import transformers

sys.path.insert(0, ".")
from preference_datasets import get_dataset, get_collate_fn, tokenize_batch_element
from slpo import slpo_adapter
from trainers import concatenated_inputs
from utils import disable_dropout, get_local_dir


def load_model(name_or_path, archive_path, dtype, cache_dir):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        name_or_path,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    disable_dropout(model)
    if archive_path:
        state = torch.load(archive_path, map_location="cpu")
        model.load_state_dict(state["state"])
        print(f"Loaded archive from step {state['step_idx']}")
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--model", default="EleutherAI/pythia-2.8b")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--dataset", default="hh")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--output", default="logprobs.csv")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--cache_dir", default=".cache")
    parser.add_argument("--max_examples", type=int, default=None, help="Stop after this many pairs (default: all)")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)

    print(f"Loading tokenizer: {args.model}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model from {args.archive}...")
    model = load_model(args.model, args.archive, dtype, args.cache_dir)
    device = next(model.parameters()).device

    print(f"Loading dataset '{args.dataset}' ({args.split} split)...")
    dataset = get_dataset(args.dataset, args.split, cache_dir=args.cache_dir)
    collate_fn = get_collate_fn(tokenizer)

    rows = []
    n = 0
    for prompt, data in tqdm.tqdm(dataset.items(), desc="Examples"):
        for pair in data["pairs"]:
            if args.max_examples is not None and n >= args.max_examples:
                break
            chosen = data["responses"][pair[0]]
            rejected = data["responses"][pair[1]]
            truncation_mode = "keep_end"  # hh uses keep_end

            elem = tokenize_batch_element(
                prompt, chosen, rejected, truncation_mode,
                tokenizer, args.max_length, args.max_prompt_length,
            )
            batch = collate_fn([elem])
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            with torch.no_grad():
                chosen_logp, rejected_logp, chosen_logp_comp, rejected_logp_comp, _, _ = (
                    slpo_adapter.concatenated_forward(model, batch, concatenated_inputs)
                )

            rows.append({
                "chosen_logp": chosen_logp.item(),
                "rejected_logp": rejected_logp.item(),
                "chosen_logp_comp": chosen_logp_comp.item(),
                "rejected_logp_comp": rejected_logp_comp.item(),
                "chosen_gt_rejected": int(chosen_logp.item() > rejected_logp.item()),
                "chosen_len": elem["chosen_input_ids"].shape[0],
                "rejected_len": elem["rejected_input_ids"].shape[0],
            })
            n += 1
        if args.max_examples is not None and n >= args.max_examples:
            break

    print(f"\nWriting {len(rows)} rows to {args.output}...")
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Quick summary
    chosen_gt = sum(r["chosen_gt_rejected"] for r in rows)
    print(f"\nSummary over {len(rows)} pairs:")
    print(f"  chosen_logp > rejected_logp: {chosen_gt}/{len(rows)} ({100*chosen_gt/len(rows):.1f}%)")
    print(f"  mean chosen_logp:   {sum(r['chosen_logp'] for r in rows)/len(rows):.2f}")
    print(f"  mean rejected_logp: {sum(r['rejected_logp'] for r in rows)/len(rows):.2f}")
    print(f"Done. Output: {args.output}")


if __name__ == "__main__":
    main()
