# File management notes

## Files

These are checkpoints of training runs. They were originally in
~/Documents/GitHub/slpo/scripts/third_party/dpo/.cache/yaoshiang

but are mostly moved to an S3 bucket now.

Optimizer state was deleted to save space using

```sh
find . -type f -name 'optimizer.pt' -delete
```

1. Setup your azure storage key

```sh
KEY=$(az storage account keys list \
  -g yh-w1 \
  -n slpo \
  --query '[0].value' \
  -o tsv)
```


1. The main SFT run. Stored in az bucket. 1.7T. 

~/Documents/GitHub/slpo/scripts/third_party/dpo/.cache/yaoshiang/pythia28_sft_anthropic_HH__2025-10-21_16-48-21_233387

This was copied to an az bucket using


```sh
RUN_ID=pythia28_sft_anthropic_HH__2025-10-21_16-48-21_233387

az storage blob upload-batch \
  --account-name slpo \
  --account-key "$KEY" \
  -d slpo/$RUN_ID \
  -s ~/Documents/GitHub/slpo/scripts/third_party/dpo/.cache/yaoshiang/$RUN_ID
```

2. The 0.1 Beta DPO run. 

~/Documents/GitHub/slpo/scripts/third_party/dpo/.cache/yaoshiang/pythia28_dpo_anthropic_hh_2025-11-03_17-17-18_601686

This was copied to an az bucket using

```sh
RUN_ID=pythia28_dpo_anthropic_hh_2025-11-03_17-17-18_601686

az storage blob upload-batch \
  --account-name slpo \
  --account-key "$KEY" \
  -d slpo/$RUN_ID \
  -s ~/Documents/GitHub/slpo/scripts/third_party/dpo/.cache/yaoshiang/$RUN_ID
```

2. The beta 0.5 run

```sh
RUN_ID=tbd


az storage blob upload-batch \
  --account-name slpo \
  --account-key "$KEY" \
  -d slpo/$RUN_ID \
  -s ~/Documents/GitHub/slpo/scripts/third_party/dpo/.cache/yaoshiang/$RUN_ID
```