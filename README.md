# RIGOR

Public code release for **RIGOR**, a prototype-guided robust multi-agent reasoning method for LLM systems.

This repository contains the RIGOR method implementation, benchmark-format datasets, inference/evaluation scripts, and sanitized OpenAI-compatible model configuration templates.

## Contents

```text
.
├── inference.py                  # Run RIGOR inference
├── evaluate.py                   # Evaluate saved inference jsonl files
├── methods/
│   ├── rigor/                    # RIGOR implementation
│   ├── mas_base/                 # Internal single-agent LLM wrapper
│   └── utils.py                  # Method utilities
├── datasets/
│   ├── data/                     # Included benchmark JSON files
│   └── build_test_dataset.py     # Optional dataset conversion/downloading script
├── evaluations/                  # Evaluation helpers
├── model_api_configs/            # Sanitized model endpoint configs
└── requirements.txt
```

## Setup

```bash
cd RIGOR
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

RIGOR uses an OpenAI-compatible chat-completions endpoint. No real API keys are stored in this repository. The default config reads `${OPENAI_API_KEY}`:

```bash
export OPENAI_API_KEY="your_api_key"
```

To use another backend, edit `model_api_configs/model_api_config.json` or add a new alias following `model_api_configs/model_api_config.example.json`.

## Quick Start

Run RIGOR on the included MATH split:

```bash
python inference.py \
  --method_name rigor \
  --method_config_name config_math \
  --model_name gpt-4o-mini \
  --test_dataset_name MATH \
  --output_path results/MATH/gpt-4o-mini/rigor_infer.jsonl \
  --max_samples 20 \
  --sequential
```

Run on MedMCQA:

```bash
python inference.py \
  --method_name rigor \
  --method_config_name config_medmcqa \
  --model_name gpt-4o-mini \
  --test_dataset_name MedMCQA \
  --output_path results/MedMCQA/gpt-4o-mini/rigor_infer.jsonl
```

## Evaluation

For math and multiple-choice tasks, use an LLM judge:

```bash
python evaluate.py \
  --eval_protocol xverify \
  --model_name gpt-4o-mini \
  --tested_dataset_name MATH \
  --tested_method_name rigor \
  --tested_mas_model_name gpt-4o-mini \
  --tested_infer_path results/MATH/gpt-4o-mini/rigor_infer.jsonl
```

## Configs

RIGOR configs live in `methods/rigor/configs/`.

Common choices:

```text
config_math.yaml                           MATH config
config_medmcqa.yaml                        MedMCQA config
```

The method name `rigor` is the public entry point.

## Datasets

The supported runnable datasets are MATH and MedMCQA under `datasets/data/`. The inference script resolves dataset names case-insensitively, so `MATH` maps to `datasets/data/MATH.json`.

To rebuild a dataset from HuggingFace:

```bash
python datasets/build_test_dataset.py --dataset_name MATH --num2sample 500
```

## Notes

The local MiniLM directory from the original workspace is not bundled because it is large. If `sentence-transformers` cannot download `all-MiniLM-L6-v2`, RIGOR falls back to a deterministic hashed text encoder so the code remains runnable.

Generated artifacts are written to `results/` and ignored by git.
