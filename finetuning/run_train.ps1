$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
python finetuning/train_qwen3_8b.py --model "unsloth/Qwen3-8B-unsloth-bnb-4bit" --data "finetuning/data/agent_sft.jsonl" --output-dir "finetuning/outputs/nexa-qwen3-8b-lora" --max-length 2048 --epochs 2 --learning-rate 0.0002 --batch-size 1 --gradient-accumulation 8 --lora-r 16
