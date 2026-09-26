$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
python finetuning/prepare_agent_sft.py --model "qwen/qwen3-8b" --thinking-mode default --max-iterations 12 --output "finetuning/data/agent_sft.jsonl"
