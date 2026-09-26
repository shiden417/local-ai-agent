# NEXA Qwen3-8B Fine-tuning

NEXA/J.A.R.V.I.S. の実運用に合わせた Qwen3-8B の QLoRA/SFT 環境です。

## 方針

- ベースモデル: Qwen/Qwen3-8B
- 方式: 4-bit QLoRA + SFT
- 学習対象: NEXAの成功したAgent実行軌跡
- 元モデルは変更せず、LoRA adapterとして保存
- Tool Callingの軌跡も学習対象に含める
- 内部思考そのものは学習データとして保存しない
- 失敗から復旧して成功した軌跡は残し、復旧能力を学習させる

TRLのSFTTrainerは conversation、tool_calls、tool role、tools schema を含むTool Callingデータを学習できます。
Qwen3はthinking / non-thinkingを切り替えられます。

## 1. 学習用データを作る

CustomAgentのルートで:

    python finetuning/prepare_agent_sft.py --model qwen/qwen3-8b --thinking-mode default

成功したBenchmark Taskだけを finetuning/data/agent_sft.jsonl に保存します。

データ生成時には実際にBenchmarkを実行するため、LM Studioが起動している必要があります。

## 2. 学習環境

PowerShell:

    python -m venv .venv-finetune
    .\\.venv-finetune\\Scripts\\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r finetuning/requirements.txt

UnslothはWindowsでローカルfine-tuningをサポートしています。

## 3. QLoRA

    python finetuning/train_qwen3_8b.py

初期設定は8GB級GPUを想定して:

- max_length=2048
- batch_size=1
- gradient_accumulation_steps=8
- LoRA r=16
- 2 epochs

です。

最初の実験では小さなadapterを作り、元モデルと同じBenchmarkで比較します。

## 4. 評価

最初から本番モデルへ置き換えません。

1. 元の qwen/qwen3-8b で Benchmark を測る
2. LoRA adapterを評価する
3. 成功Task、失敗復旧、LLM latency、Tool回数を比較する
4. 改善が確認できたadapterだけを本番へ反映する

## 5. LM Studioへ戻す

adapter評価後に必要ならGGUFへ書き出します。

Unslothには save_pretrained_gguf があり、q4_k_m などの量子化形式を指定できます。
作成したGGUFは LM Studio の lms import でローカルモデルとして取り込めます。

fine-tuningで生成速度そのものが必ず速くなるわけではありません。
速度は別途、LM Studioの context length、GPU offload、Flash Attention、continuous batching を測定して最適化します。
