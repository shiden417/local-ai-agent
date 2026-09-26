from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning for Qwen3-8B using NEXA agent trajectories."
    )
    parser.add_argument(
        "--model",
        default="unsloth/Qwen3-8B-unsloth-bnb-4bit",
        help="Hugging Face model ID.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("finetuning/data/agent_sft.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("finetuning/outputs/nexa-qwen3-8b-lora"),
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument(
        "--export-gguf",
        choices=("q4_k_m", "q5_k_m", "q8_0"),
        help="Optionally export the trained adapter merged into GGUF.",
    )
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.data.exists():
        raise SystemExit(
            f"Dataset not found: {args.data}\n"
            "Run prepare_agent_sft.py first."
        )

    if args.max_length < 512:
        raise SystemExit("--max-length must be at least 512")
    if args.batch_size < 1 or args.gradient_accumulation < 1:
        raise SystemExit("Batch size and gradient accumulation must be positive")

    from datasets import load_dataset
    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer

    dataset = load_dataset(
        "json",
        data_files=str(args.data),
        split="train",
    )

    if len(dataset) < 4:
        raise SystemExit(
            f"Only {len(dataset)} training samples are available. "
            "Generate more successful Benchmark trajectories before training."
        )

    eval_size = max(1, int(len(dataset) * 0.1))
    if eval_size >= len(dataset):
        eval_size = 1

    split = dataset.train_test_split(
        test_size=eval_size,
        seed=args.seed,
    )

    train_dataset = split["train"]
    eval_dataset = split["test"]

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")
    print(f"Model: {args.model}")
    print(f"Max length: {args.max_length}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_length,
        dtype=None,
        load_in_4bit=True,
        full_finetuning=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_r,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            output_dir=str(args.output_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            warmup_ratio=0.05,
            weight_decay=0.01,
            lr_scheduler_type="linear",
            logging_steps=1,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            optim="adamw_8bit",
            seed=args.seed,
            dataset_num_proc=1,
            max_length=args.max_length,
            assistant_only_loss=True,
            report_to="none",
        ),
    )

    trainer.train()

    adapter_dir = args.output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    if args.export_gguf:
        gguf_dir = args.output_dir / f"gguf-{args.export_gguf}"
        print()
        print(f"Exporting GGUF: {gguf_dir} ({args.export_gguf})")
        model.save_pretrained_gguf(
            str(gguf_dir),
            tokenizer,
            quantization_method=args.export_gguf,
        )
        print(f"GGUF saved: {gguf_dir}")

    print()
    print(f"LoRA adapter saved: {adapter_dir}")
    print("Base Qwen3-8B remains untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
