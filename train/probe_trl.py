"""Time-boxed probe: does TRL SFTTrainer train the same LoRA setup as smoke_lora.py?

Feeds pre-rendered input_ids/labels (assistant-only mask from smoke_lora.py) via a
custom collator. Run: python probe_trl.py --model Qwen/Qwen3.5-4B --device cuda:0
"""

import argparse
import sys

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/home/kamil/projects/here/function_calling_edge_fine_tuning/train")
from smoke_lora import build_samples, render_and_mask


class PadCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, features):
        maxlen = max(len(f["input_ids"]) for f in features)
        ids, labels, attn = [], [], []
        for f in features:
            n = maxlen - len(f["input_ids"])
            ids.append(list(f["input_ids"]) + [self.pad_id] * n)
            labels.append(list(f["labels"]) + [-100] * n)
            attn.append([1] * len(f["input_ids"]) + [0] * n)
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn),
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--steps", type=int, default=3)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": args.device}, attn_implementation="sdpa"
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules="all-linear",
        ),
    )
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    rows = []
    for ex in build_samples():
        ids, labels, mode, n = render_and_mask(tok, ex)
        rows.append({"input_ids": ids, "labels": labels})
    ds = Dataset.from_list(rows)
    print(f"[probe] dataset rows={len(ds)} modes ok")

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir="/tmp/opencode/trl_probe",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_length=2048,
            max_steps=args.steps,
            learning_rate=1e-4,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            bf16=True,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        ),
        train_dataset=ds,
        processing_class=tok,  # qwen3_5 is unified-VL: AutoProcessor would pull
        data_collator=PadCollator(tok.pad_token_id or tok.eos_token_id),  # an image processor
    )
    trainer.train()
    print("TRL PROBE RESULT: OK")


if __name__ == "__main__":
    main()
