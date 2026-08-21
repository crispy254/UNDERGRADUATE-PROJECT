"""
finetune_lora.py (v2 - local retrain)
----------------------------------------
LoRA fine-tune of Llama 3.2 (3B, Instruct) on the broadened dataset
from prepare_dataset.py, running locally on your RTX PRO 4000
(Blackwell, 24GB VRAM) instead of Colab.

What's different from the Colab version:
  - No Google Drive, no Colab-specific paths - everything local
  - Native bf16 (Blackwell supports it directly - the fp16 workaround
    was only needed for the T4, which couldn't do bf16)
  - Uses train.jsonl AND val.jsonl - tracks validation loss during
    training, so you can see overfitting starting (val loss rising
    while train loss keeps falling) instead of training blind
  - Larger batch size (24GB VRAM vs. T4's 15GB allows more headroom)

Run (in your nova_train conda environment):
    conda activate nova_train
    cd C:\\Users\\chosun\\Desktop\\PROJECTS\\UNDERGRADUATE\\backend
    python finetune_lora.py
"""
import json
import os

TRAIN_PATH = os.path.join(os.path.dirname(__file__), "training_data", "train.jsonl")
VAL_PATH = os.path.join(os.path.dirname(__file__), "training_data", "val.jsonl")
BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct"
MAX_SEQ_LENGTH = 2048
OUTPUT_LORA_DIR = os.path.join(os.path.dirname(__file__), "nova-counseling-lora-v2")
OUTPUT_GGUF_DIR = os.path.join(os.path.dirname(__file__), "nova-counseling-gguf-v2")
GGUF_QUANTIZATION = "q4_k_m"


def load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run prepare_dataset.py first."
        )
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def main():
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    train_examples = load_jsonl(TRAIN_PATH)
    val_examples = load_jsonl(VAL_PATH)
    print(f"Loaded {len(train_examples)} training examples")
    print(f"Loaded {len(val_examples)} validation examples")

    bf16_ok = is_bfloat16_supported()
    print(f"bf16 supported on this GPU: {bf16_ok}")  # should be True on Blackwell

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    def format_example(ex):
        text = tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    train_dataset = Dataset.from_list(train_examples).map(format_example)
    val_dataset = Dataset.from_list(val_examples).map(format_example)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=SFTConfig(
            per_device_train_batch_size=4,   # up from 2 on the T4 - 24GB has more headroom
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=4,
            eval_strategy="steps",
            eval_steps=100,
            warmup_steps=10,
            num_train_epochs=3,
            learning_rate=2e-4,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=os.path.join(os.path.dirname(__file__), "outputs"),
            fp16=not bf16_ok,
            bf16=bf16_ok,
            save_strategy="steps",
            save_steps=200,
            load_best_model_at_end=True,   # keep the checkpoint with the LOWEST val loss,
            metric_for_best_model="eval_loss",  # not just whatever epoch 3 happens to land on
        ),
    )

    print("Starting training...")
    trainer.train()

    print("\nFinal evaluation on validation set:")
    metrics = trainer.evaluate()
    print(metrics)
    with open(os.path.join(os.path.dirname(__file__), "training_data", "val_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    model.save_pretrained(OUTPUT_LORA_DIR)
    tokenizer.save_pretrained(OUTPUT_LORA_DIR)
    print(f"Saved LoRA adapter to {OUTPUT_LORA_DIR}/")

    print("Exporting to GGUF (this step takes several minutes)...")
    model.save_pretrained_gguf(
        OUTPUT_GGUF_DIR, tokenizer, quantization_method=GGUF_QUANTIZATION
    )
    print(f"Saved GGUF export to {OUTPUT_GGUF_DIR}/ (or {OUTPUT_GGUF_DIR}_gguf/ - "
          f"check both, Unsloth's naming has been inconsistent before)")
    print("\nDone. This is all local - no download step needed. Point Ollama's "
          "Modelfile at whichever folder actually has the .gguf file.")


if __name__ == "__main__":
    main()