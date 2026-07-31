"""
finetune_lora.py
-----------------
LoRA fine-tune of Llama 3.2 (3B, Instruct) on Nova's counseling
dialogue, using Unsloth (2-5x faster, ~60-70% less VRAM than plain
Hugging Face Trainer) so this comfortably fits a free Colab T4.

Run in Colab, AFTER prepare_dataset.py has produced
training_data/combined.jsonl:

    !pip install unsloth
    !python finetune_lora.py

Output:
    nova-counseling-lora/    - the LoRA adapter alone (~100MB)
    nova-counseling-gguf/    - merged, quantized, Ollama-ready GGUF

I can't run this myself - this sandbox has no GPU and no network
access to pull the base model or datasets. If a cell errors in Colab,
paste the traceback back and I'll fix the script from that.
"""
import json
import os

DATASET_PATH = os.path.join(os.path.dirname(__file__), "training_data", "combined.jsonl")
BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct"
MAX_SEQ_LENGTH = 2048
OUTPUT_LORA_DIR = "nova-counseling-lora"
OUTPUT_GGUF_DIR = "nova-counseling-gguf"
GGUF_QUANTIZATION = "q4_k_m"   # good balance of quality vs. size/speed on consumer hardware


def load_training_examples(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run prepare_dataset.py first to build it "
            "from the real counseling datasets (CounselChat, ESConv, etc.)."
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

    # T4 (Turing, compute capability 7.5) does NOT support bf16 - only
    # Ampere+ (A100, etc.) does. SFTConfig defaults to requesting bf16
    # if left unset, which crashes on T4 with "doesn't support bf16/gpu".
    # is_bfloat16_supported() checks the actual GPU and picks correctly
    # either way, so this same script works unmodified on a T4 or an A100.
    bf16_ok = is_bfloat16_supported()

    examples = load_training_examples(DATASET_PATH)
    print(f"Loaded {len(examples)} training examples from {DATASET_PATH}")
    if len(examples) < 20:
        print(
            "WARNING: very small training set. LoRA can still run, but "
            "expect a subtle style/tone shift rather than deep behavior "
            "change - that's normal and fine for a persona fine-tune."
        )

    # ------------------------------------------------------------
    # 1. Load base model in 4-bit (QLoRA) - this is what makes a 3B
    #    model trainable on a free T4's ~15GB VRAM.
    # ------------------------------------------------------------
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    # ------------------------------------------------------------
    # 2. Attach LoRA adapters. Only these (~1-2% of total params) get
    #    trained; the base weights stay frozen.
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # 3. Format examples with the model's own chat template, so the
    #    fine-tune reinforces the exact turn structure Ollama will use
    #    at inference time via llm.py.
    # ------------------------------------------------------------
    def format_example(ex):
        text = tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    dataset = Dataset.from_list(examples).map(format_example)

    # ------------------------------------------------------------
    # 4. Train. 3 epochs is deliberately light for a persona/style
    #    fine-tune on a few thousand examples - more risks overfitting
    #    into repeating training phrases verbatim.
    # ------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            num_train_epochs=3,
            learning_rate=2e-4,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            fp16=not bf16_ok,
            bf16=bf16_ok,
        ),
    )

    print("Starting training...")
    trainer.train()

    # ------------------------------------------------------------
    # 5. Save the LoRA adapter on its own (small, portable).
    # ------------------------------------------------------------
    model.save_pretrained(OUTPUT_LORA_DIR)
    tokenizer.save_pretrained(OUTPUT_LORA_DIR)
    print(f"Saved LoRA adapter to {OUTPUT_LORA_DIR}/")

    # ------------------------------------------------------------
    # 6. Export a merged, quantized GGUF - this is what Ollama loads
    #    directly. Unsloth handles the llama.cpp conversion internally.
    # ------------------------------------------------------------
    print("Exporting to GGUF (this step takes several minutes)...")
    model.save_pretrained_gguf(
        OUTPUT_GGUF_DIR, tokenizer, quantization_method=GGUF_QUANTIZATION
    )
    print(f"Saved GGUF export to {OUTPUT_GGUF_DIR}/")
    print("\nDone. Download the nova-counseling-gguf/ folder and follow "
          "the Ollama setup steps to load it as `nova-counseling`.")


if __name__ == "__main__":
    main()