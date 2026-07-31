"""
prepare_dataset.py
-------------------
Downloads real counseling/emotional-support datasets from Hugging Face,
converts each into the {"messages": [...]} chat format finetune_lora.py
expects, and writes them all to training_data/combined.jsonl.

Run this in Colab (or locally) BEFORE finetune_lora.py:
    pip install datasets
    python prepare_dataset.py

Design decision - crisis content is filtered OUT, not trained on
--------------------------------------------------------------
Nova's crisis handling is a fixed, human-reviewed response
(prompt_builder.CRISIS_RESPONSE), triggered by inference.py's
rule-based check BEFORE the LLM ever runs - see chatbot_engine.py.
That's intentional: crisis wording should never depend on model
sampling. So this script drops any training pair whose client/seeker
turn matches the same crisis patterns inference.py checks for, using
the same pattern list, so training data and runtime safety logic never
drift apart. The fine-tuned model should never see (and therefore
never learn to imitate) a crisis exchange - that path is handled
entirely outside the LLM.

Sources used (see citations below) - swap CONFIG at the top to add/
remove any of them.
"""
import json
import os
import re
import sys

# Reuse the exact same crisis patterns the running app uses, so the
# filter here can never quietly drift out of sync with inference.py.
# This file lives in the same flat backend/ folder as inference.py, so
# a plain import finds it as long as both were uploaded to Colab
# together (see the Colab steps for this run).
sys.path.insert(0, os.path.dirname(__file__))
try:
    from inference import CRISIS_PATTERNS
except ImportError:
    # Fallback copy, only used if inference.py wasn't uploaded alongside
    # this script (e.g. you only grabbed prepare_dataset.py by itself).
    CRISIS_PATTERNS = [
        r"\bkill myself\b", r"\bend my life\b", r"\bend it all\b",
        r"\bsuicid\w*", r"\bwant to die\b", r"\bdon'?t want to (be alive|live)\b",
        r"\bno reason to live\b", r"\bhurt myself\b", r"\bself[- ]?harm\b",
    ]

OUT_PATH = os.path.join(os.path.dirname(__file__), "training_data", "combined.jsonl")

# Matches prompt_builder.COUNSELING_PERSONA, so the fine-tune reinforces
# (rather than fights) the system prompt the app actually sends at
# inference time.
SYSTEM_PROMPT = (
    "You are Nova, a warm, empathetic peer-support companion for "
    "undergrad students dealing with stress and career worries. You are "
    "not a licensed therapist and never diagnose. You listen, validate, "
    "and offer grounded, practical support."
)

MAX_ASSISTANT_WORDS = 130  # keep training targets consistent with the live prompt's own length rule


def is_crisis(text: str) -> bool:
    text = (text or "").lower()
    return any(re.search(p, text) for p in CRISIS_PATTERNS)


def _trim(text: str, max_words: int = MAX_ASSISTANT_WORDS) -> str:
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text


def build_example(user_text: str, assistant_text: str) -> dict | None:
    user_text = (user_text or "").strip()
    assistant_text = (assistant_text or "").strip()
    if not user_text or not assistant_text:
        return None
    if is_crisis(user_text) or is_crisis(assistant_text):
        return None
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": _trim(assistant_text)},
        ]
    }


def load_counsel_chat(rows: list):
    """
    nbertagnolli/counsel-chat - 2,775 real Q&A pairs from licensed
    therapists on CounselChat.com.
    Cite: Bertagnolli, N. "Counsel Chat: Bootstrapping High-Quality
    Therapy Data," 2020.
    """
    from datasets import load_dataset
    ds = load_dataset("nbertagnolli/counsel-chat", split="train")
    n = 0
    for row in ds:
        ex = build_example(row.get("questionText"), row.get("answerText"))
        if ex:
            rows.append(ex)
            n += 1
    print(f"  counsel-chat: {n} examples")


def load_amod(rows: list):
    """
    Amod/mental_health_counseling_conversations - real one-on-one
    counseling Q&A from licensed professionals.
    NOTE: license requires a $100 donation for commercial use; fine
    for academic/non-commercial use (this project).
    """
    from datasets import load_dataset
    ds = load_dataset("Amod/mental_health_counseling_conversations", split="train")
    n = 0
    for row in ds:
        ex = build_example(row.get("Context"), row.get("Response"))
        if ex:
            rows.append(ex)
            n += 1
    print(f"  Amod counseling conversations: {n} examples")


def load_esconv(rows: list):
    """
    thu-coai/esconv - 1,300 real crowdsourced help-seeker/supporter
    dialogues annotated with support strategies (Hill's Helping Skills
    Theory: exploration -> comforting -> action).
    Cite: Liu et al., "Towards Emotional Support Dialog Systems," ACL 2021.

    Each dialogue is multi-turn; we extract adjacent (seeker, supporter)
    turn pairs rather than whole dialogues, matching finetune_lora.py's
    single-turn example format (history is handled separately at
    inference time by chatbot_service.py, not baked into training rows).
    """
    from datasets import load_dataset
    ds = load_dataset("thu-coai/esconv", split="train")
    n = 0
    for row in ds:
        try:
            dialog = json.loads(row["text"]) if isinstance(row.get("text"), str) else row
            turns = dialog.get("dialog", [])
        except (KeyError, json.JSONDecodeError, TypeError):
            continue
        for i in range(len(turns) - 1):
            a, b = turns[i], turns[i + 1]
            if a.get("speaker") == "usr" and b.get("speaker") == "sys":
                ex = build_example(a.get("content"), b.get("content"))
                if ex:
                    rows.append(ex)
                    n += 1
    print(f"  ESConv: {n} examples")


def load_mentalchat16k(rows: list, limit: int = 4000):
    """
    ShenLab/MentalChat16K - synthetic counselor-client conversations
    across 33 mental-health topics. Synthetic, so treat as volume/
    coverage filler rather than a primary source - weight it lower by
    capping how many rows get pulled in (`limit`).
    """
    from datasets import load_dataset
    ds = load_dataset("ShenLab/MentalChat16K", split="train")
    n = 0
    for row in ds.select(range(min(limit, len(ds)))):
        ex = build_example(row.get("input"), row.get("output"))
        if ex:
            rows.append(ex)
            n += 1
    print(f"  MentalChat16K (capped at {limit}): {n} examples")


DATASET_LOADERS = {
    "counsel_chat": load_counsel_chat,
    "amod": load_amod,
    "esconv": load_esconv,
    "mentalchat16k": load_mentalchat16k,
}


def main(sources: list[str] | None = None):
    sources = sources or list(DATASET_LOADERS.keys())
    rows: list[dict] = []

    print(f"Building training set from: {', '.join(sources)}")
    for name in sources:
        loader = DATASET_LOADERS.get(name)
        if not loader:
            print(f"  skipping unknown source: {name}")
            continue
        try:
            loader(rows)
        except Exception as e:
            print(f"  WARNING: failed to load {name} ({e}) - skipping it")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for ex in rows:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(rows)} training examples to {OUT_PATH}")
    if len(rows) == 0:
        print("WARNING: zero examples written - check dataset loader errors above.")


if __name__ == "__main__":
    # Edit this list to pick which sources to include, e.g.:
    #   main(["counsel_chat", "esconv"])
    main()