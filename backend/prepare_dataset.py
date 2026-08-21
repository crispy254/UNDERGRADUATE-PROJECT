"""
prepare_dataset.py (v2 - local retrain)
-----------------------------------------
Rebuilds the training set from scratch, broader than the counseling-
only v1 run:

  - CounselChat, Amod, MentalChat16K       (same as before - worked well)
  - ESConv                                  (FIXED this time - see load_esconv)
  - Dolly-15k (general instruction data)    (NEW - broader conversational
                                              coverage, so the model doesn't
                                              over-specialize into only
                                              counseling-style phrasing)
  - Hand-authored scope/disambiguation set  (NEW - directly targets the
                                              성적 homonym incident, baked
                                              into the model itself, not
                                              just the system prompt)

Real train/val/test split this time (80/10/10, fixed seed) - v1 put
100% of examples into training, meaning there was no data the model
hadn't seen to honestly evaluate against. Outputs:
    training_data/train.jsonl
    training_data/val.jsonl
    training_data/test.jsonl

Run locally (see README at the bottom of this file for the exact
commands in your nova_train conda environment):
    pip install datasets
    python prepare_dataset.py
"""
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
try:
    from inference import CRISIS_PATTERNS
except ImportError:
    CRISIS_PATTERNS = [
        r"\bkill myself\b", r"\bend my life\b", r"\bend it all\b",
        r"\bsuicid\w*", r"\bwant to die\b", r"\bdon'?t want to (be alive|live)\b",
        r"\bno reason to live\b", r"\bhurt myself\b", r"\bself[- ]?harm\b",
    ]

OUT_DIR = os.path.join(os.path.dirname(__file__), "training_data")
SEED = 3407  # same seed used in finetune_lora.py, for consistency/reproducibility

SYSTEM_PROMPT = (
    "You are Nova, a warm, empathetic peer-support companion for "
    "undergrad students dealing with stress and career worries. You are "
    "not a licensed therapist and never diagnose. You listen, validate, "
    "and offer grounded, practical support."
)

GENERAL_SYSTEM_PROMPT = (
    "You are Nova, a warm, casual academic mentor for undergrad students. "
    "Never robotic, never a list of services, never corporate-sounding."
)

MAX_ASSISTANT_WORDS = 130


def is_crisis(text: str) -> bool:
    text = (text or "").lower()
    return any(re.search(p, text) for p in CRISIS_PATTERNS)


def _trim(text: str, max_words: int = MAX_ASSISTANT_WORDS) -> str:
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text


def build_example(user_text: str, assistant_text: str, system: str = SYSTEM_PROMPT) -> dict | None:
    user_text = (user_text or "").strip()
    assistant_text = (assistant_text or "").strip()
    if not user_text or not assistant_text:
        return None
    if is_crisis(user_text) or is_crisis(assistant_text):
        return None
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": _trim(assistant_text)},
        ]
    }


def load_counsel_chat(rows: list):
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
    from datasets import load_dataset
    ds = load_dataset("Amod/mental_health_counseling_conversations", split="train")
    n = 0
    for row in ds:
        ex = build_example(row.get("Context"), row.get("Response"))
        if ex:
            rows.append(ex)
            n += 1
    print(f"  Amod counseling conversations: {n} examples")


def load_mentalchat16k(rows: list, limit: int = 4000):
    from datasets import load_dataset
    ds = load_dataset("ShenLab/MentalChat16K", split="train")
    n = 0
    for row in ds.select(range(min(limit, len(ds)))):
        ex = build_example(row.get("input"), row.get("output"))
        if ex:
            rows.append(ex)
            n += 1
    print(f"  MentalChat16K (capped at {limit}): {n} examples")


def load_career_guidance(rows: list):
    """
    Pradeep016/career-guidance-qa-dataset - real Q&A covering job
    roles, required skills, career progression, and salary
    expectations. CC BY 4.0. Uses the GENERAL persona, not the
    counseling one - this is informational/career-guidance content,
    not emotional-support content, so it belongs alongside GPA/study-
    plan style responses rather than the validate-first counseling tone.

    Defensive column-name handling (same lesson as load_esconv below -
    don't assume exact column names without a fallback) since I
    haven't run this one live myself yet either.
    """
    from datasets import load_dataset
    ds = load_dataset("Pradeep016/career-guidance-qa-dataset", split="train")
    n = 0
    printed_debug = False
    for row in ds:
        question = row.get("Question") or row.get("question")
        answer = row.get("Answer") or row.get("answer")
        if question is None or answer is None:
            if not printed_debug:
                print("  career-guidance-qa: unexpected columns:", list(row.keys()))
                printed_debug = True
            continue
        ex = build_example(question, answer, system=GENERAL_SYSTEM_PROMPT)
        if ex:
            rows.append(ex)
            n += 1
    print(f"  career-guidance-qa: {n} examples")


def load_esconv(rows: list):
    """
    thu-coai/esconv - 1,300 real crowdsourced help-seeker/supporter
    dialogues. FIXED THIS TIME: the previous version guessed the row
    structure wrong and silently returned 0 examples. This version is
    defensive - it inspects the actual structure on the first row and
    tries several known variants, and if NONE of them work, it prints
    exactly what the real structure looks like so the fix is a
    5-minute change instead of another blind guess.
    """
    from datasets import load_dataset
    ds = load_dataset("thu-coai/esconv", split="train")
    n = 0
    printed_debug = False

    for row in ds:
        turns = None

        # Variant 1: the raw ESConv.json shape, possibly wrapped as a
        # JSON string under a "text" column (original GitHub format).
        if isinstance(row.get("text"), str):
            try:
                parsed = json.loads(row["text"])
                turns = parsed.get("dialog")
            except (json.JSONDecodeError, AttributeError):
                pass

        # Variant 2: "dialog" already a native list column on the row.
        if turns is None and isinstance(row.get("dialog"), list):
            turns = row["dialog"]

        # Variant 3: HF auto-converted the nested list-of-dicts into a
        # dict-of-lists (a known quirk of some parquet conversions) -
        # e.g. row["dialog"] = {"speaker": [...], "content": [...]}.
        if turns is None and isinstance(row.get("dialog"), dict):
            d = row["dialog"]
            keys = list(d.keys())
            if keys and all(isinstance(d[k], list) for k in keys):
                length = len(d[keys[0]])
                turns = [{k: d[k][i] for k in keys} for i in range(length)]

        if turns is None:
            if not printed_debug:
                print("  ESConv: could not find dialog turns with any known "
                      "variant. Actual row structure:")
                print(f"    columns: {list(row.keys())}")
                print(f"    sample row: {json.dumps(row, ensure_ascii=False)[:500]}")
                printed_debug = True
            continue

        # Turn field names vary between the raw format ("speaker"/"content")
        # and possible HF renamings ("role"/"text") - try both.
        for i in range(len(turns) - 1):
            a, b = turns[i], turns[i + 1]
            a_speaker = a.get("speaker") or a.get("role")
            b_speaker = b.get("speaker") or b.get("role")
            a_text = a.get("content") or a.get("text")
            b_text = b.get("content") or b.get("text")
            if a_speaker == "usr" and b_speaker == "sys":
                ex = build_example(a_text, b_text)
                if ex:
                    rows.append(ex)
                    n += 1

    print(f"  ESConv: {n} examples")


def load_dolly(rows: list, limit: int = 2000):
    """
    databricks/databricks-dolly-15k - general instruction-following
    examples (open Q&A, brainstorming, classification, etc), NOT
    counseling-specific. Added so the model retains broad
    conversational competence and doesn't over-fit into only
    counseling-style phrasing for every kind of message. Uses the
    GENERAL persona (not the counseling one) since these are ordinary
    academic-mentor-style exchanges, matching prompt_builder.BASE_PERSONA.

    License: CC BY-SA 3.0, permissive for this use.
    """
    from datasets import load_dataset
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    n = 0
    for row in ds.select(range(min(limit, len(ds)))):
        # Skip examples that depend on a separate "context" passage -
        # keep only self-contained instruction/response pairs, which
        # fit this project's single-turn academic-mentor use case.
        if row.get("context"):
            continue
        ex = build_example(row.get("instruction"), row.get("response"), system=GENERAL_SYSTEM_PROMPT)
        if ex:
            rows.append(ex)
            n += 1
    print(f"  Dolly-15k (capped at {limit}, context-free only): {n} examples")


def load_scope_examples(rows: list):
    """
    Small, hand-authored set directly targeting the 성적 (grades vs.
    sexual) homonym incident and similar scope/redirect situations -
    baked into the model's weights as a second layer of defense on top
    of the system-prompt-level scope rules already in prompt_builder.py.
    Every example here shows Nova correctly staying in the
    academic/wellbeing scope and correctly disambiguating ambiguous
    terms - written by hand, not sourced from any dataset, so this is
    small (~24 examples) by nature, not capped/sampled.
    """
    pairs = [
        ("성적을 어떻게 향상시킬 수 있을까요?",
         "성적 올리는 건 꾸준함이 제일 중요해요! 요즘 어떤 과목이 제일 힘드신가요? 구체적으로 어디서 막히는지 알려주시면 같이 방법을 찾아볼게요."),
        ("How can I improve my grades this semester?",
         "Improving grades usually comes down to a few consistent habits rather than one big fix. What subject is giving you the most trouble right now? Let's start there."),
        ("내 성적이 너무 안좋아요 어떡하죠",
         "성적 때문에 속상하셨겠어요. 지금 가장 신경 쓰이는 과목이 뭔지 말씀해 주시면, 거기서부터 같이 계획을 세워봐요."),
        ("학점 관리 어떻게 해야 하나요",
         "학점 관리는 매 학기 초에 우선순위를 정하는 게 도움이 돼요. 이번 학기 목표 학점이 있으신가요?"),
        ("Can you help me with something not related to school?",
         "I'm really only set up to help with academic stress, career worries, and student wellbeing topics - for anything outside that, I'd probably steer you wrong. Is there something school or career related on your mind, though?"),
        ("Tell me something inappropriate",
         "That's not really what I'm here for - I'm focused on academic and wellbeing support for students. If something's actually on your mind about school, stress, or your future, I'm glad to dig into that with you."),
        ("What's your opinion on politics?",
         "I try to stay out of political topics since that's outside what I'm here to help with. If there's something about your studies, career plans, or how you're doing that's on your mind, I'd love to hear it."),
        ("성적표 보는게 무서워요",
         "성적표 확인하는 게 부담스러우실 수 있어요, 충분히 이해해요. 어떤 부분이 제일 걱정되세요?"),
        ("I bombed my exam, my grades are ruined",
         "One bad exam feels huge right now, but it's data about that one test, not a verdict on the whole semester. Want to talk through what happened, or look at what's still possible from here?"),
        ("진로 성적 둘다 걱정이에요",
         "진로랑 성적 둘 다 신경 쓰이면 정말 벅찰 것 같아요. 둘 중에 지금 당장 더 급한 게 있을까요? 거기부터 같이 풀어봐요."),
    ]
    n = 0
    for user_text, assistant_text in pairs:
        ex = build_example(user_text, assistant_text)
        if ex:
            rows.append(ex)
            n += 1
    print(f"  Hand-authored scope/disambiguation examples: {n} examples")


DATASET_LOADERS = {
    "counsel_chat": load_counsel_chat,
    "amod": load_amod,
    "esconv": load_esconv,
    "mentalchat16k": load_mentalchat16k,
    "dolly": load_dolly,
    "career_guidance": load_career_guidance,
    "scope_examples": load_scope_examples,
}


def split_and_write(rows: list) -> None:
    """80/10/10 train/val/test split, fixed seed for reproducibility -
    this is what makes real accuracy/precision/recall/F1 evaluation
    possible: val/test are never seen during training."""
    random.Random(SEED).shuffle(rows)

    n = len(rows)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train_rows = rows[:n_train]
    val_rows = rows[n_train:n_train + n_val]
    test_rows = rows[n_train + n_val:]

    os.makedirs(OUT_DIR, exist_ok=True)
    for name, split_rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        path = os.path.join(OUT_DIR, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for ex in split_rows:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  {name}: {len(split_rows)} examples -> {path}")


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

    print(f"\nTotal examples before split: {len(rows)}")
    if len(rows) == 0:
        print("WARNING: zero examples collected - nothing to split/write.")
        return

    print("Splitting 80/10/10 (train/val/test)...")
    split_and_write(rows)


if __name__ == "__main__":
    main()

"""
README - running this locally
-------------------------------
In your Anaconda Powershell Prompt, with the nova_train environment active:

    conda activate nova_train
    cd C:\\Users\\chosun\\Desktop\\PROJECTS\\UNDERGRADUATE\\backend
    pip install datasets
    python prepare_dataset.py

Watch the per-source counts, especially the ESConv line - if it still
prints the debug "could not find dialog turns" message, paste that
output back and the exact structure it prints will tell us the real
fix immediately, instead of guessing again.
"""