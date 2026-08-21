"""
evaluate_detection.py
-----------------------
Computes accuracy, precision, recall, and F1-score for inference.py's
emotion, stress-type, and crisis detection, against a labeled test set
with human-assigned ground truth.

This is separate from finetune_lora.py's held-out test.jsonl - that
set evaluates the GENERATIVE model (does it produce good replies).
This script evaluates the RULE-BASED detection layer (does it
correctly classify emotion/stress_type/crisis) - a different system,
a different kind of test.

Usage:
    python evaluate_detection.py labeled_test_set.json

Input format (see labeled_test_set_starter.json for a real, runnable
starter set):
    [
      {"message": "...", "true_emotion": "anxious",
       "true_stress_type": "academic", "true_crisis": false},
      ...
    ]

For the crisis column specifically, recall is reported as the primary
number to watch - see the printed note at the end explaining why.
"""
import json
import sys
from collections import defaultdict

from sklearn.metrics import classification_report, precision_recall_fscore_support

import inference


def evaluate(test_set_path: str):
    with open(test_set_path, "r", encoding="utf-8") as f:
        test_set = json.load(f)

    if not test_set:
        print("Test set is empty - nothing to evaluate.")
        return

    true_emotion, pred_emotion = [], []
    true_stress, pred_stress = [], []
    true_crisis, pred_crisis = [], []

    for item in test_set:
        message = item["message"]
        result = inference.infer_emotional_state(message)
        crisis_pred = inference.check_crisis_risk(message)

        true_emotion.append(item["true_emotion"])
        pred_emotion.append(result["emotion"])

        true_stress.append(item["true_stress_type"])
        pred_stress.append(result["stress_type"])

        true_crisis.append(bool(item["true_crisis"]))
        pred_crisis.append(bool(crisis_pred))

    print("=" * 70)
    print(f"Evaluated on {len(test_set)} labeled examples")
    print("=" * 70)

    print("\n--- EMOTION DETECTION ---")
    print(classification_report(true_emotion, pred_emotion, zero_division=0))

    print("\n--- STRESS-TYPE DETECTION ---")
    print(classification_report(true_stress, pred_stress, zero_division=0))

    print("\n--- CRISIS DETECTION ---")
    print(classification_report(true_crisis, pred_crisis, zero_division=0,
                                 target_names=["not_crisis", "crisis"]))

    # Crisis recall specifically, called out on its own - this is THE
    # number that matters most for this class. A missed crisis message
    # (false negative) is far more serious than a false alarm.
    p, r, f, _ = precision_recall_fscore_support(
        true_crisis, pred_crisis, labels=[True], zero_division=0
    )
    print(f"*** Crisis-class recall: {r[0]:.3f} "
          f"({'GOOD - catching real crisis messages' if r[0] >= 0.9 else 'REVIEW NEEDED - missing real crisis messages'}) ***")

    # Mismatches, printed individually - useful for actually reading
    # WHICH messages the system got wrong, not just the aggregate score.
    print("\n--- MISMATCHES (for manual review) ---")
    any_mismatch = False
    for item, pe, ps, pc in zip(test_set, pred_emotion, pred_stress, pred_crisis):
        wrong = []
        if pe != item["true_emotion"]:
            wrong.append(f"emotion: predicted={pe}, true={item['true_emotion']}")
        if ps != item["true_stress_type"]:
            wrong.append(f"stress_type: predicted={ps}, true={item['true_stress_type']}")
        if pc != bool(item["true_crisis"]):
            wrong.append(f"CRISIS: predicted={pc}, true={item['true_crisis']}")
        if wrong:
            any_mismatch = True
            print(f"  \"{item['message'][:60]}...\" -> {'; '.join(wrong)}")
    if not any_mismatch:
        print("  (none)")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "labeled_test_set_starter.json"
    evaluate(path)