"""
strategy_selector.py
---------------------
The RL component: a contextual bandit that picks which counseling
response STRATEGY to use for a given student context (emotion +
stress type), learns from thumbs-up/down feedback, and gets better at
matching strategy to context over time.

Algorithm: LinUCB (Li et al., 2010) - a standard, well-understood
contextual bandit. Chosen deliberately over full RL (e.g. PPO) because:
  - The action space is small and fixed (4 safe, pre-written strategies -
    see STRATEGIES below). The agent picks WHICH one to use, it never
    generates or modifies response content itself.
  - LinUCB has closed-form updates (no gradient descent, no training
    loop, no GPU) - a handful of matrix operations per request.
  - It's easy to explain and evaluate: at any point you can print out
    each strategy's learned preference per context and see exactly
    why a choice was made, which matters for a project that needs to
    be defensible to reviewers, not just work.

Context features: one-hot emotion + one-hot stress_type (see
CONTEXT_DIM below). Reward: +1 for thumbs-up feedback, -1 for
thumbs-down. No reward is ever generated automatically - only real
student feedback updates the model (see record_feedback()).

Persistence: this module holds the bandit's state (per-strategy A and
b matrices) in memory. There is no separate save file - on server
startup, call warm_start(rows) once with every past (context, strategy,
reward) tuple from the StrategyLog table, and the exact same LinUCB
update is replayed in order. The database log is the single source of
truth; the in-memory bandit is a deterministic function of it.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np

# The four safe, pre-written strategies. This list is intentionally
# small and fixed - the bandit chooses AMONG these, it can never
# invent a new one. prompt_builder.py has one rule-block per strategy.
STRATEGIES: List[str] = [
    "validate_listen",
    "validate_ask",
    "validate_suggest",
    "validate_normalize",
]

EMOTIONS = ["anxious", "overwhelmed", "sad", "angry", "hopeful", "neutral"]
STRESS_TYPES = ["academic", "career", "interpersonal", "general"]

CONTEXT_DIM = len(EMOTIONS) + len(STRESS_TYPES)  # one-hot emotion + one-hot stress_type

ALPHA = 0.8  # exploration strength - higher = more willing to try under-tried strategies

# Per-strategy LinUCB state: A (CONTEXT_DIM x CONTEXT_DIM), b (CONTEXT_DIM,).
# Initialized fresh (A = identity, b = zero) = no preference yet, pure
# exploration, until warm_start() replays real history into it.
_A: Dict[str, np.ndarray] = {s: np.identity(CONTEXT_DIM) for s in STRATEGIES}
_b: Dict[str, np.ndarray] = {s: np.zeros(CONTEXT_DIM) for s in STRATEGIES}


def _context_vector(emotion: str, stress_type: str) -> np.ndarray:
    """One-hot encode (emotion, stress_type) into a CONTEXT_DIM vector."""
    x = np.zeros(CONTEXT_DIM)
    if emotion in EMOTIONS:
        x[EMOTIONS.index(emotion)] = 1.0
    else:
        x[EMOTIONS.index("neutral")] = 1.0
    offset = len(EMOTIONS)
    if stress_type in STRESS_TYPES:
        x[offset + STRESS_TYPES.index(stress_type)] = 1.0
    else:
        x[offset + STRESS_TYPES.index("general")] = 1.0
    return x


def select_strategy(emotion: str, stress_type: str) -> Tuple[str, Dict[str, float]]:
    """
    Picks a strategy for this context using LinUCB's upper-confidence-
    bound rule: score each strategy by (predicted reward + exploration
    bonus for how little it's been tried in similar contexts), pick
    the highest.

    Returns (chosen_strategy, scores_by_strategy) - the scores are
    returned too so callers can log/inspect them, e.g. for the
    evaluation writeup (does the chosen strategy match what a human
    rater would pick?).
    """
    x = _context_vector(emotion, stress_type)
    scores = {}
    for s in STRATEGIES:
        A_inv = np.linalg.inv(_A[s])
        theta = A_inv @ _b[s]
        predicted_reward = float(theta @ x)
        exploration_bonus = ALPHA * float(np.sqrt(x @ A_inv @ x))
        scores[s] = predicted_reward + exploration_bonus

    chosen = max(scores, key=scores.get)
    return chosen, scores


def record_feedback(emotion: str, stress_type: str, strategy: str, reward: float) -> None:
    """
    Updates the bandit's weights for `strategy` given the context and
    an observed reward (+1 thumbs-up, -1 thumbs-down - see
    routers/feedback.py). This is the ONLY way the bandit's state
    changes - no automatic/simulated rewards anywhere in this module.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy!r}. Must be one of {STRATEGIES}.")
    x = _context_vector(emotion, stress_type)
    _A[strategy] += np.outer(x, x)
    _b[strategy] += reward * x


def warm_start(rows: List[Tuple[str, str, str, float]]) -> int:
    """
    Rebuilds the bandit's in-memory state from historical
    (emotion, stress_type, strategy, reward) rows - call this ONCE at
    application startup with every StrategyLog row that already has a
    reward recorded, in chronological order.

    Returns the number of rows replayed.
    """
    global _A, _b
    _A = {s: np.identity(CONTEXT_DIM) for s in STRATEGIES}
    _b = {s: np.zeros(CONTEXT_DIM) for s in STRATEGIES}
    n = 0
    for emotion, stress_type, strategy, reward in rows:
        if strategy in STRATEGIES:
            record_feedback(emotion, stress_type, strategy, reward)
            n += 1
    return n


def get_current_preferences() -> Dict[str, Dict[str, float]]:
    """
    Debug/evaluation utility: for every (emotion, stress_type) context,
    shows which strategy the bandit currently prefers and its score.
    Useful for the paper's evaluation section - e.g. printing this
    before and after a batch of feedback to show the policy shifting.
    """
    result = {}
    for emotion in EMOTIONS:
        for stress_type in STRESS_TYPES:
            _, scores = select_strategy(emotion, stress_type)
            key = f"{emotion}/{stress_type}"
            result[key] = scores
    return result


if __name__ == "__main__":
    # Quick manual sanity check: python strategy_selector.py
    print("Initial (untrained) selection for anxious/academic:")
    strategy, scores = select_strategy("anxious", "academic")
    print(f"  chosen: {strategy}, scores: {scores}")

    print("\nSimulating 10 thumbs-up for 'validate_ask' on anxious/academic...")
    for _ in range(10):
        record_feedback("anxious", "academic", "validate_ask", reward=1.0)

    strategy, scores = select_strategy("anxious", "academic")
    print(f"After feedback, chosen: {strategy}, scores: {scores}")
    assert strategy == "validate_ask", "Bandit should now prefer validate_ask for this context"
    print("OK - bandit learned the preference as expected.")