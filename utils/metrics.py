"""
Evaluation metrics (EM, F1)
"""
def exact_match(pred: str, gold: str) -> float:
    return 1.0 if pred.strip().lower() == gold.strip().lower() else 0.0
