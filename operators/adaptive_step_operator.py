from __future__ import annotations

from typing import Dict, Any

from .base_operator import BaseOperator


class AdaptiveStepOperator(BaseOperator):
    """
    A lightweight adapter that chooses the next operator based on question type.

    The executor will read `chosen_operator` and run it immediately.
    """
    name = "adaptive_step"

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        qtype = (state.get("qtype") or "Null").lower()

        if "comparison" in qtype:
            chosen = "substep"
        elif "temporal" in qtype:
            chosen = "iterative_step"
        elif "inference" in qtype:
            chosen = "iterative_step"
        else:
            chosen = "cot"

        return {"operator": self.name, "chosen_operator": chosen, "reason": f"heuristic by qtype={state.get('qtype')}"}
