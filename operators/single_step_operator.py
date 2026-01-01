from __future__ import annotations

from typing import Dict, Any

from .base_operator import BaseOperator
from models.llm_wrapper import BaseLLM
from prompts.operator_prompts import SINGLE_STEP_PROMPT


def _evidence_text(state: Dict[str, Any], max_chunks: int = 14) -> str:
    chunks = state.get("evidence_chunks", [])
    if not chunks:
        return "N/A"
    return "\n\n".join(chunks[-max_chunks:])


class SingleStepOperator(BaseOperator):
    name = "single_step"

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        question = state["question"]
        evidence = _evidence_text(state)

        raw = self.llm.generate(SINGLE_STEP_PROMPT.format(evidence=evidence, question=question)).strip()
        state["final_answer"] = raw
        return {"operator": self.name, "answer": raw, "raw": raw}
