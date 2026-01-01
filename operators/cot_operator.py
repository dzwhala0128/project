from __future__ import annotations

import re
from typing import Dict, Any

from .base_operator import BaseOperator
from models.llm_wrapper import BaseLLM
from prompts.operator_prompts import COT_PROMPT


_FINAL_RE = re.compile(r"Final Answer:\s*(.*)", re.IGNORECASE | re.DOTALL)


def _evidence_text(state: Dict[str, Any], max_chunks: int = 14) -> str:
    chunks = state.get("evidence_chunks", [])
    if not chunks:
        return "N/A"
    return "\n\n".join(chunks[-max_chunks:])


def _extract_final(text: str) -> str:
    m = _FINAL_RE.search(text or "")
    if m:
        return m.group(1).strip()
    # fallback: last non-empty line
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    return lines[-1] if lines else (text or "").strip()


class CoTOperator(BaseOperator):
    name = "cot"

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        question = state["question"]
        evidence = _evidence_text(state)

        raw = self.llm.generate(COT_PROMPT.format(evidence=evidence, question=question)).strip()
        answer = _extract_final(raw)

        state["final_answer"] = answer
        return {"operator": self.name, "answer": answer, "raw": raw}
