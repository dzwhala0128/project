from __future__ import annotations

import json
import re
from typing import Dict, Any, List

from .base_operator import BaseOperator
from models.llm_wrapper import BaseLLM
from prompts.operator_prompts import SUBSTEP_PROMPT, SUBQ_ANSWER_PROMPT


_JSON_RE = re.compile(r"\{[\s\S]*\}", re.UNICODE)


def _extract_json(text: str) -> Dict[str, Any]:
    m = _JSON_RE.search(text or "")
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def _evidence_text(state: Dict[str, Any], max_chunks: int = 12) -> str:
    chunks = state.get("evidence_chunks", [])
    if not chunks:
        return "N/A"
    return "\n\n".join(chunks[-max_chunks:])


class SubstepOperator(BaseOperator):
    name = "substep"

    def __init__(self, llm: BaseLLM, max_subquestions: int = 3):
        self.llm = llm
        self.max_subquestions = int(max_subquestions)

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        question = state["question"]
        evidence = _evidence_text(state)

        prompt = SUBSTEP_PROMPT.format(
            evidence=evidence,
            question=question,
            max_subquestions=self.max_subquestions
        )
        raw = self.llm.generate(prompt)

        data = _extract_json(raw)
        subqs = data.get("subquestions", [])
        if isinstance(subqs, str):
            subqs = [subqs]
        if not isinstance(subqs, list):
            subqs = []
        subqs = [str(s).strip() for s in subqs if str(s).strip()][: self.max_subquestions]

        # Retrieval + answer per subq
        retriever = state.get("retriever")
        per_q_top_k = int(state.get("per_hop_top_k", 3))
        sub_answers: List[Dict[str, str]] = []

        for subq in subqs:
            docs = retriever.retrieve(subq, top_k=per_q_top_k) if retriever else []
            # update evidence
            for d in docs:
                title = d.get("title", "") or ""
                state.setdefault("evidence_chunks", []).append(f"[SubQ] {subq}\nTitle: {title}\n{d.get('content','')}")
            state.setdefault("retrieval_log", []).append({"query": subq, "docs": docs})

            # answer subq
            ev = _evidence_text(state)
            ans_prompt = SUBQ_ANSWER_PROMPT.format(evidence=ev, subq=subq)
            ans = self.llm.generate(ans_prompt).strip()
            sub_answers.append({"subquestion": subq, "answer": ans})

        state.setdefault("intermediate", {})["subquestions"] = subqs
        state.setdefault("intermediate", {})["sub_answers"] = sub_answers

        return {
            "operator": self.name,
            "subquestions": subqs,
            "sub_answers": sub_answers,
            "raw": raw
        }
