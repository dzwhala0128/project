from __future__ import annotations

import json
import re
from typing import Dict, Any, List

from .base_operator import BaseOperator
from models.llm_wrapper import BaseLLM
from prompts.operator_prompts import ITER_NEXT_QUERY_PROMPT


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


class IterativeStepOperator(BaseOperator):
    name = "iterative_step"

    def __init__(self, llm: BaseLLM, max_hops: int = 3):
        self.llm = llm
        self.max_hops = int(max_hops)

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        question = state["question"]
        retriever = state.get("retriever")
        per_hop_top_k = int(state.get("per_hop_top_k", 3))

        hops: List[Dict[str, Any]] = []

        for hop_idx in range(1, self.max_hops + 1):
            evidence = _evidence_text(state)
            prompt = ITER_NEXT_QUERY_PROMPT.format(evidence=evidence, question=question)
            raw = self.llm.generate(prompt)

            data = _extract_json(raw)
            next_query = str(data.get("next_query", "")).strip()
            stop = bool(data.get("stop", False))
            note = str(data.get("note", "")).strip()

            if stop or not next_query:
                hops.append({"hop": hop_idx, "stop": True, "note": note, "raw": raw})
                break

            docs = retriever.retrieve(next_query, top_k=per_hop_top_k) if retriever else []
            for d in docs:
                title = d.get("title", "") or ""
                state.setdefault("evidence_chunks", []).append(f"[Hop {hop_idx}] Query: {next_query}\nTitle: {title}\n{d.get('content','')}")
            state.setdefault("retrieval_log", []).append({"query": next_query, "docs": docs, "hop": hop_idx})
            state["current_query"] = next_query

            hops.append({
                "hop": hop_idx,
                "query": next_query,
                "note": note,
                "docs_count": len(docs),
                "raw": raw,
                "stop": False
            })

        state.setdefault("intermediate", {})["iter_hops"] = hops
        return {"operator": self.name, "hops": hops}
