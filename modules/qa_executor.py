from __future__ import annotations

from typing import Dict, List, Any, Optional

from models.llm_wrapper import BaseLLM
from retrieval.retriever import WikipediaRetriever

from prompts.operator_prompts import FINAL_SYNTHESIS_PROMPT, COT_PROMPT

from operators import (
    CoTOperator,
    SingleStepOperator,
    IterativeStepOperator,
    SubstepOperator,
    AdaptiveStepOperator,
)

import re
_FINAL_RE = re.compile(r"Final Answer:\s*(.*)", re.IGNORECASE | re.DOTALL)

def _extract_final(text: str) -> str:
    m = _FINAL_RE.search(text or "")
    if m:
        return m.group(1).strip()
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    return lines[-1] if lines else (text or "").strip()

def _evidence_text(state: Dict[str, Any], max_chunks: int = 16) -> str:
    chunks = state.get("evidence_chunks", [])
    if not chunks:
        return "N/A"
    return "\n\n".join(chunks[-max_chunks:])


class MultiHopQAExecutor:
    """
    Multi-hop QA executor that:
    - maintains a shared state
    - supports multi-hop retrieval (iterative_step) and decomposition (substep)
    - synthesizes a final answer at the end
    """

    def __init__(self, llm: BaseLLM, retriever: Optional[WikipediaRetriever] = None, executor_cfg: Optional[Dict[str, Any]] = None):
        self.llm = llm
        self.retriever = retriever
        self.cfg = executor_cfg or {}

        self.max_hops = int(self.cfg.get("max_hops", 3))
        self.per_hop_top_k = int(self.cfg.get("per_hop_top_k", 3))
        self.max_subquestions = int(self.cfg.get("max_subquestions", 3))
        self.max_evidence_chunks = int(self.cfg.get("max_evidence_chunks", 40))

        # operator registry
        self.operators = {
            "cot": CoTOperator(llm),
            "single_step": SingleStepOperator(llm),
            "iterative_step": IterativeStepOperator(llm, max_hops=self.max_hops),
            "substep": SubstepOperator(llm, max_subquestions=self.max_subquestions),
            "adaptive_step": AdaptiveStepOperator(),
        }

    def execute(self, question: str, qtype: str, operator_plan: List[str]) -> Dict[str, Any]:
        # Initialize state
        state: Dict[str, Any] = {
            "question": question,
            "qtype": qtype,
            "current_query": question,
            "retriever": self.retriever,
            "per_hop_top_k": self.per_hop_top_k,
            "evidence_chunks": [],
            "retrieval_log": [],
            "intermediate": {},
        }

        # Seed evidence with an initial retrieval (helps stabilize later steps)
        self._seed_retrieval(state)

        trace: List[Dict[str, Any]] = []

        for op_name in operator_plan:
            op = self.operators.get(op_name)
            if not op:
                trace.append({"operator": op_name, "warning": "unknown operator, skipped"})
                continue

            out = op.execute(state)
            trace.append(out)

            # If adaptive_step chooses another operator, run it immediately
            if op_name == "adaptive_step" and isinstance(out, dict) and out.get("chosen_operator"):
                chosen = out["chosen_operator"]
                chosen_op = self.operators.get(chosen)
                if chosen_op:
                    out2 = chosen_op.execute(state)
                    out2 = {"operator": f"adaptive-> {chosen}", **out2}
                    trace.append(out2)

            # cap evidence size to avoid token explosion
            if len(state["evidence_chunks"]) > self.max_evidence_chunks:
                state["evidence_chunks"] = state["evidence_chunks"][-self.max_evidence_chunks:]

        # Final synthesis if not produced
        #print("[DEBUG] evidence head:", _evidence_text(state)[:400].replace("\n", " "))#tmp临时验证
        final_answer = state.get("final_answer")
        if not final_answer:
            ev = _evidence_text(state)
            raw_final = self.llm.generate(COT_PROMPT.format(evidence=ev, question=question)).strip()
            final_answer = _extract_final(raw_final)
            state["final_raw"] = raw_final  # 可选：方便你debug看模型到底说了啥

        return {
            "execution_trace": trace,
            "retrieval_log": state.get("retrieval_log", []),
            "evidence": _evidence_text(state),
            "final_answer": final_answer,
        }

    def _seed_retrieval(self, state: Dict[str, Any]) -> None:
        if not self.retriever:
            return
        docs = self.retriever.retrieve(state["question"], top_k=self.per_hop_top_k)
        for d in docs:
            title = d.get("title", "") or ""
            state["evidence_chunks"].append(f"[Seed] Title: {title}\n{d.get('content','')}")
        state["retrieval_log"].append({"query": state["question"], "docs": docs, "hop": 0})
