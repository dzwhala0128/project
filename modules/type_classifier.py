from __future__ import annotations

import json
import re
from typing import Dict

from models.llm_wrapper import BaseLLM


_JSON_RE = re.compile(r"\{[\s\S]*?\}", re.UNICODE)


class QuestionTypeClassifier:
    """
    BELLE-style question type annotation.

    Types:
    - Inference: needs chaining facts across hops.
    - Comparison: compare two entities/attributes.
    - Temporal: ordering in time ("what happened first/after").
    - Null: cannot be categorized reminding above.
    """

    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.types = ["Inference", "Comparison", "Temporal", "Null"]

    def _extract_json(self, text: str) -> Dict:
        # find the first JSON object candidate
        m = _JSON_RE.search(text or "")
        if not m:
            return {}
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    def classify(self, question: str) -> str:
        prompt = f"""You are classifying a question for a multi-hop QA system.
Choose exactly one label from: Inference, Comparison, Temporal, Null.

Guidelines:
- Inference: requires multi-step reasoning / chained evidence.
- Comparison: asks to compare (bigger/smaller/more/less/etc.).
- Temporal: asks about before/after/first/earlier/later.
- Null: none of the above clearly applies.

Examples:
Q: "Who is the daughter of the author of the novel that won the 2018 Booker Prize?"
A: {{"type": "Inference"}}

Q: "Which is taller, the Eiffel Tower or the Statue of Liberty?"
A: {{"type": "Comparison"}}

Q: "What happened earlier: the signing of the Declaration of Independence or the first flight by the Wright brothers?"
A: {{"type": "Temporal"}}

Q: "What is the capital of France?"
A: {{"type": "Null"}}

Now classify:
Q: "{question}"
A:"""

        resp = self.llm.generate(prompt)
        data = self._extract_json(resp)
        t = (data.get("type") or "").strip()
        if t in self.types:
            return t

        # fallback: string match
        for cand in self.types:
            if cand.lower() in (resp or "").lower():
                return cand
        return "Null"
