from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import json
import re

from models.llm_wrapper import BaseLLM
from prompts.debate_prompts import (
    AFFIRMATIVE_PROMPT,
    NEGATIVE_PROMPT,
    FAST_REVIEW_PROMPT,
    SLOW_REVIEW_PROMPT,
    JUDGE_PROMPT,
)

from prompts.meta_prompts import META_PROMPT_TEMPLATE, OPERATOR_POOL_DESCRIPTION, DEBATE_LEVELS


_JSON_RE = re.compile(r"\{[\s\S]*\}", re.UNICODE)


@dataclass
class DebateRound:
    round_num: int
    affirmative_raw: str = ""
    negative_raw: str = ""
    fast_raw: str = ""
    slow_raw: str = ""
    judge_raw: str = ""

    affirmative: Dict[str, Any] = field(default_factory=dict)
    negative: Dict[str, Any] = field(default_factory=dict)
    fast: Dict[str, Any] = field(default_factory=dict)
    slow: Dict[str, Any] = field(default_factory=dict)
    judge: Dict[str, Any] = field(default_factory=dict)


class BiLevelMultiAgentDebate:
    """
    Bi-level Multi-Agent Debate for producing an operator plan.

    Roles:
    - affirmative / negative (debate)
    - fast / slow (bi-level review)
    - judge (hard/soft convergence)
    """

    def __init__(
        self,
        llm: BaseLLM,
        operators: List[str],
        max_rounds: int = 2,
        mode: str = "hard",
        score_threshold: float = 0.85,
        max_plan_len: int = 3,
        debate_level: int = 2,
    ):
        self.llm = llm
        self.operators = operators
        self.max_rounds = int(max_rounds)
        self.mode = str(mode).lower()
        self.score_threshold = float(score_threshold)
        self.max_plan_len = int(max_plan_len)
        self.debate_level = int(debate_level)
        self.history: List[DebateRound] = []

    def debate(self, question: str, qtype: str) -> Dict[str, Any]:
        self.history = []
        scored_candidates: List[Dict[str, Any]] = []

        for round_num in range(1, self.max_rounds + 1):
            rd = DebateRound(round_num)

            # 1) Affirmative 232323
            aff_prompt = self._prompt_affirmative(question, qtype)  # ✅
            rd.affirmative_raw = self.llm.generate(aff_prompt)
            rd.affirmative = self._parse_plan_json(rd.affirmative_raw)

            # 2) Negative
            neg_prompt = self._prompt_negative(question, qtype, rd.affirmative)  # ✅
            rd.negative_raw = self.llm.generate(neg_prompt)
            rd.negative = self._parse_plan_json(rd.negative_raw)

            # 3) Fast reviewer
            fast_prompt = self._prompt_fast(question, qtype, rd.affirmative, rd.negative)  # ✅
            rd.fast_raw = self.llm.generate(fast_prompt)
            rd.fast = self._parse_review_json(rd.fast_raw)

            # 4) Slow reviewer
            slow_prompt = self._prompt_slow(question, qtype, rd.affirmative, rd.negative, rd.fast)  # ✅
            rd.slow_raw = self.llm.generate(slow_prompt)
            rd.slow = self._parse_review_json(rd.slow_raw)

            # 5) Judge 11200000
            judge_prompt = self._prompt_judge(question, qtype, rd.affirmative, rd.negative, rd.fast, rd.slow)  # ✅
            rd.judge_raw = self.llm.generate(judge_prompt)
            rd.judge = self._parse_judge_json(rd.judge_raw)

            self.history.append(rd)

            best = rd.judge.get("best_plan") or rd.slow.get("best_plan") or rd.fast.get("best_plan") or rd.affirmative.get("operators")
            best = self._normalize_ops(best)
            if best:
                scored_candidates.append({
                    "round": round_num,
                    "plan": best,
                    "score": float(rd.judge.get("score", rd.slow.get("score", rd.fast.get("score", 0.0)))),
                    "reason": rd.judge.get("reason") or rd.slow.get("reason") or rd.fast.get("reason") or ""
                })

            if self.mode == "hard":
                if self._is_converged():
                    final = self._finalize()
                    final["debate_history"] = self._history_brief()
                    return final
#1223
        # Not converged
        if self.mode == "soft" and scored_candidates:
            scored_candidates.sort(key=lambda x: x["score"], reverse=True)
            top = scored_candidates[0]
            return {
                "operator_plan": top["plan"],
                "reasoning": f"[soft-best] score={top['score']:.2f} {top.get('reason','')}",
                "debate_history": self._history_brief()
            }

        final = self._finalize()
        final["debate_history"] = self._history_brief()
        return final

    # -------------------------
    # Convergence
    # -------------------------
    def _is_converged(self) -> bool:
        if not self.history:
            return False
        last = self.history[-1]

        # Judge may explicitly mark converged
        if bool(last.judge.get("converged", False)):
            return True

        # High confidence (judge or slow)
        if float(last.judge.get("score", 0.0)) >= self.score_threshold:
            return True
        if float(last.slow.get("score", 0.0)) >= self.score_threshold:
            return True

        # Stable plan over 2 rounds
        if len(self.history) >= 2:
            prev = self.history[-2]
            lp = self._normalize_ops(last.judge.get("best_plan") or last.slow.get("best_plan") or last.fast.get("best_plan") or last.affirmative.get("operators"))
            pp = self._normalize_ops(prev.judge.get("best_plan") or prev.slow.get("best_plan") or prev.fast.get("best_plan") or prev.affirmative.get("operators"))
            if lp and pp and lp == pp:
                return True

        # Agreeing debaters is strong evidence
        aff = self._normalize_ops(last.affirmative.get("operators"))
        neg = self._normalize_ops(last.negative.get("operators"))
        if aff and neg and aff == neg:
            return True

#连续两个证据才收敛，判断是哪两个证据
        # print("[CONV]",
        #       "judge_conv=", last.judge.get("converged", False),
        #       "judge_score=", last.judge.get("score", None),
        #       "slow_score=", last.slow.get("score", None),
        #       "stable2=", stable_2rounds,
        #       "aff==neg=", aff == neg)

        return False

    def _finalize(self) -> Dict[str, Any]:
        if not self.history:
            return {"operator_plan": ["cot"], "reasoning": "empty debate history fallback"}

        last = self.history[-1]
        plan = (last.judge.get("best_plan")
                or last.slow.get("best_plan")
                or last.fast.get("best_plan")
                or last.affirmative.get("operators")
                or ["cot"])

        plan = self._normalize_ops(plan) or (["cot"] if "cot" in self.operators else self.operators[:1])
        reason = last.judge.get("reason") or last.slow.get("reason") or last.fast.get("reason") or last.affirmative.get("reason") or "fallback"
        return {"operator_plan": plan, "reasoning": reason}

    # -------------------------
    # Prompts
    # -------------------------
    
    def _meta_prompt(self, qtype: str) -> str:
        lvl = DEBATE_LEVELS.get(self.debate_level, DEBATE_LEVELS.get(2))
        return META_PROMPT_TEMPLATE.format(
            operator_pool=OPERATOR_POOL_DESCRIPTION,
            qtype=qtype,
            max_rounds=self.max_rounds,
            debate_level=lvl,
        )

    def _prev_fast(self) -> Dict[str, Any]:
        return self.history[-1].fast if self.history else {}

    def _prev_slow(self) -> Dict[str, Any]:
        return self.history[-1].slow if self.history else {}

    def _history_aff(self) -> List[Dict[str, Any]]:
        return [r.affirmative for r in self.history[-3:]]

    def _history_neg(self) -> List[Dict[str, Any]]:
        return [r.negative for r in self.history[-3:]]

    def _history_slow(self) -> List[Dict[str, Any]]:
        # keep some "memory" shows up in slow reviewer JSON if present
        return [r.slow for r in self.history[-3:]]

    def _prompt_affirmative(self, question: str, qtype: str) -> str:
        return AFFIRMATIVE_PROMPT.format(
            meta=self._meta_prompt(qtype),
            history_ad=json.dumps(self._history_aff(), ensure_ascii=False),
            prev_fast=json.dumps(self._prev_fast(), ensure_ascii=False),
            prev_slow=json.dumps(self._prev_slow(), ensure_ascii=False),
            question=question,
            operators=self.operators,
            max_plan_len=self.max_plan_len,
        )

    def _prompt_negative(self, question: str, qtype: str, affirmative: Dict[str, Any]) -> str:
        return NEGATIVE_PROMPT.format(
            meta=self._meta_prompt(qtype),
            history_nd=json.dumps(self._history_neg(), ensure_ascii=False),
            prev_fast=json.dumps(self._prev_fast(), ensure_ascii=False),
            prev_slow=json.dumps(self._prev_slow(), ensure_ascii=False),
            question=question,
            operators=self.operators,
            max_plan_len=self.max_plan_len,
            affirmative=json.dumps(affirmative, ensure_ascii=False),
        )

    def _prompt_fast(self, question: str, qtype: str, affirmative: Dict[str, Any], negative: Dict[str, Any]) -> str:
        return FAST_REVIEW_PROMPT.format(
            meta=self._meta_prompt(qtype),
            question=question,
            operators=self.operators,
            affirmative=json.dumps(affirmative, ensure_ascii=False),
            negative=json.dumps(negative, ensure_ascii=False),
        )

    def _prompt_slow(self, question: str, qtype: str, affirmative: Dict[str, Any], negative: Dict[str, Any], fast: Dict[str, Any]) -> str:
        return SLOW_REVIEW_PROMPT.format(
            meta=self._meta_prompt(qtype),
            history_slow=json.dumps(self._history_slow(), ensure_ascii=False),
            question=question,
            operators=self.operators,
            affirmative=json.dumps(affirmative, ensure_ascii=False),
            negative=json.dumps(negative, ensure_ascii=False),
            fast=json.dumps(fast, ensure_ascii=False),
        )

    def _prompt_judge(
        self,
        question: str,
        qtype: str,
        affirmative: Dict[str, Any],
        negative: Dict[str, Any],
        fast: Dict[str, Any],
        slow: Dict[str, Any],
    ) -> str:
        return JUDGE_PROMPT.format(
            meta=self._meta_prompt(qtype),
            mode=self.mode,
            question=question,
            operators=self.operators,
            affirmative=json.dumps(affirmative, ensure_ascii=False),
            negative=json.dumps(negative, ensure_ascii=False),
            fast=json.dumps(fast, ensure_ascii=False),
            slow=json.dumps(slow, ensure_ascii=False),
        )

    # -------------------------
    # Parsing helpers
    # -------------------------

    # --- backward-compatible aliases ---
    def _parse_plan_json(self, text: str):
        return self._parse_plan(text)

    def _parse_review_json(self, text: str):
        return self._parse_review(text)

    def _parse_judge_json(self, text: str):
        return self._parse_judge(text)

    def _extract_json_obj(self, text: str) -> Optional[str]:
        # Try multiple JSON candidates reminded the *first* dict that parses.
        for m in _JSON_RE.finditer(text or ""):
            s = m.group(0)
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    return s
            except Exception:
                continue
        return None

    def _safe_load(self, s: str) -> Dict[str, Any]:
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    def _normalize_ops(self, ops: Any) -> List[str]:
        if not ops:
            return []
        if isinstance(ops, str):
            ops = [ops]
        if not isinstance(ops, list):
            return []
        cleaned = [op for op in ops if isinstance(op, str) and op in self.operators]
        return cleaned[: self.max_plan_len]

    def _parse_plan(self, text: str, key: str = "operators") -> Dict[str, Any]:
        js = self._extract_json_obj(text)
        data = self._safe_load(js or "")
        ops = self._normalize_ops(data.get(key))
        if not ops:
            # safe fallback
            ops = ["substep", "iterative_step"] if ("substep" in self.operators and "iterative_step" in self.operators) else (self.operators[:1] or ["cot"])
        return {"operators": ops, "reason": data.get("reason", ""), "critique": data.get("critique", "")}

    def _parse_review(self, text: str) -> Dict[str, Any]:
        js = self._extract_json_obj(text)
        data = self._safe_load(js or "")
        best = self._normalize_ops(data.get("best_plan"))
        score = data.get("score", 0.0)
        try:
            score = float(score)
        except Exception:
            score = 0.0
        return {"best_plan": best, "score": max(0.0, min(1.0, score)), "reason": data.get("reason", "")}

    def _parse_judge(self, text: str) -> Dict[str, Any]:
        js = self._extract_json_obj(text)
        data = self._safe_load(js or "")
        best = self._normalize_ops(data.get("best_plan"))
        score = data.get("score", 0.0)
        try:
            score = float(score)
        except Exception:
            score = 0.0
        converged = bool(data.get("converged", False))
        return {"best_plan": best, "score": max(0.0, min(1.0, score)), "converged": converged, "reason": data.get("reason", "")}

    def _history_brief(self) -> List[Dict[str, Any]]:
        out = []
        for r in self.history[-3:]:
            out.append({
                "round": r.round_num,
                "aff": r.affirmative.get("operators"),
                "neg": r.negative.get("operators"),
                "fast": r.fast.get("best_plan"),
                "slow": r.slow.get("best_plan"),
                "judge": r.judge.get("best_plan"),
                "score": r.judge.get("score", r.slow.get("score", r.fast.get("score", 0.0))),
            })
        return out
