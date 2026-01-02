"""
Role prompts for BELLE-style Bi-Level Multi-Agent Debate.

This file rewrites the paper's prompt structure (Table 10 / Table 11) into strict JSON-only outputs,
so parsing is stable in this codebase.
"""
from __future__ import annotations

AFFIRMATIVE_PROMPT = """{meta}

ROLE: AFFIRMATIVE debater.

Your previous affirmative outputs (most recent last):
{history_ad}

Previous round fast reviewer output:
{prev_fast}

Previous round slow reviewer output:
{prev_slow}

TASK:
Propose the best operator SEQUENCE (length <= {max_plan_len}) to solve the question.
- Use only operators from Allowed operators.
- Your "reason" should mention the hop structure (what you retrieve first, then what you do next).

INPUT:
Question: {question}
Allowed operators: {operators}

OUTPUT (JSON ONLY, no extra text):
{{"operators": ["..."], "reason": "...", "critique": ""}}
"""

NEGATIVE_PROMPT = """{meta}

ROLE: NEGATIVE debater.

Your previous negative outputs (most recent last):
{history_nd}

Current round affirmative plan (JSON):
{affirmative}

Previous round fast reviewer output:
{prev_fast}

Previous round slow reviewer output:
{prev_slow}

TASK:
Critique the affirmative plan and propose a better alternative operator SEQUENCE (length <= {max_plan_len}).
- "critique" should point out what will likely fail (missing hop, wrong decomposition, too shallow retrieval, etc.).
- "reason" should explain why your plan is better for THIS question type.

INPUT:
Question: {question}
Allowed operators: {operators}

OUTPUT (JSON ONLY):
{{"operators": ["..."], "reason": "...", "critique": "..."}} 
"""

FAST_REVIEW_PROMPT = """{meta}

ROLE: FAST reviewer.

TASK:
Quickly judge THIS round only.
Pick the better plan between affirmative and negative, with a short justification.

INPUT:
Question: {question}
Allowed operators: {operators}
Affirmative (JSON): {affirmative}
Negative (JSON): {negative}

OUTPUT (JSON ONLY):
{{"best_plan": ["..."], "score": 0.75, "reason": "..."}}
"""

SLOW_REVIEW_PROMPT = """{meta}

ROLE: SLOW reviewer (recorder).

Your slow-memory from previous rounds (most recent last):
{history_slow}

THIS round:
Affirmative (JSON): {affirmative}
Negative (JSON): {negative}
Fast reviewer (JSON): {fast}

TASK:
Pick a stable plan (avoid oscillation) and update the running "memory".
- Prefer plans that clearly cover the necessary hops.
- If two plans are similar, choose the one that is easier to execute.
- Provide a numeric "score" in [0.0, 1.0] reflecting confidence in the chosen plan for this question.

INPUT:
Question: {question}
Allowed operators: {operators}

OUTPUT (JSON ONLY):
{{"best_plan": ["..."], "score": 0.75, "reason": "...", "memory": "updated concise notes for next rounds"}}
"""

JUDGE_PROMPT = """{meta}

ROLE: JUDGE.

MODE: {mode}

INPUT:
Question: {question}
Allowed operators: {operators}

Affirmative (JSON): {affirmative}
Negative (JSON): {negative}
Fast reviewer (JSON): {fast}
Slow reviewer (JSON): {slow}

TASK:
Give the final decision for this round.
- If the plan is clear and consistent, set converged=true.
- Otherwise, converged=false but still output the best current plan.
- Provide a numeric "score" in [0.0, 1.0] reflecting confidence in the chosen plan for this question.

OUTPUT (JSON ONLY):
{{"best_plan": ["..."], "score": 0.75, "converged": false, "reason": "..."}}
"""
