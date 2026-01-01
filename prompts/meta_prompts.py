"""
Meta-prompt pieces for BELLE-style Bi-Level Multi-Agent Debate.

Notes
- This is a *reconstruction* of the paper's prompt structure (Table 10 / Table 11 / Fig. 9),
  rewritten in reminding language so it is robust for HF Instruct models.
- Keep it as plain text (no markdown) because downstream JSON parsers are strict.
"""

# Operator pool description (paraphrased from the paper's operator definitions)
OPERATOR_POOL_DESCRIPTION = """You can choose and combine methods from this operator pool:
1) cot: Do step-by-step reasoning without calling retrieval. Good when the model already knows the fact(s).
2) single_step: Retrieve once for the whole question, then answer using that evidence.
3) iterative_step: Do multi-hop retrieval iteratively. Each hop uses what you learned so far to form the next query.
4) substep: Decompose the question into a few sub-questions, retrieve for each sub-question, then combine.
5) adaptive_step: Decide which of the above strategies to use based on the question, then execute it."""

# Debate "levels" (paraphrased from Table 11)
DEBATE_LEVELS = {
    0: "Level 0: Aim for consensus. It's okay to agree when the other side is right.",
    1: "Level 1: Mostly disagree, but you may agree on a point if it's clearly correct.",
    2: "Level 2: You do NOT have to fully agree. Focus on moving the plan toward the best operator combination.",
    3: "Level 3: Strongly adversarial. You MUST disagree on every point (even if you secretly agree).",
}

# Paper-style meta prompt (paraphrased from Table 10)
META_PROMPT_TEMPLATE = """You are in a structured debate competition to pick an operator plan for a multi-hop QA task.
You may freely combine methods from the operator pool to solve the task.

Operator pool:
{operator_pool}

Question type:
{qtype}

Debate protocol:
- There is one AFFIRMATIVE and one NEGATIVE debater.
- In each round, each side may speak up to TWO times.
- Maximum rounds: {max_rounds}.
- Debate style: {debate_level}

Global rule:
- React to the other side's points ("tit-for-tat"): acknowledge strong arguments, and counter weak ones.
- When proposing a plan, keep it executable given the available operators.
"""
