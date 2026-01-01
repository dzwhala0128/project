"""
Operator prompts (executor side).

These are short and structured prompts used inside each operator implementation.
They are not copied verbatim from the paper, but they follow the same operator semantics
(cot / single_step / iterative_step / substep) described in the BELLE workflow.
"""

# 1) CoT (no retrieval)
COT_PROMPT = """You are answering a question without calling retrieval tools.

Question:
{question}

If evidence is provided, you may use it, otherwise rely on your knowledge.
Evidence:
{evidence}

Write your final answer in one short sentence.
"""

# 2) Single-step: answer directly from already-seeded evidence
SINGLE_STEP_PROMPT = """You have evidence retrieved from a Wikipedia index. Answer the question using ONLY the evidence.
If the evidence is insufficient, say "Insufficient evidence" and suggest one missing entity to retrieve.

Question:
{question}

Evidence:
{evidence}

Answer:"""

# 3) Iterative-step: propose next query for the next hop
ITER_NEXT_QUERY_PROMPT = """You are doing multi-hop retrieval.
Based on the evidence so far, propose the next Wikipedia search query needed to move one hop forward.
If enough evidence exists to answer, set stop=true and leave next_query empty.

Question:
{question}

Evidence so far:
{evidence}

Return JSON ONLY:
{{"next_query": "...", "stop": false, "note": "what you are trying to find next"}}"""

# 4) Sub-step: decompose into sub-questions
SUBSTEP_PROMPT = """Decompose the main question into up to {max_subquestions} sub-questions that can be answered by Wikipedia retrieval.
Each sub-question should be specific enough to retrieve relevant pages.

Main question:
{question}

Evidence so far:
{evidence}

Return JSON ONLY:
{{"subquestions": ["...","..."], "note": "how these subquestions connect back to the main question"}}"""

# 5) Answer a sub-question given accumulated evidence
SUBQ_ANSWER_PROMPT = """Answer the sub-question using the evidence below.
If evidence is insufficient, answer "Insufficient evidence".

Sub-question:
{subq}

Evidence:
{evidence}

Answer:"""

# 6) Final synthesis
FINAL_SYNTHESIS_PROMPT = """Use the evidence to answer the question. Keep it concise.
If evidence is insufficient, say "Insufficient evidence".

Question:
{question}

Evidence:
{evidence}

Final Answer:"""
