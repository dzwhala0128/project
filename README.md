# BELLE Framework (Offline, BELLE-like)

This is a runnable **offline** implementation that follows the *workflow* described in:

**BELLE: A Bi-Level Multi-Agent Reasoning Framework for Multi-Hop Question Answering** (ACL 2025)

> Note: This repo is **not** the official authors' release. It is an engineering reproduction of the main pipeline:
> **Type Classifier → Bi-level Multi-Agent Debate → Multi-hop Operator Execution**.

---

## 1. Quick Start

### Install
```bash
pip install -r requirements.txt
```

### Configure
Edit `config/config.yaml`:

- `llm.model_path`: local HF model folder (e.g., Qwen3)
- `retrieval.wiki_path`: local offline wiki dump folder

If you already built a **SQLite FTS** Wikipedia database from a ZIM file (recommended for speed/coverage):

- set `retrieval.method: sqlite_fts`
- set `retrieval.sqlite_path: /path/to/wiki.sqlite`
- optionally set `retrieval.fts_table` (default `pages_fts`), `title_col`, `text_col`

Optional tuning:
- `debate.mode`: `hard` / `soft`
- `executor.max_hops`, `executor.per_hop_top_k`

### Run
```bash
python main.py
```

---

## 2. Architecture

```
QuestionTypeClassifier
  → BiLevelMultiAgentDebate (affirmative/negative + fast/slow + judge)
    → MultiHopQAExecutor (substep / iterative_step / cot / single_step / adaptive_step)
```

---

## 3. Operators

- `substep`: decompose into sub-questions, retrieve per sub-question, answer each
- `iterative_step`: hop-by-hop: propose next query → retrieve → repeat
- `cot`: final reasoning & answer using aggregated evidence
- `single_step`: direct answer using evidence
- `adaptive_step`: choose an operator based on question type (heuristic)

---

## 4. Tips

- If your wiki dump is huge, reduce `retrieval.max_docs` in config.
- For large-scale offline Wikipedia, prefer `retrieval.method: sqlite_fts` to avoid building a huge in-memory BM25 index every run.
- If you see out-of-memory, set `llm.device_map: "cpu"` or use a smaller model.
