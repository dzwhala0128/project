"""
BELLE pipeline (offline): Question type classification -> Bi-level debate -> Multi-hop execution.

This project is a practical, runnable approximation of the BELLE workflow described in:
"BELLE: A Bi-Level Multi-Agent Reasoning Framework for Multi-Hop Question Answering"
"""
from __future__ import annotations

import os
from pathlib import Path

import sqlite3
from typing import List, Dict, Any
import yaml

from models.llm_wrapper import create_llm
from modules.type_classifier import QuestionTypeClassifier
from modules.multi_agent_debate import BiLevelMultiAgentDebate
from modules.qa_executor import MultiHopQAExecutor
from retrieval.retriever import create_retriever
from utils.logger import log


class BELLEPipeline:
    """Fully offline BELLE-like framework."""

    def __init__(self, config_path: str = "config/config.yaml"):
        # Resolve config path relative to this file
        base_dir = Path(__file__).resolve().parent
        cfg_path = Path(config_path)
        if not cfg_path.is_absolute():
            cfg_path = base_dir / cfg_path

        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")

        log("Initializing BELLE pipeline...")
        self.config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

        # LLM
        self.llm = create_llm(self.config)

        # Retriever
        r_cfg = self.config.get("retrieval", {})
        self.retriever = create_retriever(r_cfg)
        log(f"Retriever: {type(self.retriever).__name__} | method={r_cfg.get('method')} | sqlite={r_cfg.get('sqlite_path')}")

        # Components
        self.classifier = QuestionTypeClassifier(self.llm)

        d_cfg = self.config.get("debate", {})
        self.debate = BiLevelMultiAgentDebate(
            llm=self.llm,
            operators=self.config.get("operators", ["cot"]),
            max_rounds=int(d_cfg.get("max_rounds", 3)),
            mode=str(d_cfg.get("mode", "hard")),
            score_threshold=float(d_cfg.get("score_threshold", 0.85)),
            max_plan_len=int(d_cfg.get("max_plan_len", 3)),
            debate_level=int(d_cfg.get("debate_level", 2)),
        )

        e_cfg = self.config.get("executor", {})
        self.executor = MultiHopQAExecutor(self.llm, self.retriever, executor_cfg=e_cfg)

        log("BELLE pipeline ready.")

    def run(self, question: str) -> Dict[str, Any]:
        log("=" * 80)
        log(f"Question: {question}")

        qtype = self.classifier.classify(question)
        log(f"Type: {qtype}")

        plan = self.debate.debate(question, qtype)
        log(f"Plan: {plan.get('operator_plan')}")

        result = self.executor.execute(question, qtype, plan["operator_plan"])
        log(f"Answer: {result['final_answer'][:200]}")
        log(f"Retrieval log: {[(x['hop'], x['query'], len(x['docs'])) for x in result['retrieval_log']]}")

        return {
            "question": question,
            "type": qtype,
            "plan": plan,
            "result": result,
        }

