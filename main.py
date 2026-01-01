#!/usr/bin/env python3
"""
Run the offline BELLE pipeline.

Usage:
  python main.py

Make sure you update config/config.yaml with:
- llm.model_path: local model folder
- retrieval.wiki_path: local wiki dump folder
"""
from __future__ import annotations

import torch
from pipeline import BELLEPipeline


def main():
    print("🚀 启动离线 BELLE 框架 (Local LM + Offline Wiki BM25)")
    print(f"💻 GPU 可用: {torch.cuda.is_available()}")

    belle = BELLEPipeline()

    questions = [
        "What is the capital of the country where the inventor of the telephone was born?",
        "Which happened earlier: the publication of the novel that introduced Sherlock Holmes or the publication of the novel that introduced Harry Potter?",
        "Who was the spouse of the U.S. president who signed the Emancipation Proclamation?"
    ]

    results = []
    for q in questions:
        try:
            results.append(belle.run(q))
        except Exception as e:
            print(f"❌ 运行失败: {e}")

    print("\n📋 结果汇总:")
    for i, r in enumerate(results, 1):
        ans = r["result"]["final_answer"]
        print(f"{i}. {r['question'][:60]}... → {ans[:80]}...")


if __name__ == "__main__":
    main()
