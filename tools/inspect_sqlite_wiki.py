"""Inspect a SQLite Wikipedia DB built from ZIM (or other sources).

Usage:
  python tools/inspect_sqlite_wiki.py /path/to/wiki.sqlite "Vermont"

It prints:
  - available tables
  - guessed FTS table & its columns
  - a sample retrieval result (title + snippet)

This is optional, but it helps you configure:
  retrieval.method: sqlite_fts
  retrieval.sqlite_path: ...
  retrieval.fts_table/title_col/text_col (if auto-detection isn't correct)
"""

from __future__ import annotations

import sys
import sqlite3

from retrieval.retriever import SQLiteFTSRetriever


def list_tables(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")
    return cur.fetchall()


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/inspect_sqlite_wiki.py /path/to/wiki.sqlite [query]")
        sys.exit(1)

    sqlite_path = sys.argv[1]
    query = sys.argv[2] if len(sys.argv) >= 3 else "Vermont"

    conn = sqlite3.connect(sqlite_path, timeout=30.0)
    conn.row_factory = sqlite3.Row

    print(f"[DB] {sqlite_path}")
    print("\n[TABLES]")
    for r in list_tables(conn):
        name = str(r[0])
        sql = str(r[1] or "")
        mark = " (looks like FTS)" if ("fts" in name.lower() or "using fts" in sql.lower()) else ""
        print(f"- {name}{mark}")

    print("\n[GUESS & SAMPLE RETRIEVAL]")
    retriever = SQLiteFTSRetriever(sqlite_path=sqlite_path, fts_table="pages_fts", top_k=3)
    docs = retriever.retrieve(query, top_k=3)
    print(f"Query: {query}")
    for i, d in enumerate(docs, 1):
        print("-" * 60)
        print(f"#{i} title: {d.get('title')}")
        print(f"score: {d.get('score')}")
        content = d.get('content', '')
        print(content[:800].replace("\n", " "))

    retriever.close()
    conn.close()


if __name__ == "__main__":
    main()
