"""
Offline Wikipedia BM25 retriever.

This retriever is intentionally permissive about the wiki dump format:
- .txt files: treated as one document each
- .json files: supports dict {"text": "..."} / {"content": "..."} / {"title":..., "text":...}
             or list of dicts/strings
- .jsonl files: one JSON object per line

It builds a BM25 index (rank_bm25.BM25Okapi) over a tokenized corpus.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import sqlite3

from rank_bm25 import BM25Okapi


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]

# class SQLiteFTSRetriever:
#     def __init__(
#         self,
#         sqlite_path: str,
#         fts_table: str = "pages_fts",
#         title_col: str = "title",
#         text_col: str = "lead",
#         top_k: int = 5,
#         snippet_chars: int = 1200,
#     ):
#         self.sqlite_path = sqlite_path
#         self.fts_table = fts_table
#         self.title_col = title_col
#         self.text_col = text_col
#         self.top_k = top_k
#         self.snippet_chars = snippet_chars
#         self.con = sqlite3.connect(self.sqlite_path, check_same_thread=False)
#
#     def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
#         k = top_k or self.top_k
#
#         # 先用最稳的：MATCH + bm25 排序（FTS5 支持 bm25）
#         sql = f"""
#         SELECT
#           {self.title_col} as title,
#           substr({self.text_col}, 1, ?) as text
#         FROM {self.fts_table}
#         WHERE {self.fts_table} MATCH ?
#         ORDER BY bm25({self.fts_table})
#         LIMIT ?;
#         """
#
#         try:
#             rows = self.con.execute(sql, (self.snippet_chars, query, k)).fetchall()
#         except sqlite3.OperationalError:
#             # 有些构建方式可能不支持 bm25，就退化成不排序的 LIMIT
#             sql2 = f"""
#             SELECT
#               {self.title_col} as title,
#               substr({self.text_col}, 1, ?) as text
#             FROM {self.fts_table}
#             WHERE {self.fts_table} MATCH ?
#             LIMIT ?;
#             """
#             rows = self.con.execute(sql2, (self.snippet_chars, query, k)).fetchall()
#
#         return [{"title": r[0], "text": r[1]} for r in rows]
#

class WikipediaRetriever:
    """Offline BM25 retriever over a local wiki dump folder."""
    def __init__(self, wiki_path: str, top_k: int = 5, max_docs: int = 200000):
        self.wiki_path = Path(wiki_path)
        self.top_k = int(top_k)
        self.max_docs = int(max_docs)

        self._docs: List[Dict[str, Any]] = []
        self._corpus_tokens: List[List[str]] = []
        self._bm25: Optional[BM25Okapi] = None
        self._built: bool = False

    # ---------- public ----------
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        self._ensure_index()

        if not self._bm25 or not self._docs:
            return []

        k = int(top_k) if top_k is not None else self.top_k
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        scores = self._bm25.get_scores(q_tokens)
        # top-k indices
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results: List[Dict[str, Any]] = []
        for idx in top_idx:
            doc = self._docs[idx]
            content = doc["content"]
            snippet = content[:800] + "..." if len(content) > 800 else content
            results.append({
                "doc_id": doc.get("doc_id", idx),
                "title": doc.get("title", ""),
                "content": snippet,
                "score": float(scores[idx]),
                "source": doc.get("source", "")
            })
        return results

    # ---------- internal ----------
    def _ensure_index(self) -> None:
        if self._built:
            return
        if not self.wiki_path.exists():
            # Graceful fallback: empty corpus
            self._docs = []
            self._corpus_tokens = []
            self._bm25 = None
            self._built = True
            return

        files = []
        for ext in ("*.txt", "*.json", "*.jsonl"):
            files.extend(self.wiki_path.rglob(ext))

        if not files:
            # Allow empty corpus
            self._docs = []
            self._corpus_tokens = []
            self._bm25 = None
            self._built = True
            return

        docs: List[Dict[str, Any]] = []
        doc_id = 0
        for fp in files:
            if len(docs) >= self.max_docs:
                break
            try:
                for d in self._load_file(fp):
                    if len(docs) >= self.max_docs:
                        break
                    d["doc_id"] = doc_id
                    doc_id += 1
                    docs.append(d)
            except Exception:
                # Skip problematic files
                continue

        self._docs = docs
        self._corpus_tokens = [_tokenize(d.get("title", "") + "\n" + d.get("content", "")) for d in docs]
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None
        self._built = True

    def _load_file(self, fp: Path) -> List[Dict[str, Any]]:
        suffix = fp.suffix.lower()
        if suffix == ".txt":
            txt = fp.read_text(encoding="utf-8", errors="ignore")
            return [{"title": fp.stem, "content": txt, "source": str(fp)}]

        if suffix == ".jsonl":
            out: List[Dict[str, Any]] = []
            with fp.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    out.extend(self._normalize_json_obj(obj, source=str(fp)))
            return out

        if suffix == ".json":
            obj = json.loads(fp.read_text(encoding="utf-8", errors="ignore"))
            return self._normalize_json_obj(obj, source=str(fp))

        return []

    def _normalize_json_obj(self, obj: Any, source: str) -> List[Dict[str, Any]]:
        """
        Normalize various wiki json formats into list of {"title","content","source"}.
        """
        out: List[Dict[str, Any]] = []

        def add(title: str, content: str):
            content = content or ""
            title = title or ""
            if content.strip():
                out.append({"title": title, "content": content, "source": source})

        if isinstance(obj, dict):
            title = obj.get("title") or obj.get("id") or obj.get("docid") or ""
            content = obj.get("text") or obj.get("content") or obj.get("body") or ""
            if isinstance(content, list):
                content = "\n".join([str(x) for x in content])
            add(str(title), str(content))
            return out

        if isinstance(obj, list):
            for it in obj:
                if isinstance(it, str):
                    add("", it)
                elif isinstance(it, dict):
                    title = it.get("title") or it.get("id") or ""
                    content = it.get("text") or it.get("content") or ""
                    if isinstance(content, list):
                        content = "\n".join([str(x) for x in content])
                    add(str(title), str(content))
            return out

        # Unknown type
        return out


# Backward-compat alias
DummyRetriever = WikipediaRetriever


class SQLiteFTSRetriever:
    """SQLite FTS retriever for offline Wikipedia databases.

    This is designed for databases you may have built from a Wikipedia .zim file.

    Expected shape (FTS5 recommended):
    - An FTS table (default: pages_fts) containing at least a title column, and ideally a text/content column.

    The retriever will try to auto-detect sensible title/text columns if not specified.
    """

    def __init__(
        self,
        sqlite_path: str,
        fts_table: str = "pages_fts",
        top_k: int = 5,
        title_col: Optional[str] = None,
        text_col: Optional[str] = None,
        snippet_chars: int = 1200,
        max_terms: int = 12,
        timeout: float = 30.0,
    ):
        self.sqlite_path = str(sqlite_path)
        self.fts_table = str(fts_table)
        self.top_k = int(top_k)
        self.title_col = title_col
        self.text_col = text_col
        self.snippet_chars = int(snippet_chars)
        self.max_terms = int(max_terms)
        self.timeout = float(timeout)

        self._conn: Optional[sqlite3.Connection] = None
        self._resolved: bool = False

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def __del__(self):
        self.close()

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        self._ensure_ready()
        if not self._conn:
            return []

        k = int(top_k) if top_k is not None else self.top_k
        tokens = _tokenize(query)
        #print("[DEBUG FTS]", query, "=>", tokens)
        if not tokens:
            return []

        # tokens 清洗 + 截断
        tokens = [t for t in tokens if len(t) >= 2][: self.max_terms]
        if not tokens:
            return []

        # ✅ AND 优先（更精准），OR 兜底（更召回）
        fts_query_and = " ".join(tokens)  # FTS 默认 AND
        fts_query_or = " OR ".join(tokens)

        # Try FTS5 bm25 + snippet first; fall back gracefully.
        title_col = self.title_col or "title"
        text_col = self.text_col or "lead"  # ✅ 你的库是 lead，不是 content

        # Helper: run SQL and normalize output
        def norm_rows(rows: List[sqlite3.Row], score_key: str = "score") -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for r in rows:
                title = (r["title"] if hasattr(r, "keys") and "title" in r.keys() else r[0])
                content = (r["content"] if hasattr(r, "keys") and "content" in r.keys() else "")
                score = float(r[score_key]) if hasattr(r, "keys") and score_key in r.keys() else 0.0
                text = str(content or "")[: self.snippet_chars]
                out.append({
                    "doc_id": (r["rowid"] if hasattr(r, "keys") and "rowid" in r.keys() else None),
                    "title": str(title or ""),
                    "content": text,  # 保留老字段
                    "text": text,  # ✅ 新增：兼容 executor 读 text 的情况
                    "score": score,
                    "source": self.sqlite_path,
                })
            return out

        cur = self._conn.cursor()

        # Attempt 0: exact title match (prefer the canonical page)
        try:
            sql0 = (
                f"SELECT rowid as rowid, {title_col} as title, {text_col} as content "
                f"FROM {self.fts_table} WHERE {title_col} = ? COLLATE NOCASE LIMIT ?"
            )
            cur.execute(sql0, (query, k))
            rows = cur.fetchall()
            if rows:
                out = norm_rows(rows, score_key="score")
                for i, d in enumerate(out):
                    d["score"] = float(k - i)
                return out
        except Exception:
            pass

        # Attempt 0.5: phrase match (good for multi-word entities)
        if " " in query.strip():
            phrase = f"\"{query.strip()}\""
            for q0 in (f"{title_col}:{phrase}", phrase):
                try:
                    sqlp = (
                        f"SELECT rowid as rowid, {title_col} as title, {text_col} as content, "
                        f"(-bm25({self.fts_table})) as score "
                        f"FROM {self.fts_table} WHERE {self.fts_table} MATCH ? "
                        f"ORDER BY bm25({self.fts_table}) LIMIT ?"
                    )
                    cur.execute(sqlp, (q0, k))
                    rows = cur.fetchall()
                    if rows:
                        return norm_rows(rows, score_key="score")
                except Exception:
                    pass

        # ✅ 关键：两次尝试，先 AND，AND 没命中再 OR
        for q in (fts_query_and, fts_query_or):

            # Attempt 1: FTS5 bm25 + snippet()
            try:
                text_col_idx = self._col_index(self.fts_table, text_col)
                sql = (
                    f"SELECT rowid as rowid, {title_col} as title, "
                    f"snippet({self.fts_table}, ?, '', '', ' ... ', 16) as content, "
                    f"(-bm25({self.fts_table})) as score "
                    f"FROM {self.fts_table} WHERE {self.fts_table} MATCH ? "
                    f"ORDER BY bm25({self.fts_table}) LIMIT ?"
                )
                cur.execute(sql, (text_col_idx, q, k))
                rows = cur.fetchall()
                if rows:
                    return norm_rows(rows, score_key="score")
            except Exception:
                pass

            # Attempt 2: no snippet, but bm25 works
            try:
                sql = (
                    f"SELECT rowid as rowid, {title_col} as title, {text_col} as content, "
                    f"(-bm25({self.fts_table})) as score "
                    f"FROM {self.fts_table} WHERE {self.fts_table} MATCH ? "
                    f"ORDER BY bm25({self.fts_table}) LIMIT ?"
                )
                cur.execute(sql, (q, k))
                rows = cur.fetchall()
                if rows:
                    return norm_rows(rows, score_key="score")
            except Exception:
                pass

            # Attempt 3: minimal MATCH with LIMIT
            try:
                sql = (
                    f"SELECT rowid as rowid, {title_col} as title, {text_col} as content "
                    f"FROM {self.fts_table} WHERE {self.fts_table} MATCH ? LIMIT ?"
                )
                cur.execute(sql, (q, k))
                rows = cur.fetchall()
                if rows:
                    out = norm_rows(rows, score_key="score")
                    # fabricate score
                    for i, d in enumerate(out):
                        d["score"] = float(k - i)
                    return out
            except Exception:
                pass

        # AND/OR 都没命中
        return []


    # ---------- internal ----------
    def _ensure_ready(self) -> None:
        if self._conn is None:
            try:
                self._conn = sqlite3.connect(self.sqlite_path, timeout=self.timeout)
                self._conn.row_factory = sqlite3.Row
            except Exception:
                self._conn = None
                return

        if self._resolved:
            return

        # Auto-detect title/text columns if not provided
        try:
            if not self._table_exists(self.fts_table):
                # Try to find any FTS-like table
                self.fts_table = self._guess_fts_table() or self.fts_table
            cols = self._table_columns(self.fts_table)
            if cols:
                if not self.title_col:
                    self.title_col = self._pick_first(cols, ["title", "name", "page_title"])
                if not self.text_col:
                    self.text_col = self._pick_first(cols, ["content", "text", "body", "article", "plaintext"])
        except Exception:
            pass

        self._resolved = True

    def _table_exists(self, table: str) -> bool:
        cur = self._conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
        return cur.fetchone() is not None

    def _guess_fts_table(self) -> Optional[str]:
        cur = self._conn.cursor()
        cur.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")
        rows = cur.fetchall()
        # Prefer explicit pages_fts
        for r in rows:
            if str(r["name"]).lower() == "pages_fts":
                return r["name"]
        # Otherwise pick something that looks like FTS
        for r in rows:
            name = str(r["name"]).lower()
            sql = str(r["sql"] or "").lower()
            if "fts" in name or "using fts" in sql:
                return r["name"]
        return None

    def _table_columns(self, table: str) -> List[str]:
        cur = self._conn.cursor()
        cur.execute(f"PRAGMA table_info({table})")
        return [str(r["name"]) for r in cur.fetchall()]

    def _pick_first(self, cols: List[str], candidates: List[str]) -> Optional[str]:
        lower = {c.lower(): c for c in cols}
        for cand in candidates:
            if cand.lower() in lower:
                return lower[cand.lower()]
        return cols[0] if cols else None

    def _col_index(self, table: str, col: str) -> int:
        cols = self._table_columns(table)
        for i, c in enumerate(cols):
            if c.lower() == col.lower():
                return i
        return 0


def create_retriever(r_cfg: Dict[str, Any]):
    """Factory to create a retriever based on config."""
    method = str(r_cfg.get("method", "bm25")).lower().strip()

    if method in {"sqlite", "sqlite_fts", "fts", "zim_sqlite"}:
        return SQLiteFTSRetriever(
            sqlite_path=r_cfg.get("sqlite_path", ""),
            fts_table=r_cfg.get("fts_table", "pages_fts"),
            top_k=int(r_cfg.get("top_k", 5)),
            title_col=r_cfg.get("title_col", "title"),
            text_col=r_cfg.get("text_col", "lead"),
            snippet_chars=int(r_cfg.get("snippet_chars", 1200)),
            max_terms=int(r_cfg.get("max_terms", 12)),
            timeout=float(r_cfg.get("timeout", 30.0)),
        )

    # default: folder-based BM25
    return WikipediaRetriever(
        wiki_path=r_cfg.get("wiki_path", ""),
        top_k=int(r_cfg.get("top_k", 5)),
        max_docs=int(r_cfg.get("max_docs", 200000)),
    )
