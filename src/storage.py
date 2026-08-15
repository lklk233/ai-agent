# -*- coding: utf-8 -*-
"""SQLite 数据层：建表、论文/分块/关系 CRUD、笔记与状态、去重查询。

设计要点（v4 计划）：
- 单文件 SQLite（sqlite3 标准库），轻量无外部依赖
- 每请求/每次调用新建连接（ThreadingHTTPServer 多线程安全）
- 向量以 scipy.sparse CSR 序列化后存 BLOB（稀疏存储防内存失控）
- meta_status 追踪 LLM 元数据生成状态，只重试 failed
"""
import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT NOT NULL,
    file_hash   TEXT NOT NULL UNIQUE,
    title       TEXT DEFAULT '',
    authors     TEXT DEFAULT '',
    year        TEXT DEFAULT '',
    journal     TEXT DEFAULT '',
    pages       TEXT DEFAULT '',
    topic       TEXT DEFAULT '',
    method      TEXT DEFAULT '',
    keywords    TEXT DEFAULT '',
    summary     TEXT DEFAULT '',
    theme       TEXT DEFAULT '',
    degree      REAL DEFAULT 0,
    status      TEXT DEFAULT '未读',
    notes       TEXT DEFAULT '',
    meta_status TEXT DEFAULT 'pending',
    parsed_at   TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS chunks (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    seq      INTEGER NOT NULL,
    text     TEXT NOT NULL,
    vec      BLOB
);
CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id);
CREATE TABLE IF NOT EXISTS relations (
    paper_a INTEGER NOT NULL,
    paper_b INTEGER NOT NULL,
    type    TEXT NOT NULL,
    weight  REAL DEFAULT 1,
    UNIQUE(paper_a, paper_b, type)
);
CREATE INDEX IF NOT EXISTS idx_rel_a ON relations(paper_a);
CREATE INDEX IF NOT EXISTS idx_rel_b ON relations(paper_b);
"""


class Storage:
    """SQLite 封装。每个实例持有一个连接；多线程场景每请求新建实例。"""

    def __init__(self, db_path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

    # ---------- 论文 ----------

    def add_paper(self, filename, file_hash, title="", authors="", year="",
                  journal="", pages="", topic="", method="", keywords="",
                  summary="", theme="", meta_status="pending",
                  parsed_at=None, notes="", status="未读"):
        cur = self.conn.execute(
            """INSERT INTO papers(filename, file_hash, title, authors, year,
               journal, pages, topic, method, keywords, summary, theme,
               meta_status, parsed_at, notes, status)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (filename, file_hash, title, authors, year, journal, pages,
             topic, method, keywords, summary, theme, meta_status,
             parsed_at or time.strftime("%Y-%m-%d %H:%M:%S"), notes, status))
        self.conn.commit()
        return cur.lastrowid

    def paper_exists(self, file_hash):
        row = self.conn.execute(
            "SELECT id FROM papers WHERE file_hash=?", (file_hash,)).fetchone()
        return row["id"] if row else None

    def get_paper(self, paper_id):
        row = self.conn.execute(
            "SELECT * FROM papers WHERE id=?", (paper_id,)).fetchone()
        return dict(row) if row else None

    def list_papers(self):
        rows = self.conn.execute(
            "SELECT * FROM papers ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    def count_papers(self):
        return self.conn.execute("SELECT COUNT(*) c FROM papers").fetchone()["c"]

    def update_paper_meta(self, paper_id, **fields):
        """更新元数据字段；fields 只含允许的列。"""
        allowed = {"title", "authors", "year", "journal", "pages", "topic",
                   "method", "keywords", "summary", "theme", "meta_status"}
        cols = [k for k in fields if k in allowed]
        if not cols:
            return
        sets = ", ".join(f"{c}=?" for c in cols)
        vals = [fields[c] for c in cols] + [paper_id]
        self.conn.execute(f"UPDATE papers SET {sets} WHERE id=?", vals)
        self.conn.commit()

    def update_notes_status(self, paper_id, notes=None, status=None):
        sets, vals = [], []
        if notes is not None:
            sets.append("notes=?"); vals.append(notes)
        if status is not None:
            sets.append("status=?"); vals.append(status)
        if not sets:
            return
        vals.append(paper_id)
        self.conn.execute(
            f"UPDATE papers SET {', '.join(sets)} WHERE id=?", vals)
        self.conn.commit()

    def update_graph_degree(self, degrees):
        """批量写入度中心性。degrees: {paper_id: degree}"""
        self.conn.executemany(
            "UPDATE papers SET degree=? WHERE id=?",
            [(d, pid) for pid, d in degrees.items()])
        self.conn.commit()

    def papers_with_meta_status(self, status):
        rows = self.conn.execute(
            "SELECT id FROM papers WHERE meta_status=?", (status,)).fetchall()
        return [r["id"] for r in rows]

    def delete_paper(self, paper_id):
        self.conn.execute("DELETE FROM papers WHERE id=?", (paper_id,))
        self.conn.execute("DELETE FROM chunks WHERE paper_id=?", (paper_id,))
        self.conn.execute(
            "DELETE FROM relations WHERE paper_a=? OR paper_b=?",
            (paper_id, paper_id))
        self.conn.commit()

    # ---------- 分块 ----------

    def add_chunks(self, paper_id, chunks):
        """chunks: [(seq, text, vec_bytes_or_None), ...]"""
        self.conn.executemany(
            "INSERT INTO chunks(paper_id, seq, text, vec) VALUES(?,?,?,?)",
            [(paper_id, seq, text, vec) for seq, text, vec in chunks])
        self.conn.commit()

    def clear_chunks(self, paper_id):
        self.conn.execute("DELETE FROM chunks WHERE paper_id=?", (paper_id,))
        self.conn.commit()

    def get_chunks(self, paper_id):
        rows = self.conn.execute(
            "SELECT id, seq, text, vec FROM chunks WHERE paper_id=? ORDER BY seq",
            (paper_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_all_chunk_vecs(self):
        rows = self.conn.execute(
            "SELECT id, paper_id, vec FROM chunks WHERE vec IS NOT NULL").fetchall()
        return [dict(r) for r in rows]

    def clear_all_chunks(self):
        self.conn.execute("DELETE FROM chunks")
        self.conn.commit()

    # ---------- 关系 ----------

    def clear_relations(self):
        self.conn.execute("DELETE FROM relations")
        self.conn.commit()

    def add_relations(self, relations):
        """relations: [(paper_a, paper_b, type, weight), ...]"""
        self.conn.executemany(
            "INSERT OR REPLACE INTO relations(paper_a, paper_b, type, weight) VALUES(?,?,?,?)",
            relations)
        self.conn.commit()

    def get_relations(self, rtype=None):
        sql = "SELECT * FROM relations"
        if rtype:
            sql += " WHERE type=?"
            rows = self.conn.execute(sql, (rtype,)).fetchall()
        else:
            rows = self.conn.execute(sql).fetchall()
        return [dict(r) for r in rows]

    # ---------- 检索 ----------

    def fulltext_search(self, query, limit=50):
        """LIKE 全文检索（含排序打分：标题>摘要区>正文，命中数加成）。

        返回 [(paper, {score, hits, snippets:[(seq,text)]}), ...] 按分降序。
        排序轻量实现（v4 计划 14）：标题命中 3 分/次，正文命中 1 分/次，
        摘要区（前 20% 块）2 分/次。
        """
        q = query.strip()
        if not q:
            return []
        like = f"%{q}%"
        papers = self.list_papers()
        results = []
        for p in papers:
            score, hits = 0, 0
            if q in (p.get("title") or ""):
                score += 3 * (p["title"].count(q) or 1)
            kw = p.get("keywords") or ""
            if q in kw:
                score += 2
            if q in (p.get("summary") or ""):
                score += 2
            # 分块中检索
            chunks = self.get_chunks(p["id"])
            snippets = []
            for c in chunks:
                if q in c["text"]:
                    n = c["text"].count(q)
                    hits += n
                    score += n
                    snippets.append((c["seq"], c["text"]))
                    if len(snippets) >= 3:
                        break
            if score > 0:
                results.append((p, {"score": score, "hits": hits,
                                    "snippets": snippets}))
        results.sort(key=lambda x: x[1]["score"], reverse=True)
        return results[:limit]

    def get_paper_text(self, paper_id):
        """拼接全文（用于 RAG 上下文/综述）。"""
        chunks = self.get_chunks(paper_id)
        return "\n".join(c["text"] for c in chunks)


def file_hash_of(path):
    """按文件内容计算 SHA-256（增量去重）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(65536), b""):
            h.update(blk)
    return h.hexdigest()


def serialize_vec(csr):
    """scipy.sparse CSR -> bytes（BLOB 存储）。"""
    import numpy as np
    payload = {
        "shape": list(csr.shape),
        "data": csr.data.tolist(),
        "indices": csr.indices.tolist(),
        "indptr": csr.indptr.tolist(),
    }
    return json.dumps(payload).encode("utf-8")


def deserialize_vec(blob):
    """bytes -> scipy.sparse CSR。"""
    if not blob:
        return None
    import numpy as np
    from scipy import sparse
    p = json.loads(blob.decode("utf-8"))
    return sparse.csr_matrix(
        (np.array(p["data"]), np.array(p["indices"]), np.array(p["indptr"])),
        shape=tuple(p["shape"]))
