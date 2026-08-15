# -*- coding: utf-8 -*-
"""知识库入库流水线：扫描→去重→解析→LLM 元数据→分块→向量→写库→去重提示。

v4 计划要点：
- 幂等：file_hash 去重；重复执行 0 新增
- meta_status 追踪：pending/done/failed，重跑只重试 failed
- LLM 失败降级正则+分词关键词，不阻塞入库
- 入库后触发全量向量重建（由调用方决定何时 build）
"""
import logging
import re
import time
from pathlib import Path

from .extractor import extract_file, scan_corpus
from .llm_client import LLMClient, OllamaError
from .storage import Storage, file_hash_of, serialize_vec
from .text_prep import normalize, split_chunks, tokenize

logger = logging.getLogger("arch.knowledge")


def _extract_head(text: str, n: int = 1500) -> str:
    """取论文前部（标题区+摘要区）。"""
    return text[:n]


def _fallback_keywords(text: str, topn: int = 8) -> str:
    """LLM 失败时的关键词降级：词频 TopN（过滤停用词）。"""
    stop = {"的", "了", "是", "在", "与", "和", "及", "对", "中", "为", "以",
            "本文", "研究", "分析", "通过", "进行", "一个", "我们", "其", "之",
            "从", "到", "等", "被", "把", "也", "都", "而", "但", "或", "并"}
    freq = {}
    for w in tokenize(text):
        w = w.strip()
        if len(w) < 2 or w in stop or w.isdigit():
            continue
        freq[w] = freq.get(w, 0) + 1
    top = sorted(freq.items(), key=lambda x: -x[1])[:topn]
    return "；".join(w for w, _ in top)


def _find_reference_zone(text: str) -> str:
    """定位参考文献区（全文尾部）。"""
    for marker in ("参考文献", "参考资料", "REFERENCES", "References"):
        idx = text.rfind(marker)
        if idx != -1:
            return text[idx:]
    return text[-1500:]


class KnowledgeBase:
    def __init__(self, cfg):
        self.cfg = cfg
        self.storage = Storage(cfg["db_path"])

    def close(self):
        self.storage.close()

    # ---------- 入库 ----------

    def ingest(self, corpus_dir=None, with_llm=True, llm=None) -> dict:
        """扫描语料目录入库。返回统计信息。

        llm 参数用于测试注入 mock；None 时按 with_llm 创建真实客户端。
        """
        corpus_dir = Path(corpus_dir or self.cfg["corpus_dir"])
        corpus_dir.mkdir(parents=True, exist_ok=True)
        files = scan_corpus(corpus_dir)
        result = {
            "scanned": len(files), "added": 0, "duplicates": 0,
            "failed": [], "dedup_hints": [],
        }
        if llm is None and with_llm:
            llm = LLMClient(self.cfg)
        if llm is not None and not getattr(llm, "health", lambda: True)():
            logger.warning("Ollama 未连接，元数据将使用降级提取")
            llm = None

        for f in files:
            fhash = file_hash_of(f)
            existing = self.storage.paper_exists(fhash)
            if existing:
                result["duplicates"] += 1
                # 已入库但元数据 failed 的，重跑时重试
                p = self.storage.get_paper(existing)
                if p and p.get("meta_status") == "failed" and llm:
                    self._retry_metadata(existing, p, llm, f)
                continue
            ok = self._ingest_one(f, fhash, llm)
            if ok:
                result["added"] += 1
            else:
                result["failed"].append(f.name)
        return result

    def _ingest_one(self, f: Path, fhash: str, llm) -> bool:
        text, method = extract_file(f, self.cfg)
        if not text or not text.strip():
            logger.warning("解析 %s 无文本，跳过", f.name)
            return False
        text = normalize(text)
        pid = self.storage.add_paper(
            filename=f.name, file_hash=fhash, meta_status="pending",
            parsed_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        # 分块（无论 LLM 成败都入库）
        chunks = split_chunks(text, self.cfg.get("chunk_size", 800),
                              self.cfg.get("chunk_overlap", 80))
        self.storage.add_chunks(pid, [(seq, t, None) for seq, t in chunks])

        meta = {}
        if llm:
            try:
                head = _extract_head(text)
                tail = _find_reference_zone(text)
                raw = llm.extract_metadata(head, tail)
                # 统一键名规范化（幂等：中文键/英文键均可）
                from .llm_client import normalize_meta_keys
                meta = normalize_meta_keys(raw if isinstance(raw, dict) else {})
                if not meta or not (meta.get("title") or "").strip():
                    # LLM 偶发返回空/异常结构：合并降级字段保证有内容；
                    # 标记 failed 以便重跑 ingest 时重试 LLM
                    fb = self._fallback_meta(text)
                    meta = {**fb, **{k: v for k, v in meta.items() if v}}
                    self.storage.update_paper_meta(pid, meta_status="failed",
                                                   **meta)
                else:
                    self.storage.update_paper_meta(pid, meta_status="done",
                                                   **meta)
            except OllamaError as e:
                logger.warning("%s 元数据 LLM 失败（降级）: %s", f.name, e)
                meta = self._fallback_meta(text)
                self.storage.update_paper_meta(pid, meta_status="failed",
                                               **meta)
        else:
            meta = self._fallback_meta(text)
            self.storage.update_paper_meta(pid, meta_status="done", **meta)
        logger.info("入库 %s (id=%d, method=%s)", f.name, pid, method)
        return True

    def _retry_metadata(self, pid, paper, llm, f: Path):
        try:
            text = self.storage.get_paper_text(pid)
            head = _extract_head(text)
            tail = _find_reference_zone(text)
            from .llm_client import normalize_meta_keys
            raw = llm.extract_metadata(head, tail)
            meta = normalize_meta_keys(raw if isinstance(raw, dict) else {})
            if not meta or not (meta.get("title") or "").strip():
                # LLM 再次失败：合并降级字段，保留 failed 标记以便下次重试
                fb = self._fallback_meta(text)
                meta = {**fb, **{k: v for k, v in meta.items() if v}}
                self.storage.update_paper_meta(pid, meta_status="failed",
                                               **meta)
                logger.warning("重试元数据仍不完整，保持 failed: %s", f.name)
            else:
                self.storage.update_paper_meta(pid, meta_status="done", **meta)
                logger.info("重试元数据成功: %s", f.name)
        except OllamaError as e:
            logger.warning("重试元数据仍失败: %s: %s", f.name, e)

    def _fallback_meta(self, text: str) -> dict:
        """无 LLM 时的降级元数据（正则标题/年份 + 词频关键词）。"""
        title = ""
        # 标题常在文本首行（可能无换行，如 PDF 提取的标题+作者同行）
        m = re.match(r"^\s*([^\n]{5,80})", text)
        if m:
            cand = m.group(1).strip().strip("《》")
            # 在摘要/关键词标记处截断
            for marker in ("摘要", "关键词", "引言", "一、", "1 ", "【"):
                idx = cand.find(marker)
                if idx > 4:
                    cand = cand[:idx]
                    break
            # 去掉可能拼接的作者与年份（中文名2-4字+年份）
            cand = re.sub(r"(19\d{2}|20\d{2}).*$", "", cand)
            cand = re.sub(r"[\u4e00-\u9fa5]{2,4}(19\d{2}|20\d{2})$", "", cand)
            cand = re.sub(r"^[\d\s]+$", "", cand).strip()
            if 4 <= len(cand) <= 40:
                title = cand
            elif len(cand) > 40:
                # 超长则取前 30 字符
                title = cand[:30]
        year = ""
        m2 = re.search(r"(19\d{2}|20\d{2})", text[:500])
        if m2:
            year = m2.group(1)
        return {
            "title": title,
            "authors": "",
            "year": year,
            "journal": "",
            "pages": "",
            "topic": "",
            "method": "",
            "keywords": _fallback_keywords(text[:3000]),
            "summary": "",
            "theme": "",
        }

    # ---------- 去重提示 ----------

    def dedup_hints(self, threshold: float = 0.8, topn: int = 3) -> list:
        """基于元数据关键词/标题的简单相似提示（供入库后展示）。

        返回 [{a_title, b_title, score}, ...] 超过阈值的相似对。
        """
        from difflib import SequenceMatcher
        papers = self.storage.list_papers()
        out = []
        for i in range(len(papers)):
            for j in range(i + 1, len(papers)):
                a, b = papers[i], papers[j]
                ta, tb = (a.get("title") or ""), (b.get("title") or "")
                if not ta or not tb:
                    continue
                s = SequenceMatcher(None, ta, tb).ratio()
                if s >= threshold:
                    out.append({"a": a["id"], "b": b["id"],
                                "a_title": ta, "b_title": tb, "score": round(s, 3)})
        out.sort(key=lambda x: -x["score"])
        return out[:topn]

    # ---------- 向量与关系图 ----------

    def rebuild_vectors(self):
        """全量重建论文向量与块向量并持久化（含 vectorizer 保存）。"""
        from .vector_store import VectorStore
        papers = self.storage.list_papers()
        if not papers:
            logger.info("语料为空，跳过向量重建")
            return {}, []
        chunks_by_paper = {}
        for p in papers:
            ch = self.storage.get_chunks(p["id"])
            chunks_by_paper[p["id"]] = [(c["seq"], c["text"]) for c in ch]
        vs = VectorStore(self.cfg)
        for p in papers:
            chs = chunks_by_paper.get(p["id"], [])
            p["text_head"] = chs[0][1][:1200] if chs else ""
        paper_vecs, chunk_vecs = vs.build(papers, chunks_by_paper)
        # 持久化块向量
        for pid, seq, text, vec in chunk_vecs:
            self.storage.conn.execute(
                "UPDATE chunks SET vec=? WHERE paper_id=? AND seq=?",
                (serialize_vec(vec), pid, seq))
        self.storage.conn.commit()
        # 持久化 vectorizer（供查询向量化）
        vs.save(self.vectorizer_path())
        return paper_vecs, chunk_vecs

    def vectorizer_path(self):
        return Path(self.cfg["index_dir"]) / "vectorizer.pkl"

    def _load_vectorizer(self):
        from .vector_store import VectorStore
        path = self.vectorizer_path()
        if not path.exists():
            return None
        return VectorStore.load(path)

    def query_top_k(self, question: str, k: int = 5) -> list:
        """RAG 检索：问题向量化后与块向量做余弦相似度，返回 TopK 块。"""
        from .storage import deserialize_vec
        vec = self._load_vectorizer()
        if vec is None:
            return []
        q_vec = vec.transform([question]).getrow(0)
        chunk_rows = self.storage.get_all_chunk_vecs()
        if not chunk_rows:
            return []
        papers_map = {p["id"]: p for p in self.storage.list_papers()}
        items = []
        for r in chunk_rows:
            v = deserialize_vec(r["vec"])
            if v is None:
                continue
            items.append((r["id"], r["paper_id"], v))
        if not items:
            return []
        from .vector_store import VectorStore
        results = VectorStore.top_k_similar(q_vec, {i: v for i, _, v in items}, k=k)
        out = []
        id2row = {i: (pid, ) for i, pid, _ in items}
        id2text = {}
        for r in self.storage.conn.execute(
                "SELECT id, paper_id, seq, text FROM chunks").fetchall():
            id2text[r["id"]] = r
        for cid, score in results:
            row = id2text.get(cid)
            if not row:
                continue
            p = papers_map.get(row["paper_id"], {})
            out.append({
                "paper_id": row["paper_id"],
                "title": p.get("title", ""),
                "seq": row["seq"],
                "text": row["text"],
                "score": round(float(score), 4),
            })
        return out
