# -*- coding: utf-8 -*-
"""向量存储：TF-IDF 稀疏向量 + 全量重建 + numpy 相似度矩阵 + TopK 检索。

v4 计划要点：
- 每次 ingest/build-graph 后全量重建（避免向量空间漂移）
- scipy.sparse 稀疏存储（防内存失控），论文级 + 块级两套向量
- numpy 向量化相似度矩阵，不写 Python 循环
- 未安装 sklearn 时降级为手写 TF-IDF（n-gram 分词兜底）
"""
import logging

import numpy as np

logger = logging.getLogger("arch.vector")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAS_SKLEARN = True
except Exception:  # pragma: no cover
    HAS_SKLEARN = False

from .text_prep import tokenize  # noqa: E402


class VectorStore:
    """论文级与块级 TF-IDF 向量。

    论文向量：标题 + 摘要区 + 关键词加权拼接后的文档向量（用于相似度/聚类）。
    块向量：每个分块的文档向量（用于 RAG 检索）。
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._vec = None

    def build(self, papers, chunks_by_paper):
        """全量重建向量空间。

        papers: [{id, title, summary, keywords, text_head}, ...]
        chunks_by_paper: {paper_id: [(seq, text), ...]}
        返回 (paper_vecs, chunk_vecs)：
          paper_vecs: {paper_id: scipy.sparse.csr}
          chunk_vecs: [(paper_id, seq, text, scipy.sparse.csr), ...]
        """
        # ---- 论文级文档 ----
        doc_ids = []
        docs = []
        for p in papers:
            title = p.get("title", "") or ""
            summary = p.get("summary", "") or ""
            keywords = p.get("keywords", "") or ""
            head = p.get("text_head", "") or ""
            # 关键词加权：重复拼接提升权重（TF-IDF 中提升词频）
            weighted = f"{title} {title} {summary} {keywords} {keywords} {head}"
            docs.append(weighted)
            doc_ids.append(p["id"])

        # ---- 块级文档 ----
        chunk_items = []  # (paper_id, seq, text)
        for pid, chunks in chunks_by_paper.items():
            for seq, text in chunks:
                chunk_items.append((pid, seq, text))

        if not docs and not chunk_items:
            return {}, []

        # 统一词典拟合（论文 + 块同空间，避免维度不一致）
        all_texts = docs + [c[2] for c in chunk_items]
        self._vec = self._fit_vectorizer(all_texts)
        vec = self._vec

        paper_vecs = {}
        if docs:
            X = vec.transform(docs)
            for i, pid in enumerate(doc_ids):
                paper_vecs[pid] = X.getrow(i)

        chunk_vecs = []
        if chunk_items:
            Xc = vec.transform([c[2] for c in chunk_items])
            for i, (pid, seq, text) in enumerate(chunk_items):
                chunk_vecs.append((pid, seq, text, Xc.getrow(i)))

        logger.info("向量重建完成：论文 %d 篇，分块 %d 条",
                    len(paper_vecs), len(chunk_vecs))
        return paper_vecs, chunk_vecs

    def _fit_vectorizer(self, texts):
        if HAS_SKLEARN:
            return TfidfVectorizer(tokenizer=self._tok,
                                   token_pattern=None,
                                   max_features=20000,
                                   sublinear_tf=True).fit(texts)
        # 手写 TF-IDF 兜底（轻量化降级路径）
        return _HandTfidf().fit(texts)

    def save(self, path):
        """持久化 vectorizer（joblib 优先，pickle 兜底）。"""
        import pickle
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._vec, f)

    @classmethod
    def load(cls, path):
        import pickle
        obj = cls.__new__(cls)
        with open(path, "rb") as f:
            obj._vec = pickle.load(f)
        return obj

    def transform(self, texts):
        return self._vec.transform(texts)

    @staticmethod
    def _tok(text):
        return tokenize(text)

    # ---------- 相似度与检索 ----------

    @staticmethod
    def similarity_matrix(vecs):
        """numpy 余弦相似度矩阵（稀疏输入，稠密输出）。"""
        if not vecs:
            return np.zeros((0, 0))
        from scipy import sparse
        mat = sparse.vstack(list(vecs)).tocsr()
        norm = np.sqrt(np.asarray(mat.multiply(mat).sum(axis=1)).ravel())
        norm[norm == 0] = 1.0
        mat_n = mat.multiply(1.0 / norm[:, None]).tocsr()
        sim = mat_n @ mat_n.T
        return np.asarray(sim.toarray())

    @staticmethod
    def top_k_similar(vec, vecs, k=5, exclude_ids=None):
        """返回与 vec 最相似的 TopK 项。vecs: {key: csr}"""
        if not vecs:
            return []
        from scipy import sparse
        keys = list(vecs.keys())
        mat = sparse.vstack([vecs[k] for k in keys]).tocsr()
        v = vec
        vn = np.sqrt(v.multiply(v).sum())
        if vn == 0:
            return []
        v = v.multiply(1.0 / vn)
        norms = np.sqrt(np.asarray(mat.multiply(mat).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        mat_n = mat.multiply(1.0 / norms[:, None]).tocsr()
        sims = np.asarray((mat_n @ v.T).toarray()).ravel()
        order = np.argsort(-sims)
        out = []
        for idx in order:
            key = keys[idx]
            if exclude_ids and key in exclude_ids:
                continue
            out.append((key, float(sims[idx])))
            if len(out) >= k:
                break
        return out


class _HandTfidf:
    """sklearn 缺失时的手写 TF-IDF 兜底（仅支持 fit/transform）。"""

    def __init__(self):
        self.vocab_ = None
        self.idf_ = None

    def fit(self, texts):
        from collections import Counter
        dfs = Counter()
        token_sets = []
        for t in texts:
            toks = set(tokenize(t))
            token_sets.append(toks)
            for w in toks:
                dfs[w] += 1
        self.vocab_ = {w: i for i, w in enumerate(dfs)}
        n = len(texts)
        import math
        self.idf_ = np.array([
            math.log((1 + n) / (1 + dfs[w])) + 1 for w in self.vocab_])
        return self

    def transform(self, texts):
        from scipy import sparse
        from collections import Counter
        rows, cols, vals = [], [], []
        for r, t in enumerate(texts):
            tf = Counter(tokenize(t))
            for w, c in tf.items():
                if w in self.vocab_:
                    rows.append(r)
                    cols.append(self.vocab_[w])
                    vals.append(c)
        mat = sparse.csr_matrix(
            (vals, (rows, cols)), shape=(len(texts), len(self.vocab_)))
        # TF 归一化后乘 IDF
        from sklearn.preprocessing import normalize
        return mat.multiply(self.idf_).tocsr()
