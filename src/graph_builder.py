# -*- coding: utf-8 -*-
"""关系图构建：相似边 + 主题聚类 + 引文边 → graph.json（原子写入）。

v4 计划要点：
- 相似边：numpy 余弦相似度矩阵，> 阈值取 TopK
- 聚类：KMeans（簇数 min(max_clusters, n//2)），LLM 命名主题（降级高频词）
- 引文边：LLM 提取参考文献标题优先，正则兜底，difflib 0.8 匹配
- 原子写：临时文件 → rename，失败保留旧图
- networkx 计算度中心性
"""
import json
import logging
import re
import tempfile
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

logger = logging.getLogger("arch.graph")


def _regex_reference_titles(tail: str) -> list:
    """正则提取参考文献标题（兜底路径）。

    匹配 [n] 作者. 标题[J/M/D/C]. 期刊, 年(期): 页码.
    """
    titles = []
    for line in tail.split("\n"):
        line = line.strip()
        if not re.match(r"^\[\d+\]?\s*", line) and not re.match(r"^\d+\.", line):
            continue
        # 去掉序号与作者段，取标题段（在 [J]/[M] 等标记前的书名号或连续文本）
        m = re.search(r"《([^》]{4,60})》", line)
        if m:
            titles.append(m.group(1))
            continue
        body = re.sub(r"^(\[\d+\]|\d+\.)\s*", "", line)
        parts = re.split(r"\.\s*", body)
        if len(parts) >= 2:
            # 作者段之后到 [J] 标记前是标题
            t = parts[0]
            if len(t) >= 4 and len(t) <= 80:
                titles.append(t)
    return titles


def _clean_title(t: str) -> str:
    return re.sub(r"[\s《》。，,]", "", t or "").strip()


def _match_citations(ref_titles, papers):
    """将参考文献标题与库内论文标题模糊匹配（difflib 阈值 0.8）。"""
    pairs = []
    paper_titles = {p["id"]: _clean_title(p.get("title") or "") for p in papers}
    for ref in ref_titles:
        rc = _clean_title(ref)
        if len(rc) < 4:
            continue
        best_id, best_score = None, 0.0
        for pid, pt in paper_titles.items():
            if not pt:
                continue
            s = SequenceMatcher(None, rc, pt).ratio()
            if s > best_score:
                best_id, best_score = pid, s
        if best_id and best_score >= 0.8:
            pairs.append((best_id, best_score))
    return pairs


def build_graph(cfg, storage, paper_vecs, llm=None) -> dict:
    """构建关系图。返回 graph 数据 dict 并原子写入 graph.json。"""
    papers = storage.list_papers()
    n = len(papers)
    if n == 0:
        graph = {"nodes": [], "edges": []}
        _atomic_write(cfg["graph_path"], graph)
        return graph

    # ---------- 1. 相似边 ----------
    similar_edges = []
    if paper_vecs:
        from .vector_store import VectorStore
        ids = [p["id"] for p in papers]
        vecs = [paper_vecs.get(pid) for pid in ids]
        present = [(i, v) for i, v in enumerate(vecs) if v is not None]
        if len(present) >= 2:
            idxs = [i for i, _ in present]
            sub = [v for _, v in present]
            sim = VectorStore.similarity_matrix(sub)
            thr = cfg.get("similarity_threshold", 0.35)
            topk = cfg.get("top_k_edges", 8)
            for a in range(len(sub)):
                order = np.argsort(-sim[a])
                cnt = 0
                for b in order:
                    if b == a:
                        continue
                    if sim[a, b] < thr:
                        break
                    pa, pb = ids[idxs[a]], ids[idxs[b]]
                    similar_edges.append(
                        {"source": pa, "target": pb, "type": "similar",
                         "weight": round(float(sim[a, b]), 4)})
                    cnt += 1
                    if cnt >= topk:
                        break

    # ---------- 2. 主题聚类 ----------
    themes = _cluster_themes(cfg, papers, paper_vecs, llm)

    # ---------- 3. 引文边 ----------
    citation_edges = []
    # 逐篇论文提取其参考文献标题（记录引用者 source）
    for p in papers:
        text = storage.get_paper_text(p["id"])
        tail = _find_ref_zone(text)
        refs = []
        if llm:
            try:
                refs = llm.extract_reference_titles(tail)
            except Exception as e:  # noqa: BLE001
                logger.warning("LLM 引文提取失败，使用正则: %s", e)
        if not refs:
            refs = _regex_reference_titles(tail)
        if not refs:
            continue
        for target_id, score in _match_citations(refs, papers):
            if target_id == p["id"]:
                continue  # 自引忽略
            citation_edges.append(
                {"source": p["id"], "target": target_id, "type": "citation",
                 "weight": round(float(score), 3)})

    # ---------- 4. 度中心性与主题写回 ----------
    degrees = _compute_degrees(papers, similar_edges, citation_edges)
    storage.update_graph_degree(degrees)
    # 主题写回数据库（综述导出按 theme 分组）
    for p in papers:
        t = themes.get(p["id"], "")
        if t and t != p.get("theme", ""):
            storage.update_paper_meta(p["id"], theme=t)

    # ---------- 5. 组装节点 ----------
    nodes = []
    for p in papers:
        nodes.append({
            "id": p["id"],
            "title": p.get("title") or p["filename"],
            "authors": p.get("authors", ""),
            "year": p.get("year", ""),
            "theme": themes.get(p["id"], ""),
            "degree": degrees.get(p["id"], 0),
            "status": p.get("status", "未读"),
            "summary": p.get("summary", ""),
            "keywords": p.get("keywords", ""),
        })

    graph = {"nodes": nodes, "edges": similar_edges + citation_edges,
             "stats": {"paper_count": n, "edge_count": len(similar_edges) + len(citation_edges),
                       "similar_edges": len(similar_edges),
                       "citation_edges": len(citation_edges)}}
    _atomic_write(cfg["graph_path"], graph)
    logger.info("graph.json 已生成：%d 节点, %d 相似边, %d 引文边",
                n, len(similar_edges), len(citation_edges))
    return graph


def _find_ref_zone(text: str) -> str:
    for marker in ("参考文献", "参考资料", "REFERENCES", "References"):
        idx = text.rfind(marker)
        if idx != -1:
            return text[idx:]
    return text[-1500:]


def _cluster_themes(cfg, papers, paper_vecs, llm):
    """KMeans 聚类 + 主题命名。返回 {paper_id: theme}。"""
    ids = [p["id"] for p in papers]
    vecs = [paper_vecs.get(pid) for pid in ids]
    if len([v for v in vecs if v is not None]) < 2:
        return {pid: "" for pid in ids}
    try:
        from scipy import sparse
        from sklearn.cluster import KMeans
        mat = sparse.vstack([v for v in vecs if v is not None]).tocsr()
        k = min(cfg.get("max_clusters", 8), max(2, len(ids) // 2))
        k = min(k, len(ids))
        km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(mat)
        labels = km.labels_
    except Exception as e:  # noqa: BLE001
        logger.warning("KMeans 失败，主题留空: %s", e)
        return {pid: "" for pid in ids}

    # 每簇命名
    clusters = {}
    for i, pid in enumerate(ids):
        if vecs[i] is not None:
            clusters.setdefault(int(labels[i]), []).append(pid)
    themes = {}
    paper_by_id = {p["id"]: p for p in papers}
    for cid, pids in clusters.items():
        titles = [paper_by_id[pid]["title"] or "" for pid in pids]
        titles = [t for t in titles if t]
        name = ""
        if llm and titles:
            try:
                name = llm.name_cluster(titles)
            except Exception:  # noqa: BLE001
                name = ""
        if not name or name == "未命名主题":
            name = _fallback_theme(titles)
        for pid in pids:
            themes[pid] = name
    return themes


def _fallback_theme(titles) -> str:
    """无 LLM 时主题命名降级：取标题最长公共子串（简化：最高频 2-gram 词）。"""
    from collections import Counter
    freq = Counter()
    for t in titles:
        for w in re.findall(r"[\u4e00-\u9fa5]{2,4}", t):
            freq[w] += 1
    if freq:
        return freq.most_common(1)[0][0]
    return "未命名主题"


def _compute_degrees(papers, similar_edges, citation_edges):
    degrees = {p["id"]: 0 for p in papers}
    for e in similar_edges:
        degrees[e["source"]] = degrees.get(e["source"], 0) + 1
        degrees[e["target"]] = degrees.get(e["target"], 0) + 1
    for e in citation_edges:
        if e.get("target") is not None:
            degrees[e["target"]] = degrees.get(e["target"], 0) + 2  # 被引加权
    return degrees


def _atomic_write(path, data):
    """原子写 JSON：临时文件 → rename，失败保留旧文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with open(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        tmp_path = Path(tmp)
        if path.exists():
            path.unlink()
        tmp_path.rename(path)
    except Exception:
        try:
            Path(tmp).unlink()
        except OSError:
            pass
        raise
