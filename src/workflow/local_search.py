# -*- coding: utf-8 -*-
"""本地语料检索服务。

代替 Paper-Agent 的在线论文检索（arXiv/OpenAlex/Semantic Scholar），
改为在本地语料库（SQLite + TF-IDF 向量）中检索。这样：
1. 完全离线，符合"本地部署模型"的定位；
2. 检索的是用户自己收集的考古学论文，更贴合研究场景。

对外提供两个方法：
- search_by_keywords：按关键词全文检索（标题/摘要加权）
- search_by_similarity：按向量相似度检索（语义相关）
"""
import logging

logger = logging.getLogger("arch.workflow.search")


class LocalSearchService:
    """本地语料检索服务，封装 storage 的全文检索与向量检索。"""

    def __init__(self, storage, cfg):
        self.storage = storage
        self.cfg = cfg

    def search_by_keywords(self, keywords, max_results=10, excluded=None):
        """按关键词在本地语料中全文检索。

        关键词是 SearchAgent 生成的中文/英文关键词列表，
        这里对每个关键词做全文检索，然后按命中论文去重合并。
        """
        excluded = set(excluded or [])
        merged = {}
        for kw in keywords:
            if not kw or len(kw) < 2:
                continue
            try:
                results = self.storage.fulltext_search(kw, limit=max_results * 2)
            except Exception as e:  # noqa: BLE001
                logger.warning("关键词检索失败 %r: %s", kw, e)
                continue
            for paper, info in results:
                pid = paper["id"]
                if pid in excluded:
                    continue
                if pid not in merged:
                    merged[pid] = {
                        "paper": paper,
                        "score": info["score"],
                        "matched_keywords": [kw],
                    }
                else:
                    merged[pid]["score"] += info["score"]
                    merged[pid]["matched_keywords"].append(kw)
        # 按总得分排序
        ranked = sorted(merged.values(),
                        key=lambda x: x["score"], reverse=True)
        return [item["paper"] for item in ranked[:max_results]]

    def search_by_similarity(self, query, max_results=10):
        """按向量相似度检索（语义相关，需已重建向量）。"""
        try:
            from ..knowledge import KnowledgeBase
            kb = KnowledgeBase(self.cfg)
            try:
                hits = kb.query_top_k(query, k=max_results)
            finally:
                kb.close()
            papers = []
            for h in hits:
                paper = self.storage.get_paper(h["paper_id"])
                if paper:
                    paper["_sim_score"] = h["score"]
                    papers.append(paper)
            return papers
        except Exception as e:  # noqa: BLE001
            logger.warning("向量检索失败（可能是未重建向量）: %s", e)
            return []

    def combined_search(self, keywords, query, max_results=10, excluded=None):
        """关键词检索 + 向量检索合并，用于检索节点。

        优先关键词命中，不足时用向量检索补充。
        """
        by_kw = self.search_by_keywords(keywords, max_results, excluded)
        if len(by_kw) >= max_results:
            return by_kw[:max_results]
        by_sim = self.search_by_similarity(query, max_results * 2)
        seen = {p["id"] for p in by_kw}
        for p in by_sim:
            if p["id"] in seen:
                continue
            by_kw.append(p)
            if len(by_kw) >= max_results:
                break
        return by_kw[:max_results]
