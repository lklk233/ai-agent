# -*- coding: utf-8 -*-
"""导出：综述 Markdown + GB/T 7714 引文格式化。

v4 计划要点：
- 综述：按主题分组 → LLM 段落（失败降级纯模板）→ Markdown（含引文列表）
- 引文：GB/T 7714 格式，缺失字段标 [未提取]
"""
import logging
import time
from pathlib import Path

logger = logging.getLogger("arch.export")


def format_gbt7714(p: dict) -> str:
    """GB/T 7714-2015 期刊引文格式。缺失字段显式标注。"""
    authors = p.get("authors") or "[未提取]"
    title = p.get("title") or "[未提取]"
    journal = p.get("journal") or "[未提取]"
    year = p.get("year") or "[未提取]"
    pages = p.get("pages") or "[未提取]"
    return f"{authors}. {title}[J]. {journal}, {year}: {pages}."


def export_citations(papers: list) -> str:
    """导出多篇论文的引文列表（按年份排序）。"""
    items = [format_gbt7714(p) for p in papers]
    return "\n".join(items)


def group_by_theme(papers: list) -> dict:
    """按主题分组。无主题的归入「未分类」。"""
    groups = {}
    for p in papers:
        theme = (p.get("theme") or "").strip() or "未分类"
        groups.setdefault(theme, []).append(p)
    return groups


def build_review_markdown(cfg, papers: list, llm=None) -> str:
    """生成综述 Markdown 文本（不写文件，由调用方保存/返回）。"""
    groups = group_by_theme(papers)
    lines = []
    now = time.strftime("%Y-%m-%d")
    lines.append(f"# 考古学文献研究现状综述（{now}）")
    lines.append("")
    lines.append(f"共收录论文 {len(papers)} 篇，分为 {len(groups)} 个主题。")
    lines.append("")
    lines.append("---")
    lines.append("")

    for theme, plist in sorted(groups.items(), key=lambda x: -len(x[1])):
        lines.append(f"## 主题：{theme}")
        lines.append("")
        # LLM 段落
        section = ""
        if llm:
            try:
                section = llm.write_review_section(theme, plist)
            except Exception as e:  # noqa: BLE001
                logger.warning("综述段落 LLM 失败（降级模板）: %s", e)
                section = ""
        if section:
            lines.append(section)
            lines.append("")
        lines.append(f"**该主题论文（{len(plist)} 篇）：**")
        lines.append("")
        for p in sorted(plist, key=lambda x: str(x.get("year", ""))):
            lines.append(f"- 《{p.get('title') or '(未命名)'}》"
                         f"（{p.get('authors') or '作者未提取'}，"
                         f"{p.get('year') or '年份未提取'}）"
                         f"：{p.get('summary') or '（无摘要）'}")
        lines.append("")
        # 该主题引文
        lines.append("参考文献：")
        lines.append("")
        for p in plist:
            lines.append(f"{format_gbt7714(p)}")
        lines.append("")
        lines.append("---")
        lines.append("")

    # 全局引文汇总
    lines.append("## 全部参考文献（GB/T 7714）")
    lines.append("")
    for p in sorted(papers, key=lambda x: str(x.get("year", ""))):
        lines.append(f"{format_gbt7714(p)}")
    lines.append("")
    return "\n".join(lines)


def save_review(cfg, papers: list, llm=None) -> Path:
    """生成并保存综述到 data/index/综述_日期.md。返回路径。"""
    md = build_review_markdown(cfg, papers, llm)
    out = Path(cfg["index_dir"]) / f"综述_{time.strftime('%Y%m%d_%H%M')}.md"
    out.write_text(md, encoding="utf-8")
    logger.info("综述已保存: %s", out)
    return out
