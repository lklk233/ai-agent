# -*- coding: utf-8 -*-
"""五个文献调研 Agent：Search / Read / Analyse / WritingOutline / Writing。

借鉴 Paper-Agent 的多智能体设计：每个 Agent 接收共享状态，返回局部状态更新。
所有 LLM 调用都走本地 Ollama（llm_client.LLMClient），保留本地部署方式。

每个 Agent 的职责：
- SearchAgent：把主题拆成研究子方向与关键词（检索计划）
- ReadAgent：读论文摘要，三维匹配判断相关性，生成阅读笔记
- AnalyseAgent：先按子主题横向比较，再做全局综合（八字段）
- WritingOutlineAgent：生成综述大纲与证据映射
- WritingAgent：按大纲逐节写作，最后生成摘要
"""
import json
import logging
import time

from . import prompts

logger = logging.getLogger("arch.workflow.agents")

# 综合分析的八个固定字段（与提示词一致）
OVERALL_FIELDS = [
    "领域整体研究概况", "领域全域共性研究共识", "领域核心研究争议与矛盾体系",
    "领域系统性研究空白与局限", "领域研究时序演化脉络",
    "领域技术与研究方法迭代脉络", "各子主题横向差异对比分析",
    "领域整体总结与研究展望",
]


def _note_slim(note: dict) -> str:
    """把阅读笔记压缩为短文，避免超长输入导致 LLM 空响应。"""
    try:
        parts = []
        if note.get("short_summary"):
            parts.append(f"摘要：{note['short_summary']}")
        if note.get("main_question"):
            parts.append(f"问题：{note['main_question']}")
        if note.get("methods"):
            parts.append("方法：" + "、".join(note["methods"][:3]))
        if note.get("contributions"):
            parts.append("贡献：" + "；".join(note["contributions"][:2]))
        return "；".join(parts)[:400]
    except Exception:  # noqa: BLE001
        return ""


class _AgentBase:
    """所有 Agent 的公共基类：持有 LLM 客户端与提示词。"""

    def __init__(self, llm):
        self.llm = llm

    def _chat_json(self, system, user, max_tokens=1200, require_keys=None,
                   retries=3):
        """调用本地 LLM 并解析 JSON。

        require_keys: 期望返回 dict 中必须存在的键（deepseek-r1 偶发返回空 {}，
        此时按失败重试，避免把空结果写进状态）。重试间加短暂延迟，
        减少连续快速空响应的概率。
        """
        last = None
        for attempt in range(retries + 1):
            try:
                data = self.llm.chat(
                    [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
                    format_json=True, temperature=0.2, max_tokens=max_tokens)
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(1.0)
                continue
            if not isinstance(data, dict):
                last = ValueError("LLM 未返回 JSON 对象")
                time.sleep(1.0)
                continue
            if require_keys:
                missing = [k for k in require_keys if not data.get(k)]
                if missing:
                    last = ValueError(f"LLM 返回缺少字段: {missing}")
                    # 记录原始键名，便于定位 LLM 返回结构不符的问题
                    logger.warning(
                        "LLM 返回键名不符: 期望 %s, 实际键 %s, 内容前200字: %s",
                        missing, list(data.keys())[:10], str(data)[:200])
                    # 空响应时等更久再重试（给 Ollama 模型留出稳定时间）
                    time.sleep(2.0 if attempt < 2 else 0.5)
                    continue
            return data
        raise last or ValueError("LLM 调用失败")


class SearchAgent(_AgentBase):
    """检索规划 Agent：主题 -> 子方向 + 关键词。"""

    def run(self, state):
        request = state["request"]
        topic = request.get("topic", "")
        constraints = request.get("constraints", {})
        user = (
            f"用户主题：{topic}\n"
            f"用户约束（JSON，可能为空）：{json.dumps(constraints, ensure_ascii=False)}"
        )
        data = self._chat_json(prompts.SEARCH_AGENT_SYSTEM_PROMPT, user,
                               max_tokens=800, require_keys=["subtopics"])
        subtopics = []
        for st in data.get("subtopics", []):
            if isinstance(st, dict) and st.get("subtopic") and st.get("keyword"):
                subtopics.append({
                    "subtopic": str(st["subtopic"]).strip(),
                    "keyword": str(st["keyword"]).strip(),
                })
        if not subtopics:
            # 兜底：至少保留一个方向
            subtopics = [{"subtopic": topic, "keyword": topic}]
        intent = {
            "topic": topic,
            "subtopics": subtopics,
            "keywords": list({st["keyword"] for st in subtopics}),
            "max_results": int(request.get("max_results", 8)),
        }
        return {"search_intent": intent,
                "current_step": "search"}


class ReadAgent(_AgentBase):
    """阅读 Agent：逐篇读摘要，三维匹配，生成笔记。"""

    def run(self, state):
        intent = state["search_intent"] or {}
        topic = intent.get("topic", "")
        constraints = dict(state["request"].get("constraints") or {})
        papers = state.get("search_results", [])
        read_results = []
        for idx, paper in enumerate(papers):
            title = paper.get("title") or paper.get("filename", "")
            abstract = paper.get("summary") or ""
            head = paper.get("text_head") or ""
            user = (
                "【用户主题】\n" + topic + "\n\n"
                "【用户要求】\n" + json.dumps(constraints, ensure_ascii=False) + "\n\n"
                "【论文标题】\n" + title + "\n\n"
                "【论文摘要】\n" + (abstract or "（无摘要）") + "\n\n"
                "【论文开头】\n" + (head[:800] or "")
            )
            try:
                note = self._chat_json(prompts.READ_AGENT_SYSTEM_PROMPT, user,
                                       max_tokens=700,
                                       require_keys=["main_question",
                                                     "short_summary"])
            except Exception as e:  # noqa: BLE001
                logger.warning("阅读论文 %s 失败，跳过: %s", title, e)
                continue
            match = note.get("match_levels", {})
            # 三维中至少两维匹配才算相关（保留给后续分析）
            vals = [match.get(k) for k in
                    ("research_question", "research_object_or_scene",
                     "method_or_technical_route")]
            relevant = sum(1 for v in vals if v == "match") >= 1 or \
                sum(1 for v in vals if v in ("match", "partial_match")) >= 2
            read_results.append({
                "paper_id": paper["id"],
                "title": title,
                "year": paper.get("year", ""),
                "authors": paper.get("authors", ""),
                "summary": paper.get("summary", ""),
                "note": note,
                "relevant": relevant,
                "index": f"P{idx + 1}",
            })
        return {"read_results": read_results,
                "current_step": "read"}


class AnalyseAgent(_AgentBase):
    """分析 Agent：先按子主题分析，再做全局综合。"""

    def run(self, state):
        intent = state.get("search_intent") or {}
        subtopics = intent.get("subtopics", [])
        read_results = state.get("read_results", [])
        # 只分析相关的论文
        relevant = [r for r in read_results if r.get("relevant")]
        if not relevant:
            relevant = read_results  # 全部相关时兜底

        subtopic_analyses = []
        # 按子主题分组：每篇论文归属到所有子主题（简化：全部分析一次）
        for st in subtopics:
            papers_blob = "\n\n".join(
                f"[{r['index']}]《{r['title']}》（{r.get('authors', '')}，"
                f"{r.get('year', '')}）：{r.get('summary', '')}" + (
                    f" 阅读笔记：{_note_slim(r.get('note') or {})}"
                    if r.get("note") else "")
                for r in relevant)
            user = (
                f"子主题：{st.get('subtopic', '')}\n\n"
                f"论文列表（编号从P1开始，按输入顺序）：\n{papers_blob}"
            )
            try:
                analysis = self._chat_json(
                    prompts.ANALYSE_SUBTOPIC_SYSTEM_PROMPT, user,
                    max_tokens=1200, require_keys=["研究现状"])
                subtopic_analyses.append({
                    "subtopic": st.get("subtopic", ""),
                    "keyword": st.get("keyword", ""),
                    "analysis": analysis,
                })
            except Exception as e:  # noqa: BLE001
                logger.warning("子主题分析失败 %s: %s", st.get("subtopic"), e)

        # 全局综合
        overall = {}
        if subtopic_analyses:
            blob = json.dumps(subtopic_analyses, ensure_ascii=False, indent=1)
            try:
                overall = self._chat_json(
                    prompts.ANALYSE_OVERALL_SYSTEM_PROMPT, blob,
                    max_tokens=1500,
                    require_keys=["领域整体研究概况"])
                # 规整为八字段
                overall = {k: overall.get(k, "证据不足") for k in OVERALL_FIELDS}
            except Exception as e:  # noqa: BLE001
                logger.warning("全局综合分析失败: %s", e)
                overall = {k: "证据不足" for k in OVERALL_FIELDS}

        return {"analysis_report": {
                    "subtopic_analyses": subtopic_analyses,
                    "overall_analysis": overall,
                },
                "current_step": "analyse"}


class WritingOutlineAgent(_AgentBase):
    """大纲 Agent：分析结果 -> 章节大纲。"""

    def run(self, state):
        analysis = state.get("analysis_report", {})
        overall = analysis.get("overall_analysis", {})
        subtopics = analysis.get("subtopic_analyses", [])
        user = (
            "【全域综合分析（八字段）】\n"
            + json.dumps(overall, ensure_ascii=False, indent=1) + "\n\n"
            "【子主题分析】\n"
            + json.dumps(subtopics, ensure_ascii=False, indent=1)
        )
        outline = {}
        try:
            outline = self._chat_json(prompts.WRITING_OUTLINE_AGENT_SYSTEM_PROMPT,
                                      user, max_tokens=1200,
                                      require_keys=["Chapter1"])
        except Exception as e:  # noqa: BLE001
            logger.warning("大纲生成失败: %s", e)
            outline = {}
        return {"writing_outline": outline,
                "current_step": "writing_outline"}


class WritingAgent(_AgentBase):
    """写作 Agent：按大纲逐节写作，最后生成摘要。"""

    def run(self, state):
        outline = state.get("writing_outline", {})
        analysis = state.get("analysis_report", {})
        overall = analysis.get("overall_analysis", {})
        relevant = [r for r in state.get("read_results", [])
                    if r.get("relevant")] or state.get("read_results", [])
        sections = []
        # 章节展开为小节列表
        for chapter_key in sorted(outline.keys()):
            chapter = outline[chapter_key]
            if not isinstance(chapter, dict):
                continue
            title = chapter.get("title", chapter_key)
            desc = chapter.get("description", "")
            secs = chapter.get("Sections", {})
            for sec_key in sorted(secs.keys()):
                sec = secs[sec_key]
                if not isinstance(sec, dict):
                    continue
                sections.append({
                    "chapter": title,
                    "chapter_desc": desc,
                    "title": sec.get("title", sec_key),
                    "task": sec.get("task", ""),
                    "word_count": sec.get("word-count", 600),
                    "evidence_map": sec.get("evidence-map", []),
                })
        if not sections:
            sections = [{
                "chapter": "综述", "chapter_desc": "",
                "title": "研究现状", "task": "综合梳理该主题研究现状",
                "word_count": 800, "evidence_map": OVERALL_FIELDS,
            }]

        papers_blob = "\n\n".join(
            f"[{r['index']}]《{r['title']}》（{r.get('authors', '')}，"
            f"{r.get('year', '')}）：{r.get('summary', '')}" + (
                f"\n阅读笔记：{_note_slim(r.get('note') or {})}"
                if r.get("note") else "")
            for r in relevant)

        written = []
        for sec in sections:
            # 收集该小节需要的证据字段内容
            evidence_blob = "\n".join(
                f"{k}：{overall.get(k, '证据不足')}"
                for k in (sec.get("evidence_map") or [])
                if k in overall)
            user = (
                f"【小节标题】{sec['title']}\n"
                f"【写作任务】{sec['task']}\n"
                f"【目标字数】{sec['word_count']}\n\n"
                f"【可参考的全局分析字段】\n{evidence_blob or '（无，依据子主题分析）'}\n\n"
                f"【相关论文（编号从P1开始）】\n{papers_blob}"
            )
            try:
                data = self._chat_json(prompts.WRITING_AGENT_SYSTEM_PROMPT,
                                       user, max_tokens=1200,
                                       require_keys=["content"])
                content = data.get("content", "")
                paper_ids = data.get("paperIds", [])
            except Exception as e:  # noqa: BLE001
                logger.warning("写作小节失败 %s: %s", sec["title"], e)
                content, paper_ids = "", []
            written.append({
                "chapter": sec["chapter"],
                "title": sec["title"],
                "content": content,
                "paperIds": paper_ids,
                "task": sec["task"],
            })

        # 摘要
        abstract = ""
        if written:
            # 精简输入：只传标题与正文前 300 字，避免超长导致模型回显输入
            slim_sections = []
            for s in written:
                content = s["content"] or ""
                slim_sections.append({
                    "title": s["title"],
                    "content": content[:300],
                })
            blob = json.dumps({
                "user_topic": state["request"].get("topic", ""),
                "sections": slim_sections,
            }, ensure_ascii=False)
            try:
                ab = self._chat_json(prompts.WRITING_ABSTRACT_SYSTEM_PROMPT,
                                     blob, max_tokens=500,
                                     require_keys=["content"])
                abstract = ab.get("content", "")
            except Exception as e:  # noqa: BLE001
                logger.warning("摘要生成失败: %s", e)

        return {"writing_sections": written,
                "final_report": {
                    "abstract": abstract,
                    "sections": written,
                    "topic": state["request"].get("topic", ""),
                },
                "current_step": "writing"}
