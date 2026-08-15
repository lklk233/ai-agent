# -*- coding: utf-8 -*-
"""工作流共享状态定义。

借鉴 Paper-Agent 的 LangGraph State 设计，但保持轻量：
用普通 dict 保存各节点之间的共享数据，节点之间通过明确的键名传递。
"""
from dataclasses import dataclass, field
from typing import Any

# 一次调研任务的完整状态（dict 的键说明）：
#   request          用户发起的调研请求 {topic, constraints, language}
#   search_intent    检索计划 {topic, keywords, subtopics, max_results}
#   search_results   本地语料检索到的候选论文列表
#   read_results     阅读节点筛选后的论文及笔记
#   analysis_report  分析节点的输出（子主题分析 + 全局综合）
#   writing_outline  写作大纲（章节结构）
#   writing_sections 逐节写好的正文
#   final_report     最终产物（综述全文 + 引用）
#   current_step     当前进行到哪个阶段（用于进度展示）
#   events           运行过程中产生的事件记录（SSE 推送用）
#   diagnostics      各节点的诊断/错误信息

DEFAULT_STATE = {
    "request": None,
    "search_intent": None,
    "search_results": [],
    "search_summary": {},
    "read_results": [],
    "read_summary": {},
    "analysis_report": {},
    "writing_outline": {},
    "writing_sections": [],
    "final_report": {},
    "current_step": "init",
    "events": [],
    "diagnostics": {},
}


def make_state(request=None) -> dict:
    """创建一个全新的工作流状态。"""
    state = dict(DEFAULT_STATE)
    if request:
        state["request"] = request
    return state
