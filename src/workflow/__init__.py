# -*- coding: utf-8 -*-
"""多智能体文献调研工作流（借鉴 Paper-Agent 设计，本地化实现）。

子模块：
- state.py        共享状态
- prompts.py      五 Agent 系统提示词（本地化）
- agents.py       五个 Agent 实现
- local_search.py 本地语料检索服务
- runtime.py      工作流运行时（节点编排 + SSE 事件 + 持久化）
"""
