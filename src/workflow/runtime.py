# -*- coding: utf-8 -*-
"""多智能体工作流运行时。

按 Paper-Agent 的工作流编排五个节点：
  search -> read -> analyse -> writing_outline -> writing

运行时职责：
1. 按顺序执行各节点，共享 State 字典；
2. 每个节点开始/结束/出错时产生事件（供 SSE 推送与前端进度展示）；
3. 任务持久化：把状态与产物写入 data/index/research/{run_id}/；
4. 支持中断：检查 cancellation 标志，节点之间安全停止。

事件结构（与前端约定）：
  {"type": "stage", "stage": "search", "status": "started|done|error",
   "message": "..."}
  {"type": "progress", "stage": "search", "message": "...", "data": {...}}
  {"type": "artifact", "stage": "read", "name": "...", "data": {...}}
"""
import json
import logging
import threading
import time
import uuid
from pathlib import Path

logger = logging.getLogger("arch.workflow.runtime")

# 五个节点的展示名
STAGE_NAMES = {
    "search": "🔍 检索规划与本地检索",
    "read": "📖 摘要阅读与相关性判断",
    "analyse": "🔬 分层分析（子主题 + 全局综合）",
    "writing_outline": "✍️ 综述大纲生成",
    "writing": "📝 综述正文写作",
}


class WorkflowCancellation:
    """保存一次运行是否收到用户停止请求。"""

    def __init__(self):
        self._flag = threading.Event()

    def request(self):
        self._flag.set()

    def is_requested(self):
        return self._flag.is_set()

    def raise_if_requested(self):
        if self._flag.is_set():
            raise InterruptedError("用户请求停止工作流")


class ResearchTask:
    """一次调研任务：状态 + 事件队列 + 产物。"""

    def __init__(self, run_id, topic, constraints=None):
        self.run_id = run_id
        self.topic = topic
        self.constraints = constraints or {}
        self.state = None          # 最终/当前共享状态
        self.status = "pending"    # pending/running/done/error/cancelled
        self.error = None
        self.created_at = time.time()
        self.finished_at = None
        self.events = []           # 全部事件（供重放）
        self._lock = threading.Lock()

    def emit(self, event: dict):
        """记录一条事件（加时间戳与 run_id）。"""
        with self._lock:
            ev = dict(event)
            ev.setdefault("run_id", self.run_id)
            ev.setdefault("ts", time.time())
            self.events.append(ev)
        return ev

    def to_dict(self):
        return {
            "run_id": self.run_id,
            "topic": self.topic,
            "status": self.status,
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "event_count": len(self.events),
            "current_step": (self.state or {}).get("current_step", "init"),
        }


class WorkflowRuntime:
    """工作流运行时：持有任务注册表与本地资源。"""

    def __init__(self, cfg, storage):
        self.cfg = cfg
        self.storage = storage
        self.tasks = {}            # run_id -> ResearchTask
        self._lock = threading.Lock()

    # ---------- 任务管理 ----------

    def create_task(self, topic, constraints=None):
        run_id = uuid.uuid4().hex[:12]
        task = ResearchTask(run_id, topic, constraints)
        with self._lock:
            self.tasks[run_id] = task
        self._save_task(task)
        return task

    def get_task(self, run_id):
        with self._lock:
            task = self.tasks.get(run_id)
        if task is None:
            task = self._load_task(run_id)  # 从磁盘恢复
        return task

    def list_tasks(self):
        # 先从内存取，再从磁盘目录扫描（崩溃恢复）
        with self._lock:
            tasks = list(self.tasks.values())
        seen = {t.run_id for t in tasks}
        base = Path(self.cfg["index_dir"]) / "research"
        if base.exists():
            for d in sorted(base.iterdir(), reverse=True):
                if d.is_dir() and d.name not in seen:
                    t = self._load_task(d.name)
                    if t:
                        tasks.append(t)
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return tasks

    # ---------- 执行 ----------

    def run(self, run_id):
        """在工作线程中执行完整工作流（同步，节点内调 LLM）。"""
        task = self.get_task(run_id)
        if task is None:
            return
        task.status = "running"
        task.emit({"type": "stage", "stage": "init", "status": "started",
                   "message": "调研任务已启动"})
        task.emit({"type": "progress", "stage": "init",
                   "message": f"主题：{task.topic}"})

        from ..llm_client import LLMClient, OllamaError
        from .agents import (AnalyseAgent, ReadAgent, SearchAgent,
                             WritingAgent, WritingOutlineAgent)
        from .local_search import LocalSearchService

        llm = LLMClient(self.cfg)
        if not llm.health():
            task.status = "error"
            task.error = "本地模型（Ollama）未连接，无法执行调研工作流"
            task.finished_at = time.time()
            task.emit({"type": "stage", "stage": "init", "status": "error",
                       "message": task.error})
            self._save_task(task)
            return

        cancellation = WorkflowCancellation()
        search = LocalSearchService(self.storage, self.cfg)
        agents = {
            "search": SearchAgent(llm),
            "read": ReadAgent(llm),
            "analyse": AnalyseAgent(llm),
            "writing_outline": WritingOutlineAgent(llm),
            "writing": WritingAgent(llm),
        }
        state = {
            "request": {"topic": task.topic, "constraints": task.constraints},
            "search_results": [],
            "read_results": [],
            "analysis_report": {},
            "writing_outline": {},
            "writing_sections": [],
            "final_report": {},
            "current_step": "init",
            "diagnostics": {},
        }

        try:
            # ---- 1. 检索节点 ----
            self._run_stage(task, "search", cancellation, state, agents)
            intent = state.get("search_intent") or {}
            keywords = intent.get("keywords", [])
            query = intent.get("topic", task.topic)
            task.emit({"type": "progress", "stage": "search",
                       "message": f"开始本地语料检索（关键词 {len(keywords)} 组）",
                       "data": {"keywords": keywords}})
            papers = search.combined_search(
                keywords, query,
                max_results=int(state.get("search_intent", {}).get(
                    "max_results", 8)))
            # 补充全文开头片段给阅读节点
            for p in papers:
                chunks = self.storage.get_chunks(p["id"])
                p["text_head"] = chunks[0]["text"][:1200] if chunks else ""
            state["search_results"] = papers
            task.emit({"type": "artifact", "stage": "search",
                       "name": "检索结果",
                       "data": [{"id": p["id"], "title": p.get("title") or p["filename"],
                                 "year": p.get("year", "")} for p in papers]})

            # ---- 2. 阅读节点 ----
            self._run_stage(task, "read", cancellation, state, agents)

            # ---- 3. 分析节点 ----
            self._run_stage(task, "analyse", cancellation, state, agents)

            # ---- 4. 大纲节点 ----
            self._run_stage(task, "writing_outline", cancellation, state,
                            agents)

            # ---- 5. 写作节点 ----
            self._run_stage(task, "writing", cancellation, state, agents)

            task.state = state
            task.status = "done"
            task.emit({"type": "stage", "stage": "done", "status": "done",
                       "message": "🎉 调研完成！综述已生成"})
        except InterruptedError:
            task.status = "cancelled"
            task.emit({"type": "stage", "stage": state.get("current_step"),
                       "status": "cancelled", "message": "任务已取消"})
        except Exception as e:  # noqa: BLE001
            logger.exception("调研工作流 %s 失败", run_id)
            task.status = "error"
            task.error = str(e)
            task.emit({"type": "stage", "stage": state.get("current_step"),
                       "status": "error", "message": f"执行出错：{e}"})
        finally:
            task.finished_at = time.time()
            task.state = state
            self._save_task(task)

    def cancel(self, run_id):
        task = self.get_task(run_id)
        if task:
            task.status = "cancelled"
            task.emit({"type": "stage", "stage": task.state.get("current_step")
                       if task.state else "init", "status": "cancelled",
                       "message": "任务已取消"})
            task.finished_at = time.time()
            self._save_task(task)

    # ---------- 内部 ----------

    def _run_stage(self, task, stage, cancellation, state, agents):
        """执行单个节点，带进度事件与错误兜底。"""
        cancellation.raise_if_requested()
        task.emit({"type": "stage", "stage": stage, "status": "started",
                   "message": STAGE_NAMES.get(stage, stage)})
        try:
            update = agents[stage].run(state)
        except InterruptedError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.error("节点 %s 失败: %s", stage, e)
            task.emit({"type": "progress", "stage": stage,
                       "message": f"⚠️ 节点失败（降级继续）：{e}"})
            state.setdefault("diagnostics", {})[stage] = str(e)
            update = {"current_step": stage}  # 兜底：只更新步骤，不阻塞
        state.update(update)
        task.emit({"type": "stage", "stage": stage, "status": "done",
                   "message": f"{STAGE_NAMES.get(stage, stage)} 完成"})

    def _save_task(self, task):
        """持久化任务（状态 + 产物）到 data/index/research/{run_id}/。"""
        base = Path(self.cfg["index_dir"]) / "research" / task.run_id
        try:
            base.mkdir(parents=True, exist_ok=True)
            meta = task.to_dict()
            (base / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
            if task.state:
                (base / "state.json").write_text(
                    json.dumps(task.state, ensure_ascii=False, indent=1,
                               default=str), encoding="utf-8")
            events = [{"type": e.get("type"), "stage": e.get("stage"),
                       "status": e.get("status"), "message": e.get("message"),
                       "ts": e.get("ts")} for e in task.events]
            (base / "events.json").write_text(
                json.dumps(events, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            logger.warning("任务持久化失败 %s: %s", task.run_id, e)

    def _load_task(self, run_id):
        """从磁盘恢复任务（服务重启后）。"""
        base = Path(self.cfg["index_dir"]) / "research" / run_id
        meta_path = base / "meta.json"
        if not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            task = ResearchTask(meta.get("run_id", run_id),
                                meta.get("topic", ""))
            task.status = meta.get("status", "pending")
            task.error = meta.get("error")
            task.created_at = meta.get("created_at", time.time())
            task.finished_at = meta.get("finished_at")
            state_path = base / "state.json"
            if state_path.exists():
                task.state = json.loads(state_path.read_text(encoding="utf-8"))
            ev_path = base / "events.json"
            if ev_path.exists():
                task.events = json.loads(ev_path.read_text(encoding="utf-8"))
            with self._lock:
                self.tasks[run_id] = task
            return task
        except Exception as e:  # noqa: BLE001
            logger.warning("任务恢复失败 %s: %s", run_id, e)
            return None
