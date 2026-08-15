# -*- coding: utf-8 -*-
"""HTTP 服务：静态托管 + REST API（纯标准库 http.server，零第三方依赖）。

v4 计划要点：
- 绑定 127.0.0.1（默认），防路径穿越（归一化 + 目录校验）
- 统一错误契约 {"error": msg}
- 每请求新建 SQLite 连接（线程安全）
- Ollama 未启动时：元数据/问答/综述返回 503 提示，其余功能照常
"""
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger("arch.server")

MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png", ".jpg": "image/jpeg", ".svg": "image/svg+xml",
    ".ico": "image/x-icon", ".txt": "text/plain; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
}


class App:
    """应用上下文：共享配置与懒加载服务。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.web_root = Path(cfg.get("web_dir",
                             str(Path(__file__).resolve().parent.parent / "web")))
        self._kb = None
        self._llm = None
        self._workflow = None
        self._lock = threading.Lock()
        self._current_model = cfg.get("llm_model", "deepseek-r1:8b")

    def knowledge(self):
        if self._kb is None:
            from .knowledge import KnowledgeBase
            self._kb = KnowledgeBase(self.cfg)
        return self._kb

    def workflow(self):
        """懒加载多智能体工作流运行时。"""
        if self._workflow is None:
            from .workflow.runtime import WorkflowRuntime
            self._workflow = WorkflowRuntime(self.cfg, self.knowledge().storage)
        return self._workflow

    def llm(self):
        if self._llm is None:
            from .llm_client import LLMClient
            self._llm = LLMClient(self.cfg)
            self._llm.set_model(self._current_model)
        return self._llm

    def llm_available(self) -> bool:
        try:
            return self.llm().health()
        except Exception:
            return False

    # ---------- 模型选择 ----------

    def current_model(self) -> str:
        return self._current_model

    def set_current_model(self, model: str) -> bool:
        """运行时切换模型并持久化到 config.json。返回是否成功。"""
        model = (model or "").strip()
        if not model:
            return False
        self._current_model = model
        self.cfg["llm_model"] = model  # 后续新建的 LLM 客户端沿用
        if self._llm is not None:
            self._llm.set_model(model)  # 已有单例立即生效
        try:
            self._persist_model(model)
        except Exception as e:
            logger.warning("模型选择未能持久化到 config.json: %s", e)
        return True

    def _persist_model(self, model: str):
        """更新项目根 config.json 的 llm_model 字段（保留其余配置）。"""
        cfg_path = Path(self.cfg.get("config_path",
                        str(Path(__file__).resolve().parent.parent / "config.json")))
        data = {}
        if cfg_path.exists():
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
        data["llm_model"] = model
        cfg_path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        logger.info("模型选择已持久化: %s", model)


class Handler(BaseHTTPRequestHandler):
    server_version = "ArchAgent/1.0"
    app: App = None  # 由工厂注入

    # ---------- 基础 ----------

    def log_message(self, fmt, *args):
        logger.info("%s - %s", self.address_string(), fmt % args)

    def _send(self, status, body: bytes, ctype="application/json; charset=utf-8"):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self._send(status, body)

    def _error(self, status, msg):
        self._json(status, {"error": msg})

    def _read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    # ---------- 路由 ----------

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path.startswith("/api/"):
            self._route_api_get(path, parse_qs(parsed.query))
        else:
            self._serve_static(path)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self._route_api_post(parsed.path, self._read_body())
        else:
            self._error(404, "not found")

    def do_PUT(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self._route_api_put(parsed.path, self._read_body())
        else:
            self._error(404, "not found")

    # ---------- 静态文件（防路径穿越） ----------

    def _serve_static(self, path):
        root = Path(self.app.web_root).resolve()
        if path in ("/", ""):
            path = "/index.html"
        try:
            rel = Path(path.lstrip("/"))
            # 防路径穿越：归一化后必须仍在 web_root 内
            target = (root / rel).resolve()
            target.relative_to(root)
        except (ValueError, OSError):
            self._error(403, "forbidden")
            return
        if not target.is_file():
            self._error(404, "not found")
            return
        ctype = MIME.get(target.suffix.lower(), "application/octet-stream")
        try:
            body = target.read_bytes()
        except OSError as e:
            self._error(500, str(e))
            return
        self._send(200, body, ctype)

    # ---------- GET API ----------

    def _route_api_get(self, path, query):
        kb = self.app.knowledge()
        try:
            if path == "/api/graph":
                graph_path = Path(self.app.cfg["graph_path"])
                if graph_path.exists():
                    self._json(200, json.loads(
                        graph_path.read_text(encoding="utf-8")))
                else:
                    self._json(200, {"nodes": [], "edges": [],
                                     "stats": {"paper_count": 0}})
            elif path == "/api/papers":
                papers = []
                for p in kb.storage.list_papers():
                    papers.append({k: p.get(k, "") for k in
                                   ("id", "title", "authors", "year",
                                    "journal", "topic", "method", "keywords",
                                    "summary", "theme", "degree", "status",
                                    "notes")})
                self._json(200, {"papers": papers})
            elif path.startswith("/api/papers/"):
                try:
                    pid = int(path.split("/")[-1])
                except ValueError:
                    self._error(400, "invalid id"); return
                p = kb.storage.get_paper(pid)
                if not p:
                    self._error(404, "paper not found"); return
                p = dict(p)
                p["chunks"] = [{"seq": c["seq"], "text": c["text"]}
                               for c in kb.storage.get_chunks(pid)]
                self._json(200, p)
            elif path == "/api/search":
                q = (query.get("q") or [""])[0].strip()
                if not q:
                    self._json(200, {"results": []}); return
                results = []
                for paper, info in kb.storage.fulltext_search(q):
                    results.append({
                        "paper": {k: paper.get(k, "") for k in
                                  ("id", "title", "authors", "year", "status")},
                        "score": info["score"],
                        "hits": info["hits"],
                        "snippets": [{"seq": s, "text": t[:200]}
                                     for s, t in info["snippets"]],
                    })
                self._json(200, {"results": results})
            elif path == "/api/health":
                self._json(200, {
                    "ok": True,
                    "ollama": self.app.llm_available(),
                    "papers": kb.storage.count_papers(),
                })
            elif path == "/api/models":
                models = self.app.llm().list_models()
                self._json(200, {
                    "models": models,
                    "current": self.app.current_model(),
                    "available": bool(models),
                })
            elif path.startswith("/api/export/citations"):
                ids = (query.get("ids") or [""])[0]
                try:
                    pid_list = [int(x) for x in ids.split(",") if x.strip()]
                except ValueError:
                    self._error(400, "invalid ids"); return
                citations = []
                for pid in pid_list:
                    p = kb.storage.get_paper(pid)
                    if not p:
                        continue
                    citations.append(format_gbt7714(p))
                body = "\n".join(citations).encode("utf-8")
                self._send(200, body, "text/plain; charset=utf-8")
            elif path == "/api/research":
                # 调研任务列表
                tasks = [t.to_dict() for t in self.app.workflow().list_tasks()]
                self._json(200, {"tasks": tasks})
            elif path.startswith("/api/research/"):
                self._route_research_get(path, query)
            else:
                self._error(404, "unknown api")
        except Exception as e:  # noqa: BLE001
            logger.exception("API GET %s 失败", path)
            self._error(500, str(e))

    def _route_research_get(self, path, query):
        """调研任务的 GET 路由：详情 / 事件流(SSE) / 结果产物。"""
        parts = path.split("/")  # ['', 'api', 'research', run_id, ...]
        if len(parts) < 4:
            self._error(400, "missing run_id"); return
        run_id = parts[3]
        task = self.app.workflow().get_task(run_id)
        if task is None:
            self._error(404, "research task not found"); return
        rest = parts[4] if len(parts) > 4 else ""
        if rest == "events":
            self._stream_events(task)
        elif rest == "result":
            # 最终产物（综述正文等）
            state = task.state or {}
            self._json(200, {
                "run_id": run_id,
                "topic": task.topic,
                "status": task.status,
                "search_results": state.get("search_results", []),
                "analysis_report": state.get("analysis_report", {}),
                "writing_outline": state.get("writing_outline", {}),
                "final_report": state.get("final_report", {}),
            })
        else:
            # 任务详情（元信息 + 事件摘要）
            self._json(200, {
                **task.to_dict(),
                "events": task.events[-100:],
            })

    def _stream_events(self, task):
        """SSE 事件流：按时间顺序推送任务事件（含历史重放）。

        前端用 EventSource 连接，收到 stage/status 事件更新进度条。
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        last = 0
        deadline = time.time() + 3600  # 最多挂 1 小时
        try:
            while time.time() < deadline:
                events = list(task.events)
                for ev in events[last:]:
                    try:
                        payload = json.dumps(ev, ensure_ascii=False)
                        self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                        self.wfile.flush()
                        last += 1
                    except Exception:
                        return
                if task.status in ("done", "error", "cancelled") and \
                        last >= len(events):
                    break
                time.sleep(0.3)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            try:
                self.wfile.write(b"data: {\"type\":\"eof\"}\n\n")
                self.wfile.flush()
            except Exception:
                pass

    # ---------- POST API ----------

    def _route_api_post(self, path, body):
        try:
            if path == "/api/ingest":
                # 入库可能耗时较长，Ollama 不可用也允许（降级元数据）
                kb = self.app.knowledge()
                result = kb.ingest(with_llm=True)
                if result["added"] or result["duplicates"]:
                    kb.rebuild_vectors()
                    self._rebuild_graph()
                self._json(200, result)
            elif path == "/api/query":
                q = (body.get("question") or "").strip()
                if not q:
                    self._error(400, "question required"); return
                if not self.app.llm_available():
                    self._error(503, "本地模型（Ollama）未连接，无法问答；请先启动 Ollama")
                    return
                kb = self.app.knowledge()
                hits = kb.query_top_k(q, k=self.app.cfg.get("top_k_retrieval", 5))
                if not hits:
                    self._json(200, {"answer": "未在本地语料中找到相关论文。",
                                     "sources": []})
                    return
                context = "\n\n".join(f"[{h['title']}]\n{h['text']}" for h in hits)
                answer = self.app.llm().answer_question(q, context)
                sources = [{"title": h["title"], "paper_id": h["paper_id"],
                            "score": h["score"]} for h in hits]
                self._json(200, {"answer": answer, "sources": sources})
            elif path == "/api/export/review":
                kb = self.app.knowledge()
                papers = kb.storage.list_papers()
                if not papers:
                    self._error(400, "语料为空，无法生成综述"); return
                md = self._build_review(papers)
                self._send(200, md.encode("utf-8"),
                           "text/markdown; charset=utf-8")
            elif path == "/api/research":
                # 发起一次多智能体文献调研任务
                topic = (body.get("topic") or "").strip()
                if not topic:
                    self._error(400, "topic required"); return
                constraints = body.get("constraints") or {}
                max_results = int(body.get("max_results", 8) or 8)
                task = self.app.workflow().create_task(
                    topic, {**constraints, "max_results": max_results})
                # 后台线程执行工作流
                threading.Thread(
                    target=self.app.workflow().run, args=(task.run_id,),
                    daemon=True).start()
                self._json(200, {"run_id": task.run_id, "topic": topic,
                                 "status": "running",
                                 "events_url": f"/api/research/{task.run_id}/events"})
            else:
                self._error(404, "unknown api")
        except Exception as e:  # noqa: BLE001
            logger.exception("API POST %s 失败", path)
            self._error(500, str(e))

    # ---------- PUT API ----------

    def _route_api_put(self, path, body):
        try:
            if path == "/api/model":
                model = (body.get("model") or "").strip()
                if not model:
                    self._error(400, "model required"); return
                if not self.app.set_current_model(model):
                    self._error(400, "invalid model"); return
                self._json(200, {"ok": True, "model": model})
            elif path.startswith("/api/papers/"):
                pid = int(path.split("/")[-1])
                kb = self.app.knowledge()
                p = kb.storage.get_paper(pid)
                if not p:
                    self._error(404, "paper not found"); return
                notes = body.get("notes")
                status = body.get("status")
                kb.storage.update_notes_status(pid, notes=notes, status=status)
                self._json(200, {"ok": True})
            else:
                self._error(404, "unknown api")
        except Exception as e:  # noqa: BLE001
            logger.exception("API PUT %s 失败", path)
            self._error(500, str(e))

    # ---------- 综述 ----------

    def _build_review(self, papers):
        """生成综述 Markdown：按主题分组 + LLM 段落（失败降级纯模板）。"""
        from .export import build_review_markdown
        llm = self.app.llm() if self.app.llm_available() else None
        return build_review_markdown(self.app.cfg, papers, llm)

    def _rebuild_graph(self):
        """向量重建后构建图（与 knowledge 联动）。"""
        from .graph_builder import build_graph
        kb = self.app.knowledge()
        paper_vecs, _ = kb.rebuild_vectors()
        llm = self.app.llm() if self.app.llm_available() else None
        build_graph(self.app.cfg, kb.storage, paper_vecs, llm=llm)


def format_gbt7714(p: dict) -> str:
    """GB/T 7714 引文格式化（缺失字段标 [未提取]）。"""
    authors = p.get("authors") or "[未提取]"
    title = p.get("title") or "[未提取]"
    journal = p.get("journal") or "[未提取]"
    year = p.get("year") or "[未提取]"
    pages = p.get("pages") or "[未提取]"
    return f"{authors}. {title}[J]. {journal}, {year}: {pages}."


def make_app(cfg):
    return App(cfg)


def run_server(cfg, host=None, port=None):
    host = host or cfg.get("server_host", "127.0.0.1")
    port = port or int(cfg.get("server_port", 8000))
    app = App(cfg)
    Handler.app = app
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"
    logger.info("文献图谱服务已启动: %s  （Ctrl+C 停止）", url)
    logger.info("打开浏览器访问 %s 查看 3D 关系图", url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("服务停止")
    finally:
        httpd.server_close()
    return url
