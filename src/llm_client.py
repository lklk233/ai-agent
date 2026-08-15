# -*- coding: utf-8 -*-
"""Ollama LLM 客户端：懒加载、JSON 模式、低温、重试、超时、串行。

v4 计划要点：
- format="json" + temperature 0.2 + 重试 + 120s 超时 + 正则截取 JSON 兜底
- 串行调用（Ollama 单队列），进程内互斥锁
- 读取 OLLAMA_HOST 环境变量（容忍无 http:// 前缀）
- 懒加载：仅元数据/问答/综述时实例化使用，搜索/图/笔记/导出不依赖
"""
import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger("arch.llm")


def _http_post_json(url: str, payload: dict, timeout: float):
    """标准库 HTTP POST（JSON），返回响应 JSON 或抛出异常。

    显式禁用代理：本机检测到系统配置了本地代理（如 127.0.0.1:10090），
    而 Ollama 是本机服务，走代理必然失败。
    """
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_get_json(url: str, timeout: float):
    """标准库 HTTP GET，返回响应 JSON（禁用代理）。"""
    req = urllib.request.Request(url, method="GET")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class OllamaError(Exception):
    """Ollama 不可用或请求失败。"""


class LLMClient:
    def __init__(self, cfg):
        self.base = cfg["ollama_base"]
        self.model = cfg.get("llm_model", "deepseek-r1:8b")
        self.timeout = cfg.get("llm_timeout_s", 120)
        self.retries = cfg.get("llm_retries", 2)
        self._lock = threading.Lock()  # 串行调用

    # ---------- 基础 ----------

    def health(self) -> bool:
        try:
            _http_post_json(f"{self.base}/api/show", {"name": self.model},
                            timeout=5)
            return True
        except Exception:
            return False

    def set_model(self, model: str):
        """运行时切换模型（无需重建客户端）。"""
        self.model = model

    def list_models(self) -> list:
        """列出 Ollama 已安装模型。返回 [{name, size_GB}, ...]（按名称排序）。"""
        try:
            data = _http_get_json(f"{self.base}/api/tags", timeout=10)
        except Exception as e:
            logger.warning("列出 Ollama 模型失败: %s", e)
            return []
        models = []
        for m in data.get("models", []):
            name = m.get("name", "")
            size = m.get("size", 0)
            models.append({"name": name, "size_GB": round(size / 1e9, 1)})
        models.sort(key=lambda x: x["name"])
        return models

    def chat(self, messages, format_json=False, temperature=0.2,
             max_tokens=None):
        """串行调用 /api/chat。返回文本或解析后的 JSON（format_json=True）。"""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if format_json:
            payload["format"] = "json"
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        last_err = None
        with self._lock:  # Ollama 单队列，串行
            for attempt in range(self.retries + 1):
                try:
                    data = _http_post_json(
                        f"{self.base}/api/chat", payload, self.timeout)
                    content = data.get("message", {}).get("content", "")
                    # 统一剥离 <think> 推理块（deepseek-r1 会输出思考过程；
                    # 处理未闭合块——token 预算不足时 </think> 可能缺失）
                    content = re.sub(r"<think>.*?(</think>|$)", "",
                                     content, flags=re.S).strip()
                    if format_json:
                        return self._parse_json(content)
                    return content
                except (urllib.error.URLError, OSError,
                        json.JSONDecodeError) as e:
                    last_err = OllamaError(f"Ollama 请求失败: {e}")
                    time.sleep(1.5)
        raise last_err or OllamaError("Ollama 请求失败")

    def _parse_json(self, content: str):
        """解析 LLM 输出为 JSON：format 约束优先，正则截取兜底。"""
        content = content.strip()
        # 去掉可能的 <think> 推理块
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.S)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        m = re.search(r"\{.*\}", content, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        raise OllamaError(f"LLM 未返回合法 JSON: {content[:200]}")

    # ---------- 任务封装 ----------

    def extract_metadata(self, head_text: str, tail_text: str) -> dict:
        """从论文文本提取结构化元数据（JSON）。

        失败时抛出 OllamaError，由调用方降级处理。
        """
        head = head_text[:1500]
        tail = tail_text[-1200:] if tail_text else ""
        prompt = (
            "你是考古学文献管理助手。根据给定论文文本（前部与参考文献区），"
            "提取结构化元数据，输出严格 JSON，字段："
            "{\"title\":\"标题\",\"authors\":\"作者\",\"year\":\"年份\","
            "\"journal\":\"期刊名\",\"pages\":\"卷期页码\","
            "\"topic\":\"研究对象（一句话）\",\"method\":\"研究方法\","
            "\"keywords\":[\"关键词1\",\"关键词2\"],"
            "\"summary\":\"论文核心结论（50字内）\"}。"
            "无法确定的字段填空字符串或空数组，不要编造。\n"
            f"【论文前部】\n{head}\n\n【参考文献区】\n{tail}"
        )
        data = self.chat([{"role": "user", "content": prompt}],
                         format_json=True, max_tokens=512)
        if not isinstance(data, dict):
            raise OllamaError("元数据 JSON 结构错误")
        return normalize_meta_keys(data)

    def extract_reference_titles(self, tail_text: str) -> list:
        """从参考文献区提取文献标题列表（优先 LLM，供引文匹配）。"""
        tail = tail_text[-2000:] if tail_text else ""
        if not tail.strip():
            return []
        prompt = (
            "以下是论文的参考文献区文本。请提取每条文献的标题，"
            "输出 JSON 数组：[\"标题1\",\"标题2\",...]。"
            "无法识别的条目跳过，输出空数组。\n"
            f"【参考文献区】\n{tail}"
        )
        data = self.chat([{"role": "user", "content": prompt}],
                         format_json=True, max_tokens=300)
        if isinstance(data, list):
            return [str(t).strip() for t in data if str(t).strip()]
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return [str(t).strip() for t in v if str(t).strip()]
        return []

    def name_cluster(self, titles: list) -> str:
        """为主题簇命名（一句话主题名）。

        deepseek-r1 在非 JSON 模式下会输出超长推理，故统一走 JSON 模式。
        """
        joined = "；".join(titles[:6])
        prompt = (
            f"以下论文属于同一研究主题簇：{joined}\n"
            "请用不超过10个字概括该主题（如\"新石器陶器类型学\"）。"
            "输出 JSON：{\"name\":\"主题名\"}"
        )
        try:
            data = self.chat([{"role": "user", "content": prompt}],
                             format_json=True, temperature=0.3,
                             max_tokens=128)
            if isinstance(data, dict):
                return str(data.get("name", "")).strip() or "未命名主题"
            return "未命名主题"
        except OllamaError:
            return "未命名主题"

    def answer_question(self, question: str, context: str) -> str:
        """RAG 问答：基于检索上下文生成带引用回答。"""
        prompt = (
            "你是考古学文献研究助手。根据提供的本地论文片段回答用户问题。"
            "回答需基于上下文内容，可引用论文标题。若上下文不足以回答，"
            "明确说明。保持简洁（200字内）。"
            "输出 JSON：{\"answer\":\"回答内容\"}\n\n"
            f"【论文片段】\n{context[:6000]}\n\n"
            f"【问题】{question}"
        )
        data = self.chat([{"role": "user", "content": prompt}],
                         format_json=True, temperature=0.3, max_tokens=600)
        if isinstance(data, dict):
            return str(data.get("answer", "")).strip()
        return ""

    def write_review_section(self, theme: str, papers: list) -> str:
        """为综述的一个主题撰写段落。papers: [{title, authors, year, summary}]"""
        items = "\n".join(
            f"- 《{p.get('title', '')}》（{p.get('authors', '')}，"
            f"{p.get('year', '')}）：{p.get('summary', '')}" for p in papers)
        prompt = (
            f"为考古学文献综述撰写主题「{theme}」的研究现状段落。"
            f"相关论文：\n{items}\n\n"
            "要求：1) 概述该主题的研究脉络与代表性工作；2) 归纳方法流派；"
            "3) 指出研究空白或争议；4) 150字以内，学术语体。"
            "输出 JSON：{\"section\":\"段落内容\"}"
        )
        data = self.chat([{"role": "user", "content": prompt}],
                         format_json=True, temperature=0.3, max_tokens=400)
        if isinstance(data, dict):
            return str(data.get("section", "")).strip()
        return ""


def normalize_meta_keys(data: dict) -> dict:
    """兼容 LLM 返回的中文/英文键（deepseek-r1 实测返回中文键）。"""
    key_map = {
        "标题": "title", "title": "title", "题名": "title",
        "作者": "authors", "authors": "authors", "作者列表": "authors",
        "年份": "year", "year": "year", "年代": "year",
        "期刊": "journal", "journal": "journal", "期刊名": "journal",
        "页码": "pages", "pages": "pages", "卷期页码": "pages",
        "研究对象": "topic", "topic": "topic", "主题": "topic",
        "方法": "method", "method": "method", "研究方法": "method",
        "关键词": "keywords", "keywords": "keywords",
        "摘要": "summary", "summary": "summary", "总结": "summary",
        "核心结论": "summary", "结论": "summary",
    }
    norm = {}
    for k, v in data.items():
        key = key_map.get(k)
        if not key:
            continue
        if key == "keywords" and isinstance(v, list):
            v = "；".join(str(x).strip() for x in v if str(x).strip())
        elif key == "keywords" and isinstance(v, str):
            v = v.replace(",", "；").replace("，", "；")
        elif key == "authors" and isinstance(v, list):
            v = "、".join(str(x).strip() for x in v if str(x).strip())
        elif isinstance(v, (list, dict)):
            # 其他字段若为容器（如 year 偶发返回列表），取首个元素或空
            if isinstance(v, list):
                v = str(v[0]).strip() if v else ""
            else:
                v = ""
        if v is None:
            v = ""
        v = str(v) if key != "keywords" else v
        # 清理 LLM 的占位词（"未知"/"不详"/"无"等视为空）
        if isinstance(v, str) and v.strip() in ("未知", "不详", "无", "暂无",
                                                "未提取", "N/A", "n/a"):
            v = ""
        norm[key] = v
    return norm
