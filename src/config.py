# -*- coding: utf-8 -*-
"""配置加载：config.json + 路径规范（pathlib，兼容中文/空格路径）。"""
import json
import logging
import logging.handlers
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path=None):
    """轻量加载项目根目录的 .env 文件（零第三方依赖）。

    规则（与 python-dotenv 一致）：
    - 每行 KEY=VALUE；以 # 开头为注释；支持 export 前缀与双引号；
    - 已存在的环境变量优先，.env 不覆盖；
    - 找不到 .env 时静默跳过。
    """
    env_file = Path(path) if path else PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


DEFAULT_CONFIG = {
    "corpus_dir": "data/papers",
    "samples_dir": "data/samples",
    "index_dir": "data/index",
    "ocr_cache_dir": "data/ocr_cache",
    "db_path": "data/index/library.db",
    "graph_path": "data/index/graph.json",
    "log_path": "data/index/app.log",
    "ollama_base": "auto",
    "llm_model": "deepseek-r1:8b",
    "llm_timeout_s": 120,
    "llm_retries": 2,
    "chunk_size": 800,
    "chunk_overlap": 80,
    "similarity_threshold": 0.35,
    "top_k_edges": 8,
    "top_k_retrieval": 5,
    "max_clusters": 8,
    "server_host": "127.0.0.1",
    "server_port": 8000,
    "tesseract_cmd": "tesseract",
    "ocr_lang": "chi_sim+eng",
    "min_chars_for_text_layer": 10,
}


def load_config(config_path=None):
    load_dotenv()  # 先读 .env（如存在），供下方环境变量读取使用
    cfg = dict(DEFAULT_CONFIG)
    path = Path(config_path) if config_path else PROJECT_ROOT / "config.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        cfg.update(user_cfg)
    # 相对路径解析到项目根
    for key in ("corpus_dir", "samples_dir", "index_dir", "ocr_cache_dir",
                "db_path", "graph_path", "log_path"):
        p = Path(cfg[key])
        if not p.is_absolute():
            cfg[key] = str(PROJECT_ROOT / p)
    # 确保目录存在
    for key in ("corpus_dir", "samples_dir", "index_dir", "ocr_cache_dir"):
        Path(cfg[key]).mkdir(parents=True, exist_ok=True)
    Path(cfg["db_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg["log_path"]).parent.mkdir(parents=True, exist_ok=True)
    # Ollama 地址：auto -> 环境变量 OLLAMA_HOST -> 默认
    if cfg["ollama_base"] == "auto":
        host = os.environ.get("OLLAMA_HOST", "").strip()
        if not host:
            cfg["ollama_base"] = "http://127.0.0.1:11434"
        else:
            host = host.replace("http://", "").replace("https://", "")
            # 0.0.0.0 表示本机任意接口，客户端应连回环地址
            if host in ("0.0.0.0", "::"):
                host = "127.0.0.1"
            if ":" not in host:  # 无端口则补默认 11434
                host = f"{host}:11434"
            cfg["ollama_base"] = f"http://{host}"
    return cfg


def setup_logging(cfg):
    logger = logging.getLogger("arch")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    fh = logging.FileHandler(cfg["log_path"], encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger
