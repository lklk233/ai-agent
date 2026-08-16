# 本地考古学文献梳理 AI-Agent

轻量、实用的本地考古学论文文献管理智能体：文献梳理入库、3D 论文关系图、全文检索、阅读笔记、综述/引文导出、RAG 问答。全程离线（LLM 用本地 Ollama），无需联网与 API 费用。


## 功能一览

| 功能 | 说明 |
|---|---|
| 📥 文献入库 | 扫描语料目录（PDF/DOCX/TXT，扫描件自动 OCR），LLM 提取标题/作者/年份/关键词/摘要，SQLite 持久化 |
| 🌐 3D 论文关系图 | 自包含单页 3D 力导向图：拖拽旋转/缩放/点击详情/搜索/主题着色，零外部 JS 依赖 |
| 🤖 AI 文献调研 | **多智能体流水线**（借鉴 Paper-Agent）：输入研究主题 → 五步自动执行（检索规划→摘要阅读→分层分析→大纲→写作）→ 生成带证据引用的领域综述，SSE 实时进度追踪 |
| 🔍 全文检索 | 跨全部论文搜索，按标题>摘要>正文加权排序，定位到段落 |
| 📝 阅读笔记 + 状态 | 每篇论文 未读/在读/已读/重点 状态 + 自由笔记，点击 3D 图节点即可编辑 |
| 📄 综述导出 | 按主题分组自动生成研究现状综述 Markdown（含 GB/T 7714 引文） |
| 📖 引文导出 | 按 GB/T 7714 标准导出参考文献条目 |
| ⚠️ 去重提示 | 入库时按文件哈希去重 + 标题相似度 Top-3 提示 |
| 💬 智能问答 | 基于本地语料的 RAG 问答（需 Ollama 运行） |
| 🧠 模型可切换 | Web 界面下拉选择 Ollama 已安装模型，切换即时生效并持久化 |
| ♻️ 增量学习 | 新增论文后自动重建向量与关系图（幂等，重跑 0 新增） |

## 快速开始

### 1. 安装依赖（仅首次）

```bat
pip install --user jieba charset_normalizer scikit-learn networkx
```

> 建议在 Python 3.12+ 虚拟环境中安装（如 conda：`conda create -n localdocument_comb python=3.12`）。

需要本机运行 Ollama 并已拉取模型：

```bat
ollama pull deepseek-r1:8b      # 默认模型（考古领域效果良好）
# 也可使用其他模型，例如：
ollama pull qwen2.5:7b          # 通用中文模型，速度更快
ollama pull llama3.1:8b         # 英文为主的通用模型
```

> **模型可自由选择**：系统会自动列出 Ollama 中已安装的全部模型，在 Web 界面的工具栏下拉框即可切换（切换立即生效并持久化到 `config.json` 的 `llm_model` 字段，重启后保持）。

### 2. 准备语料

把你的考古学论文（PDF / DOCX / TXT / MD）放入：

```
data\papers\
```

> 首次体验可先用内置示例（`data\samples\`，3 篇模拟考古文献）。

### 3. 入库

```bat
python -m src.cli ingest
```

- 默认扫描 `data/papers/`；扫描示例用 `--dir data/samples`
- LLM 元数据失败不阻塞入库（自动降级正则提取），可用 `--no-llm` 完全跳过
- 入库后自动重建向量与关系图（`graph.json` 更新）

### 4. 启动 Web 界面

**方式一：双击 `start.bat`**（自动打开 http://127.0.0.1:8000）

**方式二：命令行**
```bat
python -m src.cli serve
```

浏览器访问 **http://127.0.0.1:8000** 查看 3D 关系图。

### 5. 命令行常用操作

```bat
REM 问答
python -m src.cli query "陶器类型学的研究方法是什么？"

REM 生成综述（保存到 data\index\综述_日期.md）
python -m src.cli summarize

REM 导出 GB/T 7714 引文（论文 id 见 /api/papers）
python -m src.cli export-citations --ids 1,2,3

REM 单元测试
python -m src.cli test

REM 备份（数据库 + 语料）
python -m src.cli backup
```

### 6. AI 文献调研（多智能体工作台）

在 Web 界面点击「🤖 AI 调研」进入工作台：

1. 输入研究主题（如"新石器时代陶器类型学研究"）；
2. 点击「🚀 开始调研」，后台自动执行五步流水线：
   - 🔍 **检索规划**：LLM 把主题拆成研究子方向与关键词 → 在本地语料库检索
   - 📖 **摘要阅读**：逐篇读摘要，按 研究问题/研究对象/方法 三维匹配判断相关性
   - 🔬 **分层分析**：先按子主题横向比较，再做全局综合（概况/共识/争议/空白/演化八维度）
   - ✍️ **大纲生成**：基于分析结果规划综述章节结构与证据映射
   - 📝 **综述写作**：按大纲逐节写作（证据约束、带 [P1] 引用），最后生成摘要
3. 通过 SSE 实时追踪每一步进度；完成后查看综述全文、全局分析与检索论文。

> 说明：调研全程使用本地语料库与本地 Ollama 模型，完全离线；任务结果持久化在 `data/index/research/`。

## 项目结构

```
local_document_comb/
├── config.json          # 配置（语料目录/Ollama/阈值等）
├── start.bat            # 一键启动
├── requirements.txt     # 依赖清单
├── data/
│   ├── papers/          # ★ 你的论文放这里
│   ├── samples/         # 内置示例（2 PDF + 1 DOCX）
│   ├── ocr_cache/       # OCR 结果缓存
│   └── index/           # library.db / graph.json / 综述 / 日志
├── src/
│   ├── extractor.py     # 文档解析（PDF/DOCX/TXT/OCR）
│   ├── text_prep.py     # 清洗/去页眉页脚/分块/分词
│   ├── llm_client.py    # Ollama 客户端（JSON 模式/重试/串行）
│   ├── storage.py       # SQLite 数据层
│   ├── vector_store.py  # TF-IDF 稀疏向量 + 相似度
│   ├── knowledge.py     # 入库流水线 + 检索
│   ├── graph_builder.py # 关系图构建
│   ├── export.py        # 综述/引文导出
│   ├── workflow/        # ★ 多智能体文献调研（融合 Paper-Agent）
│   │   ├── agents.py    #   五 Agent（Search/Read/Analyse/Outline/Writing）
│   │   ├── prompts.py   #   五 Agent 系统提示词（本地化）
│   │   ├── runtime.py   #   工作流运行时（编排/事件/持久化）
│   │   └── local_search.py # 本地语料检索服务
│   ├── server.py        # HTTP 服务（纯标准库）
│   ├── cli.py           # 命令行入口
│   └── tests/           # unittest
├── web/index.html       # 自包含 3D 图 + AI 调研工作台
└── scripts/             # 便捷脚本
```

## 技术要点

- **零框架后端**：Python 标准库 `http.server` + SQLite，无 FastAPI/uvicorn
- **轻量前端**：单 HTML 内联 CSS/JS，Canvas 2D 手写 3D 投影 + 力导向布局，零 CDN
- **LLM 懒加载**：仅元数据/问答/综述调 Ollama；搜索/图/笔记/导出完全离线可用
- **模型可切换**：`config.json` 的 `llm_model` 为默认模型；Web 界面下拉框可随时切换 Ollama 已安装模型（后端 `GET /api/models` 列出、`PUT /api/model` 切换并持久化）
- **向量策略**：TF-IDF + scipy.sparse 稀疏存储，每次入库全量重建避免空间漂移
- **可靠性**：LLM JSON 模式 + 重试 + 降级链；graph.json 原子写入；SQLite 事务
- **环境变量**：`.env` 文件（已被 git 忽略）提供 `OLLAMA_HOST`、`PYTHONIOENCODING` 等配置，启动时自动加载；模板见 `.env.example`，已存在的系统环境变量优先

## 备份与迁移

备份 = 复制两个位置即可完整迁移：
- `data/index/library.db`（数据库）
- `data/papers/`（原始论文）

或使用 `python -m src.cli backup` 一键打包。

## 常见问题

| 问题 | 解决 |
|---|---|
| 问答提示"Ollama 未连接" | 启动 Ollama（`ollama serve`），确认 `ollama list` 有可用模型 |
| 想换别的模型 | 1) `ollama pull 模型名` 安装；2) 在 Web 界面工具栏下拉框选择；或编辑 `config.json` 的 `llm_model` 字段 |
| 扫描 PDF 识别不出文字 | 扫描件会自动走 OCR（Tesseract，已装 chi_sim 中文包） |
| 重新入库 | 幂等设计：同文件重复扫描 0 新增，只处理新文件 |
| 关掉 Ollama 还能用吗 | 能。搜索/3D 图/笔记/导出完全离线；仅元数据生成与问答依赖 LLM |
| 修改端口/阈值 | 编辑 `config.json` 后重启服务 |


