# -*- coding: utf-8 -*-
"""命令行入口：ingest / build-graph / serve / query / summarize / export / test / backup

用法：
  python cli.py ingest [--dir DIR] [--no-llm]
  python cli.py build-graph
  python cli.py serve [--port PORT]
  python cli.py query "问题"
  python cli.py summarize
  python cli.py export-citations --ids 1,2,3
  python cli.py test
  python cli.py backup
"""
import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, setup_logging  # noqa: E402


def cmd_ingest(cfg, args):
    from src.knowledge import KnowledgeBase
    kb = KnowledgeBase(cfg)
    try:
        result = kb.ingest(corpus_dir=args.dir, with_llm=not args.no_llm)
        print(f"扫描 {result['scanned']} 个文件：新增 {result['added']}，"
              f"重复 {result['duplicates']}，失败 {len(result['failed'])}")
        for name in result["failed"]:
            print(f"  ✗ {name}")
        if result["added"]:
            print("重建向量与关系图…")
            paper_vecs, _ = kb.rebuild_vectors()
            from src.graph_builder import build_graph
            from src.llm_client import LLMClient
            llm = LLMClient(cfg) if not args.no_llm else None
            build_graph(cfg, kb.storage, paper_vecs, llm=llm)
            print("✅ 入库完成，graph.json 已更新")
    finally:
        kb.close()


def cmd_build_graph(cfg, _args):
    from src.knowledge import KnowledgeBase
    from src.graph_builder import build_graph
    from src.llm_client import LLMClient
    kb = KnowledgeBase(cfg)
    try:
        paper_vecs, _ = kb.rebuild_vectors()
        llm = LLMClient(cfg)
        graph = build_graph(cfg, kb.storage, paper_vecs,
                            llm=llm if llm.health() else None)
        print(f"✅ graph.json 已生成：{graph['stats']}")
    finally:
        kb.close()


def cmd_serve(cfg, args):
    from src.server import run_server
    run_server(cfg, port=args.port)


def cmd_query(cfg, args):
    from src.knowledge import KnowledgeBase
    from src.llm_client import LLMClient
    kb = KnowledgeBase(cfg)
    try:
        llm = LLMClient(cfg)
        if not llm.health():
            print("❌ Ollama 未连接，无法问答")
            return 1
        hits = kb.query_top_k(args.question,
                              k=cfg.get("top_k_retrieval", 5))
        if not hits:
            print("未在本地语料中找到相关论文。")
            return 0
        context = "\n\n".join(f"[{h['title']}]\n{h['text']}" for h in hits)
        answer = llm.answer_question(args.question, context)
        print("【回答】")
        print(answer)
        print("\n【依据】")
        for h in hits:
            print(f"  - {h['title']} (相似度 {h['score']})")
    finally:
        kb.close()


def cmd_summarize(cfg, _args):
    from src.knowledge import KnowledgeBase
    from src.export import save_review
    from src.llm_client import LLMClient
    kb = KnowledgeBase(cfg)
    try:
        papers = kb.storage.list_papers()
        if not papers:
            print("语料为空，无法生成综述。请先 ingest。")
            return 1
        llm = LLMClient(cfg)
        llm = llm if llm.health() else None
        path = save_review(cfg, papers, llm)
        print(f"✅ 综述已保存: {path}")
    finally:
        kb.close()


def cmd_export_citations(cfg, args):
    from src.knowledge import KnowledgeBase
    from src.export import export_citations
    kb = KnowledgeBase(cfg)
    try:
        papers = [kb.storage.get_paper(int(i)) for i in args.ids.split(",")]
        papers = [p for p in papers if p]
        text = export_citations(papers)
        out = Path(cfg["index_dir"]) / "引文_GB_T_7714.txt"
        out.write_text(text, encoding="utf-8")
        print(f"✅ 引文已导出: {out}")
        print(text)
    finally:
        kb.close()


def cmd_test(cfg, _args):
    import unittest
    test_dir = Path(__file__).resolve().parent / "tests"
    loader = unittest.TestLoader()
    suite = loader.discover(str(test_dir), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


def cmd_backup(cfg, _args):
    out = Path(cfg["index_dir"]) / f"backup_{time_str()}.zip"
    db = Path(cfg["db_path"])
    papers_dir = Path(cfg["corpus_dir"])
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        if db.exists():
            zf.write(db, "index/library.db")
        if papers_dir.exists():
            for f in papers_dir.rglob("*"):
                if f.is_file():
                    zf.write(f, f"papers/{f.relative_to(papers_dir)}")
    print(f"✅ 备份完成: {out}")


def time_str():
    import time
    return time.strftime("%Y%m%d_%H%M%S")


def main():
    cfg = load_config()
    setup_logging(cfg)

    parser = argparse.ArgumentParser(
        description="本地考古学文献梳理 AI-Agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="扫描语料入库")
    p.add_argument("--dir", default=None, help="语料目录（默认 data/papers）")
    p.add_argument("--no-llm", action="store_true", help="跳过 LLM 元数据")

    sub.add_parser("build-graph", help="重建向量与关系图")

    p = sub.add_parser("serve", help="启动 Web 服务")
    p.add_argument("--port", type=int, default=None, help="端口（默认 8000）")

    p = sub.add_parser("query", help="RAG 问答")
    p.add_argument("question", help="问题")

    sub.add_parser("summarize", help="生成综述")

    p = sub.add_parser("export-citations", help="导出 GB/T 7714 引文")
    p.add_argument("--ids", required=True, help="论文 id 列表，逗号分隔")

    sub.add_parser("test", help="运行单元测试")

    sub.add_parser("backup", help="备份数据库与语料")

    args = parser.parse_args()
    fn = {
        "ingest": cmd_ingest, "build-graph": cmd_build_graph,
        "serve": cmd_serve, "query": cmd_query, "summarize": cmd_summarize,
        "export-citations": cmd_export_citations, "test": cmd_test,
        "backup": cmd_backup,
    }[args.cmd]
    sys.exit(fn(cfg, args) or 0)


if __name__ == "__main__":
    main()
