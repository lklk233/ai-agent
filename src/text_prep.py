# -*- coding: utf-8 -*-
"""文本预处理：清洗、去页眉页脚、断行合并、分块、中文分词。

- 清洗：全角转半角、去多余空白、去页眉页脚（高频行检测）、合并断行
- 分块：按段落，目标块大小 chunk_size 字符，相邻块重叠 chunk_overlap
- 分词：优先 jieba，未安装时用字符 n-gram 兜底（轻量化降级路径）
"""
import re
from pathlib import Path

try:
    import jieba
    HAS_JIEBA = True
    # 预加载词典，避免首次调用卡顿
    jieba.initialize()
except Exception:  # pragma: no cover
    HAS_JIEBA = False

_FULLWIDTH = {ord(c): ord(c) - 0xFEE0 for c in
              "！＂＃＄％＆＇（）＊＋，－．／：；＜＝＞？＠［＼］＾＿｀｛｜｝～"}
_FULLWIDTH[ord("　")] = ord(" ")

_HEADER_LINE = re.compile(r"^(第\s*\d+\s*[页卷]|[A-Za-z\s\-]*\d+\s*$|"  # noqa: W605
                          r"《[^》]{2,20}》\s*(第\s*\d+\s*期)?$|"
                          r"考古\s*20\d{2}\s*年|文物\s*20\d{2}\s*年)")


def clean_text(text: str) -> str:
    """基础清洗：全角→半角、规范化空白。"""
    if not text:
        return ""
    text = text.translate(_FULLWIDTH)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t\u3000]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_page_headers(text: str) -> str:
    """去页眉页脚：按行统计高频短行（出现在≥30%行组中）剔除。

    对学术 PDF 常见的期刊名/页码页眉有效，避免污染向量与检索。
    """
    lines = text.split("\n")
    if len(lines) < 8:
        return text
    from collections import Counter
    cand = Counter()
    for ln in lines:
        s = ln.strip()
        if 1 <= len(s) <= 30 and not s.isdigit():
            cand[s] += 1
    threshold = max(3, int(len(lines) * 0.25))
    noise = {k for k, v in cand.items() if v >= threshold and _HEADER_LINE.match(k)}
    if not noise:
        return text
    out = [ln for ln in lines if ln.strip() not in noise]
    return "\n".join(out)


def merge_lines(text: str) -> str:
    """合并断行：行尾无标点的换行视为断行并入下一行（中文排版）。"""
    lines = text.split("\n")
    merged = []
    buf = ""
    for ln in lines:
        s = ln.strip()
        if not s:
            if buf:
                merged.append(buf)
                buf = ""
            continue
        if buf:
            buf += s
        else:
            buf = s
        # 行尾为句读/冒号/问叹号等，视为段落结束
        if re.search(r"[。；：！？）”」』]\s*$", buf) or len(buf) > 200:
            merged.append(buf)
            buf = ""
    if buf:
        merged.append(buf)
    return "\n".join(merged)


def normalize(text: str) -> str:
    """完整预处理流水线。"""
    t = clean_text(text)
    t = remove_page_headers(t)
    t = merge_lines(t)
    return t


def split_chunks(text: str, chunk_size: int = 800, overlap: int = 80):
    """按段落分块，相邻块重叠 overlap 字符。

    返回 [(seq, text), ...]。
    """
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 1 <= chunk_size:
            buf = f"{buf}\n{p}" if buf else p
            continue
        if buf:
            chunks.append(buf)
        # 超长段落按字符硬切
        while len(p) > chunk_size:
            chunks.append(p[:chunk_size])
            p = p[chunk_size - overlap:]
        buf = p
    if buf:
        chunks.append(buf)
    return [(i, c) for i, c in enumerate(chunks)]


def tokenize(text: str, ngram: int = 2):
    """分词。jieba 优先；缺失时字符 n-gram 兜底（轻量化降级）。"""
    if HAS_JIEBA:
        return [w for w in jieba.lcut(text) if w.strip()]
    # n-gram 兜底：对中文按字符 n-gram，对英文按单词
    toks = []
    for seg in re.split(r"([A-Za-z0-9]+)", text):
        if not seg:
            continue
        if re.match(r"^[A-Za-z0-9]+$", seg):
            toks.append(seg.lower())
        else:
            chars = [c for c in seg if not c.isspace()]
            for i in range(len(chars) - ngram + 1):
                toks.append("".join(chars[i:i + ngram]))
    return toks
