# -*- coding: utf-8 -*-
"""生成示例考古学论文语料：2 篇 PDF（fitz 创建）+ 1 篇 DOCX。

内容为模拟考古学文献（陶器类型学方向），互相引用，覆盖多种格式解析，
用于端到端验收：入库、关系构建、检索、问答、去重。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config  # noqa: E402

SAMPLE_TEXTS = [
    {
        "file": "新石器时代陶器类型学研究述评.pdf",
        "title": "新石器时代陶器类型学研究述评",
        "authors": "李启文",
        "year": "2018",
        "journal": "考古学报",
        "pages": "2018(3): 245-268",
        "abstract": "陶器类型学是考古学分期断代的基础方法之一。本文系统梳理了新石器时代陶器类型学研究的理论源流与发展脉络，讨论了类型划分的层级标准、型式序列的建立方法及其在考古学文化谱系研究中的应用，并对当前研究中的若干争议问题提出了新的认识。",
        "keywords": "陶器类型学；新石器时代；考古学方法；文化谱系",
        "body": [
            "一、引言",
            "陶器是考古遗存中最常见的人工制品，其形态演变具有相对稳定的序列性，因此类型学方法在考古学分期研究中长期占据核心地位。自二十世纪中叶以来，中国考古学界围绕陶器类型学的理论与操作层面展开了持续的讨论，形成了以形态特征为纲、以演进逻辑为线的研究传统。",
            "二、类型学的基本原理",
            "类型学研究的理论前提是：器物形态的变化具有方向性与阶段性，同一文化传统中器物形态的递变可以反映相对年代关系。具体操作上，研究者通常先依据陶质、陶色、纹饰等宏观特征划分类型，再在同一类型内部依据口沿、腹部、底部等部位形态差异划分亚型与式别，从而建立完整的型式序列。",
            "三、新石器时代陶器研究中的类型学实践",
            "在新石器时代研究中，类型学方法广泛用于陶器分期、文化因素分析以及聚落变迁研究。仰韶文化半坡类型与庙底沟类型的划分，龙山文化陶器群的分期研究，均以类型学分析为基础。近年来，随着定量统计方法的引入，类型划分的客观性得到了显著提升，但类型标准的统一问题仍待解决。",
            "四、研究展望",
            "未来陶器类型学研究应当加强与科技检测手段的结合，在类型划分中融入工艺学视角，同时注意避免类型学方法的滥用与僵化。本文认为，类型学作为考古学的基础方法，其生命力在于不断吸收新的研究理念与技术手段。",
        ],
        "refs": [
            "[1] 张玉良. 仰韶文化彩陶纹饰的考古学观察[J]. 文物, 2015(2): 34-51.",
            "[2] 王明远. 龙山文化蛋壳黑陶工艺与技术特征[J]. 考古, 2016(4): 78-95.",
            "[3] 陈国栋. 中国新石器时代文化谱系研究[M]. 北京: 科学出版社, 2010.",
        ],
    },
    {
        "file": "仰韶文化彩陶纹饰的考古学观察.pdf",
        "title": "仰韶文化彩陶纹饰的考古学观察",
        "authors": "张玉良",
        "year": "2015",
        "journal": "文物",
        "pages": "2015(2): 34-51",
        "abstract": "彩陶是仰韶文化最具代表性的文化因素之一。本文以类型学方法对仰韶文化彩陶的纹饰母题进行分类梳理，探讨鱼纹、人面纹、几何纹等母题的演变规律与区域差异，并在此基础上讨论彩陶纹饰所反映的社会组织与精神信仰问题。",
        "keywords": "仰韶文化；彩陶纹饰；类型学；母题",
        "body": [
            "一、问题的提出",
            "仰韶文化以发达的彩陶著称，彩陶纹饰不仅是装饰艺术，更是考古学文化研究的重要标尺。本文尝试在类型学框架下，对仰韶文化彩陶纹饰进行系统的分类与排比，以期揭示其演变规律。",
            "二、纹饰母题的类型划分",
            "根据纹饰母题的不同，可将仰韶文化彩陶划分为鱼纹、人面纹、几何纹、花瓣纹等若干类型。鱼纹多见于半坡类型，人面纹与鱼纹组合是半坡彩陶的典型特征；几何纹则广泛分布于庙底沟类型彩陶中，包括圆点、弧线三角、勾叶等母题。",
            "三、纹饰演变的类型学序列",
            "从类型学角度看，仰韶文化彩陶纹饰呈现出由具象向抽象演变的趋势：早期鱼纹写实性强，晚期则简化为抽象的几何线条。这一演变序列与陶器形制的演化相互印证，为仰韶文化的分期提供了重要依据。",
            "四、余论",
            "彩陶纹饰的类型学分析表明，仰韶文化内部存在明显的地域差异与时代差异，其背后可能反映了不同区域人群之间的交流与互动。纹饰母题的传播与变异，是探讨考古学文化关系的重要切入点。",
        ],
        "refs": [
            "[1] 李启文. 新石器时代陶器类型学研究述评[J]. 考古学报, 2018(3): 245-268.",
            "[2] 王明远. 龙山文化蛋壳黑陶工艺与技术特征[J]. 考古, 2016(4): 78-95.",
        ],
    },
    {
        "file": "龙山文化蛋壳黑陶工艺与技术特征.docx",
        "title": "龙山文化蛋壳黑陶工艺与技术特征",
        "authors": "王明远",
        "year": "2016",
        "journal": "考古",
        "pages": "2016(4): 78-95",
        "abstract": "蛋壳黑陶是龙山文化最具辨识度的陶器品类，其胎壁之薄、工艺之精代表了新石器时代制陶技术的最高水平。本文从原料选择、成型工艺、烧成制度等方面分析蛋壳黑陶的技术特征，并讨论其产生与消亡的技术动因。",
        "keywords": "龙山文化；蛋壳黑陶；制陶工艺；烧成技术",
        "body": [
            "一、引言",
            "龙山文化以磨光黑陶和蛋壳黑陶著称。蛋壳黑陶胎薄如蛋壳，器表漆黑光亮，是龙山文化制陶工艺的杰出代表，也是探讨史前技术演进的重要标本。",
            "二、原料与成型",
            "蛋壳黑陶的原料经过精细淘洗，胎土细腻均匀。成型采用快轮拉坯技术，器物口沿与腹部厚薄一致，体现了高超的轮制水平。部分器物表面可见细密的轮旋痕迹，是快轮成型的重要证据。",
            "三、烧成工艺",
            "蛋壳黑陶采用渗碳工艺烧成，在窑内高温阶段控制气氛，使碳粒渗入胎体，形成漆黑光亮的表面效果。烧成温度经检测约在900-1000摄氏度之间，在当时的技术条件下已达到较高水平。",
            "四、结语",
            "蛋壳黑陶的出现标志着新石器时代制陶技术发展到一个新的高度。其工艺特征的研究，对于理解龙山时代的社会分工与技术进步具有重要价值，也为此后青铜时代制陶技术的发展奠定了技术基础。",
        ],
        "refs": [
            "[1] 张玉良. 仰韶文化彩陶纹饰的考古学观察[J]. 文物, 2015(2): 34-51.",
            "[2] 李启文. 新石器时代陶器类型学研究述评[J]. 考古学报, 2018(3): 245-268.",
        ],
    },
]


def _find_cjk_font():
    """查找可用的中文字体文件（.ttf 优先于 .ttc）。"""
    import os
    fonts_dir = Path(os.environ.get("WINDIR", "C:\\Windows")) / "Fonts"
    candidates = ["simhei.ttf", "simkai.ttf", "simfang.ttf",
                  "msyh.ttc", "simsun.ttc", "Deng.ttf"]
    for name in candidates:
        p = fonts_dir / name
        if p.exists():
            return str(p)
    return None


def render_pdf(path: Path, text: str):
    """用 fitz 从纯文本渲染 PDF（嵌入系统中文字体）。"""
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4
    fontfile = _find_cjk_font()
    if fontfile:
        page.insert_font(fontname="cjk", fontfile=fontfile)
        fontname = "cjk"
    else:
        fontname = "helv"  # 无中文字体时退化为拉丁字体（仅作兜底）
    y = 60
    for line in text.split("\n"):
        if y > 800:
            page = doc.new_page(width=595, height=842)
            y = 60
        page.insert_text((50, y), line, fontsize=10.5, fontname=fontname)
        y += 16
    doc.save(str(path))
    doc.close()


def build_sample_text(item: dict) -> str:
    lines = [item["title"], item["authors"], item["year"],
             f"摘要：{item['abstract']}", f"关键词：{item['keywords']}", ""]
    lines += item["body"]
    lines += ["", "参考文献："] + item["refs"]
    return "\n".join(lines)


def make_samples(cfg=None):
    cfg = cfg or load_config()
    samples_dir = Path(cfg["samples_dir"])
    samples_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for item in SAMPLE_TEXTS:
        out = samples_dir / item["file"]
        if out.exists():
            continue
        text = build_sample_text(item)
        if out.suffix.lower() == ".pdf":
            render_pdf(out, text)
        elif out.suffix.lower() == ".docx":
            import docx
            d = docx.Document()
            for line in text.split("\n"):
                if line.strip():
                    d.add_paragraph(line)
            d.save(str(out))
        else:
            out.write_text(text, encoding="utf-8")
        made.append(out.name)
    return made


if __name__ == "__main__":
    created = make_samples()
    print("已生成示例论文:", created or "（已存在，跳过）")
