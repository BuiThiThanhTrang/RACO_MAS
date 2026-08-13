from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


SOURCE = Path(
    r"C:\Users\23521\.codex\attachments\0eb37e66-6089-4a5c-8d39-64386f48d932\pasted-text.txt"
)
OUTPUT = Path(r"D:\SideProject\ChatDev\output\docx\proposal_MAS_generalization_IEEE.docx")

FONT = "Times New Roman"
BLUE = "0000FF"
BLACK = "000000"
TABLE_HEADER = "E7E6E6"
CONTENT_WIDTH_DXA = 9072  # 6.30 in: matches the A4 reference PDF's text block.


# IEEE numbering follows first appearance in the proposal body.
CITATION_NUMBER = {
    "Yan et al. 2025": 1,
    "Sun et al. 2025": 2,
    "Tran et al. 2025": 3,
    "Zhu et al. 2022, 1-48": 4,
    "Zhang et al. 2025": 5,
    "Liu et al. 2025": 6,
    "Chen et al. 2026": 7,  # GoAgent; the Five-Ws survey is disambiguated below.
    "Shu et al. 2024": 8,
    "Yang et al. 2025": 9,
    "Adimulam et al. 2026": 10,
    "Charalambous et al. 2025, 6576-6611": 11,
    "Mason et al. 2024, 1566-1581": 12,
    "Moore 2025": 13,
    "Jia and Pei 2025": 14,
    "Li et al. 2025": 15,
    "Yuan et al. 2023, 1-17": 16,
    "Dubey et al. 2025, 1-7": 17,
}


REFERENCES = [
    "B. Yan, X. Zhang, L. Zhang, L. Zhang, Z. Zhou, D. Miao, and C. Li, “Beyond Self-Talk: A Communication-Centric Survey of LLM-Based Multi-Agent Systems,” *arXiv preprint arXiv:2502.14321*, 2025, doi: 10.1007/s11704-026-50857-y.",
    "L. Sun, Y. Yang, Q. Duan, Y. Shi, C. Lyu, Y.-C. Chang, C.-T. Lin, and Y. Shen, “Multi-Agent Coordination across Diverse Applications: A Survey,” *arXiv preprint arXiv:2502.14743*, 2025, doi: 10.48550/arXiv.2502.14743.",
    "K.-T. Tran, D. Dao, M.-D. Nguyen, Q.-V. Pham, B. O’Sullivan, and H. D. Nguyen, “Multi-Agent Collaboration Mechanisms: A Survey of LLMs,” *arXiv preprint arXiv:2501.06322*, 2025, doi: 10.48550/arXiv.2501.06322.",
    "C. Zhu, M. Dastani, and S. Wang, “A survey of multi-agent deep reinforcement learning with communication,” *Autonomous Agents and Multi-Agent Systems*, vol. 38, pp. 1-48, 2022, doi: 10.1007/s10458-023-09633-6.",
    "X. Zhang, J. Yu, and Z. Zhong, “Learning Efficient Communication Protocols for Multi-Agent Reinforcement Learning,” *arXiv preprint arXiv:2511.09171*, 2025, doi: 10.48550/arXiv.2511.09171.",
    "Z. Liu, Y. Li, J. Wang, J. Tu, Y. Hong, F. Li, Y. Liu, T. Sugawara, and Y. Tang, “Robust and Efficient Communication in Multi-Agent Reinforcement Learning,” *Chaos*, vol. 36, no. 2, 2025, doi: 10.48550/arXiv.2511.11393.",
    "H. Chen, X. Zheng, Y. Liu, P. Jiao, S. Li, H. Liu, Z. Zhao, Z. Xu, I. Khalil, and S. Pan, “GoAgent: Group-of-Agents Communication Topology Generation for LLM-based Multi-Agent Systems,” *arXiv preprint arXiv:2603.19677*, 2026, doi: 10.48550/arXiv.2603.19677.",
    "R. Shu, N. Das, M. Yuan, M. Sunkara, and Y. Zhang, “Towards Effective GenAI Multi-Agent Collaboration: Design and Evaluation for Enterprise Applications,” *arXiv preprint arXiv:2412.05449*, 2024, doi: 10.48550/arXiv.2412.05449.",
    "Y. Yang, H. Chai, S. Shao, Y. Song, S. Qi, R. Rui, and W. Zhang, “AgentNet: Decentralized Evolutionary Coordination for LLM-based Multi-Agent Systems,” *arXiv preprint arXiv:2504.00587*, 2025, doi: 10.48550/arXiv.2504.00587.",
    "A. Adimulam, R. Gupta, and S. Kumar, “The Orchestration of Multi-Agent Systems: Architectures, Protocols, and Enterprise Adoption,” *arXiv preprint arXiv:2601.13671*, 2026, doi: 10.48550/arXiv.2601.13671.",
    "T. Charalambous, N. Pappas, N. Nomikos, and R. Wichman, “Toward Goal-Oriented Communication in Multi-Agent Systems: An Overview,” *IEEE Open Journal of the Communications Society*, vol. 7, pp. 6576-6611, 2025, doi: 10.1109/OJCOMS.2026.3703748.",
    "F. Mason, F. Chiariotti, A. Zanella, and P. Popovski, “Multi-Agent Reinforcement Learning for Coordinating Communication and Control,” *IEEE Transactions on Cognitive Communications and Networking*, vol. 10, pp. 1566-1581, 2024, doi: 10.1109/TCCN.2024.3384492.",
    "D. Moore, “A Taxonomy of Hierarchical Multi-Agent Systems: Design Patterns, Coordination Mechanisms, and Industrial Applications,” *arXiv preprint arXiv:2508.12683*, 2025, doi: 10.48550/arXiv.2508.12683.",
    "L. Jia and Y. Pei, “Recent Advances in Multi-Agent Reinforcement Learning for Intelligent Automation and Control of Water Environment Systems,” *Machines*, vol. 13, no. 6, 2025, doi: 10.3390/machines13060503.",
    "Z. Li, S. Campos, and N. Wang, “Language-Driven Coordination and Learning in Multi-Agent Simulation Environments,” *arXiv preprint arXiv:2506.04251*, 2025, doi: 10.48550/arXiv.2506.04251.",
    "L. Yuan, F. Chen, Z. Zhang, and Y. Yu, “Communication-robust multi-agent learning by adaptable auxiliary multi-agent adversary generation,” *Frontiers of Computer Science*, vol. 18, pp. 1-17, 2023, doi: 10.1007/s11704-023-2733-5.",
    "A. Dubey, R. Gautam, and S. Mishra, “Multi-Agent Systems for Collaborative and Distributed Decision-Making: Design and Applications,” in *Proc. 7th Int. Conf. Artificial Intelligence and Speech Technology (AIST)*, 2025, pp. 1-7, doi: 10.1109/AIST68591.2025.11441571.",
    "J. Chen, H. Yang, Z. Liu, and C. Joe-Wong, “The Five Ws of Multi-Agent Communication: Who Talks to Whom, When, What, and Why - A Survey from MARL to Emergent Language and LLMs,” *Transactions on Machine Learning Research*, 2026, doi: 10.48550/arXiv.2602.11583.",
]


def set_run_font(run, size: float = 12, *, bold=None, italic=None, color=BLACK):
    run.font.name = FONT
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), FONT)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), FONT)
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), FONT)
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    run.font.color.rgb = RGBColor.from_string(color)


def set_style_font(style, size: float, *, bold=False, color=BLACK):
    style.font.name = FONT
    style._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), FONT)
    style._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), FONT)
    style._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), FONT)
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.color.rgb = RGBColor.from_string(color)


def set_widow_control(paragraph, enabled=True):
    ppr = paragraph._p.get_or_add_pPr()
    node = ppr.find(qn("w:widowControl"))
    if node is None:
        node = OxmlElement("w:widowControl")
        ppr.append(node)
    node.set(qn("w:val"), "1" if enabled else "0")


def add_page_field(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    set_run_font(run, 11)
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instr, separate, text, end])


def add_numbering_definition(doc, kind: str) -> int:
    numbering = doc.part.numbering_part.element
    abstract_ids = [int(x.get(qn("w:abstractNumId"))) for x in numbering.findall(qn("w:abstractNum"))]
    num_ids = [int(x.get(qn("w:numId"))) for x in numbering.findall(qn("w:num"))]
    abstract_id = max(abstract_ids, default=0) + 1
    num_id = max(num_ids, default=0) + 1

    abstract = OxmlElement("w:abstractNum")
    abstract.set(qn("w:abstractNumId"), str(abstract_id))
    multi = OxmlElement("w:multiLevelType")
    multi.set(qn("w:val"), "singleLevel")
    abstract.append(multi)
    lvl = OxmlElement("w:lvl")
    lvl.set(qn("w:ilvl"), "0")
    start = OxmlElement("w:start")
    start.set(qn("w:val"), "1")
    lvl.append(start)
    numfmt = OxmlElement("w:numFmt")
    numfmt.set(qn("w:val"), "bullet" if kind == "bullet" else "decimal")
    lvl.append(numfmt)
    lvltext = OxmlElement("w:lvlText")
    lvltext.set(qn("w:val"), "•" if kind == "bullet" else ("[%1]" if kind == "reference" else "%1."))
    lvl.append(lvltext)
    jc = OxmlElement("w:lvlJc")
    jc.set(qn("w:val"), "left")
    lvl.append(jc)
    ppr = OxmlElement("w:pPr")
    tabs = OxmlElement("w:tabs")
    tab = OxmlElement("w:tab")
    tab.set(qn("w:val"), "num")
    tab.set(qn("w:pos"), "720" if kind != "reference" else "600")
    tabs.append(tab)
    ppr.append(tabs)
    ind = OxmlElement("w:ind")
    if kind == "reference":
        ind.set(qn("w:left"), "600")
        ind.set(qn("w:hanging"), "600")
    else:
        ind.set(qn("w:left"), "720")
        ind.set(qn("w:hanging"), "360")
    ppr.append(ind)
    lvl.append(ppr)
    rpr = OxmlElement("w:rPr")
    rfonts = OxmlElement("w:rFonts")
    for attr in ("ascii", "hAnsi", "eastAsia"):
        rfonts.set(qn(f"w:{attr}"), FONT)
    rpr.append(rfonts)
    lvl.append(rpr)
    abstract.append(lvl)
    # OOXML requires every abstractNum before the first concrete num. Appending
    # an abstract after existing num elements makes Word repair the numbering
    # part and can collapse otherwise distinct bullet/decimal/reference lists.
    first_num = numbering.find(qn("w:num"))
    if first_num is None:
        numbering.append(abstract)
    else:
        numbering.insert(list(numbering).index(first_num), abstract)

    num = OxmlElement("w:num")
    num.set(qn("w:numId"), str(num_id))
    ref = OxmlElement("w:abstractNumId")
    ref.set(qn("w:val"), str(abstract_id))
    num.append(ref)
    numbering.append(num)
    return num_id


def apply_numbering(paragraph, num_id: int):
    ppr = paragraph._p.get_or_add_pPr()
    numpr = ppr.find(qn("w:numPr"))
    if numpr is None:
        numpr = OxmlElement("w:numPr")
        ppr.append(numpr)
    ilvl = OxmlElement("w:ilvl")
    ilvl.set(qn("w:val"), "0")
    num = OxmlElement("w:numId")
    num.set(qn("w:val"), str(num_id))
    numpr.extend([ilvl, num])


def add_bookmark(paragraph, name: str, bookmark_id: int):
    start = OxmlElement("w:bookmarkStart")
    start.set(qn("w:id"), str(bookmark_id))
    start.set(qn("w:name"), name)
    end = OxmlElement("w:bookmarkEnd")
    end.set(qn("w:id"), str(bookmark_id))
    ppr = paragraph._p.find(qn("w:pPr"))
    insert_at = 1 if ppr is not None else 0
    paragraph._p.insert(insert_at, start)
    paragraph._p.append(end)


def add_internal_hyperlink(paragraph, text: str, anchor: str):
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("w:anchor"), anchor)
    run = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    rfonts = OxmlElement("w:rFonts")
    for attr in ("ascii", "hAnsi", "eastAsia"):
        rfonts.set(qn(f"w:{attr}"), FONT)
    color = OxmlElement("w:color")
    color.set(qn("w:val"), BLUE)
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "none")
    size = OxmlElement("w:sz")
    size.set(qn("w:val"), "24")
    rpr.extend([rfonts, color, underline, size])
    run.append(rpr)
    t = OxmlElement("w:t")
    t.text = text
    run.append(t)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def normalize_text(text: str) -> str:
    replacements = {
        "\u00a0": " ",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return re.sub(r"[ \t]+", " ", text).strip()


def citation_numbers(group: str) -> list[int]:
    group = normalize_text(group)
    # The source uses the same author-year key for two Chen et al. papers.
    # This exact co-citation occurs in the sentence quoting the Five-Ws framing.
    if group == "Chen et al. 2026; Tran et al. 2025":
        return [3, 18]
    nums = []
    for item in [x.strip() for x in group.split(";")]:
        if item not in CITATION_NUMBER:
            raise ValueError(f"Unmapped citation: {item}")
        nums.append(CITATION_NUMBER[item])
    return sorted(set(nums))


def add_citation(paragraph, group: str):
    nums = citation_numbers(group)
    left = paragraph.add_run("[")
    set_run_font(left, color=BLUE)
    for i, num in enumerate(nums):
        if i:
            sep = paragraph.add_run(", ")
            set_run_font(sep, color=BLUE)
        add_internal_hyperlink(paragraph, str(num), f"ref_{num}")
    right = paragraph.add_run("]")
    set_run_font(right, color=BLUE)


INLINE_RE = re.compile(r"(\*\*.+?\*\*|(?<!\*)\*[^*]+?\*(?!\*)|`[^`]+`)")
CIT_RE = re.compile(r"\s*\(([^()]*(?:19|20)\d{2}[^()]*)\)")


def add_markup(paragraph, text: str, *, size=12, default_bold=False):
    text = normalize_text(text)
    for part in INLINE_RE.split(text):
        if not part:
            continue
        bold = default_bold
        italic = False
        if part.startswith("**") and part.endswith("**"):
            part = part[2:-2]
            bold = True
        elif part.startswith("*") and part.endswith("*"):
            part = part[1:-1]
            italic = True
        elif part.startswith("`") and part.endswith("`"):
            part = part[1:-1]
        run = paragraph.add_run(part)
        set_run_font(run, size, bold=bold, italic=italic)


def add_rich(paragraph, text: str, *, size=12, default_bold=False):
    pos = 0
    for match in CIT_RE.finditer(text):
        add_markup(paragraph, text[pos : match.start()], size=size, default_bold=default_bold)
        add_citation(paragraph, match.group(1))
        pos = match.end()
    add_markup(paragraph, text[pos:], size=size, default_bold=default_bold)


def set_body_format(paragraph, *, first_line=True):
    fmt = paragraph.paragraph_format
    fmt.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    fmt.left_indent = Pt(0)
    fmt.right_indent = Pt(0)
    fmt.first_line_indent = Inches(0.25) if first_line else Pt(0)
    fmt.space_before = Pt(0)
    fmt.space_after = Pt(0)
    fmt.line_spacing = 1.30
    set_widow_control(paragraph, True)


def add_heading(doc, text: str, level: int, *, first=False):
    paragraph = doc.add_paragraph(style=f"Heading {level}")
    paragraph.paragraph_format.space_before = Pt(0 if first else (18 if level == 1 else 12))
    add_rich(paragraph, normalize_text(text), default_bold=True)
    return paragraph


def add_body(doc, text: str, *, first_after_heading=False):
    paragraph = doc.add_paragraph(style="Normal")
    set_body_format(paragraph, first_line=not first_after_heading)
    add_rich(paragraph, text)
    return paragraph


def add_list_item(doc, text: str, num_id: int, *, numbered=False):
    paragraph = doc.add_paragraph(style="Normal")
    apply_numbering(paragraph, num_id)
    fmt = paragraph.paragraph_format
    fmt.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    fmt.space_before = Pt(0)
    fmt.space_after = Pt(4)
    fmt.line_spacing = 1.25
    set_widow_control(paragraph, True)
    add_rich(paragraph, text)
    return paragraph


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120):
    tc = cell._tc
    tcpr = tc.get_or_add_tcPr()
    tcmar = tcpr.first_child_found_in("w:tcMar")
    if tcmar is None:
        tcmar = OxmlElement("w:tcMar")
        tcpr.append(tcmar)
    for m, v in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tcmar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tcmar.append(node)
        node.set(qn("w:w"), str(v))
        node.set(qn("w:type"), "dxa")


def shade_cell(cell, fill):
    tcpr = cell._tc.get_or_add_tcPr()
    shd = tcpr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tcpr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_width(cell, width_dxa):
    tcpr = cell._tc.get_or_add_tcPr()
    tcw = tcpr.find(qn("w:tcW"))
    if tcw is None:
        tcw = OxmlElement("w:tcW")
        tcpr.append(tcw)
    tcw.set(qn("w:w"), str(width_dxa))
    tcw.set(qn("w:type"), "dxa")


def set_table_geometry(table, widths):
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    tblpr = table._tbl.tblPr
    tblw = tblpr.find(qn("w:tblW"))
    tblw.set(qn("w:w"), str(sum(widths)))
    tblw.set(qn("w:type"), "dxa")
    tblind = tblpr.find(qn("w:tblInd"))
    if tblind is None:
        tblind = OxmlElement("w:tblInd")
        tblpr.append(tblind)
    tblind.set(qn("w:w"), "0")
    tblind.set(qn("w:type"), "dxa")
    layout = tblpr.find(qn("w:tblLayout"))
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        tblpr.append(layout)
    layout.set(qn("w:type"), "fixed")
    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)
    for row in table.rows:
        for idx, cell in enumerate(row.cells):
            set_cell_width(cell, widths[idx])


def repeat_table_header(row):
    trpr = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    trpr.append(header)


def add_table(doc, rows):
    caption = doc.add_paragraph()
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.paragraph_format.space_before = Pt(6)
    caption.paragraph_format.space_after = Pt(5)
    run = caption.add_run("Bảng 1. Tóm tắt các thành phần triển khai")
    set_run_font(run, 11, bold=True)

    table = doc.add_table(rows=len(rows), cols=3)
    table.style = "Table Grid"
    widths = [2050, 4300, 2722]
    set_table_geometry(table, widths)
    for r_idx, row_data in enumerate(rows):
        for c_idx, text in enumerate(row_data):
            cell = table.cell(r_idx, c_idx)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell)
            if r_idx == 0:
                shade_cell(cell, TABLE_HEADER)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if r_idx == 0 else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.10
            add_rich(p, text, size=10, default_bold=(r_idx == 0))
        if r_idx == 0:
            repeat_table_header(table.rows[0])
    after = doc.add_paragraph()
    after.paragraph_format.space_after = Pt(0)
    return table


def configure_document(doc):
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.2)
    section.left_margin = Cm(2.8)
    section.right_margin = Cm(2.2)
    section.header_distance = Cm(1.25)
    section.footer_distance = Cm(1.15)

    normal = doc.styles["Normal"]
    set_style_font(normal, 12)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(0)
    normal.paragraph_format.line_spacing = 1.30

    h1 = doc.styles["Heading 1"]
    set_style_font(h1, 17.2, bold=True)
    h1.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    h1.paragraph_format.left_indent = Pt(0)
    h1.paragraph_format.first_line_indent = Pt(0)
    h1.paragraph_format.space_before = Pt(18)
    h1.paragraph_format.space_after = Pt(10)
    h1.paragraph_format.line_spacing = 1.0
    h1.paragraph_format.keep_with_next = True
    h1.paragraph_format.keep_together = True

    h2 = doc.styles["Heading 2"]
    set_style_font(h2, 14.35, bold=True)
    h2.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    h2.paragraph_format.left_indent = Pt(0)
    h2.paragraph_format.first_line_indent = Pt(0)
    h2.paragraph_format.space_before = Pt(12)
    h2.paragraph_format.space_after = Pt(7)
    h2.paragraph_format.line_spacing = 1.0
    h2.paragraph_format.keep_with_next = True
    h2.paragraph_format.keep_together = True

    title = doc.styles.add_style("Proposal Title", WD_STYLE_TYPE.PARAGRAPH)
    set_style_font(title, 15.5, bold=True)
    title.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(0)
    title.paragraph_format.space_after = Pt(14)
    title.paragraph_format.line_spacing = 1.15
    title.paragraph_format.keep_with_next = True

    refs = doc.styles.add_style("IEEE Reference", WD_STYLE_TYPE.PARAGRAPH)
    set_style_font(refs, 12)
    refs.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    refs.paragraph_format.space_before = Pt(0)
    refs.paragraph_format.space_after = Pt(7)
    refs.paragraph_format.line_spacing = 1.22
    refs.paragraph_format.widow_control = True

    footer = section.footer
    footer.is_linked_to_previous = False
    p = footer.paragraphs[0]
    for run in p.runs:
        run._element.getparent().remove(run._element)
    add_page_field(p)

    doc.core_properties.title = "Proposal MAS Generalization"
    doc.core_properties.subject = "Multi-agent system generalization"
    doc.core_properties.author = ""


def parse_table(lines, start_index):
    rows = []
    i = start_index
    while i < len(lines) and lines[i].strip().startswith("|"):
        cells = [x.strip() for x in lines[i].strip().strip("|").split("|")]
        if not all(re.fullmatch(r"-+", x) for x in cells):
            rows.append(cells)
        i += 1
    return rows, i


def build():
    raw = SOURCE.read_text(encoding="utf-8")
    start = raw.index("**Tên đề tài:**")
    end = raw.index("Nếu bạn muốn **file Word thực sự**")
    lines = raw[start:end].strip().splitlines()

    doc = Document()
    configure_document(doc)
    bullet_num_id = add_numbering_definition(doc, "bullet")
    decimal_num_id = add_numbering_definition(doc, "decimal")
    reference_num_id = add_numbering_definition(doc, "reference")

    after_heading = False
    section_3_2_added = False
    i = 0
    while i < len(lines):
        raw_line = lines[i].strip()
        i += 1
        if not raw_line:
            continue
        line = normalize_text(raw_line)

        if line.startswith("**Tên đề tài:**"):
            title_text = line[len("**Tên đề tài:**") :].strip()
            p = doc.add_paragraph(style="Proposal Title")
            add_rich(p, title_text, default_bold=True)
            after_heading = True
            continue

        if line.startswith("**1. ") and line.endswith("**"):
            add_heading(doc, line[2:-2], 1, first=False)
            after_heading = True
            continue
        if line.startswith("**2. ") and line.endswith("**"):
            add_heading(doc, line[2:-2], 1)
            after_heading = True
            continue
        if re.match(r"\*\*2\.[123]\. ", line) and line.endswith("**"):
            add_heading(doc, line[2:-2], 2)
            after_heading = True
            continue

        if line == "## Mục Tiêu Và Kế Hoạch":
            add_heading(doc, "3. Mục tiêu, nội dung và kế hoạch nghiên cứu", 1)
            add_heading(doc, "3.1. Mục tiêu", 2)
            after_heading = False
            continue

        if line == "## Bảng Tóm Tắt Triển Khai":
            add_heading(doc, "3.3. Bảng tóm tắt triển khai", 2)
            after_heading = False
            continue

        if line == "## Kết Quả Dự Kiến":
            add_heading(doc, "3.4. Kết quả dự kiến", 2)
            after_heading = True
            continue

        if line.startswith("| Thành phần"):
            table_rows, i = parse_table(lines, i - 1)
            add_table(doc, table_rows)
            after_heading = False
            continue

        if line.startswith("**Figure 1:**"):
            continue

        if line.startswith("- "):
            item = line[2:].strip()
            if item.startswith("**Nội dung 1:") and not section_3_2_added:
                add_heading(doc, "3.2. Nội dung và phương pháp nghiên cứu", 2)
                section_3_2_added = True
            add_list_item(doc, item, bullet_num_id)
            after_heading = False
            continue

        if line.startswith("Tính mới thứ nhất là"):
            parts = re.split(r"(?=Tính mới thứ (?:hai|ba|tư) là)", line)
            for part in parts:
                part = re.sub(r"^Tính mới thứ (?:nhất|hai|ba|tư) là\s*", "", part).strip()
                add_list_item(doc, part, decimal_num_id, numbered=True)
            after_heading = False
            continue

        add_body(doc, line, first_after_heading=after_heading)
        after_heading = False

    doc.add_page_break()
    add_heading(doc, "References", 1, first=True)
    for idx, reference in enumerate(REFERENCES, start=1):
        p = doc.add_paragraph(style="IEEE Reference")
        apply_numbering(p, reference_num_id)
        add_bookmark(p, f"ref_{idx}", 100 + idx)
        add_markup(p, reference, size=12)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build()
