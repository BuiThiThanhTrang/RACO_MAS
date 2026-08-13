from __future__ import annotations

import copy
import re
import sys
import zipfile
from pathlib import Path

from lxml import etree


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}
W = f"{{{W_NS}}}"


NEW_REFERENCES = {
    10: (
        '[10] B. Yan et al., “Beyond Self-Talk: A Communication-Centric Survey of '
        'LLM-Based Multi-Agent Systems,” arXiv preprint arXiv:2502.14321, 2025, '
        "doi: 10.48550/arXiv.2502.14321."
    ),
    11: (
        '[11] S. Iqbal, R. Costales, and F. Sha, “ALMA: Hierarchical Learning for '
        'Composite Multi-Agent Tasks,” arXiv preprint arXiv:2205.14205, 2022, '
        "doi: 10.48550/arXiv.2205.14205."
    ),
    12: (
        '[12] Y. Yue et al., “MasRouter: Learning to Route LLMs for Multi-Agent '
        'Systems,” in Proc. ACL, 2025, pp. 15549-15572, '
        "doi: 10.18653/v1/2025.acl-long.757."
    ),
    13: (
        '[13] Y. Zeng et al., “S2-MAD: Breaking the Token Barrier to Enhance '
        'Multi-Agent Debate Efficiency,” in Proc. NAACL-HLT, 2025, '
        "pp. 9393-9408, doi: 10.18653/v1/2025.naacl-long.475."
    ),
    14: (
        '[14] Z. Wang et al., “AgentDropout: Dynamic Agent Elimination for '
        'Token-Efficient and High-Performance LLM-Based Multi-Agent '
        'Collaboration,” in Proc. ACL, 2025, pp. 24013-24035, '
        "doi: 10.18653/v1/2025.acl-long.1170."
    ),
    15: (
        '[15] D. X. Long, D. N. Yen, A. T. Luu, K. Kawaguchi, M.-Y. Kan, and '
        'N. F. Chen, “Multi-expert Prompting Improves Reliability, Safety and '
        'Usefulness of Large Language Models,” in Proc. EMNLP, 2024, '
        "pp. 20370-20401, doi: 10.18653/v1/2024.emnlp-main.1135."
    ),
}


# The leading text uniquely identifies the existing paragraph. Citation strings
# are appended before its final full stop; no proposal wording is replaced.
CITATIONS_TO_APPEND = {
    "Các mô hình ngôn ngữ lớn": [10],
    "Mahajan et al. gọi khả năng": [11],
    "Đề tài đề xuất một framework Capability-Aware": [12],
    "Trong quá trình huấn luyện, thành phần đội": [12],
    "Early-stop/Answer Gate:": [13, 14],
    "MAS-aware prompts:": [6, 10, 15],
    "Kết hợp điều phối, kiểm soát dừng": [12, 13, 14, 15],
    "Mỗi agent nhận MAS-aware prompt": [6, 10, 12, 13, 14, 15],
    "Chỉ số đánh giá gồm accuracy": [13, 14],
}


def paragraph_text(paragraph: etree._Element) -> str:
    return "".join(paragraph.xpath(".//w:t/text()", namespaces=NS))


def clone_run_properties(paragraph: etree._Element) -> etree._Element | None:
    runs = paragraph.xpath("./w:r", namespaces=NS)
    if not runs:
        runs = paragraph.xpath(".//w:r", namespaces=NS)
    if not runs:
        return None
    rpr = runs[-1].find(f"{W}rPr")
    return copy.deepcopy(rpr) if rpr is not None else None


def make_run(text: str, rpr: etree._Element | None = None) -> etree._Element:
    run = etree.Element(f"{W}r")
    if rpr is not None:
        run.append(copy.deepcopy(rpr))
    text_el = etree.SubElement(run, f"{W}t")
    if text.startswith(" ") or text.endswith(" "):
        text_el.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    text_el.text = text
    return run


def make_internal_hyperlink(
    text: str, anchor: str, rpr: etree._Element | None = None
) -> etree._Element:
    hyperlink = etree.Element(f"{W}hyperlink")
    hyperlink.set(f"{W}anchor", anchor)
    hyperlink.set(f"{W}history", "1")
    run = make_run(text, rpr)
    run_rpr = run.find(f"{W}rPr")
    if run_rpr is None:
        run_rpr = etree.Element(f"{W}rPr")
        run.insert(0, run_rpr)
    color = etree.SubElement(run_rpr, f"{W}color")
    color.set(f"{W}val", "0563C1")
    underline = etree.SubElement(run_rpr, f"{W}u")
    underline.set(f"{W}val", "single")
    hyperlink.append(run)
    return hyperlink


def remove_final_period(paragraph: etree._Element) -> None:
    for text_el in reversed(paragraph.xpath(".//w:t", namespaces=NS)):
        if text_el.text and text_el.text.rstrip().endswith("."):
            stripped = text_el.text.rstrip()
            trailing = text_el.text[len(stripped) :]
            text_el.text = stripped[:-1] + trailing
            return
    raise RuntimeError(f"Paragraph has no final period: {paragraph_text(paragraph)!r}")


def append_citations(paragraph: etree._Element, numbers: list[int]) -> None:
    remove_final_period(paragraph)
    rpr = clone_run_properties(paragraph)
    paragraph.append(make_run(" ", rpr))
    for index, number in enumerate(numbers):
        if index:
            paragraph.append(make_run(", ", rpr))
        paragraph.append(
            make_internal_hyperlink(f"[{number}]", f"ref_{number}", rpr)
        )
    paragraph.append(make_run(".", rpr))


def add_reference_bookmark(
    paragraph: etree._Element, number: int, bookmark_id: int
) -> None:
    bookmark_start = etree.Element(f"{W}bookmarkStart")
    bookmark_start.set(f"{W}id", str(bookmark_id))
    bookmark_start.set(f"{W}name", f"ref_{number}")
    bookmark_end = etree.Element(f"{W}bookmarkEnd")
    bookmark_end.set(f"{W}id", str(bookmark_id))

    ppr = paragraph.find(f"{W}pPr")
    insert_at = 1 if ppr is not None else 0
    paragraph.insert(insert_at, bookmark_start)
    paragraph.append(bookmark_end)


def replace_reference_text(paragraph: etree._Element, text: str) -> None:
    runs = paragraph.xpath("./w:r", namespaces=NS)
    if not runs:
        raise RuntimeError("Reference paragraph has no direct run.")
    template_run = runs[0]
    for child in list(paragraph):
        if child.tag != f"{W}pPr":
            paragraph.remove(child)
    new_run = copy.deepcopy(template_run)
    for child in list(new_run):
        if child.tag != f"{W}rPr":
            new_run.remove(child)
    text_el = etree.SubElement(new_run, f"{W}t")
    text_el.text = text
    paragraph.append(new_run)


def link_existing_citations(paragraph: etree._Element) -> None:
    full_text = paragraph_text(paragraph)
    if not re.search(r"\[\d+\]", full_text):
        return
    direct_runs = paragraph.xpath("./w:r", namespaces=NS)
    if len(direct_runs) != 1:
        return
    run = direct_runs[0]
    rpr = run.find(f"{W}rPr")
    rpr = copy.deepcopy(rpr) if rpr is not None else None
    ppr = paragraph.find(f"{W}pPr")
    for child in list(paragraph):
        if child is not ppr:
            paragraph.remove(child)
    cursor = 0
    for match in re.finditer(r"\[(\d+)\]", full_text):
        if match.start() > cursor:
            paragraph.append(make_run(full_text[cursor : match.start()], rpr))
        number = int(match.group(1))
        paragraph.append(
            make_internal_hyperlink(match.group(0), f"ref_{number}", rpr)
        )
        cursor = match.end()
    if cursor < len(full_text):
        paragraph.append(make_run(full_text[cursor:], rpr))


def patch_document_xml(xml_bytes: bytes) -> bytes:
    parser = etree.XMLParser(remove_blank_text=False)
    root = etree.fromstring(xml_bytes, parser)
    body = root.find(f"{W}body")
    if body is None:
        raise RuntimeError("word/document.xml has no w:body.")

    paragraphs = body.xpath("./w:p", namespaces=NS)
    matches: dict[str, etree._Element] = {}
    for prefix in CITATIONS_TO_APPEND:
        found = [p for p in paragraphs if paragraph_text(p).startswith(prefix)]
        if len(found) != 1:
            raise RuntimeError(
                f"Expected one paragraph starting {prefix!r}, found {len(found)}."
            )
        matches[prefix] = found[0]

    for prefix, numbers in CITATIONS_TO_APPEND.items():
        append_citations(matches[prefix], numbers)

    paragraphs = body.xpath("./w:p", namespaces=NS)
    reference_paragraphs: dict[int, etree._Element] = {}
    for paragraph in paragraphs:
        match = re.match(r"^\[(\d+)\]\s", paragraph_text(paragraph))
        if match:
            reference_paragraphs[int(match.group(1))] = paragraph
    if sorted(reference_paragraphs) != list(range(1, 10)):
        raise RuntimeError(
            f"Expected references 1-9, found {sorted(reference_paragraphs)}."
        )

    last_reference = reference_paragraphs[9]
    insertion_point = last_reference
    for number, text in NEW_REFERENCES.items():
        new_paragraph = copy.deepcopy(last_reference)
        replace_reference_text(new_paragraph, text)
        insertion_point.addnext(new_paragraph)
        insertion_point = new_paragraph
        reference_paragraphs[number] = new_paragraph

    existing_ids = [
        int(value)
        for value in root.xpath("//w:bookmarkStart/@w:id", namespaces=NS)
        if value.isdigit()
    ]
    next_bookmark_id = max(existing_ids, default=0) + 1
    for number in range(1, 16):
        add_reference_bookmark(
            reference_paragraphs[number], number, next_bookmark_id
        )
        next_bookmark_id += 1

    # Convert pre-existing numeric citations in simple body paragraphs into
    # internal links. Newly appended citations are already hyperlinks.
    for paragraph in body.xpath("./w:p", namespaces=NS):
        text = paragraph_text(paragraph)
        if re.match(r"^\[\d+\]\s", text):
            continue
        link_existing_citations(paragraph)

    return etree.tostring(
        root, xml_declaration=True, encoding="UTF-8", standalone="yes"
    )


def patch_docx(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(source, "r") as zin:
        document_xml = zin.read("word/document.xml")
        patched_xml = patch_document_xml(document_xml)
        with zipfile.ZipFile(
            destination, "w", compression=zipfile.ZIP_DEFLATED
        ) as zout:
            for item in zin.infolist():
                payload = (
                    patched_xml
                    if item.filename == "word/document.xml"
                    else zin.read(item.filename)
                )
                zout.writestr(item, payload)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: cite_proposal_ieee.py INPUT.docx OUTPUT.docx")
    source = Path(sys.argv[1]).resolve()
    destination = Path(sys.argv[2]).resolve()
    patch_docx(source, destination)
    print(destination)


if __name__ == "__main__":
    main()
