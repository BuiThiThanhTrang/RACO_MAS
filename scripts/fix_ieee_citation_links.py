from __future__ import annotations

import copy
import re
import zipfile
from pathlib import Path

from lxml import etree


INPUT = Path(r"D:\SideProject\ChatDev\tmp\citation_edit\proposal_generalization_mas_working.docx")
OUTPUT = Path(r"D:\SideProject\ChatDev\tmp\citation_edit\proposal_generalization_mas_fixed.docx")

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML = "http://www.w3.org/XML/1998/namespace"
NS = {"w": W}


def qn(local: str) -> str:
    return f"{{{W}}}{local}"


def paragraph_text(paragraph) -> str:
    return "".join(paragraph.xpath(".//w:t/text()", namespaces=NS))


def styled_rpr(base_rpr, *, link: bool):
    rpr = copy.deepcopy(base_rpr) if base_rpr is not None else etree.Element(qn("rPr"))
    if not link:
        return rpr
    color = rpr.find(qn("color"))
    if color is None:
        color = etree.Element(qn("color"))
        rpr.append(color)
    color.set(qn("val"), "0000FF")
    underline = rpr.find(qn("u"))
    if underline is None:
        underline = etree.Element(qn("u"))
        rpr.append(underline)
    underline.set(qn("val"), "none")
    return rpr


def make_run(text: str, base_rpr, *, citation_style: bool = False):
    run = etree.Element(qn("r"))
    run.append(styled_rpr(base_rpr, link=citation_style))
    node = etree.SubElement(run, qn("t"))
    if text.startswith(" ") or text.endswith(" "):
        node.set(f"{{{XML}}}space", "preserve")
    node.text = text
    return run


def make_internal_link(label: str, number: int, base_rpr):
    hyperlink = etree.Element(qn("hyperlink"))
    hyperlink.set(qn("anchor"), f"ref_{number}")
    hyperlink.set(qn("history"), "1")
    hyperlink.append(make_run(label, base_rpr, citation_style=True))
    return hyperlink


def citation_nodes(raw: str, base_rpr):
    if raw == "[4-6]":
        return [
            make_internal_link("[4]", 4, base_rpr),
            make_run("–", base_rpr, citation_style=True),
            make_internal_link("[6]", 6, base_rpr),
        ]
    if raw == "[3, 7-9]":
        return [
            make_internal_link("[3]", 3, base_rpr),
            make_run(", ", base_rpr, citation_style=True),
            make_internal_link("[7]", 7, base_rpr),
            make_run("–", base_rpr, citation_style=True),
            make_internal_link("[9]", 9, base_rpr),
        ]
    if raw in {"[1]", "[2]"}:
        number = int(raw[1:-1])
        return [make_internal_link(raw, number, base_rpr)]
    raise ValueError(f"Unexpected citation: {raw}")


def replace_citation_in_paragraph(paragraph, raw: str) -> None:
    runs = paragraph.xpath("./w:r", namespaces=NS)
    containing = [r for r in runs if raw in "".join(r.xpath(".//w:t/text()", namespaces=NS))]
    if len(containing) != 1:
        raise ValueError(f"Expected one run containing {raw!r}, found {len(containing)}")
    old_run = containing[0]
    old_text = "".join(old_run.xpath(".//w:t/text()", namespaces=NS))
    if old_text.count(raw) != 1:
        raise ValueError(f"Expected one occurrence of {raw!r} in its run")
    before, after = old_text.split(raw, 1)
    base_rpr = old_run.find(qn("rPr"))
    replacement = []
    if before:
        replacement.append(make_run(before, base_rpr))
    replacement.extend(citation_nodes(raw, base_rpr))
    if after:
        replacement.append(make_run(after, base_rpr))
    parent = old_run.getparent()
    index = parent.index(old_run)
    parent.remove(old_run)
    for offset, node in enumerate(replacement):
        parent.insert(index + offset, node)


def add_reference_bookmark(paragraph, number: int) -> None:
    start = etree.Element(qn("bookmarkStart"))
    start.set(qn("id"), str(1000 + number))
    start.set(qn("name"), f"ref_{number}")
    end = etree.Element(qn("bookmarkEnd"))
    end.set(qn("id"), str(1000 + number))
    ppr = paragraph.find(qn("pPr"))
    insert_at = 1 if ppr is not None else 0
    paragraph.insert(insert_at, start)
    paragraph.append(end)


def patch_document_xml(xml_bytes: bytes) -> tuple[bytes, dict]:
    parser = etree.XMLParser(remove_blank_text=False)
    root = etree.fromstring(xml_bytes, parser)
    paragraphs = root.xpath(".//w:body//w:p", namespaces=NS)
    before_text = [paragraph_text(p) for p in paragraphs]

    replacements = {
        "[4-6]": "[4]–[6]",
        "[3, 7-9]": "[3], [7]–[9]",
        "[1]": "[1]",
        "[2]": "[2]",
    }
    changed = []
    references_started = False
    for paragraph in paragraphs:
        text = paragraph_text(paragraph)
        if text.strip() == "Tài liệu tham khảo":
            references_started = True
            continue
        if not references_started:
            for raw in replacements:
                if raw in text:
                    replace_citation_in_paragraph(paragraph, raw)
                    changed.append(raw)
                    break

    if sorted(changed) != sorted(replacements):
        raise ValueError(f"Citation coverage mismatch: {changed}")

    bookmark_count = 0
    for paragraph in paragraphs:
        text = paragraph_text(paragraph).strip()
        match = re.match(r"^\[(\d+)\]\s", text)
        if match:
            number = int(match.group(1))
            if 1 <= number <= 9:
                add_reference_bookmark(paragraph, number)
                bookmark_count += 1
    if bookmark_count != 9:
        raise ValueError(f"Expected 9 reference bookmarks, found {bookmark_count}")

    after_text = [paragraph_text(p) for p in paragraphs]
    changed_paragraphs = []
    for idx, (old, new) in enumerate(zip(before_text, after_text)):
        if old != new:
            changed_paragraphs.append((idx, old, new))
    expected_pairs = {(raw, new) for raw, new in replacements.items() if raw != new}
    for idx, old, new in changed_paragraphs:
        if not any(raw in old and old.replace(raw, repl) == new for raw, repl in replacements.items()):
            raise ValueError(f"Unexpected text change in paragraph {idx}: {old!r} -> {new!r}")

    out = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    return out, {
        "citations_patched": changed,
        "bookmarks_added": bookmark_count,
        "text_changes": changed_paragraphs,
    }


def main() -> None:
    with zipfile.ZipFile(INPUT, "r") as zin:
        document_xml, report = patch_document_xml(zin.read("word/document.xml"))
        with zipfile.ZipFile(OUTPUT, "w") as zout:
            for item in zin.infolist():
                data = document_xml if item.filename == "word/document.xml" else zin.read(item.filename)
                zout.writestr(item, data)
    print(report)
    print(OUTPUT)


if __name__ == "__main__":
    main()
