from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


DOCX_PATH = Path(r"D:\SideProject\ChatDev\output\bao_cao_tien_do_puppeteer_generalization.docx")

BLUE = RGBColor(31, 77, 120)
HEADING_BLUE = RGBColor(46, 116, 181)
BODY = RGBColor(38, 50, 56)
MUTED = RGBColor(89, 89, 89)


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_border(cell, color="D9E2F3", sz="6"):
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right"):
        node = borders.find(qn(f"w:{edge}"))
        if node is None:
            node = OxmlElement(f"w:{edge}")
            borders.append(node)
        node.set(qn("w:val"), "single")
        node.set(qn("w:sz"), sz)
        node.set(qn("w:space"), "0")
        node.set(qn("w:color"), color)


def set_cell_margins(cell, top=90, start=130, bottom=90, end=130):
    tc_pr = cell._tc.get_or_add_tcPr()
    margins = tc_pr.first_child_found_in("w:tcMar")
    if margins is None:
        margins = OxmlElement("w:tcMar")
        tc_pr.append(margins)
    for name, value in [("top", top), ("start", start), ("bottom", bottom), ("end", end)]:
        node = margins.find(qn(f"w:{name}"))
        if node is None:
            node = OxmlElement(f"w:{name}")
            margins.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_run_style(run, size=8.2, color=BODY, bold=False, italic=False):
    run.font.name = "Calibri"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.bold = bold
    run.italic = italic


def set_paragraph_text(paragraph, text, size=10.2, color=BODY, bold=False, italic=False):
    paragraph.text = ""
    paragraph.paragraph_format.space_after = Pt(4)
    run = paragraph.add_run(text)
    set_run_style(run, size=size, color=color, bold=bold, italic=italic)


def set_mixed_prefix(paragraph, prefix, body):
    paragraph.text = ""
    paragraph.paragraph_format.space_after = Pt(4)
    r1 = paragraph.add_run(prefix)
    set_run_style(r1, size=10.2, color=BLUE, bold=True)
    r2 = paragraph.add_run(body)
    set_run_style(r2, size=10.2, color=BODY)


def fill_cell(cell, text, header=False, size=8.1):
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run(text)
    set_run_style(run, size=size, color=BLUE if header else BODY, bold=header)
    set_cell_border(cell)
    set_cell_margins(cell)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    if header:
        set_cell_shading(cell, "F2F4F7")


def style_existing_table(table, widths=None, header_fill="F2F4F7", font_size=8.1):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for ri, row in enumerate(table.rows):
        for ci, cell in enumerate(row.cells):
            set_cell_border(cell)
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if ri == 0:
                set_cell_shading(cell, header_fill)
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_after = Pt(0)
                for run in paragraph.runs:
                    set_run_style(
                        run,
                        size=font_size,
                        color=BLUE if ri == 0 else BODY,
                        bold=ri == 0,
                    )
    if widths:
        for row in table.rows:
            for i, width in enumerate(widths):
                if i < len(row.cells):
                    row.cells[i].width = Inches(width)


def clear_cell_content(cell):
    # Keeps cell properties but removes all paragraphs and nested tables.
    cell._tc.clear_content()
    cell.add_paragraph()


def add_caption(cell, text):
    p = cell.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    set_run_style(run, size=9.5, color=BLUE, bold=True)


def add_note(cell, text):
    p = cell.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_after = Pt(5)
    run = p.add_run(text)
    set_run_style(run, size=8.2, color=MUTED, italic=True)


def rebuild_placeholder1(cell):
    clear_cell_content(cell)
    # Remove the default empty paragraph after clear.
    for p in list(cell.paragraphs):
        p._element.getparent().remove(p._element)

    add_caption(cell, "Bảng 1. Các thay đổi so với thiết lập paper gốc")
    t1 = cell.add_table(rows=1, cols=4)
    headers = ["Thành phần", "Thiết lập paper gốc", "Thiết lập hiện tại", "Ảnh hưởng / ghi chú"]
    for i, header in enumerate(headers):
        fill_cell(t1.rows[0].cells[i], header, header=True, size=8.2)
    rows = [
        [
            "State representation / reward model",
            "Paper sử dụng reward model lớn để tạo state representation và hỗ trợ tín hiệu reward.",
            "Bản hiện tại dùng embedding API hoặc reward model nhỏ hơn để giảm yêu cầu GPU.",
            "Dễ chạy hơn trong môi trường local/API, nhưng không hoàn toàn tương đương thiết lập gốc.",
        ],
        [
            "Agent pool và LLM",
            "Agent pool theo cấu hình Mimas/Titan trong paper.",
            "Agent pool đã thay đổi LLM; một số role chuyển sang Gemini/Gemma hoặc được rút gọn để giảm quota/cost.",
            "Action space và capability distribution khác paper, nên cần xem đây là reproduce có điều chỉnh.",
        ],
        [
            "Quy mô train",
            "Paper train/evaluate trên quy mô benchmark lớn hơn.",
            "Hiện mới train 38 sample GSM-Hard.",
            "Kết quả hiện là sanity check cho pipeline, chưa đủ để kết luận generalization hoặc reproduce đầy đủ.",
        ],
    ]
    for row in rows:
        cells = t1.add_row().cells
        for i, text in enumerate(row):
            fill_cell(cells[i], text, size=7.9)
    style_existing_table(t1, widths=[1.3, 1.7, 1.8, 1.7], font_size=7.9)

    add_note(cell, "Ghi chú: các thay đổi này giúp pipeline chạy được trong điều kiện hiện tại, nhưng cần được báo cáo rõ khi so sánh với kết quả paper.")

    add_caption(cell, "Bảng 2. Kết quả reproduce paper trên GSM-Hard")
    t2 = cell.add_table(rows=1, cols=7)
    headers = ["Setting", "Policy mode", "#Train", "#Eval", "Accuracy", "Avg token cost", "Ghi chú"]
    for i, header in enumerate(headers):
        fill_cell(t2.rows[0].cells[i], header, header=True, size=7.8)
    rows = [
        ["Paper - Puppeteer Mono", "Initialized", "—", "—", "0.2467", "—", "Theo bảng paper; token cost cần bổ sung nếu có."],
        ["Paper - Puppeteer Mono", "Evolved", "—", "—", "0.4800", "—", "Theo bảng paper."],
        ["Paper - Puppeteer", "Initialized", "—", "—", "0.5600", "—", "Theo bảng paper."],
        ["Paper - Puppeteer", "Evolved", "—", "—", "0.5400", "—", "Theo bảng paper."],
        ["Reproduce hiện tại", "Initialized", "0", "38", "[điền]", "[chưa kiểm tra]", "Baseline trước train trên cùng 38 sample."],
        ["Reproduce hiện tại", "Evolved/checkpoint", "38", "38", "[điền]", "[chưa kiểm tra]", "Đã thấy accuracy tăng; cần điền số cụ thể."],
        ["Reproduce mở rộng", "Evolved/checkpoint", "38+", "500 / 1000 / full", "[điền sau]", "[điền sau]", "Sau khi có A100 hoặc quota đủ."],
    ]
    for row in rows:
        cells = t2.add_row().cells
        for i, text in enumerate(row):
            fill_cell(cells[i], text, size=7.4)
    style_existing_table(t2, widths=[1.15, 0.85, 0.55, 0.75, 0.65, 0.8, 1.75], font_size=7.4)


def iter_all_cells(doc):
    def iter_table(table):
        for row in table.rows:
            for cell in row.cells:
                yield cell
                for nested in cell.tables:
                    yield from iter_table(nested)

    for table in doc.tables:
        yield from iter_table(table)


def polish_text(doc):
    replacements = {
        "Tiến độ hiện tại đã chứng minh pipeline chạy được và có tín hiệu học trên GSM-Hard ở quy mô nhỏ. Tuy nhiên, để phát triển đóng góp nghiên cứu rõ hơn, bước tiếp theo sẽ chuyển từ reproduce accuracy sang phân tích generalization khi agent pool thay đổi. ": "Tiến độ hiện tại cho thấy pipeline đã chạy được và có tín hiệu học trên GSM-Hard ở quy mô nhỏ. Tuy nhiên, để tạo đóng góp nghiên cứu rõ hơn, bước tiếp theo nên chuyển trọng tâm từ reproduce accuracy sang phân tích generalization khi agent pool thay đổi.",
        "Tiến độ hiện tại đã chứng minh pipeline chạy được và có tín hiệu học trên GSM-Hard ở quy mô nhỏ. Tuy nhiên, để phát triển đóng góp nghiên cứu rõ hơn, bước tiếp theo sẽ chuyển từ reproduce accuracy sang phân tích generalization khi agent pool thay đổi.": "Tiến độ hiện tại cho thấy pipeline đã chạy được và có tín hiệu học trên GSM-Hard ở quy mô nhỏ. Tuy nhiên, để tạo đóng góp nghiên cứu rõ hơn, bước tiếp theo nên chuyển trọng tâm từ reproduce accuracy sang phân tích generalization khi agent pool thay đổi.",
    }
    for p in doc.paragraphs:
        text = p.text
        if text in replacements:
            set_paragraph_text(p, replacements[text])

    # Ensure the issue paragraphs keep bold labels after previous rewrite.
    issue_prefixes = [
        "Agent pool cố định và MLP gắn với thứ tự personas: ",
        "Không dừng sớm khi non-terminator đã có final answer: ",
        "Agent chưa nhận thức rõ mình đang ở trong MAS: ",
        "Role specialization chưa rõ ràng: ",
    ]
    for p in doc.paragraphs:
        for prefix in issue_prefixes:
            if p.text.startswith(prefix):
                body = p.text[len(prefix) :]
                set_mixed_prefix(p, prefix, body)


def main():
    doc = Document(DOCX_PATH)
    polish_text(doc)

    for cell in iter_all_cells(doc):
        if "Bảng 1. Các thay đổi so với thiết lập paper gốc" in cell.text:
            rebuild_placeholder1(cell)
            break

    # Style top-level solution table and any remaining nested tables.
    for table in doc.tables:
        if len(table.rows) > 1 and len(table.columns) == 3 and "Hướng cải thiện" in table.rows[0].cells[0].text:
            style_existing_table(table, widths=[1.45, 3.15, 1.9], font_size=8.2)
        elif len(table.rows) == 1 and len(table.columns) == 1:
            cell = table.cell(0, 0)
            set_cell_border(cell, color="A9BCD6" if "Tóm tắt ngắn" in cell.text else "B8C2D1", sz="8")
            set_cell_margins(cell, top=110, bottom=110, start=150, end=150)

    # Also style nested tables in Placeholder 1 after rebuilding.
    for cell in iter_all_cells(doc):
        for nested in cell.tables:
            if nested.rows and nested.rows[0].cells:
                style_existing_table(nested, font_size=7.8)

    doc.save(DOCX_PATH)
    print(DOCX_PATH)


if __name__ == "__main__":
    main()
