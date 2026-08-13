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
MUTED = RGBColor(89, 89, 89)
BODY = RGBColor(38, 50, 56)


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


def remove_children(parent):
    for child in list(parent):
        parent.remove(child)


def clear_cell(cell):
    remove_children(cell._tc)
    cell._tc.append(OxmlElement("w:tcPr"))
    cell.add_paragraph()


def set_text(paragraph, text, bold=False, size=10.2, color=BODY, italic=False):
    paragraph.text = ""
    run = paragraph.add_run(text)
    run.bold = bold
    run.italic = italic
    run.font.name = "Calibri"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    run.font.size = Pt(size)
    run.font.color.rgb = color
    return run


def replace_paragraph_text(paragraph, text, bold_prefix=None):
    paragraph.text = ""
    paragraph.paragraph_format.space_after = Pt(4)
    if bold_prefix and text.startswith(bold_prefix):
        run = paragraph.add_run(bold_prefix)
        run.bold = True
        run.font.color.rgb = BLUE
        run.font.size = Pt(10.2)
        rest = paragraph.add_run(text[len(bold_prefix):])
        rest.font.size = Pt(10.2)
        rest.font.color.rgb = BODY
    else:
        run = paragraph.add_run(text)
        run.font.size = Pt(10.2)
        run.font.color.rgb = BODY


def add_bullet_after(paragraph, text):
    p = paragraph.insert_paragraph_before(text, style="List Bullet")
    return p


def add_caption(cell, text):
    p = cell.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(4)
    set_text(p, text, bold=True, size=9.5, color=BLUE)


def fill_cell(cell, text, bold=False, fill=None, size=8.2, color=BODY):
    if fill:
        set_cell_shading(cell, fill)
    set_cell_margins(cell)
    set_cell_border(cell)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    set_text(p, text, bold=bold, size=size, color=color)


def style_table(table, widths=None):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for row_idx, row in enumerate(table.rows):
        for cell in row.cells:
            set_cell_border(cell)
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_after = Pt(0)
                for run in paragraph.runs:
                    run.font.name = "Calibri"
                    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
                    run.font.size = Pt(8.2)
                    if row_idx == 0:
                        run.bold = True
                        run.font.color.rgb = BLUE
            if row_idx == 0:
                set_cell_shading(cell, "F2F4F7")
    if widths:
        for row in table.rows:
            for idx, width in enumerate(widths):
                row.cells[idx].width = Inches(width)


def rebuild_placeholder1_container(cell):
    clear_cell(cell)
    # Remove the default empty paragraph created by python-docx after clearing.
    for paragraph in list(cell.paragraphs):
        paragraph._element.getparent().remove(paragraph._element)

    add_caption(cell, "Bảng 1. Các thay đổi so với thiết lập paper gốc")
    t1 = cell.add_table(rows=1, cols=4)
    for i, header in enumerate(["Thành phần", "Paper gốc", "Thiết lập hiện tại", "Ảnh hưởng / ghi chú"]):
        fill_cell(t1.rows[0].cells[i], header, bold=True, fill="F2F4F7", size=8.3, color=BLUE)
    rows = [
        [
            "State representation / reward model",
            "Dùng reward model lớn để biểu diễn state và hỗ trợ tín hiệu reward.",
            "Dùng embedding API hoặc reward model nhỏ hơn để giảm yêu cầu GPU.",
            "Thuận tiện cho local/API run, nhưng chưa hoàn toàn tương đương thiết lập gốc.",
        ],
        [
            "Agent pool và LLM",
            "Agent pool theo cấu hình Mimas/Titan trong paper.",
            "Agent pool đã thay đổi; một số role/model chuyển sang Gemini/Gemma hoặc được rút gọn để giảm quota/cost.",
            "Action space và phân bố capability khác paper, nên cần ghi rõ đây là reproduce có điều chỉnh.",
        ],
        [
            "Quy mô train",
            "Paper train/evaluate trên quy mô benchmark lớn hơn.",
            "Hiện mới train 38 sample GSM-Hard.",
            "Kết quả hiện là sanity check cho pipeline, chưa đủ để kết luận generalization.",
        ],
    ]
    for row in rows:
        cells = t1.add_row().cells
        for i, text in enumerate(row):
            fill_cell(cells[i], text)
    style_table(t1, widths=[1.25, 1.65, 1.85, 1.75])

    spacer = cell.add_paragraph()
    spacer.paragraph_format.space_after = Pt(5)

    add_caption(cell, "Bảng 2. So sánh personas / agent pool")
    t2 = cell.add_table(rows=1, cols=5)
    for i, header in enumerate(["Nhóm agent", "Paper gốc", "Hiện tại", "Thay đổi chính", "Tác động"]):
        fill_cell(t2.rows[0].cells[i], header, bold=True, fill="F2F4F7", size=8.1, color=BLUE)
    rows = [
        [
            "Tool agents",
            "Có các agent như File/Bing/Website/Arxiv tùy cấu hình.",
            "Rút gọn trong personas_gsm_local để giảm tool call và quota.",
            "Giảm agent không cần thiết cho GSM-Hard.",
            "Có thể giảm cost nhưng cũng làm khác agent pool gốc.",
        ],
        [
            "Reasoning agents",
            "Nhiều role reasoning/reflection/summarization/conclusion.",
            "Giữ các role cốt lõi: Planner, Reasoning, Reflect, Summarizer, Concluder, Python, Terminator.",
            "Tập trung vào giải toán GSM-Hard.",
            "Dễ chạy hơn, nhưng capability distribution đã thay đổi.",
        ],
        [
            "Base LLM",
            "Theo model trong paper/bảng Mimas hoặc Titan.",
            "Dùng Gemini/Gemma qua API cho một số role.",
            "Thay đổi năng lực và cost từng agent.",
            "Kết quả cần được xem là reproduce có điều kiện.",
        ],
    ]
    for row in rows:
        cells = t2.add_row().cells
        for i, text in enumerate(row):
            fill_cell(cells[i], text, size=7.9)
    style_table(t2, widths=[1.05, 1.35, 1.55, 1.25, 1.3])

    spacer = cell.add_paragraph()
    spacer.paragraph_format.space_after = Pt(5)

    add_caption(cell, "Bảng 3. Kết quả reproduce paper trên GSM-Hard")
    t3 = cell.add_table(rows=1, cols=7)
    headers = ["Setting", "Policy mode", "#Train", "#Eval", "Accuracy", "Avg token cost", "Ghi chú"]
    for i, header in enumerate(headers):
        fill_cell(t3.rows[0].cells[i], header, bold=True, fill="F2F4F7", size=7.9, color=BLUE)
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
        cells = t3.add_row().cells
        for i, text in enumerate(row):
            fill_cell(cells[i], text, size=7.6)
    style_table(t3, widths=[1.2, 0.9, 0.55, 0.75, 0.65, 0.8, 1.65])


def polish_doc():
    if not DOCX_PATH.exists():
        raise FileNotFoundError(DOCX_PATH)

    doc = Document(DOCX_PATH)

    # Style tuning
    section = doc.sections[0]
    section.top_margin = Inches(0.7)
    section.bottom_margin = Inches(0.7)
    section.left_margin = Inches(0.85)
    section.right_margin = Inches(0.85)

    styles = doc.styles
    styles["Normal"].font.name = "Calibri"
    styles["Normal"]._element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    styles["Normal"].font.size = Pt(10.2)
    styles["Normal"].paragraph_format.space_after = Pt(4)
    styles["Normal"].paragraph_format.line_spacing = 1.06
    for name, size, before, after in [
        ("Heading 1", 15, 10, 5),
        ("Heading 2", 12.5, 7, 3),
    ]:
        style = styles[name]
        style.font.name = "Calibri"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
        style.font.size = Pt(size)
        style.font.color.rgb = HEADING_BLUE
        style.font.bold = True
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)

    # Paragraph rewrite by exact old text.
    replacements = {
        "Đã cấu hình và chạy được pipeline trên GSM-Hard với chế độ initialized/evolved theo checkpoint policy.": "Đã cấu hình và chạy ổn định pipeline Puppeteer trên GSM-Hard, bao gồm cả chế độ initialized và evolved/checkpoint policy.",
        "Đã train được 38 sample GSM-Hard. Đây là bước xác nhận training loop, logging, checkpoint và embedding-based state representation hoạt động được trong môi trường hiện tại.": "Đã train thử nghiệm 38 sample GSM-Hard, qua đó xác nhận training loop, logging, checkpoint và embedding-based state representation có thể hoạt động trong môi trường hiện tại.",
        "Đã test lại trên 38 sample đã train và ghi nhận accuracy tăng so với trước khi train. Token cost chưa được kiểm tra đầy đủ, nên kết luận hiện tại mới dừng ở hiệu quả accuracy nội bộ.": "Đã test lại trên chính 38 sample đã train và ghi nhận accuracy tăng so với baseline trước khi train. Token cost vẫn chưa được thống kê đầy đủ, vì vậy kết luận hiện tại mới phản ánh tín hiệu accuracy ở quy mô nhỏ.",
        "Cần đăng ký dùng A100 của trường để mở rộng số sample, giảm thời gian chạy và có điều kiện reproduce bảng kết quả lớn hơn.": "Đang xin đăng ký sử dụng A100 của trường để mở rộng số sample, rút ngắn thời gian chạy và tiến tới reproduce bảng kết quả ở quy mô lớn hơn.",
        "Agent pool cố định và MLP gắn với thứ tự personas: Output neuron thứ i của MLP được map trực tiếp sang agent thứ i trong file personas. Nếu thêm/xóa agent thì output_dim mismatch; nếu đổi thứ tự nhưng giữ số agent, checkpoint có thể vẫn load được nhưng semantic mapping bị sai. Điều này làm policy khó generalize khi thay đổi team composition.": "Agent pool cố định và MLP gắn với thứ tự personas: Trong implementation hiện tại, output neuron thứ i của MLP được ánh xạ trực tiếp sang agent thứ i trong file personas. Khi thêm hoặc xóa agent, output_dim sẽ mismatch; còn nếu chỉ đổi thứ tự personas nhưng giữ nguyên số agent, checkpoint có thể vẫn chạy nhưng semantic mapping đã bị lệch. Đây là điểm khiến policy khó generalize khi team composition thay đổi.",
        "Không dừng sớm khi non-terminator đã có final answer: Một agent thường có thể sinh FINAL ANSWER, nhưng path chỉ dừng khi policy chọn TerminatorAgent hoặc đạt max_step_num. Điều này gây tốn API/token và có thể làm trajectory dài hơn cần thiết, đặc biệt với GSM-Hard.": "Không dừng sớm khi non-terminator đã có final answer: Nhiều agent không phải Terminator vẫn có thể sinh FINAL ANSWER, nhưng path chỉ thật sự dừng khi policy chọn TerminatorAgent hoặc khi đạt max_step_num. Vì vậy hệ thống có thể tiếp tục gọi thêm agent dù đã có đáp án hợp lệ, làm tăng API/token cost và khiến trajectory dài hơn cần thiết, đặc biệt trên GSM-Hard.",
        "Agent không biết rõ mình đang ở trong MAS: Prompt của từng agent chủ yếu chứa role của chính nó, câu hỏi và kết quả trước đó. Agent gần như không biết agent pool có những role nào, không biết ranh giới nhiệm vụ của mình trong hệ phối hợp, nên dễ làm thay vai trò của agent khác.": "Agent chưa nhận thức rõ mình đang ở trong MAS: Prompt của từng agent chủ yếu mô tả role của chính agent, câu hỏi hiện tại và kết quả trước đó. Agent gần như không biết đầy đủ agent pool gồm những role nào, cũng chưa được nhấn mạnh ranh giới nhiệm vụ của mình trong hệ phối hợp, nên dễ lấn sang nhiệm vụ của agent khác.",
        "Role specialization chưa sạch: Nhiều prompt yêu cầu cả Planner/Reasoner/Reflect/Summarizer đều kết thúc bằng FINAL ANSWER. Điều này giúp benchmark có đáp án nhưng làm mờ ý nghĩa phối hợp: agent không chỉ tạo intermediate result mà thường tự solve luôn bài toán.": "Role specialization chưa rõ ràng: Nhiều prompt yêu cầu Planner, Reasoner, Reflect hoặc Summarizer đều kết thúc bằng FINAL ANSWER. Cách thiết kế này giúp benchmark dễ thu được đáp án, nhưng làm mờ vai trò phối hợp: thay vì chỉ tạo intermediate result theo chuyên môn, nhiều agent có xu hướng tự giải trọn bài toán.",
        "Mahajan et al. (Generalization in Cooperative Multi-Agent Systems) xem generalization trong cooperative MAS như khả năng thích nghi khi capability, số lượng agent, hoặc team composition thay đổi. Từ góc nhìn này, Puppeteer của Dang et al. có ưu điểm là điều phối agent động theo state, nhưng vẫn chưa phải một MAS có combinatorial generalization mạnh vì agent pool và action space của policy vẫn cố định.": "Mahajan et al. trong Generalization in Cooperative Multi-Agent Systems xem generalization của cooperative MAS là khả năng thích nghi khi capability, số lượng agent hoặc team composition thay đổi. Nhìn từ khung này, Puppeteer của Dang et al. có ưu điểm ở điều phối động theo state, nhưng vẫn chưa thể xem là một MAS có combinatorial generalization mạnh vì agent pool và action space của policy còn cố định.",
        "Các yêu cầu baseline hiện còn thiếu gồm: capability grounding cho từng agent; policy phụ thuộc vào team composition; kiến trúc không gắn cứng với số lượng/thứ tự agent; capability inference từ lịch sử chạy (ochestrator cần hiểu năng lực của từng agent để lựa chọn thay vì reinforcement learning theo cost và đáp án)": "Các yêu cầu baseline hiện còn thiếu gồm: capability grounding cho từng agent; policy có điều kiện theo team composition; kiến trúc không gắn cứng với số lượng hoặc thứ tự agent; và capability inference từ lịch sử chạy, tức orchestrator cần học được năng lực thực tế của từng agent thay vì chỉ tối ưu gián tiếp qua reward, cost và final answer.",
        "Các phương pháp cross-task generalization và combinatorial generalization hiện tại chủ yếu dùng game starcraft và các game khác như FRUIT FORAGE, PREDATOR PREY, chưa thấy có phương pháp test trên các dataset như bài của dang et al": "Các benchmark cross-task generalization và combinatorial generalization hiện nay chủ yếu dựa trên môi trường game như StarCraft, Fruit Forage hoặc Predator Prey. Hiện chưa thấy nhiều protocol đánh giá trực tiếp trên các dataset dạng reasoning benchmark giống Dang et al.; đây có thể là khoảng trống để thiết kế benchmark generalization riêng cho Puppeteer.",
        "Tiến độ hiện tại đã chứng minh pipeline chạy được và có tín hiệu học trên GSM-Hard ở quy mô nhỏ. Tuy nhiên, để phát triển đóng góp nghiên cứu rõ hơn, bước tiếp theo sẽ chuyển từ reproduce accuracy sang phân tích generalization khi agent pool thay đổi. ": "Tiến độ hiện tại cho thấy pipeline đã chạy được và có tín hiệu học trên GSM-Hard ở quy mô nhỏ. Tuy nhiên, để tạo đóng góp nghiên cứu rõ hơn, bước tiếp theo nên chuyển trọng tâm từ reproduce accuracy sang phân tích generalization khi agent pool thay đổi.",
    }

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text in replacements:
            if ":" in replacements[text] and text.startswith(("Agent", "Không", "Role")):
                prefix = replacements[text].split(":", 1)[0] + ": "
                replace_paragraph_text(paragraph, replacements[text], bold_prefix=prefix)
            else:
                replace_paragraph_text(paragraph, replacements[text])

    # Normalize heading capitalization.
    for paragraph in doc.paragraphs:
        if paragraph.text.strip() == "5. benchmark":
            replace_paragraph_text(paragraph, "5. Benchmark đề xuất")
            paragraph.style = doc.styles["Heading 1"]
        elif paragraph.text.strip() == "3. Combinatorial Generalization":
            replace_paragraph_text(paragraph, "3. Combinatorial Generalization")
            paragraph.style = doc.styles["Heading 1"]

    # Rebuild placeholder/table container if it contains the first caption.
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if "Bảng 1. Các thay đổi so với thiết lập paper gốc" in cell.text and "Bảng 3. Kết quả reproduce" in cell.text:
                    rebuild_placeholder1_container(cell)

    # Beautify solution table.
    for table in doc.tables:
        if table.rows and table.rows[0].cells and "Hướng cải thiện" in table.rows[0].cells[0].text:
            style_table(table, widths=[1.45, 3.15, 1.9])

    # Beautify single-cell image/callout tables without touching image run.
    for table in doc.tables:
        if len(table.rows) == 1 and len(table.columns) == 1:
            cell = table.cell(0, 0)
            set_cell_border(cell, color="A9BCD6" if "Tóm tắt ngắn" in cell.text else "B8C2D1", sz="8")
            set_cell_margins(cell, top=110, bottom=110, start=150, end=150)

    doc.save(DOCX_PATH)
    print(DOCX_PATH)


if __name__ == "__main__":
    polish_doc()
