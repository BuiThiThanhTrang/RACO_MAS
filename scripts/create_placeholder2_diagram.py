from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


OUT = Path(r"D:\SideProject\ChatDev\output\placeholder_2_so_do_diem_yeu_puppeteer.png")


def get_font(size, bold=False):
    candidates = []
    if bold:
        candidates.extend(
            [
                r"C:\Windows\Fonts\arialbd.ttf",
                r"C:\Windows\Fonts\calibrib.ttf",
                r"C:\Windows\Fonts\segoeuib.ttf",
            ]
        )
    candidates.extend(
        [
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\calibri.ttf",
            r"C:\Windows\Fonts\segoeui.ttf",
        ]
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


W, H = 2400, 1350
COL = {
    "dark": "#0B2545",
    "blue": "#2E74B5",
    "lightblue": "#E8EEF5",
    "green": "#EAF4EC",
    "green_b": "#2E7D32",
    "red": "#FCE8E6",
    "red_b": "#B3261E",
    "amber": "#FFF4D6",
    "amber_b": "#A06A00",
    "gray_b": "#B8C2D1",
    "text": "#263238",
    "muted": "#6B7280",
    "white": "#FFFFFF",
}


img = Image.new("RGB", (W, H), "#F7F9FC")
d = ImageDraw.Draw(img)

F_TITLE = get_font(46, True)
F_SUB = get_font(28)
F_BOX_TITLE = get_font(26, True)
F_BOX = get_font(22)
F_SMALL = get_font(20)
F_MONO = get_font(21)
F_FIX = get_font(24, True)


def text_width(text, font):
    box = d.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def text_height(text, font):
    box = d.textbbox((0, 0), text, font=font)
    return box[3] - box[1]


def wrap_text(text, font, max_width):
    words = text.split()
    lines = []
    current = ""
    for word in words:
        test = (current + " " + word).strip()
        if text_width(test, font) <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def multiline_center(text, x, y, max_width, font, fill, line_gap=4):
    lines = []
    for part in text.split("\n"):
        lines.extend(wrap_text(part, font, max_width) if part.strip() else [""])
    total_height = sum(text_height(line, font) for line in lines) + line_gap * (len(lines) - 1)
    cy = y - total_height / 2
    for line in lines:
        tw = text_width(line, font)
        d.text((x - tw / 2, cy), line, font=font, fill=fill)
        cy += text_height(line, font) + line_gap


def center_text(text, y, font, fill):
    d.text(((W - text_width(text, font)) / 2, y), text, font=font, fill=fill)


def round_box(x, y, w, h, title, body="", fc="#FFFFFF", ec="#B8C2D1", title_col="#0B2545", lw=4):
    d.rounded_rectangle([x + 5, y + 5, x + w + 5, y + h + 5], radius=24, fill="#DDE4EE")
    d.rounded_rectangle([x, y, x + w, y + h], radius=24, fill=fc, outline=ec, width=lw)
    multiline_center(title, x + w / 2, y + 40, w - 45, F_BOX_TITLE, title_col, line_gap=2)
    if body:
        multiline_center(body, x + w / 2, y + h / 2 + 18, w - 55, F_BOX, COL["text"], line_gap=8)


def arrow(x1, y1, x2, y2, color="#52677A", width=5):
    import math

    d.line([x1, y1, x2, y2], fill=color, width=width)
    angle = math.atan2(y2 - y1, x2 - x1)
    length = 22
    a1 = angle + math.pi * 0.82
    a2 = angle - math.pi * 0.82
    points = [
        (x2, y2),
        (x2 + length * math.cos(a1), y2 + length * math.sin(a1)),
        (x2 + length * math.cos(a2), y2 + length * math.sin(a2)),
    ]
    d.polygon(points, fill=color)


def build():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    center_text("Placeholder 2 - Sơ đồ các điểm yếu trong orchestration hiện tại", 60, F_TITLE, COL["dark"])
    center_text(
        "Puppeteer chọn agent động, nhưng policy/action space vẫn gắn với agent pool cố định và workflow chưa có early-stop tốt.",
        130,
        F_SUB,
        "#4B5563",
    )

    y = 305
    boxes = [
        (80, y, 310, 150, "Task + history", "Question\n+ previous outputs", COL["white"], COL["gray_b"]),
        (465, y, 330, 150, "State context", "dialog_history\nđược gom thành text", COL["lightblue"], COL["blue"]),
        (875, y, 310, 150, "Embedding", "API embedding /\nreward-model state", COL["white"], COL["blue"]),
        (1260, y, 310, 150, "MLP policy", "Softmax trên\nfixed output slots", COL["amber"], COL["amber_b"]),
        (1645, y, 285, 150, "Selected agent", "slot index →\nagent hash/role", COL["white"], COL["blue"]),
        (2010, y, 250, 150, "Next step", "agent chạy\nvà update path", COL["green"], COL["green_b"]),
    ]
    for box in boxes:
        round_box(*box)
    for left, right in zip(boxes, boxes[1:]):
        x, yy, w, h = left[:4]
        nx, ny, _, nh = right[:4]
        arrow(x + w, yy + h / 2, nx, ny + nh / 2)

    sx, sy, sw, sh = 1230, 575, 430, 360
    round_box(sx, sy, sw, sh, "MLP output slots", "", COL["white"], COL["amber_b"])
    slots = [
        ("output[0]", "TerminatorAgent"),
        ("output[1]", "PythonAgent"),
        ("output[2]", "PlannerAgent"),
        ("output[3]", "ReasoningAgent"),
        ("...", "..."),
    ]
    for i, (left, right) in enumerate(slots):
        yy = sy + 112 + i * 45
        d.text((sx + 55, yy), left, font=F_MONO, fill="#374151")
        d.text((sx + 190, yy), "→", font=F_MONO, fill=COL["muted"])
        d.text((sx + 230, yy), right, font=F_SMALL, fill="#111827")
    arrow(1415, y + 150, 1440, sy, COL["amber_b"], 4)

    round_box(
        95,
        610,
        560,
        250,
        "Vấn đề 1: Agent không MAS-aware",
        "System prompt chủ yếu chứa role của chính agent + task + kết quả trước. Agent không thấy rõ agent pool, nên dễ làm thay role khác.",
        COL["red"],
        COL["red_b"],
        COL["red_b"],
    )
    arrow(315, y + 150, 360, 610, COL["red_b"], 4)

    round_box(
        720,
        940,
        620,
        230,
        "Vấn đề 2: Fixed slots phụ thuộc personas",
        "Thêm/xóa agent → output_dim mismatch. Đổi thứ tự personas nhưng số agent giữ nguyên → có thể không crash, nhưng semantic mapping của checkpoint bị sai.",
        COL["red"],
        COL["red_b"],
        COL["red_b"],
    )
    arrow(sx + 170, sy + sh, 1020, 940, COL["red_b"], 4)

    round_box(
        1510,
        940,
        620,
        230,
        "Vấn đề 3: Không early-stop khi có answer",
        "Non-terminator có thể sinh FINAL ANSWER, nhưng path vẫn tiếp tục đến khi chọn TerminatorAgent hoặc chạm max_step_num. Hệ quả: tăng token/API cost.",
        COL["red"],
        COL["red_b"],
        COL["red_b"],
    )
    arrow(2140, y + 150, 1840, 940, COL["red_b"], 4)

    d.rounded_rectangle([95, 1220, 2305, 1290], radius=22, fill=COL["green"], outline=COL["green_b"], width=4)
    fix = (
        "Hướng sửa: capability-aware scorer f(state, agent_embedding) + permutation-invariant agent encoder "
        "+ answer gate/early-stop + MAS-aware prompts"
    )
    multiline_center(fix, W / 2, 1255, 2140, F_FIX, "#1B5E20", 4)

    d.rectangle([80, 240, 92, 252], fill=COL["blue"])
    d.text((105, 228), "Luồng hiện tại", font=F_SMALL, fill=COL["blue"])
    d.text((100, 1180), "Gợi ý: dùng ảnh này thay cho Placeholder 2 trong báo cáo Word.", font=F_SMALL, fill=COL["muted"])

    img.save(OUT, quality=95)
    print(OUT)


if __name__ == "__main__":
    build()
