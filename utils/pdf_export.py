import os
import re
import html
from dotenv import load_dotenv
from dependencies import get_encrypted_notes_connection

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

load_dotenv()

FONT_PATH = os.path.join(os.path.dirname(__file__), "NotoSans-Regular.ttf")

try:
    pdfmetrics.registerFont(TTFont("NotoSans", FONT_PATH))
    ACTIVE_FONT = "NotoSans"
except Exception:
    ACTIVE_FONT = "Helvetica"


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont(ACTIVE_FONT, 9)
        self.setFillColor(colors.HexColor("#7F8C8D"))
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(A4[0] - 40, 25, page_text)
        self.restoreState()


def generate_evaluation_report_pdf(content_text: str, output_path: str, include_page_numbers: bool = True):
    document = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=45 if include_page_numbers else 35,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Normal"], fontName=ACTIVE_FONT, fontSize=18, leading=22,
        textColor=colors.HexColor("#1A365D"), spaceAfter=10, alignment=1, fontStyle="bold"
    )

    question_style = ParagraphStyle(
        "QStyle", parent=styles["Normal"], fontName=ACTIVE_FONT, fontSize=11, leading=15,
        textColor=colors.HexColor("#0D9488"), spaceBefore=10, spaceAfter=4, keepWithNext=True
    )

    answer_style = ParagraphStyle(
        "AStyle", parent=styles["Normal"], fontName=ACTIVE_FONT, fontSize=10, leading=14,
        textColor=colors.HexColor("#1F2937"), spaceAfter=6
    )

    source_style = ParagraphStyle(
        "SrcStyle", parent=styles["Normal"], fontName=ACTIVE_FONT, fontSize=9, leading=12,
        textColor=colors.HexColor("#EA580C"), spaceAfter=10
    )

    body_style = ParagraphStyle(
        "GeneralBody", parent=styles["Normal"], fontName=ACTIVE_FONT, fontSize=10, leading=14,
        textColor=colors.HexColor("#374151"), spaceAfter=8
    )

    story = [
        Paragraph("<b>Evaluation Output Report</b>", title_style),
        HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0D9488"), spaceAfter=15)
    ]

    blocks = content_text.split("---")
    for block in blocks:
        block_clean = block.strip()
        if not block_clean:
            continue

        lines = block_clean.split("\n")
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            escaped = html.escape(re.sub(r"[*#_]", "", line_str).strip())

            if escaped.startswith("QUESTION:"):
                q_text = escaped.replace("QUESTION:", "").strip()
                story.append(Paragraph(f"<b>QUESTION:</b> {q_text}", question_style))
            elif escaped.startswith("ANSWER:"):
                a_text = escaped.replace("ANSWER:", "").strip()
                story.append(Paragraph(f"<b>ANSWER:</b> {a_text}", answer_style))
            elif re.search(r'^(source\s*pages?|pages?)\s*:', escaped, re.IGNORECASE):
                if include_page_numbers:
                    s_text = re.sub(r'^(source\s*pages?|pages?)\s*:', '', escaped, flags=re.IGNORECASE).strip()
                    story.append(Paragraph(f"<b>SOURCE PAGES:</b> {s_text}", source_style))
            else:
                story.append(Paragraph(escaped, body_style))

        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#E5E7EB"), spaceAfter=12))

    if include_page_numbers:
        document.build(story, canvasmaker=NumberedCanvas)
    else:
        document.build(story)

    return output_path


def export_pdf_from_vectors(session_id: str, user_id: str, start_page: int, end_page: int, output_path: str, include_page_numbers: bool = True):
    db = get_encrypted_notes_connection(user_id)

    cursor = db.pdf_pages.find({
        "session_id": session_id,
        "page_number": {"$gte": start_page, "$lte": end_page}
    }).sort("page_number", 1)

    records = list(cursor)

    if not records:
        records = [{"page_number": start_page, "content": f"No text content retrieved for pages {start_page} to {end_page}."}]

    document = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=45 if include_page_numbers else 35,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle", parent=styles["Normal"], fontName=ACTIVE_FONT, fontSize=16, leading=20,
        textColor=colors.HexColor("#2C3E50"), spaceAfter=12, alignment=1,
    )

    page_header_style = ParagraphStyle(
        "PageHeader", parent=styles["Normal"], fontName=ACTIVE_FONT, fontSize=11, leading=15,
        textColor=colors.HexColor("#D35400"), spaceAfter=6, keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "ItemBody", parent=styles["Normal"], fontName=ACTIVE_FONT, fontSize=10, leading=14,
        textColor=colors.HexColor("#34495E"), spaceAfter=8,
    )

    story = [
        Paragraph(f"<b>Exported Segment (Pages {start_page} - {end_page})</b>", title_style),
        HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#BDC3C7"), spaceAfter=15)
    ]

    current_p_num = None

    for item in records:
        p_num = item.get("page_number")
        text_content = item.get("content", "")

        if p_num != current_p_num:
            if include_page_numbers:
                story.append(Spacer(1, 8))
                story.append(Paragraph(f"<b>--- PAGE {p_num} ---</b>", page_header_style))
            current_p_num = p_num

        lines = text_content.split("\n")
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
            escaped = html.escape(re.sub(r"[*#_]", "", line_str).strip())
            story.append(Paragraph(escaped, body_style))

        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#ECF0F1"), spaceAfter=10))

    if include_page_numbers:
        document.build(story, canvasmaker=NumberedCanvas)
    else:
        document.build(story)

    return output_path