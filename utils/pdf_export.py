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
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

load_dotenv()

FONT_PATH = os.path.join(os.path.dirname(__file__), "NotoSans-Regular.ttf")

try:
    pdfmetrics.registerFont(TTFont("NotoSans", FONT_PATH))
    ACTIVE_FONT = "NotoSans"
except Exception:
    ACTIVE_FONT = "Helvetica"


def export_pdf_from_vectors(session_id: str, user_id: str, start_page: int, end_page: int, output_path: str):
    db = get_encrypted_notes_connection(user_id)

    # Fetch page range chunks from MongoDB
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
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        "DocTitle", parent=styles["Normal"], fontName=ACTIVE_FONT, fontSize=18, leading=22,
        textColor=colors.HexColor("#2C3E50"), spaceAfter=15, alignment=1,
    )

    page_header_style = ParagraphStyle(
        "PageHeader", parent=styles["Normal"], fontName=ACTIVE_FONT, fontSize=12, leading=16,
        textColor=colors.HexColor("#D35400"), spaceAfter=6, keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "ItemBody", parent=styles["Normal"], fontName=ACTIVE_FONT, fontSize=10, leading=14,
        textColor=colors.HexColor("#34495E"), spaceAfter=10,
    )

    story = []
    story.append(Paragraph(f"<b>Exported Segment (Pages {start_page} - {end_page})</b>", title_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#BDC3C7"), spaceAfter=15))

    current_p_num = None

    for item in records:
        p_num = item.get("page_number")
        text_content = item.get("content", "")

        if p_num != current_p_num:
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

    document.build(story)
    return output_path