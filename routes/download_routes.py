import os
import json
import tempfile
import traceback
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from utils.pdf_export import generate_evaluation_report_pdf

router = APIRouter()

@router.post("/download")
async def download_evaluation_report(request: Request):
    try:
        raw_body = await request.body()
        raw_text = raw_body.decode("utf-8", errors="ignore")
        
        report_text = ""
        include_pages_flag = True

        if raw_text.strip():
            try:
                data = json.loads(raw_text)
                report_text = data.get("answer") or data.get("content") or data.get("text") or ""
                if "include_pages" in data:
                    include_pages_flag = bool(data["include_pages"])
            except Exception:
                report_text = raw_text

        if not report_text.strip() or report_text == "No context data available":
            report_text = "Study Buddy Evaluation Report\n\nNo evaluation content found."

        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_pdf.close()

        pdf_path = generate_evaluation_report_pdf(
            content_text=report_text,
            output_path=temp_pdf.name,
            include_page_numbers=include_pages_flag
        )

        return FileResponse(
            path=pdf_path,
            media_type="application/pdf",
            filename="Study_Buddy_Solution.pdf"
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"PDF Download Error: {str(e)}")