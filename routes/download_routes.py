import os
import tempfile
import traceback
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
from utils.pdf_export import export_pdf_from_vectors
from dependencies import verify_firebase_token

router = APIRouter()

class PDFRangeRequest(BaseModel):
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    current_page: Optional[int] = None  # Accepts ANY dynamic page number chosen by user (1, 20, 50, etc.)


@router.post("/export-range")
async def export_pdf_range(
    data: PDFRangeRequest,
    decoded_token: dict = Depends(verify_firebase_token)
):
    try:
        firebase_uid = decoded_token["uid"].strip().lower()
        session_id = f"session_{firebase_uid}"

        # 1. DYNAMIC PAGE LOGIC HANDLING
        # Mode A: User wants to download ONLY the active current page (e.g., page 20)
        if data.current_page is not None:
            target_start = data.current_page
            target_end = data.current_page
            export_filename = f"exported_page_{data.current_page}.pdf"
            
        # Mode B: User selected a custom page range (e.g., page 10 to page 25)
        elif data.start_page is not None and data.end_page is not None:
            target_start = data.start_page
            target_end = data.end_page
            export_filename = f"exported_pages_{data.start_page}_to_{data.end_page}.pdf"
            
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid payload: Provide either 'current_page' OR both 'start_page' and 'end_page'."
            )

        temp_pdf = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        )
        temp_pdf.close()

        # 2. Fetch pages dynamically from MongoDB and generate PDF
        pdf_path = export_pdf_from_vectors(
            session_id=session_id,
            user_id=firebase_uid,
            start_page=target_start,
            end_page=target_end,
            output_path=temp_pdf.name
        )

        if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) == 0:
            raise HTTPException(
                status_code=500,
                detail="Vector PDF segment compilation failed."
            )

        return FileResponse(
            path=pdf_path,
            media_type="application/pdf",
            filename=export_filename
        )

    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Export Pipeline Exception: {str(e)}"
        )