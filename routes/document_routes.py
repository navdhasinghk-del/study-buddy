import os
import shutil
import uuid
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from dependencies import verify_firebase_token, get_encrypted_notes_connection
from utils.upload_helpers import parse_any_file_to_pages

load_dotenv()

router = APIRouter()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class PageFetchRequest(BaseModel):
    page_number: int

@router.post("/upload-pdf-session")
async def upload_pdf_session(
    file: UploadFile = File(...),
    decoded_token: dict = Depends(verify_firebase_token)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400, 
            detail="Only PDF files are supported in Document Reader Mode."
        )

    firebase_uid = decoded_token["uid"].strip().lower()
    session_id = f"session_{firebase_uid}"
    temp_path = os.path.join(UPLOAD_DIR, f"temp_{uuid.uuid4()}_{file.filename}")

    db = get_encrypted_notes_connection(firebase_uid)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        extracted_pages = await parse_any_file_to_pages(temp_path, client)
        total_pages = len(extracted_pages)

        if total_pages == 0:
            raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")

        db.pdf_pages.delete_many({"session_id": session_id})

        documents_to_insert = []
        page_1_text = extracted_pages[0] if total_pages > 0 else "[Empty Page]"

        for page_idx, clean_text in enumerate(extracted_pages):
            documents_to_insert.append({
                "session_id": session_id,
                "user_id": firebase_uid,
                "page_number": page_idx + 1,
                "content": clean_text
            })

        if documents_to_insert:
            db.pdf_pages.insert_many(documents_to_insert)

        db.pdf_pages.create_index([("session_id", 1), ("page_number", 1)])

        return {
            "status": "success",
            "total_pages": total_pages,
            "page_text": page_1_text,
            "session_id": session_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF Parsing Failure: {str(e)}")

    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

@router.post("/get-page-text")
async def get_page_text(
    data: PageFetchRequest,
    decoded_token: dict = Depends(verify_firebase_token)
):
    firebase_uid = decoded_token["uid"].strip().lower()
    session_id = f"session_{firebase_uid}"
    db = get_encrypted_notes_connection(firebase_uid)

    try:
        record = db.pdf_pages.find_one({
            "session_id": session_id,
            "page_number": data.page_number
        })

        if not record:
            return {
                "status": "not_found",
                "page_text": "[No text content found for this page index.]"
            }

        return {
            "status": "success",
            "page_number": data.page_number,
            "page_text": record.get("content", "[Empty Page]")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database Query Error: {str(e)}")

@router.post("/jump-page")
async def jump_page_text(
    data: PageFetchRequest,
    decoded_token: dict = Depends(verify_firebase_token)
):
    firebase_uid = decoded_token["uid"].strip().lower()
    session_id = f"session_{firebase_uid}"
    db = get_encrypted_notes_connection(firebase_uid)

    try:
        record = db.pdf_pages.find_one({
            "session_id": session_id,
            "page_number": data.page_number
        })

        if not record:
            return {
                "status": "not_found",
                "page_text": "[No text content found for this page index.]"
            }

        return {
            "status": "success",
            "page_number": data.page_number,
            "page_text": record.get("content", "[Empty Page]")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database Query Error: {str(e)}")

@router.post("/clear-pdf-session")
async def clear_pdf_session(
    decoded_token: dict = Depends(verify_firebase_token)
):
    firebase_uid = decoded_token["uid"].strip().lower()
    session_id = f"session_{firebase_uid}"
    db = get_encrypted_notes_connection(firebase_uid)

    try:
        result = db.pdf_pages.delete_many({"session_id": session_id})
        return {
            "status": "success", 
            "message": f"Cleared {result.deleted_count} pages from MongoDB for user: {firebase_uid}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))