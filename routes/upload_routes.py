import os
import asyncio
import shutil
import httpx
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
from utils.upload_helpers import parse_any_file_to_pages, smart_question_sanitizer
from dependencies import verify_firebase_token, get_user_premium_status

router = APIRouter()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_NAME = "gemini-2.5-flash"
UPLOAD_DIR = "uploads"
AI_WORKER_URL = os.getenv("AI_WORKER_URL")

class ActiveSession:
    def __init__(self):
        self.is_indexing_active: bool = False

session_state = ActiveSession()

SYSTEM_PROMPT = """
You are a strict academic evaluator. Your job is to answer the question using ONLY the most exact and directly relevant textbook context provided.
Cross-reference the exact meaning. If the context contains mixed or irrelevant information from other chapters that does not directly map to the core definition of the question, IGNORE IT.
If the exact answer is not found or is ambiguous, reply "Context not found in textbook."
Strict Rule for Language: Aapka ANSWER aur output usi exact language aur simple tone mein hona chahiye jis language (jaise Hinglish) mein neeche diya gaya CONTEXT likha hai. Apne man se shuddh Hindi ya extra technical English words ka use mat kijiye.
Strict Rule for Format:
Aapko har hal mein neeche दिए गए format ko line-by-line follow karna hai:
QUESTION: [Yahan user ka question likhein]
ANSWER: [Yahan targeted response likhein jo context se exact match kare]
PAGES: [Yahan explicit page numbers likhein jaise context mein diya gaya hai, e.g., Page 1, Page 2]
"""

async def forward_to_worker_for_indexing(file_path: str, firebase_uid: str):
    try:
        session_state.is_indexing_active = True
        texts = await parse_any_file_to_pages(file_path, client)
        texts = [t for t in texts if t.strip() and t != "[Empty Page]"]
        if not texts:
            return
        async with httpx.AsyncClient(timeout=60.0) as httpx_client:
            payload = {"user_id": firebase_uid, "texts": texts}
            await httpx_client.post(f"{AI_WORKER_URL}/index-text", json=payload)
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        session_state.is_indexing_active = False
        if os.path.exists(file_path):
            os.remove(file_path)

@router.post("/get-upload-url")
async def handle_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    filename: str = Form(...),
    decoded_token: dict = Depends(verify_firebase_token)
):
    firebase_uid = decoded_token["uid"].strip().lower()
    
    if not get_user_premium_status(firebase_uid):
        file.file.seek(0, os.SEEK_END)
        file_size = file.file.tell()
        file.file.seek(0)
        
        if file_size > 10 * 1024 * 1024:
            raise HTTPException(status_code=403, detail="Free Tier Limit Reached. File size exceeds 10MB limit.")

    path = ""
    try:
        clean_name = filename.replace(" ", "_")
        path = os.path.join(UPLOAD_DIR, clean_name)
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        if clean_name.startswith("book_"):
            background_tasks.add_task(forward_to_worker_for_indexing, path, firebase_uid)
            return {"status": "uploaded", "file_id": clean_name, "message": "Gateway text parsing engine active."}
        return {"status": "uploaded", "file_id": clean_name}
    except Exception as e:
        if path and os.path.exists(path):
            os.remove(path)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process-solution")
async def process_solution(question_file_id: str = Form(...), decoded_token: dict = Depends(verify_firebase_token)):
    q_path = ""
    firebase_uid = decoded_token["uid"].strip().lower()
    try:
        raw_q_id = question_file_id.replace("\\", "/")
        question_id = raw_q_id.split("/")[-1].replace(" ", "_")
        q_path = os.path.join(UPLOAD_DIR, question_id)
        file_wait_retry = 0
        while not os.path.exists(q_path) and file_wait_retry < 10:
            await asyncio.sleep(0.5)
            file_wait_retry += 1
        if not os.path.exists(q_path):
            raise HTTPException(status_code=404, detail="Question paper file not found.")
        while session_state.is_indexing_active:
            await asyncio.sleep(2.0)
        ext = os.path.splitext(q_path)[-1].lower()
        if ext in [".png", ".jpg", ".jpeg", ".webp"]:
            from utils.upload_helpers import process_image_via_vision_ai
            questions = await process_image_via_vision_ai(q_path, client)
        else:
            parsed_q_pages = await parse_any_file_to_pages(q_path, client)
            questions = smart_question_sanitizer(parsed_q_pages)
        if not questions:
            return {"status": "success", "answer": "No valid questions found to process."}

        async def process_single_question(q: str) -> str:
            combined_context = ""
            try:
                async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                    search_payload = {"user_id": firebase_uid, "question": q, "top_k": 3}
                    resp = await httpx_client.post(f"{AI_WORKER_URL}/search-vectors", json=search_payload)
                    if resp.status_code == 200:
                        combined_context = resp.json().get("context", "")
            except Exception:
                pass
            prompt_content = f"TEXTBOOK CONTEXT:\n{combined_context}\n\nTARGET QUESTION TO EVALUATE:\n{q}"
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt_content,
                    config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.0),
                )
            )
            return response.text.strip() if response.text else f"QUESTION: {q}\nANSWER: Context not found in textbook.\nPAGES: N/A"

        results = await asyncio.gather(*(process_single_question(q) for q in questions))
        return {"status": "success", "answer": "\n\n---\n\n".join(results)}
    except Exception as e:
        return JSONResponse(content={"status": "error", "answer": str(e)}, status_code=500)
    finally:
        if q_path and os.path.exists(q_path):
            os.remove(q_path)
        try:
            async with httpx.AsyncClient(timeout=10.0) as httpx_client:
                await httpx_client.post(f"{AI_WORKER_URL}/clear-vectors?user_id={firebase_uid}")
        except Exception:
            pass

@router.post("/flush-session")
async def flush_session(decoded_token: dict = Depends(verify_firebase_token)):
    firebase_uid = decoded_token["uid"].strip().lower()
    try:
        async with httpx.AsyncClient(timeout=10.0) as httpx_client:
            await httpx_client.post(f"{AI_WORKER_URL}/clear-vectors?user_id={firebase_uid}")
        return {"status": "success", "message": f"Cloud storage flushed for user: {firebase_uid}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))