import os
import re
import json
import uuid
import asyncio
import shutil
import traceback
import httpx
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse
from groq import AsyncGroq
from openai import AsyncOpenAI
from utils.upload_helpers import parse_any_file_to_pages, smart_exam_question_parser, process_image_via_vision_ai
from dependencies import verify_firebase_token, get_redis_client

router = APIRouter()

groq_client = AsyncGroq(api_key=os.getenv("GROQ_PREMIUM_API_KEY"))
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

GROQ_TEXT_MODEL = "openai/gpt-oss-120b"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

SYSTEM_PROMPT = """
You are a precise academic evaluator and textbook solver. Your job is to answer the target question using ONLY the provided Textbook Context.

CRITICAL RULES:
1. STRICT LANGUAGE MATCHING:
   - Output MUST be in the exact same language, script, dialect, and tone (Hindi, Hinglish, English, Bengali, Marathi, etc.) as the Textbook Context.
   - Do NOT translate terms into standard English or Hindi if the context uses regional language or Hinglish.
2. DYNAMIC LENGTH & MARKS ADAPTATION:
   - If the question contains an explicit limit or marks tag: Strictly follow that word count and depth.
   - If NO limit tag is present on the question: Provide the MAXIMUM, most comprehensive, rich, and detailed answer possible using the textbook context.
3. If the answer is not present or ambiguous in the context, reply "Context not found in textbook."

Strict Output Format:
QUESTION: [Target question statement without instruction tags]
ANSWER: [Targeted answer strictly matching the required depth and language]
PAGES: [Explicit source page numbers, e.g., Page 1, Page 3]
"""

async def generate_openai_vector(text: str):
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding, None
    except Exception as e:
        return [], f"OpenAI Embedding Error: {str(e)}"

@router.post("/upload-chunk")
async def handle_chunk_upload(
    file: UploadFile = File(...),
    chunk_index: int = Form(...),
    start_page: int = Form(...),
    end_page: int = Form(...),
    decoded_token: dict = Depends(verify_firebase_token),
    redis = Depends(get_redis_client)
):
    user_id = decoded_token["uid"].strip().lower()

    unique_id = uuid.uuid4().hex[:8]
    chunk_filename = f"chunk_{user_id}_{chunk_index}_{unique_id}.pdf"
    chunk_path = os.path.join(UPLOAD_DIR, chunk_filename)

    with open(chunk_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    task_payload = {
        "user_id": user_id,
        "file_path": chunk_path,
        "chunk_index": chunk_index,
        "start_page": start_page,
        "end_page": end_page,
        "target_table": "textbook_vectors_premium"
    }

    await redis.lpush("pdf_chunk_queue", json.dumps(task_payload))

    return {
        "status": "queued",
        "chunk_index": chunk_index,
        "message": f"Pages {start_page} to {end_page} successfully queued"
    }

@router.post("/get-upload-url")
async def handle_upload(
    file: UploadFile = File(...),
    filename: str = Form(...),
    decoded_token: dict = Depends(verify_firebase_token)
):
    firebase_uid = decoded_token["uid"].strip().lower()

    try:
        file_ext = os.path.splitext(filename)[1]
        base_clean = os.path.splitext(os.path.basename(filename))[0].replace(" ", "_")
        clean_name = f"{firebase_uid}_{uuid.uuid4().hex[:6]}_{base_clean}{file_ext}"
        path = os.path.join(UPLOAD_DIR, clean_name)
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        return {"status": "uploaded", "file_id": clean_name, "message": "File received successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def fetch_vector_context(http_client: httpx.AsyncClient, firebase_uid: str, question: str):
    try:
        clean_q = re.sub(r'\[.*?\]', '', question).strip()
        query_vector, embed_err = await generate_openai_vector(clean_q)
        if not query_vector:
            return None, f"[Callback: Embedding Generation Failed -> {embed_err}]"

        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json"
        }
        
        rpc_payload = {
            "query_embedding": query_vector,
            "match_threshold": -1.0,
            "match_count": 3,
            "filter_user_id": firebase_uid
        }

        resp = await http_client.post(
            f"{SUPABASE_URL}/rest/v1/rpc/match_textbook_vectors_premium",
            headers=headers,
            json=rpc_payload
        )

        if resp.status_code != 200:
            return None, f"[Callback: Supabase RPC match_textbook_vectors_premium Failed -> HTTP {resp.status_code}: {resp.text}]"

        matches = resp.json()
        if not matches:
            return None, f"[Callback: Supabase matched 0 chunks for user: {firebase_uid}]"

        context_chunks = []
        for item in matches:
            page_no = item.get("page_number", "N/A")
            txt = item.get("content", "")
            context_chunks.append(f"[Page {page_no}]\n{txt}")
            
        return "\n\n".join(context_chunks), None

    except Exception as vector_err:
        return None, f"[Callback: Vector Fetch Exception -> {str(vector_err)}]"

async def evaluate_single_question_parallel(http_client: httpx.AsyncClient, firebase_uid: str, q: str, semaphore: asyncio.Semaphore) -> str:
    async with semaphore:
        clean_q = re.sub(r'\[.*?\]', '', q).strip()
        combined_context, error_msg = await fetch_vector_context(http_client, firebase_uid, q)
        
        if error_msg or not combined_context:
            return f"QUESTION: {clean_q}\nANSWER: {error_msg}\nPAGES: N/A"
            
        prompt_content = f"TEXTBOOK CONTEXT:\n{combined_context}\n\nTARGET QUESTION STATEMENT & INSTRUCTION:\n{q}"

        try:
            response = await groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_content}
                ],
                model=GROQ_TEXT_MODEL,
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as groq_err:
            return f"QUESTION: {clean_q}\nANSWER: [Callback: Groq Execution Error -> {str(groq_err)}]\nPAGES: N/A"

def find_target_question_file(raw_id: str) -> str:
    if not raw_id:
        return ""
    base_name = os.path.basename(raw_id.replace("\\", "/")).replace(" ", "_")
    direct_path = os.path.join(UPLOAD_DIR, base_name)
    if os.path.exists(direct_path):
        return direct_path
        
    for fname in os.listdir(UPLOAD_DIR):
        if base_name in fname or fname in base_name:
            return os.path.join(UPLOAD_DIR, fname)
    return direct_path

@router.post("/process-solution")
async def process_solution(
    file: UploadFile = File(None),
    question_file_id: str = Form(None),
    decoded_token: dict = Depends(verify_firebase_token)
):
    q_path = ""
    firebase_uid = decoded_token["uid"].strip().lower()

    try:
        if file:
            clean_name = f"qp_{firebase_uid}_{os.path.basename(file.filename).replace(' ', '_')}"
            q_path = os.path.join(UPLOAD_DIR, clean_name)
            with open(q_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        elif question_file_id:
            file_wait_retry = 0
            while file_wait_retry < 15:
                q_path = find_target_question_file(question_file_id)
                if os.path.exists(q_path):
                    break
                await asyncio.sleep(0.5)
                file_wait_retry += 1
        else:
            raise HTTPException(status_code=400, detail="No question paper file provided.")

        if not q_path or not os.path.exists(q_path):
            all_files = os.listdir(UPLOAD_DIR) if os.path.exists(UPLOAD_DIR) else []
            return JSONResponse(
                content={"status": "error", "answer": f"Question paper file '{question_file_id}' not found. Current files: {all_files}"},
                status_code=200
            )
            
        ext = os.path.splitext(q_path)[-1].lower()
        if ext in [".png", ".jpg", ".jpeg", ".webp"]:
            questions = await process_image_via_vision_ai(q_path, is_premium=True)
        else:
            parsed_q_pages = await parse_any_file_to_pages(q_path)
            questions = await smart_exam_question_parser(parsed_q_pages, is_premium=True)
            
        if not questions:
            return {"status": "success", "answer": "No valid questions found to process."}

        semaphore = asyncio.Semaphore(10)

        async with httpx.AsyncClient(timeout=60.0) as http_client:
            tasks = [
                evaluate_single_question_parallel(http_client, firebase_uid, q, semaphore) 
                for q in questions
            ]
            results = await asyncio.gather(*tasks)

        return {"status": "success", "answer": "\n\n---\n\n".join(results)}

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"status": "error", "answer": f"[Callback: Processing Error -> {str(e)}]"}, status_code=200)
    finally:
        if q_path and os.path.exists(q_path):
            try:
                os.remove(q_path)
            except Exception:
                pass

@router.post("/flush-session")
async def flush_session(decoded_token: dict = Depends(verify_firebase_token)):
    firebase_uid = decoded_token["uid"].strip().lower()

    try:
        async with httpx.AsyncClient(timeout=10.0) as httpx_client:
            headers = {
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}"
            }
            await httpx_client.delete(
                f"{SUPABASE_URL}/rest/v1/textbook_vectors_premium?user_id=eq.{firebase_uid}",
                headers=headers
            )
        return {"status": "success", "message": f"Cloud storage flushed for user: {firebase_uid}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))