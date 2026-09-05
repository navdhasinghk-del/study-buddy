import os
import re
import asyncio
import shutil
import traceback
import httpx
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from groq import AsyncGroq
from openai import AsyncOpenAI
from utils.upload_helpers import parse_any_file_to_pages, smart_exam_question_parser, process_image_via_vision_ai
from dependencies import verify_firebase_token, get_user_premium_status

router = APIRouter()

groq_free_client = AsyncGroq(api_key=os.getenv("GROQ_FREE_API_KEY"))
groq_premium_client = AsyncGroq(api_key=os.getenv("GROQ_PREMIUM_API_KEY"))
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

GROQ_TEXT_MODEL = "openai/gpt-oss-120b"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

active_indexing_users = set()

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

async def generate_gemini_vector(http_client: httpx.AsyncClient, text: str):
    url = f"{GEMINI_EMBED_URL}?key={GEMINI_API_KEY}"
    payload = {
        "model": "models/gemini-embedding-001",
        "content": {
            "parts": [{"text": text}]
        }
    }
    try:
        response = await http_client.post(url, json=payload, timeout=20.0)
        if response.status_code == 200:
            values = response.json().get("embedding", {}).get("values", [])
            return values, None
        return [], f"Gemini API Error: HTTP {response.status_code} - {response.text}"
    except Exception as e:
        return [], f"Gemini Request Exception: {str(e)}"

async def generate_openai_vector(text: str):
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding, None
    except Exception as e:
        return [], f"OpenAI Embedding Error: {str(e)}"

async def get_embedding_by_tier(http_client: httpx.AsyncClient, text: str, is_premium: bool):
    if is_premium:
        return await generate_openai_vector(text)
    return await generate_gemini_vector(http_client, text)

async def forward_to_worker_for_indexing(file_path: str, firebase_uid: str, is_premium: bool):
    try:
        active_indexing_users.add(firebase_uid)
        texts = await parse_any_file_to_pages(file_path)
        
        valid_pages = []
        for idx, t in enumerate(texts):
            if t.strip() and t != "[Empty Page]":
                valid_pages.append((idx + 1, t))

        if not valid_pages:
            print(f"[Indexing Diagnostic] No text extracted from: {file_path}")
            return

        target_table = "textbook_vectors_premium" if is_premium else "textbook_vectors_free"

        async with httpx.AsyncClient(timeout=60.0) as httpx_client:
            headers = {
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json"
            }

            records = []
            for page_num, text in valid_pages:
                embedding, err = await get_embedding_by_tier(httpx_client, text, is_premium)
                if embedding:
                    records.append({
                        "user_id": firebase_uid,
                        "page_number": page_num,
                        "content": text,
                        "embedding": embedding
                    })
                else:
                    print(f"[Indexing Embedding Fail Page {page_num}]: {err}")

            if records:
                resp = await httpx_client.post(
                    f"{SUPABASE_URL}/rest/v1/{target_table}",
                    headers=headers,
                    json=records
                )
                print(f"[Indexing DB Insert Status]: HTTP {resp.status_code}")

    except Exception as e:
        print(f"[Indexing Exception Diagnostic] {str(e)}")
    finally:
        active_indexing_users.discard(firebase_uid)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass

@router.post("/get-upload-url")
async def handle_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    filename: str = Form(...),
    decoded_token: dict = Depends(verify_firebase_token)
):
    firebase_uid = decoded_token["uid"].strip().lower()
    is_premium = get_user_premium_status(firebase_uid)
    
    if not is_premium:
        file.file.seek(0, os.SEEK_END)
        file_size = file.file.tell()
        file.file.seek(0)
        if file_size > 10 * 1024 * 1024:
            raise HTTPException(status_code=403, detail="Free Tier Limit Reached. File size exceeds 10MB limit.")

    try:
        clean_name = os.path.basename(filename).replace(" ", "_")
        path = os.path.join(UPLOAD_DIR, clean_name)
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        is_question_paper = any(k in clean_name.lower() for k in ["question", "prashna", "paper", "qp_"])
        
        if not is_question_paper:
            background_tasks.add_task(forward_to_worker_for_indexing, path, firebase_uid, is_premium)
            return {"status": "uploaded", "file_id": clean_name, "message": "Document indexing started."}
            
        return {"status": "uploaded", "file_id": clean_name, "message": "Question paper received."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def fetch_vector_context(http_client: httpx.AsyncClient, firebase_uid: str, question: str, is_premium: bool):
    try:
        clean_q = re.sub(r'\[.*?\]', '', question).strip()
        query_vector, embed_err = await get_embedding_by_tier(http_client, clean_q, is_premium)
        if not query_vector:
            return None, f"[Callback: Embedding Generation Failed -> {embed_err}]"

        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json"
        }
        
        rpc_function = "match_textbook_vectors_premium" if is_premium else "match_textbook_vectors_free"
        
        rpc_payload = {
            "query_embedding": query_vector,
            "match_threshold": -1.0,
            "match_count": 3,
            "filter_user_id": firebase_uid
        }

        resp = await http_client.post(
            f"{SUPABASE_URL}/rest/v1/rpc/{rpc_function}",
            headers=headers,
            json=rpc_payload
        )

        if resp.status_code != 200:
            return None, f"[Callback: Supabase RPC {rpc_function} Failed -> HTTP {resp.status_code}: {resp.text}]"

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

async def evaluate_single_question_parallel(http_client: httpx.AsyncClient, firebase_uid: str, q: str, is_premium: bool) -> str:
    clean_q = re.sub(r'\[.*?\]', '', q).strip()
    combined_context, error_msg = await fetch_vector_context(http_client, firebase_uid, q, is_premium)
    
    if error_msg or not combined_context:
        return f"QUESTION: {clean_q}\nANSWER: {error_msg}\nPAGES: N/A"
        
    prompt_content = f"TEXTBOOK CONTEXT:\n{combined_context}\n\nTARGET QUESTION STATEMENT & INSTRUCTION:\n{q}"
    selected_groq = groq_premium_client if is_premium else groq_free_client

    try:
        response = await selected_groq.chat.completions.create(
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
    is_premium = get_user_premium_status(firebase_uid)

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
            
        while firebase_uid in active_indexing_users:
            await asyncio.sleep(1.0)
            
        ext = os.path.splitext(q_path)[-1].lower()
        if ext in [".png", ".jpg", ".jpeg", ".webp"]:
            questions = await process_image_via_vision_ai(q_path, is_premium=is_premium)
        else:
            parsed_q_pages = await parse_any_file_to_pages(q_path)
            questions = await smart_exam_question_parser(parsed_q_pages, is_premium=is_premium)
            
        if not questions:
            return {"status": "success", "answer": "No valid questions found to process."}

        async with httpx.AsyncClient(timeout=45.0) as http_client:
            tasks = [evaluate_single_question_parallel(http_client, firebase_uid, q, is_premium) for q in questions]
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
    is_premium = get_user_premium_status(firebase_uid)
    target_table = "textbook_vectors_premium" if is_premium else "textbook_vectors_free"

    try:
        async with httpx.AsyncClient(timeout=10.0) as httpx_client:
            headers = {
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}"
            }
            await httpx_client.delete(
                f"{SUPABASE_URL}/rest/v1/{target_table}?user_id=eq.{firebase_uid}",
                headers=headers
            )
        return {"status": "success", "message": f"Cloud storage flushed for user: {firebase_uid}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))