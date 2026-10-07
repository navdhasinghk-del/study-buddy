import os
import re
import uuid
import json
import shutil
import asyncio
import edge_tts
import httpx
from typing import List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Request
from fastapi.responses import StreamingResponse
from groq import AsyncGroq
from openai import AsyncOpenAI
from utils.upload_helpers import parse_any_file_to_pages
from dependencies import verify_firebase_token, get_user_premium_status

router = APIRouter()

groq_free_client = AsyncGroq(api_key=os.getenv("GROQ_FREE_API_KEY"))
groq_premium_client = AsyncGroq(api_key=os.getenv("GROQ_PREMIUM_API_KEY"))
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

GROQ_TEXT_MODEL = "openai/gpt-oss-120b"
GROQ_WHISPER_MODEL = "whisper-large-v3-turbo"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

VOICE_SYSTEM_PROMPT = """
You are an expert oral exam evaluator. You will receive a Textbook Reference Context, an Evaluation Question, and the transcribed text of what the student spoke.
Compare the semantic core meaning of the student's answer against the provided Reference Context.
Ignore minor grammatical slips, fillers, or pronunciation-based transcription artifacts.
You MUST respond strictly in valid JSON format with no markdown blocks:
{
  "percentage": 85,
  "verdict": "correct"
}
If semantic match is 75% or above, set "verdict": "correct". If below 75%, set "verdict": "retry".
"""

async def generate_gemini_vector(http_client: httpx.AsyncClient, text: str):
    url = f"{GEMINI_EMBED_URL}?key={GEMINI_API_KEY}"
    payload = {
        "model": "models/gemini-embedding-001",
        "content": {"parts": [{"text": text}]}
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

async def fetch_vector_context(http_client: httpx.AsyncClient, firebase_uid: str, question: str, is_premium: bool):
    try:
        clean_q = re.sub(r'\[.*?\]', '', question).strip()
        query_vector, embed_err = await get_embedding_by_tier(http_client, clean_q, is_premium)
        if not query_vector:
            return "", f"Question Embedding Error: {embed_err}"

        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json"
        }
        
        rpc_function = "match_voice_vectors_premium" if is_premium else "match_voice_vectors_free"

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
            return "", f"Supabase RPC Error: {resp.text}"

        matches = resp.json()
        if not matches:
            return "", "No context matches found in database."

        context_chunks = [item.get("content", "") for item in matches]
        return "\n\n".join(context_chunks), None

    except Exception as vector_err:
        return "", f"Vector Search Exception: {str(vector_err)}"

def auto_detect_language_voice(text: str) -> str:
    if re.search(r'[\u0B80-\u0BFF]', text):
        return "ta-IN-PallaviNeural"
    elif re.search(r'[\u0C00-\u0C7F]', text):
        return "te-IN-MohanNeural"
    elif re.search(r'[\u0C80-\u0CFF]', text):
        return "kn-IN-GaganNeural"
    elif re.search(r'[\u0D00-\u0D7F]', text):
        return "ml-IN-MidhunNeural"
    elif re.search(r'[\u0980-\u09FF]', text):
        return "bn-IN-BashkarNeural"
    elif re.search(r'[\u0A80-\u0AFF]', text):
        return "gu-IN-NiranjanNeural"
    elif re.search(r'[\u0A00-\u0A7F]', text):
        return "pa-IN-OjasNeural"
    elif re.search(r'[\u0B00-\u0B7F]', text):
        return "or-IN-SubhasiniNeural"
    elif re.search(r'[\u0900-\u097F]', text):
        marathi_markers = r'\b(aahe|nahi|kasa|aani|mhanje|aamhi|tumhi|kela|hotat|hota|aahet|karan|kay)\b'
        if re.search(marathi_markers, text.lower()):
            return "mr-IN-AarohiNeural"
        return "hi-IN-MadhurNeural"
    elif re.search(r'[\u0600-\u06FF]', text):
        return "ur-IN-SalmanNeural"
    elif re.search(r'[\u4E00-\u9FFF]', text):
        return "zh-CN-XiaoxiaoNeural"
    elif re.search(r'[\u3040-\u309F\u30A0-\u30FF]', text):
        return "ja-JP-NanamiNeural"
    elif re.search(r'[\uAC00-\uD7AF]', text):
        return "ko-KR-SunHiNeural"
    elif re.search(r'[\u0E00-\u0E7F]', text):
        return "th-TH-PremwadeeNeural"
    elif re.search(r'[\u0400-\u04FF]', text):
        return "ru-RU-SvetlanaNeural"
    elif re.search(r'[\u0370-\u03FF]', text):
        return "el-GR-AthinaNeural"
    elif re.search(r'[\u0590-\u05FF]', text):
        return "he-IL-AvriNeural"
    elif re.search(r'\b(que|para|por|con|como|este|esta|pero|del|los|las)\b', text.lower()):
        return "es-ES-AlvaroNeural"
    elif re.search(r'\b(pour|avec|dans|plus|tout|faire|mais|nous|vous)\b', text.lower()):
        return "fr-FR-HenriNeural"
    elif re.search(r'\b(und|nicht|oder|aber|wieder|haben|werden|sein)\b', text.lower()):
        return "de-DE-ConradNeural"
    elif re.search(r'\b(sono|essere|questo|tutto|anche|della|nella)\b', text.lower()):
        return "it-IT-DiegoNeural"
    elif re.search(r'\b(para|com|mais|como|muito|tudo|isso|pelo)\b', text.lower()):
        return "pt-BR-AntonioNeural"
    elif re.search(r'\b(bir|olan|için|daha|olarak|kadar|sonra)\b', text.lower()):
        return "tr-TR-AhmetNeural"
    elif re.search(r'\b(yang|dan|di|dari|untuk|pada|adalah|ini)\b', text.lower()):
        return "id-ID-ArdiNeural"
    elif re.search(r'\b(kya|ka|ki|ke|hai|ho|hain|ko|se|par|me|mein|hu|hoon|hoga|kare|karte|kaunsi|kis|cheez|bhi|nahi)\b', text.lower()):
        return "en-IN-PrabhatNeural"
    else:
        return "en-IN-PrabhatNeural"

@router.post("/process-voice-material")
async def process_voice_material(file: UploadFile = File(...), decoded_token: dict = Depends(verify_firebase_token)):
    firebase_uid = decoded_token["uid"].strip().lower()
    is_premium = get_user_premium_status(firebase_uid)
    target_table = "voice_vectors_premium" if is_premium else "voice_vectors_free"
    selected_groq = groq_premium_client if is_premium else groq_free_client

    temp_voice_doc_path = ""
    try:
        clean_name = f"voice_ref_{uuid.uuid4()}_{file.filename.replace(' ', '_')}"
        temp_voice_doc_path = os.path.join(UPLOAD_DIR, clean_name)
        with open(temp_voice_doc_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        parsed_pages = await parse_any_file_to_pages(temp_voice_doc_path)
        valid_pages = []
        for idx, t in enumerate(parsed_pages):
            if t.strip() and t != "[Empty Page]":
                valid_pages.append((idx + 1, t))

        if not valid_pages:
            raise HTTPException(status_code=400, detail="Document layout completely unreadable.")

        async with httpx.AsyncClient(timeout=60.0) as httpx_client:
            headers = {
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json"
            }
            
            await httpx_client.delete(
                f"{SUPABASE_URL}/rest/v1/{target_table}?user_id=eq.{firebase_uid}",
                headers=headers
            )

            records = []
            for page_num, text in valid_pages:
                embedding, _ = await get_embedding_by_tier(httpx_client, text, is_premium)
                if embedding:
                    records.append({
                        "user_id": firebase_uid,
                        "page_number": page_num,
                        "content": text,
                        "embedding": embedding
                    })

            if records:
                await httpx_client.post(
                    f"{SUPABASE_URL}/rest/v1/{target_table}",
                    headers=headers,
                    json=records
                )

        full_text_context = "\n".join([t[1] for t in valid_pages])
        prompt = (
            f"Analyze the extracted text content carefully:\n\n{full_text_context[:4000]}\n\n"
            f"Extract all logical academic questions present in this text. Respond STRICTLY in this JSON format: {{\"questions\": [\"Question 1\", \"Question 2\"]}}"
        )
        chat_completion = await selected_groq.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a precise JSON extractor. Output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            model=GROQ_TEXT_MODEL,
            response_format={"type": "json_object"},
            temperature=0.0
        )
        parsed_json = json.loads(chat_completion.choices[0].message.content.strip())
        detected_questions = parsed_json.get("questions", [])
        
        if not detected_questions:
            raise HTTPException(status_code=400, detail="Could not extract questions from document.")

        async with httpx.AsyncClient(timeout=30.0) as httpx_client:
            headers = {
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json"
            }
            dummy_vector = [0.0] * (1536 if is_premium else 768)
            questions_record = [{
                "user_id": firebase_uid,
                "page_number": 0,
                "content": json.dumps(detected_questions),
                "embedding": dummy_vector
            }]
            await httpx_client.post(
                f"{SUPABASE_URL}/rest/v1/{target_table}",
                headers=headers,
                json=questions_record
            )
            
        return {"status": "success", "questions": detected_questions}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_voice_doc_path and os.path.exists(temp_voice_doc_path):
            try:
                os.remove(temp_voice_doc_path)
            except Exception:
                pass

@router.get("/study-buddy-tts")
async def get_study_buddy_tts(text: str, decoded_token: dict = Depends(verify_firebase_token)):
    try:
        selected_voice = auto_detect_language_voice(text)
        communicate = edge_tts.Communicate(text, selected_voice)
        
        async def audio_stream_generator():
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]
                    
        return StreamingResponse(audio_stream_generator(), media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluate-audio-recall")
async def evaluate_audio_recall(
    question_index: int = Form(...),
    audio_file: UploadFile = File(...),
    decoded_token: dict = Depends(verify_firebase_token)
):
    firebase_uid = decoded_token["uid"].strip().lower()
    is_premium = get_user_premium_status(firebase_uid)
    target_table = "voice_vectors_premium" if is_premium else "voice_vectors_free"
    selected_groq = groq_premium_client if is_premium else groq_free_client

    target_question = None

    async with httpx.AsyncClient(timeout=15.0) as http_client:
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}"
        }
        res = await http_client.get(
            f"{SUPABASE_URL}/rest/v1/{target_table}?user_id=eq.{firebase_uid}&page_number=eq.0&select=content",
            headers=headers
        )
        if res.status_code == 200:
            rows = res.json()
            if rows and len(rows) > 0:
                questions_list = json.loads(rows[0]["content"])
                if 0 <= question_index < len(questions_list):
                    target_question = questions_list[question_index]

    if not target_question:
        return {"status": "error", "message": f"Question index {question_index} not found in Supabase session."}

    temp_audio_name = f"spoken_{uuid.uuid4()}.m4a"
    temp_audio_path = os.path.join(UPLOAD_DIR, temp_audio_name)

    try:
        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)

        with open(temp_audio_path, "rb") as f:
            transcription = await selected_groq.audio.transcriptions.create(
                file=(temp_audio_name, f.read()),
                model=GROQ_WHISPER_MODEL,
                response_format="json"
            )

        student_speech = transcription.text.strip()
        if not student_speech:
            return {"status": "error", "message": "No audible speech recognized in audio."}

        async with httpx.AsyncClient(timeout=30.0) as http_client:
            combined_context, _ = await fetch_vector_context(http_client, firebase_uid, target_question, is_premium)

        eval_prompt = (
            f"TEXTBOOK CONTEXT:\n{combined_context}\n\n"
            f"EVALUATION QUESTION:\n{target_question}\n\n"
            f"STUDENT SPOKEN ANSWER:\n{student_speech}"
        )
        
        eval_resp = await selected_groq.chat.completions.create(
            messages=[
                {"role": "system", "content": VOICE_SYSTEM_PROMPT},
                {"role": "user", "content": eval_prompt}
            ],
            model=GROQ_TEXT_MODEL,
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        evaluation_result = json.loads(eval_resp.choices[0].message.content.strip())
        
        return {
            "status": "success",
            "transcription": student_speech,
            "percentage": evaluation_result.get("percentage", 0),
            "verdict": evaluation_result.get("verdict", "retry")
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except Exception:
                pass

@router.post("/flush-voice-session")
async def flush_voice_session(decoded_token: dict = Depends(verify_firebase_token)):
    firebase_uid = decoded_token["uid"].strip().lower()
    is_premium = get_user_premium_status(firebase_uid)
    target_table = "voice_vectors_premium" if is_premium else "voice_vectors_free"

    try:
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            headers = {
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}"
            }
            await http_client.delete(
                f"{SUPABASE_URL}/rest/v1/{target_table}?user_id=eq.{firebase_uid}",
                headers=headers
            )
        return {"status": "success", "message": "Vectors and session flushed from Supabase."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))