import os
import re
import uuid
import json
import shutil
import asyncio
import edge_tts
import httpx
from typing import List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import StreamingResponse
from groq import AsyncGroq
from openai import AsyncOpenAI
from utils.upload_helpers import parse_any_file_to_pages
from dependencies import verify_firebase_token, get_redis_client, get_user_premium_status

router = APIRouter()

groq_client = AsyncGroq(api_key=os.getenv("GROQ_PREMIUM_API_KEY"))
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

GROQ_TEXT_MODEL = "openai/gpt-oss-120b"
GROQ_WHISPER_MODEL = "whisper-large-v3-turbo"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

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

async def generate_openai_vector(text: str):
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding, None
    except Exception as e:
        return [], f"OpenAI Embedding Error: {str(e)}"

async def fetch_vector_context(http_client: httpx.AsyncClient, firebase_uid: str, question: str):
    try:
        clean_q = re.sub(r'\[.*?\]', '', question).strip()
        query_vector, embed_err = await generate_openai_vector(clean_q)
        if not query_vector:
            return "", f"Question Embedding Error: {embed_err}"

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
            f"{SUPABASE_URL}/rest/v1/rpc/match_voice_vectors_premium",
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
        return "hi-IN-MadhurNeural"
    else:
        return "en-IN-PrabhatNeural"

@router.post("/process-voice-chunk")
async def process_voice_chunk(
    file: UploadFile = File(...),
    chunk_index: int = Form(...),
    start_page: int = Form(...),
    end_page: int = Form(...),
    is_first_chunk: bool = Form(False),
    decoded_token: dict = Depends(verify_firebase_token),
    redis = Depends(get_redis_client)
):
    user_id = decoded_token["uid"].strip().lower()

    unique_id = uuid.uuid4().hex[:8]
    chunk_filename = f"voice_chunk_{user_id}_{chunk_index}_{unique_id}.pdf"
    chunk_path = os.path.join(UPLOAD_DIR, chunk_filename)

    with open(chunk_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    task_payload = {
        "user_id": user_id,
        "file_path": chunk_path,
        "chunk_index": chunk_index,
        "start_page": start_page,
        "end_page": end_page,
        "is_first_chunk": is_first_chunk
    }

    await redis.lpush("voice_chunk_queue", json.dumps(task_payload))

    return {
        "status": "queued",
        "chunk_index": chunk_index,
        "message": f"Voice chunk {chunk_index} queued successfully"
    }

@router.post("/finalize-voice-session")
async def finalize_voice_session(
    decoded_token: dict = Depends(verify_firebase_token)
):
    firebase_uid = decoded_token["uid"].strip().lower()

    async with httpx.AsyncClient(timeout=30.0) as http_client:
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}"
        }
        res = await http_client.get(
            f"{SUPABASE_URL}/rest/v1/voice_vectors_premium?user_id=eq.{firebase_uid}&select=content&order=page_number.asc",
            headers=headers
        )
        if res.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to fetch extracted content from Supabase.")
        
        rows = res.json()
        all_text = "\n".join([r.get("content", "") for r in rows if r.get("page_number", 0) > 0])

    if not all_text.strip():
        raise HTTPException(status_code=400, detail="No readable content found in uploaded session.")

    prompt = (
        f"Analyze the extracted text content carefully:\n\n{all_text[:5000]}\n\n"
        f"Extract all logical academic questions present in this text. Respond STRICTLY in this JSON format: {{\"questions\": [\"Question 1\", \"Question 2\"]}}"
    )
    
    chat_completion = await groq_client.chat.completions.create(
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

    async with httpx.AsyncClient(timeout=30.0) as http_client:
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json"
        }
        dummy_vector = [0.0] * 1536
        questions_record = [{
            "user_id": firebase_uid,
            "page_number": 0,
            "content": json.dumps(detected_questions),
            "embedding": dummy_vector
        }]
        await http_client.post(
            f"{SUPABASE_URL}/rest/v1/voice_vectors_premium",
            headers=headers,
            json=questions_record
        )

    return {"status": "success", "questions": detected_questions}

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

    if not is_premium and question_index >= 2:
        return {
            "status": "error",
            "percentage": 0,
            "verdict": "upgrade_required",
            "is_last_question": True,
            "transition_message": "Free tier limit: Aap kewal 2 questions practice kar sakte hain. Baki questions ke liye Premium lein."
        }

    target_question = None
    total_questions = 0

    async with httpx.AsyncClient(timeout=15.0) as http_client:
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}"
        }
        res = await http_client.get(
            f"{SUPABASE_URL}/rest/v1/voice_vectors_premium?user_id=eq.{firebase_uid}&page_number=eq.0&select=content",
            headers=headers
        )
        if res.status_code == 200:
            rows = res.json()
            if rows and len(rows) > 0:
                questions_list = json.loads(rows[0]["content"])
                total_questions = len(questions_list)
                if 0 <= question_index < total_questions:
                    target_question = questions_list[question_index]

    if not target_question:
        return {"status": "error", "message": f"Question index {question_index} not found in Supabase session."}

    temp_audio_name = f"spoken_{uuid.uuid4()}.m4a"
    temp_audio_path = os.path.join(UPLOAD_DIR, temp_audio_name)

    try:
        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)

        with open(temp_audio_path, "rb") as f:
            transcription = await groq_client.audio.transcriptions.create(
                file=(temp_audio_name, f.read()),
                model=GROQ_WHISPER_MODEL,
                response_format="json"
            )

        student_speech = transcription.text.strip()
        if not student_speech:
            return {"status": "error", "message": "No audible speech recognized in audio."}

        async with httpx.AsyncClient(timeout=30.0) as http_client:
            combined_context, _ = await fetch_vector_context(http_client, firebase_uid, target_question)

        eval_prompt = (
            f"TEXTBOOK CONTEXT:\n{combined_context}\n\n"
            f"EVALUATION QUESTION:\n{target_question}\n\n"
            f"STUDENT SPOKEN ANSWER:\n{student_speech}"
        )
        
        eval_resp = await groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": VOICE_SYSTEM_PROMPT},
                {"role": "user", "content": eval_prompt}
            ],
            model=GROQ_TEXT_MODEL,
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        evaluation_result = json.loads(eval_resp.choices[0].message.content.strip())
        
        if not is_premium:
            is_last_question = (question_index + 1 >= 2) or (question_index + 1 >= total_questions)
        else:
            is_last_question = (question_index + 1) >= total_questions

        verdict = evaluation_result.get("verdict", "retry")

        if is_last_question and verdict == "correct":
            if not is_premium and total_questions > 2:
                transition_message = "Free trial ke 2 questions complete ho gaye. Aage practice karne ke liye Premium lein."
            else:
                transition_message = "All questions completed. Your study session is complete."
        elif verdict == "correct":
            transition_message = "Next question is on the way."
        else:
            transition_message = "Please try again."

        return {
            "status": "success",
            "transcription": student_speech,
            "percentage": evaluation_result.get("percentage", 0),
            "verdict": verdict,
            "is_last_question": is_last_question,
            "transition_message": transition_message
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

    try:
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            headers = {
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}"
            }
            await http_client.delete(
                f"{SUPABASE_URL}/rest/v1/voice_vectors_premium?user_id=eq.{firebase_uid}",
                headers=headers
            )
        return {"status": "success", "message": "Vectors and session flushed from Supabase."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))