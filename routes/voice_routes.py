import os
import io
import re
import uuid
import json
import shutil
import asyncio
import edge_tts
import httpx
from typing import List
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from google import genai
from google.genai import types
from utils.upload_helpers import parse_any_file_to_pages
from dependencies import verify_firebase_token, get_user_premium_status

router = APIRouter()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

STUDY_BUDDY_ENGINE = "gemini-2.5-flash"
UPLOAD_DIR = "uploads"
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

class QuestionListSchema(BaseModel):
    questions: List[str]

VOICE_SYSTEM_PROMPT = """
You are an expert oral exam evaluator. You will receive a Textbook Reference Context, an Evaluation Question, and an Audio File of the user speaking their answer.
Analyze the user's spoken audio directly. Compare the semantic core meaning of what they said against the provided Reference Context.
Ignore micro-stutters, natural conversational fillers, background static, or language pronunciation limits.
You MUST respond strictly in this JSON structure, with absolutely no markdown formatting or extra text:
{
  "percentage": 85,
  "verdict": "correct" 
}
Note: If semantic match is 80% or above, set "verdict": "correct". If below 80%, set "verdict": "retry".
"""

async def save_supabase_voice_session(user_id: str, context: str, questions: list):
    async with httpx.AsyncClient(timeout=30.0) as http_client:
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates"
        }
        payload = {
            "user_id": user_id,
            "full_text_context": context,
            "questions": questions,
            "attempted_count": 0
        }
        resp = await http_client.post(
            f"{SUPABASE_URL}/rest/v1/voice_sessions?on_conflict=user_id",
            headers=headers,
            json=payload
        )
        if resp.status_code not in [200, 201]:
            print(f"Supabase save error: {resp.text}")

async def get_supabase_voice_session(user_id: str) -> dict:
    async with httpx.AsyncClient(timeout=30.0) as http_client:
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}"
        }
        resp = await http_client.get(
            f"{SUPABASE_URL}/rest/v1/voice_sessions?user_id=eq.{user_id}&select=*",
            headers=headers
        )
        if resp.status_code == 200 and resp.json():
            return resp.json()[0]
    return {"full_text_context": "", "attempted_count": 0, "questions": []}

async def update_voice_attempt(user_id: str, new_count: int):
    async with httpx.AsyncClient(timeout=30.0) as http_client:
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json"
        }
        await http_client.patch(
            f"{SUPABASE_URL}/rest/v1/voice_sessions?user_id=eq.{user_id}",
            headers=headers,
            json={"attempted_count": new_count}
        )

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
    temp_voice_doc_path = ""
    try:
        clean_name = f"voice_ref_{uuid.uuid4()}_{file.filename.replace(' ', '_')}"
        temp_voice_doc_path = os.path.join(UPLOAD_DIR, clean_name)
        with open(temp_voice_doc_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        ext = os.path.splitext(temp_voice_doc_path)[-1].lower()
        
        detected_questions = []
        full_text_context = ""
        if ext in [".png", ".jpg", ".jpeg", ".webp"]:
            from utils.upload_helpers import process_image_via_vision_ai
            detected_questions = await process_image_via_vision_ai(temp_voice_doc_path, client)
            parsed_pages = await parse_any_file_to_pages(temp_voice_doc_path, client)
            full_text_context = "\n".join(parsed_pages)
        else:
            parsed_pages = await parse_any_file_to_pages(temp_voice_doc_path, client)
            extracted_text = "\n".join(parsed_pages)
            if not extracted_text.strip() or extracted_text == "[Empty Page]":
                raise HTTPException(status_code=400, detail="Document layout completely unreadable.")
            full_text_context = extracted_text
            intelligence_prompt = (
                f"Analyze the following extracted document text content carefully:\n\n"
                f"{full_text_context}\n\n"
                f"TASK:\n"
                f"Identify, reconstruct, and extract a clean list of all logical, academic, or evaluation questions "
                f"present in this text. If a question is plain or short, capture it exactly. "
                f"Strictly filter out any metadata, introductory headers, page indices, or background clutter."
            )
            loop = asyncio.get_running_loop()
            ai_response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=STUDY_BUDDY_ENGINE,
                    contents=intelligence_prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=QuestionListSchema,
                        temperature=0.0
                    ),
                )
            )
            if ai_response.text:
                parsed_json = json.loads(ai_response.text.strip())
                detected_questions = parsed_json.get("questions", [])
                
        if not detected_questions:
            raise HTTPException(status_code=400, detail="Could not extract questions from document.")
            
        await save_supabase_voice_session(firebase_uid, full_text_context, detected_questions)
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
async def evaluate_audio_recall(question: str = Form(...), file: UploadFile = File(...), decoded_token: dict = Depends(verify_firebase_token)):
    firebase_uid = decoded_token["uid"].strip().lower()
    
    current_session = await get_supabase_voice_session(firebase_uid)
    attempted_count = current_session.get("attempted_count", 0)
    
    if not get_user_premium_status(firebase_uid) and attempted_count >= 2:
        raise HTTPException(status_code=403, detail="Free Tier Practice Limit Reached. Max 2 questions allowed.")

    temp_audio_path = ""
    uploaded_audio_file = None
    try:
        temp_audio_name = f"{uuid.uuid4()}.m4a"
        temp_audio_path = os.path.join(UPLOAD_DIR, temp_audio_name)
        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        uploaded_audio_file = client.files.upload(file=temp_audio_path)
        
        prompt_payload = [
            uploaded_audio_file,
            f"TEXTBOOK CONTEXT ENGINE:\n{current_session.get('full_text_context', '')[:4000]}\n\nEVALUATION QUESTION:\n{question}\n"
        ]
        response = client.models.generate_content(
            model=STUDY_BUDDY_ENGINE,
            contents=prompt_payload,
            config=types.GenerateContentConfig(
                system_instruction=VOICE_SYSTEM_PROMPT,
                temperature=0.0,
                response_mime_type="application/json"
            ),
        )
        evaluation_result = json.loads(response.text)
        await update_voice_attempt(firebase_uid, attempted_count + 1)
        
        return {
            "status": "success",
            "percentage": evaluation_result.get("percentage", 0),
            "verdict": evaluation_result.get("verdict", "retry")
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if uploaded_audio_file:
            try:
                client.files.delete(name=uploaded_audio_file.name)
            except Exception:
                pass
        if temp_audio_path and os.path.exists(temp_audio_path):
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
            await http_client.delete(f"{SUPABASE_URL}/rest/v1/voice_sessions?user_id=eq.{firebase_uid}", headers=headers)
        return {"status": "success", "message": "Supabase voice session cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))