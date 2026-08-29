import os
import time
from collections import defaultdict
from typing import Optional
from fastapi import APIRouter, HTTPException, Request, Header, status
from pydantic import BaseModel
import firebase_admin
from firebase_admin import auth
from google import genai
from google.genai import types

router = APIRouter(prefix="/demo", tags=["Dual Mode Evaluator"])

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_NAME = "gemini-2.5-flash"

ip_request_history = defaultdict(list)
RATE_LIMIT_WINDOW_SECONDS = 60
MAX_REQUESTS_PER_WINDOW = 1

class EvaluationRequest(BaseModel):
    sample_context: str
    sample_question: str

def check_optional_jwt(authorization: Optional[str]) -> Optional[str]:
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split("Bearer ")[1].strip()
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token.get("uid")
    except Exception:
        return None

def enforce_ip_rate_limit(request: Request):
    client_ip = request.client.host if request.client else "unknown_client"
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()

    current_time = time.time()
    valid_timestamps = [ts for ts in ip_request_history[client_ip] if current_time - ts < RATE_LIMIT_WINDOW_SECONDS]
    ip_request_history[client_ip] = valid_timestamps

    if len(valid_timestamps) >= MAX_REQUESTS_PER_WINDOW:
        retry_after = int(RATE_LIMIT_WINDOW_SECONDS - (current_time - valid_timestamps[0]))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded for public access. 1 request per minute allowed. Try again in {max(1, retry_after)}s or login via App."
        )

    ip_request_history[client_ip].append(current_time)

@router.post("/evaluate-preview")
async def evaluate_preview(
    payload: EvaluationRequest,
    request: Request,
    authorization: Optional[str] = Header(None)
):
    authenticated_uid = check_optional_jwt(authorization)
    
    if not authenticated_uid:
        enforce_ip_rate_limit(request)
        mode = "public_demo (1 req/min rate-limited)"
    else:
        mode = f"authenticated_app_user (unlimited) - UID: {authenticated_uid}"

    context = payload.sample_context.strip()
    question = payload.sample_question.strip()

    if not context or not question:
        raise HTTPException(status_code=400, detail="Both 'sample_context' and 'sample_question' are required.")

    if not authenticated_uid and (len(context) > 3000 or len(question) > 300):
        raise HTTPException(status_code=400, detail="Public demo payload limit exceeded.")

    system_prompt = (
        "You are an academic evaluator engine. "
        "Answer the question strictly and directly using the provided context."
    )
    prompt_content = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt_content,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.0
            )
        )
        return {
            "status": "success",
            "access_mode": mode,
            "evaluator_output": response.text.strip() if response.text else "No response generated."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")