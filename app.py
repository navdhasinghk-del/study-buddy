import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from psycopg2.extras import RealDictCursor

from routes import (
    upload_routes, 
    download_routes, 
    voice_routes, 
    notepad_routes, 
    document_routes, 
    referral_routes, 
    demo_routes
)
from dependencies import verify_firebase_token, get_postgres_db, verify_secure_bypass_key

load_dotenv()

RAW_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = [origin.strip() for origin in RAW_ORIGINS.split(",") if origin.strip()]

app = FastAPI(
    title="Study Buddy Gateway Engine",
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if "*" not in ALLOWED_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return response

@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/demo/")

@app.post("/admin/toggle-premium")
async def toggle_user_premium(
    target_user_id: str, 
    status_value: int, 
    bypass_key: str = None,
    expiry_date: str = None
):
    if not verify_secure_bypass_key(bypass_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="Access Denied. Invalid Security Bypass Key."
        )
    
    if not expiry_date:
        expiry_date = (datetime.utcnow() + timedelta(days=31)).strftime("%Y-%m-%d")
        
    clean_uid = str(target_user_id).strip().lower()
        
    with get_postgres_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                INSERT INTO user_profiles (user_id, is_premium, expiry_date)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_id) 
                DO UPDATE SET is_premium = EXCLUDED.is_premium, expiry_date = EXCLUDED.expiry_date;
                """,
                (clean_uid, status_value, expiry_date)
            )
            conn.commit()

    return {"status": "success", "target": clean_uid, "expiry": expiry_date}

@app.get("/subscription-status")
async def get_subscription_status(decoded_token: dict = Depends(verify_firebase_token)):
    user_id = decoded_token["uid"].strip().lower()
    today = datetime.utcnow().date()
    
    with get_postgres_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT is_premium, expiry_date FROM user_profiles WHERE LOWER(user_id) = %s;", (user_id,))
            row = cursor.fetchone()
    
    if row:
        is_premium = row["is_premium"] == 1
        expiry_str = row["expiry_date"]
        days_remaining = None
        
        if is_premium and expiry_str:
            try:
                expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
                remaining = (expiry_date - today).days
                days_remaining = max(0, remaining)
                if remaining < 0:
                    is_premium = False
            except Exception:
                days_remaining = 30
        elif is_premium:
            days_remaining = 30
            
        return {"is_premium": is_premium, "days_remaining": days_remaining}
        
    return {"is_premium": False, "days_remaining": None}

@app.post("/cancel-subscription")
async def cancel_subscription(decoded_token: dict = Depends(verify_firebase_token)):
    user_id = decoded_token["uid"].strip().lower()
    
    with get_postgres_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute("UPDATE user_profiles SET is_premium = 0, expiry_date = NULL WHERE LOWER(user_id) = %s;", (user_id,))
            conn.commit()
    
    return {"status": "success"}

app.include_router(upload_routes.router)
app.include_router(download_routes.router)
app.include_router(voice_routes.router)
app.include_router(notepad_routes.router)
app.include_router(document_routes.router)
app.include_router(referral_routes.router)
app.include_router(demo_routes.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=False)