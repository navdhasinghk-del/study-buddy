import os
import secrets
from datetime import datetime
from contextlib import contextmanager
from dotenv import load_dotenv
from fastapi import Header, HTTPException, status, Request
import firebase_admin
from firebase_admin import credentials, auth
from pymongo import MongoClient
import psycopg2
import psycopg2.pool
from psycopg2.extras import RealDictCursor

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("CRITICAL ERROR: DATABASE_URL is missing in .env file!")

try:
    db_pool = psycopg2.pool.ThreadedConnectionPool(
        minconn=2,
        maxconn=20,
        dsn=DATABASE_URL,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5
    )
except Exception as e:
    raise RuntimeError(f"Database Pool Initialization Error: {str(e)}")

@contextmanager
def get_postgres_db():
    conn = None
    try:
        conn = db_pool.getconn()
        if conn.closed != 0:
            db_pool.putconn(conn, close=True)
            conn = db_pool.getconn()
        
        try:
            with conn.cursor() as test_cur:
                test_cur.execute("SELECT 1;")
        except Exception:
            db_pool.putconn(conn, close=True)
            conn = db_pool.getconn()

        yield conn
    except Exception:
        if conn and conn.closed == 0:
            try:
                conn.rollback()
            except Exception:
                pass
        raise
    finally:
        if conn:
            if conn.closed != 0:
                try:
                    db_pool.putconn(conn, close=True)
                except Exception:
                    pass
            else:
                try:
                    db_pool.putconn(conn)
                except Exception:
                    pass

def init_postgres_user_db():
    try:
        with get_postgres_db() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_profiles (
                        user_id TEXT PRIMARY KEY,
                        is_premium INTEGER DEFAULT 0,
                        expiry_date TEXT,
                        referred_by TEXT DEFAULT NULL,
                        referral_count INTEGER DEFAULT 0,
                        bonus_requests INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_limits (
                        limit_id SERIAL PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        feature_name TEXT NOT NULL,
                        usage_date TEXT NOT NULL,
                        usage_count INTEGER DEFAULT 1,
                        UNIQUE (user_id, feature_name, usage_date)
                    );
                """)
                conn.commit()
    except Exception as e:
        raise RuntimeError(f"Table Initialization Error: {str(e)}")

init_postgres_user_db()

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("CRITICAL CRASH: MONGO_URI missing.")

mongo_client = MongoClient(MONGO_URI)
db = mongo_client["notepad_database"]

def get_encrypted_notes_connection(user_id: str):
    return db

FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase_creds.json.json")
if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
    firebase_admin.initialize_app(cred)

def get_crypto_storage_key(user_id: str) -> str:
    return os.getenv("DB_PEPPER_KEY", "")

def verify_secure_bypass_key(provided_key: str) -> bool:
    expected_key = os.getenv("DB_PEPPER_KEY", "")
    if not provided_key or not expected_key:
        return False
    return secrets.compare_digest(provided_key.strip(), expected_key.strip())

def get_user_premium_status(user_id: str) -> bool:
    if not user_id:
        return False
    try:
        with get_postgres_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    "SELECT is_premium, expiry_date FROM user_profiles WHERE LOWER(user_id) = LOWER(%s);",
                    (user_id.strip(),)
                )
                row = cursor.fetchone()
                if row and row["is_premium"] == 1:
                    if row["expiry_date"]:
                        expiry = datetime.strptime(row["expiry_date"], "%Y-%m-%d").date()
                        if datetime.utcnow().date() <= expiry:
                            return True
                    else:
                        return True
    except Exception:
        return False
    return False

def check_and_increment_limit(user_id: str, feature_name: str, max_limit: int) -> bool:
    if get_user_premium_status(user_id):
        return True
    clean_uid = user_id.lower().strip()
    today = datetime.utcnow().strftime("%Y-%m-%d")

    with get_postgres_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                "SELECT usage_count FROM user_limits WHERE LOWER(user_id) = %s AND feature_name = %s AND usage_date = %s;",
                (clean_uid, feature_name, today)
            )
            row = cursor.fetchone()

            if row:
                current_count = row["usage_count"]
                if current_count < max_limit:
                    cursor.execute(
                        "UPDATE user_limits SET usage_count = usage_count + 1 WHERE LOWER(user_id) = %s AND feature_name = %s AND usage_date = %s;",
                        (clean_uid, feature_name, today)
                    )
                    conn.commit()
                    return True
            else:
                cursor.execute(
                    "INSERT INTO user_limits (user_id, feature_name, usage_date, usage_count) VALUES (%s, %s, %s, 1);",
                    (clean_uid, feature_name, today)
                )
                conn.commit()
                return True

            cursor.execute("SELECT bonus_requests FROM user_profiles WHERE LOWER(user_id) = %s;", (clean_uid,))
            bonus_row = cursor.fetchone()
            if bonus_row and bonus_row["bonus_requests"] and bonus_row["bonus_requests"] > 0:
                cursor.execute(
                    "UPDATE user_profiles SET bonus_requests = bonus_requests - 1 WHERE LOWER(user_id) = %s;",
                    (clean_uid,)
                )
                conn.commit()
                return True

    return False

async def verify_firebase_token(request: Request, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Authorization token missing or invalid format"
        )
    
    token = authorization.split("Bearer ")[1].strip()
    try:
        decoded_token = auth.verify_id_token(token)
        user_id = decoded_token["uid"].strip().lower()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail=f"Token verification failed: {str(e)}"
        )

    if get_user_premium_status(user_id):
        return decoded_token
        
    path = request.url.path
    method = request.method

    if "/notepad" in path:
        if method in ["POST", "PUT", "DELETE"]:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Creating or editing notes is a premium feature.")
        elif method == "GET" and any(action in path for action in ["/note/", "/open", "/details"]):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Opening existing notes is a premium feature.")

    elif "/process-voice-material" in path:
        if not check_and_increment_limit(user_id, "voice_upload", 1):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Daily free limit reached for voice feature.")

    elif "/get-upload-url" in path:
        form_data = await request.form()
        filename = form_data.get("filename", "")
        file_size = form_data.get("file_size")
        
        if file_size and int(file_size) > 10 * 1024 * 1024:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="File size exceeds 10MB limit for free tier.")

        if filename.startswith("book_"):
            if not check_and_increment_limit(user_id, "get_answer_upload", 1):
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Daily free limit reached for AI workspace.")

    return decoded_token