import os
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from psycopg2.extras import RealDictCursor
from dependencies import verify_firebase_token, get_postgres_db, verify_secure_bypass_key

router = APIRouter(prefix="/referral", tags=["Referral Engine"])

class RegisterReferralRequest(BaseModel):
    ref_user_id: str

@router.post("/register-attribution")
async def register_referral_attribution(
    data: RegisterReferralRequest, 
    decoded_token: dict = Depends(verify_firebase_token)
):
    new_user_id = decoded_token["uid"].strip().lower()
    referrer_id = data.ref_user_id.strip().lower()

    if new_user_id == referrer_id:
        return {"status": "ignored", "reason": "Self referral ignored"}

    with get_postgres_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT referred_by FROM user_profiles WHERE LOWER(user_id) = %s;", (new_user_id,))
            row = cursor.fetchone()

            if not row:
                cursor.execute(
                    "INSERT INTO user_profiles (user_id, is_premium, referred_by) VALUES (%s, 0, %s);",
                    (new_user_id, referrer_id)
                )
            elif row["referred_by"] is None:
                cursor.execute(
                    "UPDATE user_profiles SET referred_by = %s WHERE LOWER(user_id) = %s;",
                    (referrer_id, new_user_id)
                )
            conn.commit()

    return {"status": "success", "message": "Referral linked successfully"}

@router.post("/process-premium-reward")
async def process_premium_reward(
    buying_user_id: str,
    bypass_key: str
):
    if not verify_secure_bypass_key(bypass_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="Invalid Security Bypass Key"
        )

    clean_buyer_id = buying_user_id.strip().lower()
    expiry = (datetime.utcnow() + timedelta(days=30)).strftime("%Y-%m-%d")

    with get_postgres_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                INSERT INTO user_profiles (user_id, is_premium, expiry_date)
                VALUES (%s, 1, %s)
                ON CONFLICT (user_id) 
                DO UPDATE SET is_premium = 1, expiry_date = %s;
                """,
                (clean_buyer_id, expiry, expiry)
            )

            cursor.execute("SELECT referred_by FROM user_profiles WHERE LOWER(user_id) = %s;", (clean_buyer_id,))
            row = cursor.fetchone()

            if row and row["referred_by"]:
                referrer_id = row["referred_by"]

                cursor.execute(
                    """
                    UPDATE user_profiles 
                    SET bonus_requests = COALESCE(bonus_requests, 0) + 3,
                        referral_count = COALESCE(referral_count, 0) + 1
                    WHERE LOWER(user_id) = %s;
                    """,
                    (referrer_id,)
                )

                cursor.execute("SELECT referral_count FROM user_profiles WHERE LOWER(user_id) = %s;", (referrer_id,))
                ref_row = cursor.fetchone()

                if ref_row and ref_row["referral_count"] >= 15:
                    ref_expiry = (datetime.utcnow() + timedelta(days=30)).strftime("%Y-%m-%d")
                    cursor.execute(
                        "UPDATE user_profiles SET is_premium = 1, expiry_date = %s WHERE LOWER(user_id) = %s;",
                        (ref_expiry, referrer_id)
                    )
            conn.commit()

    return {"status": "success", "message": "Payment & Referral reward processed"}

@router.get("/stats")
async def get_referral_stats(decoded_token: dict = Depends(verify_firebase_token)):
    user_id = decoded_token["uid"].strip().lower()
    
    with get_postgres_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT referral_count, bonus_requests FROM user_profiles WHERE LOWER(user_id) = %s;", (user_id,))
            row = cursor.fetchone()

    total_referrals = row["referral_count"] if row and row["referral_count"] else 0
    bonus_requests = row["bonus_requests"] if row and row["bonus_requests"] else 0

    return {
        "user_id": user_id,
        "successful_referrals": total_referrals,
        "bonus_requests_left": bonus_requests,
        "referral_link": f"https://yourapp.com/signup?ref={user_id}"
    }