import os
import json
import asyncio
import httpx
import redis.asyncio as aioredis
from dotenv import load_dotenv
from openai import AsyncOpenAI
from utils.upload_helpers import parse_any_file_to_pages

load_dotenv()

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
REDIS_URL = os.getenv("REDIS_URL")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

PDF_QUEUE = "pdf_chunk_queue"
VOICE_QUEUE = "voice_chunk_queue"

async def generate_openai_vector(text: str):
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding, None
    except Exception as e:
        return [], f"OpenAI Embedding Error: {str(e)}"

async def process_pdf_chunk(task_data: dict):
    file_path = task_data.get("file_path")
    user_id = task_data.get("user_id")
    start_page = task_data.get("start_page", 1)
    end_page = task_data.get("end_page", 1)
    chunk_index = task_data.get("chunk_index", 1)

    print(f"\n[Worker PDF] Processing Chunk #{chunk_index} for User: {user_id} (Pages {start_page}-{end_page})...")

    try:
        pages_text = await parse_any_file_to_pages(file_path)
        records = []

        for idx, text in enumerate(pages_text):
            actual_page_num = start_page + idx
            clean_text = text.strip() if text else ""

            if clean_text and clean_text != "[Empty Page]":
                embedding, _ = await generate_openai_vector(clean_text)
                if embedding:
                    records.append({
                        "user_id": user_id,
                        "page_number": actual_page_num,
                        "content": clean_text,
                        "embedding": embedding
                    })

        if records:
            async with httpx.AsyncClient(timeout=60.0) as client:
                headers = {
                    "apikey": SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                    "Content-Type": "application/json"
                }
                response = await client.post(
                    f"{SUPABASE_URL}/rest/v1/textbook_vectors_premium",
                    headers=headers,
                    json=records
                )
                if response.status_code in [200, 201]:
                    print(f"[Worker PDF SUCCESS] Chunk #{chunk_index} saved to Supabase!")
                else:
                    print(f"[Worker PDF ERROR] Supabase rejected chunk: {response.text}")
    except Exception as err:
        print(f"[Worker PDF FAILURE] Error on Chunk #{chunk_index}: {err}")
    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass

async def process_voice_chunk(task_data: dict):
    file_path = task_data.get("file_path")
    user_id = task_data.get("user_id")
    start_page = task_data.get("start_page", 1)
    end_page = task_data.get("end_page", 1)
    chunk_index = task_data.get("chunk_index", 1)
    is_first_chunk = task_data.get("is_first_chunk", False)

    print(f"\n[Worker Voice] Processing Voice Chunk #{chunk_index} for User: {user_id}...")

    try:
        pages_text = await parse_any_file_to_pages(file_path)
        records = []

        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json"
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            if is_first_chunk:
                await client.delete(
                    f"{SUPABASE_URL}/rest/v1/voice_vectors_premium?user_id=eq.{user_id}",
                    headers=headers
                )

            for idx, text in enumerate(pages_text):
                actual_page_num = start_page + idx
                clean_text = text.strip() if text else ""

                if clean_text and clean_text != "[Empty Page]":
                    embedding, _ = await generate_openai_vector(clean_text)
                    if embedding:
                        records.append({
                            "user_id": user_id,
                            "page_number": actual_page_num,
                            "content": clean_text,
                            "embedding": embedding
                        })

            if records:
                response = await client.post(
                    f"{SUPABASE_URL}/rest/v1/voice_vectors_premium",
                    headers=headers,
                    json=records
                )
                if response.status_code in [200, 201]:
                    print(f"[Worker Voice SUCCESS] Chunk #{chunk_index} saved to Supabase!")
                else:
                    print(f"[Worker Voice ERROR] Supabase rejected chunk: {response.text}")
    except Exception as err:
        print(f"[Worker Voice FAILURE] Error on Chunk #{chunk_index}: {err}")
    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass

async def start_worker():
    if not REDIS_URL:
        print("[Worker Fatal] REDIS_URL environment variable missing!")
        return

    redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    print(f"[Worker Online] Listening to queues: '{PDF_QUEUE}' and '{VOICE_QUEUE}'...")

    while True:
        try:
            result = await redis.brpop([PDF_QUEUE, VOICE_QUEUE], timeout=5)
            if result:
                queue_name, task_json = result
                task_data = json.loads(task_json)
                if queue_name == PDF_QUEUE:
                    await process_pdf_chunk(task_data)
                elif queue_name == VOICE_QUEUE:
                    await process_voice_chunk(task_data)
        except asyncio.CancelledError:
            print("[Worker] Stopping worker gracefully...")
            break
        except Exception as queue_err:
            print(f"[Worker Queue Loop Error] {queue_err}")
            await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(start_worker())