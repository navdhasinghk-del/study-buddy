import os
import io
import re
import fitz
import json
import base64
import asyncio
import unicodedata
import httpx
from typing import List
from groq import AsyncGroq

OCR_SPACE_API_KEY = os.getenv("OCR_SPACE_API_KEY", "")

KRUTI_DEV_MAPPINGS = {
    "k": "ा", "d": "क", "f": "ि", "g": "ह", "h": "ी", "j": "र",
    "l": "त", "m": "म", "n": "न", "o": "द", "p": "च", "r": "प",
    "s": "े", "t": "त", "u": "ु", "v": "अ", "w": "ै", "x": "ग",
    "y": "ल", "z": "्र", "A": "ा", "B": "ी", "D": "्", "F": "ँ",
    "G": "ा", "H": "ी", "K": "ा", "L": "स", "M": "ं", "N": "छ",
    "O": "इ", "P": "फ", "R": "ज्ञ", "S": "े", "T": "ू", "U": "ू",
    "V": "ट", "W": "ै", "X": "घ", "Y": "भ", "Z": "र्", "1": "1",
    "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7",
    "8": "8", "9": "9", "0": "0"
}

def decode_kruti_dev_text(text: str) -> str:
    if not text:
        return text
    decoded = ""
    for char in text:
        decoded += KRUTI_DEV_MAPPINGS.get(char, char)
    return decoded

def is_scrambled_legacy_font(text: str) -> bool:
    if not text:
        return False
    scrambled_patterns = [r"fo'k;", r"d{kk", r"ekWMy", r"iz\"u", r"oLrqfu'B", r"v/;k;"]
    for pattern in scrambled_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False

def protect_and_normalize_font(text_stream: str) -> str:
    if not text_stream or text_stream.strip() == "[Empty Page]":
        return "[Empty Page]"
    if is_scrambled_legacy_font(text_stream):
        text_stream = decode_kruti_dev_text(text_stream)
    normalized_text = unicodedata.normalize('NFC', text_stream)
    if "cid:" in normalized_text or "unknown:" in normalized_text:
        return "[Empty Page]"
    lines = normalized_text.split("\n")
    valid_chunks = [line.strip() for line in lines if line.strip()]
    final_output = "\n".join(valid_chunks)
    if not final_output.strip() or len(final_output.strip()) < 5:
        return "[Empty Page]"
    return final_output

def extract_layer1_direct_code(page: fitz.Page) -> str:
    try:
        raw_text = page.get_text("text")
        clean_text = raw_text.strip()
        if len(clean_text) >= 20 and not is_scrambled_legacy_font(clean_text):
            return clean_text
    except Exception:
        pass
    return ""

def extract_layer2_layout_blocks(page: fitz.Page) -> str:
    try:
        blocks = page.get_text("blocks")
        blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
        text_chunks = [b[4].strip() for b in blocks if b[4].strip()]
        combined = "\n\n".join(text_chunks).strip()
        if len(combined) >= 20 and not is_scrambled_legacy_font(combined):
            return combined
    except Exception:
        pass
    return ""

async def extract_text_via_ocr_space(img_bytes: bytes) -> str:
    if not OCR_SPACE_API_KEY:
        return "[Empty Page]"
    try:
        url = "https://api.ocr.space/parse/image"
        files = {"file": ("page.png", img_bytes, "image/png")}
        data = {
            "apikey": OCR_SPACE_API_KEY,
            "language": "hin",
            "isOverlayRequired": False,
            "OCREngine": "2"
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, files=files, data=data)
            if resp.status_code == 200:
                res_data = resp.json()
                parsed = res_data.get("ParsedResults", [])
                if parsed:
                    return parsed[0].get("ParsedText", "").strip()
    except Exception:
        pass
    return "[Empty Page]"

async def parse_any_file_to_pages(file_path: str) -> List[str]:
    ext = os.path.splitext(file_path)[-1].lower()
    pages_text = []
    
    if ext == ".pdf":
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            l1_text = extract_layer1_direct_code(page)
            if l1_text:
                pages_text.append(protect_and_normalize_font(l1_text))
                continue
                
            l2_text = extract_layer2_layout_blocks(page)
            if l2_text:
                pages_text.append(protect_and_normalize_font(l2_text))
                continue
                
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = pix.tobytes("png")
            ocr_result = await extract_text_via_ocr_space(img_data)
            pages_text.append(protect_and_normalize_font(ocr_result))
            
        doc.close()
    elif ext in [".png", ".jpg", ".jpeg", ".webp"]:
        with open(file_path, "rb") as f:
            img_data = f.read()
        ocr_result = await extract_text_via_ocr_space(img_data)
        pages_text.append(protect_and_normalize_font(ocr_result))
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()
        pages_text.append(protect_and_normalize_font(raw_text))
    else:
        try:
            doc = fitz.open(file_path)
            for page in doc:
                txt = extract_layer1_direct_code(page) or extract_layer2_layout_blocks(page)
                if txt:
                    pages_text.append(protect_and_normalize_font(txt))
                else:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img_data = pix.tobytes("png")
                    ocr_result = await extract_text_via_ocr_space(img_data)
                    pages_text.append(protect_and_normalize_font(ocr_result))
            doc.close()
        except Exception:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                pages_text.append(protect_and_normalize_font(f.read()))
                
    return [p if p.strip() else "[Empty Page]" for p in pages_text]

async def smart_exam_question_parser(raw_pages: list, is_premium: bool = False) -> list:
    full_text = "\n\n".join([p for p in raw_pages if p != "[Empty Page]"])
    if not full_text.strip():
        return []

    if is_premium:
        selected_key = os.getenv("GROQ_PREMIUM_API_KEY") or os.getenv("GROQ_API_KEY")
    else:
        selected_key = os.getenv("GROQ_FREE_API_KEY") or os.getenv("GROQ_API_KEY")

    if not selected_key:
        return smart_fallback_sanitizer(raw_pages)

    prompt = f"""
You are an intelligent Exam Paper and Question Extractor.
Analyze the following document text and extract all standalone evaluation questions:

DOCUMENT CONTENT:
\"\"\"{full_text[:12000]}\"\"\"

STRICT RULES:
1. FILTER OUT METADATA: Ignore time limits, maximum marks header, roll number fields, general exam instructions, and section headers (e.g. 'समय', 'पूर्णांक', 'निर्देश', 'खण्ड-अ').
2. EXTRACT EVERY QUESTION & 'अथवा' (OR) OPTION: Extract main questions as well as 'अथवा' / OR alternative questions as separate, independent items.
3. DYNAMIC MARKS & LENGTH MAPPING:
   - If instructions specify word limits or marks for a question (e.g., '2 अंक, शब्द सीमा 30 शब्द', '5 अंक, शब्द सीमा 150 शब्द', '1 अंक / वस्तुनिष्ठ'), tag it at the end like: [Limit: 30 words, 2 Marks] or [Type: 1 Mark Objective].
   - If NO exam limits/marks exist in the paper (simple question list), do NOT attach any limit tag.
4. LANGUAGE: Keep the exact original language, script, and spelling of the questions without translating.

Return strictly valid JSON only:
{{
  "questions": [
    "राज्य शब्द का प्रयोग सर्वप्रथम किसने किया? [Type: 1 Mark Objective]"
  ]
}}
"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            client = AsyncGroq(api_key=selected_key, http_client=http_client)
            completion = await client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a precise academic question parser. Output valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                model="openai/gpt-oss-120b",
                response_format={"type": "json_object"},
                temperature=0.0
            )
            parsed = json.loads(completion.choices[0].message.content.strip())
            extracted = parsed.get("questions", [])
            if extracted:
                return extracted
    except Exception as e:
        pass

    return smart_fallback_sanitizer(raw_pages)

def smart_fallback_sanitizer(raw_pages: list) -> list:
    compiled_questions = []
    full_text_stream = "\n".join([p for p in raw_pages if p and p != "[Empty Page]"])
    lines = full_text_stream.split("\n")
    current_buffer = []
    for line in lines:
        clean = line.strip()
        if not clean or any(clean.startswith(ignore) for ignore in ["निर्देश", "समय", "पूर्णांक", "कक्षा", "खण्ड"]):
            continue
        current_buffer.append(clean)
        if clean.endswith("?") or clean.endswith("।") or clean.endswith(":"):
            q = " ".join(current_buffer).strip()
            if len(q) > 6:
                compiled_questions.append(q)
            current_buffer = []
    if current_buffer:
        q = " ".join(current_buffer).strip()
        if len(q) > 6:
            compiled_questions.append(q)
    return compiled_questions

async def process_image_via_vision_ai(file_path: str, is_premium: bool = False) -> list:
    pages = await parse_any_file_to_pages(file_path)
    return await smart_exam_question_parser(pages, is_premium=is_premium)

def smart_question_sanitizer(raw_pages: list) -> list:
    return smart_fallback_sanitizer(raw_pages)