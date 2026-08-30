import os
import io
import re
import fitz
import json
import asyncio
import unicodedata
from typing import List
from pydantic import BaseModel, Field
from google.genai import types

MODEL_NAME = "gemini-2.5-flash"

class ExtractedQuestions(BaseModel):
    questions: List[str] = Field(
        description="List of fully reconstructed academic or evaluation questions. UI elements, page numbers, and filenames must be strictly excluded."
    )

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
    valid_chunks = []
    
    for line in lines:
        clean_line = line.strip()
        if clean_line:
            valid_chunks.append(clean_line)
            
    final_output = "\n".join(valid_chunks)
    if not final_output.strip() or len(final_output.strip()) < 5:
        return "[Empty Page]"
        
    return final_output

def extract_clean_blocks_from_page(page: fitz.Page) -> str:
    try:
        blocks = page.get_text("blocks")
        blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
        text_chunks = []
        for b in blocks:
            text = b[4].strip()
            if text:
                text_chunks.append(text)
        return "\n\n".join(text_chunks)
    except Exception:
        return page.get_text()

async def extract_text_via_gemini_ocr(page: fitz.Page, client) -> str:
    try:
        zoom = 2
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        image_part = types.Part.from_bytes(
            data=img_data,
            mime_type="image/png"
        )
        
        ocr_prompt = (
            "Extract all readable text from this image page in clean Unicode Hindi (Devanagari) or English. "
            "Preserve matras, conjunct characters, layout structure, and paragraphs exactly. "
            "Do not omit Hindi words or characters. Do not add any commentary, just return the extracted text."
        )
        
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model=MODEL_NAME,
                contents=[image_part, ocr_prompt]
            )
        )
        return response.text.strip() if response.text else "[Empty Page]"
    except Exception:
        return "[Empty Page]"

async def extract_direct_image_file(file_path: str, client) -> str:
    try:
        with open(file_path, "rb") as f:
            img_data = f.read()
            
        ext = os.path.splitext(file_path)[-1].lower().replace(".", "")
        mime_type = f"image/{ext}" if ext in ["png", "jpeg", "jpg", "webp"] else "image/png"
        
        image_part = types.Part.from_bytes(
            data=img_data,
            mime_type=mime_type
        )
        
        ocr_prompt = (
            "Extract all readable text from this document image in clean Unicode Hindi or English. "
            "Maintain complete sentences, Hindi matras, and paragraph structure. Return only the extracted text."
        )
        
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model=MODEL_NAME,
                contents=[image_part, ocr_prompt]
            )
        )
        return response.text.strip() if response.text else "[Empty Page]"
    except Exception:
        return "[Empty Page]"

async def parse_any_file_to_pages(file_path: str, client) -> List[str]:
    ext = os.path.splitext(file_path)[-1].lower()
    pages_text = []
    
    if ext == ".pdf":
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            content = extract_clean_blocks_from_page(page)
            
            if is_scrambled_legacy_font(content) or not content.strip() or len(content.strip()) < 15:
                ocr_result = await extract_text_via_gemini_ocr(page, client)
                text_content = protect_and_normalize_font(ocr_result)
            else:
                text_content = protect_and_normalize_font(content)
                
            pages_text.append(text_content)
        doc.close()
        
    elif ext in [".png", ".jpg", ".jpeg", ".webp"]:
        raw_text = await extract_direct_image_file(file_path, client)
        pages_text.append(protect_and_normalize_font(raw_text))
        
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()
        pages_text.append(protect_and_normalize_font(raw_text))
        
    else:
        try:
            doc = fitz.open(file_path)
            for page in doc:
                pages_text.append(protect_and_normalize_font(extract_clean_blocks_from_page(page)))
            doc.close()
        except Exception:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                pages_text.append(protect_and_normalize_font(f.read()))
                
    return [p if p.strip() else "[Empty Page]" for p in pages_text]

async def process_image_via_vision_ai(file_path: str, client) -> list:
    try:
        with open(file_path, "rb") as f:
            img_data = f.read()
            
        ext = os.path.splitext(file_path)[-1].lower().replace(".", "")
        mime_type = f"image/{ext}" if ext in ["png", "jpeg", "jpg", "webp"] else "image/png"
        
        image_part = types.Part.from_bytes(
            data=img_data,
            mime_type=mime_type
        )
        
        vision_prompt = (
            "Analyze this screenshot or document image carefully.\n\n"
            "TASK:\n"
            "Identify and extract ONLY the actual academic, logical, or evaluation questions present in the main content area.\n\n"
            "STRICT CRITERIA:\n"
            "- If a single question is broken down into multiple lines or has arbitrary line breaks due to layout, "
            "you MUST intelligently merge them into one single continuous string sequence.\n"
            "- Completely IGNORE and FILTER OUT all user interface elements, application headers, top action bars, back arrows, "
            "search icons, file names, page counters, and bottom navigation dock icons.\n"
            "- Do not include raw layout artifacts."
        )
        
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model=MODEL_NAME,
                contents=[image_part, vision_prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ExtractedQuestions,
                    temperature=0.0
                )
            )
        )
        
        if response.text:
            cleaned_json = json.loads(response.text.strip())
            return cleaned_json.get("questions", [])
        return []
    except Exception:
        return []

def smart_question_sanitizer(raw_pages: list) -> list:
    compiled_questions = []
    
    full_text_stream = ""
    for page_text in raw_pages:
        if page_text and page_text != "[Empty Page]":
            full_text_stream += page_text + "\n"
            
    lines = full_text_stream.split("\n")
    current_question_buffer = []
    
    for line in lines:
        clean_line = line.strip()
        if not clean_line:
            continue
            
        current_question_buffer.append(clean_line)
        if clean_line.endswith("?") or clean_line.endswith("।"):
            combined_q = " ".join(current_question_buffer).strip()
            if combined_q and len(combined_q) > 5:
                compiled_questions.append(combined_q)
            current_question_buffer = []
            
    if current_question_buffer:
        combined_q = " ".join(current_question_buffer).strip()
        if combined_q and len(combined_q) > 5:
            compiled_questions.append(combined_q)
            
    return compiled_questions