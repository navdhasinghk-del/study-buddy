import os
import json
import traceback
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, Request, Form, status
from fastapi.responses import HTMLResponse, JSONResponse
from groq import AsyncGroq
from dependencies import get_postgres_db

router = APIRouter(prefix="/demo", tags=["Demo Portal"])

groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
GROQ_TEXT_MODEL = "openai/gpt-oss-120b"

def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown_guest"

def check_and_enforce_ip_daily_limit(client_ip: str, feature_name: str):
    clean_id = f"ip_{client_ip.replace(':', '_').replace('.', '_')}"
    today = datetime.utcnow().strftime("%Y-%m-%d")
    
    with get_postgres_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT usage_count FROM user_limits WHERE user_id = %s AND feature_name = %s AND usage_date = %s;",
                (clean_id, feature_name, today)
            )
            row = cursor.fetchone()
            if row and row[0] >= 1:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Daily free limit reached for '{feature_name}'. 1 request per day allowed. Try again tomorrow or login via App."
                )
            elif row:
                cursor.execute(
                    "UPDATE user_limits SET usage_count = usage_count + 1 WHERE user_id = %s AND feature_name = %s AND usage_date = %s;",
                    (clean_id, feature_name, today)
                )
            else:
                cursor.execute(
                    "INSERT INTO user_limits (user_id, feature_name, usage_date, usage_count) VALUES (%s, %s, %s, 1);",
                    (clean_id, feature_name, today)
                )
            conn.commit()

@router.get("/", response_class=HTMLResponse)
async def serve_demo_ui():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Study Buddy - Platform & Live Demo</title>
        <style>
            * { box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
            body { background: #080c14; color: #f1f5f9; margin: 0; padding: 30px 15px; display: flex; flex-direction: column; align-items: center; }
            .wrapper { max-width: 800px; width: 100%; }
            .header { text-align: center; margin-bottom: 30px; }
            .header h1 { color: #38bdf8; margin: 0 0 10px 0; font-size: 28px; }
            .header p { color: #94a3b8; font-size: 14px; margin: 0; line-height: 1.5; }
            
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 16px; margin-bottom: 35px; }
            .feature-card { background: #111827; border: 1px solid #1f293d; border-radius: 12px; padding: 18px; position: relative; }
            .feature-card h3 { margin: 0 0 8px 0; font-size: 15px; color: #f8fafc; }
            .feature-card p { margin: 0; font-size: 12px; color: #94a3b8; line-height: 1.4; }
            .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: bold; margin-bottom: 8px; }
            .badge-free { background: #064e3b; color: #34d399; }
            .badge-native { background: #1e3a5f; color: #38bdf8; }
            .badge-prem { background: #4c1d95; color: #c084fc; }

            .playground-card { background: #111827; border: 1px solid #38bdf8; border-radius: 16px; padding: 25px; box-shadow: 0 8px 25px rgba(56,189,248,0.1); }
            .playground-card h2 { color: #38bdf8; margin: 0 0 6px 0; font-size: 20px; }
            .playground-desc { font-size: 13px; color: #94a3b8; margin-bottom: 20px; }
            
            label { display: block; font-size: 13px; font-weight: 600; color: #cbd5e1; margin-bottom: 6px; }
            select, textarea { width: 100%; background: #080c14; border: 1px solid #334155; color: #f1f5f9; border-radius: 8px; padding: 12px; margin-bottom: 16px; font-size: 14px; outline: none; }
            select:focus, textarea:focus { border-color: #38bdf8; }
            textarea { resize: vertical; min-height: 80px; }
            button { width: 100%; background: #0284c7; color: white; border: none; padding: 14px; border-radius: 8px; font-size: 15px; font-weight: bold; cursor: pointer; transition: 0.2s; }
            button:hover { background: #0369a1; }
            button:disabled { background: #334155; cursor: not-allowed; }
            #resBox { display: none; margin-top: 20px; padding: 16px; border-radius: 8px; background: #080c14; border: 1px solid #1e293b; white-space: pre-wrap; font-size: 14px; line-height: 1.5; }
            .badge-200 { color: #4ade80; font-weight: bold; }
            .badge-429 { color: #f87171; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="wrapper">
            <div class="header">
                <h1>Study Buddy Architecture & Gateway</h1>
                <p>High-performance AI engine for academic extraction, semantic oral evaluation, and structured note storage.</p>
            </div>

            <div class="grid">
                <div class="feature-card">
                    <span class="badge badge-free">FREE TIER AVAILABLE</span>
                    <h3>1. AI Q&A Workspace</h3>
                    <p>Vector search combined with strict context verification for academic textbooks and questions.</p>
                </div>
                <div class="feature-card">
                    <span class="badge badge-native">ON-DEVICE (CLIENT-SIDE)</span>
                    <h3>2. Document Reader</h3>
                    <p>Native Android PDF rendering, local slicing, and Devanagari ML Kit OCR with zero backend load.</p>
                </div>
                <div class="feature-card">
                    <span class="badge badge-free">FREE TIER AVAILABLE</span>
                    <h3>3. Voice Recall Evaluator</h3>
                    <p>Real-time oral exam evaluation comparing spoken student audio against textbook context.</p>
                </div>
                <div class="feature-card">
                    <span class="badge badge-prem">APP EXCLUSIVE (PREMIUM)</span>
                    <h3>4. Encrypted Notepad</h3>
                    <p>MongoDB storage for categorization, personal revision notes, and keypoint management.</p>
                </div>
                <div class="feature-card">
                    <span class="badge badge-prem">APP EXCLUSIVE (PREMIUM)</span>
                    <h3>5. Vector PDF Export</h3>
                    <p>Dynamic evaluation report generation and custom PDF re-compilation directly from the cloud.</p>
                </div>
                <div class="feature-card">
                    <span class="badge badge-prem">APP EXCLUSIVE (PREMIUM)</span>
                    <h3>6. Text-to-Speech Engine</h3>
                    <p>Multilingual Neural TTS supporting regional Indian languages and global scripts seamlessly.</p>
                </div>
            </div>

            <div class="playground-card">
                <h2>Live Free Tier Playground</h2>
                <div class="playground-desc">Test free backend features directly. Enforced limit: 1 request / feature / day per IP.</div>
                
                <label for="feature">Select Feature To Test:</label>
                <select id="feature" onchange="renderForm()">
                    <option value="qa_workspace">1. AI Q&A Workspace (Context + Question)</option>
                    <option value="voice_evaluator">2. Voice Recall Evaluator (Simulated Oral Exam)</option>
                </select>

                <div id="dynamic-inputs"></div>

                <button id="submitBtn" onclick="executeFeature()">Run Feature Demo</button>
                <div id="resBox"></div>
            </div>
        </div>

        <script>
            function renderForm() {
                const feat = document.getElementById('feature').value;
                const container = document.getElementById('dynamic-inputs');
                let html = '';

                if (feat === 'qa_workspace') {
                    html = `
                        <label>Textbook / Document Context:</label>
                        <textarea id="ctx" placeholder="Paste textbook context here..."></textarea>
                        <label>Target Question:</label>
                        <textarea id="q" style="min-height:50px;" placeholder="Ask specific question based on above text..."></textarea>
                    `;
                } else if (feat === 'voice_evaluator') {
                    html = `
                        <label>Reference Textbook Context:</label>
                        <textarea id="ctx" placeholder="Enter reference paragraph..."></textarea>
                        <label>Evaluation Question:</label>
                        <textarea id="q" style="min-height:40px;" placeholder="e.g., Explain the greenhouse effect"></textarea>
                        <label>Spoken Answer Text (Transcript Simulation):</label>
                        <textarea id="voiceText" style="min-height:50px;" placeholder="What the student spoke during evaluation..."></textarea>
                    `;
                }
                container.innerHTML = html;
            }

            renderForm();

            async function executeFeature() {
                const feat = document.getElementById('feature').value;
                const resBox = document.getElementById('resBox');
                const btn = document.getElementById('submitBtn');

                btn.disabled = true;
                btn.innerText = "Processing Gateway Engine...";
                resBox.style.display = "block";
                resBox.innerHTML = "Submitting request...";

                const formData = new FormData();
                formData.append("feature_name", feat);

                const ctx = document.getElementById('ctx') ? document.getElementById('ctx').value.trim() : '';
                const q = document.getElementById('q') ? document.getElementById('q').value.trim() : '';
                const voiceText = document.getElementById('voiceText') ? document.getElementById('voiceText').value.trim() : '';
                
                if (!ctx) {
                    alert("Context text is required.");
                    btn.disabled = false;
                    btn.innerText = "Run Feature Demo";
                    resBox.style.display = "none";
                    return;
                }
                formData.append("context", ctx);
                formData.append("question", q);
                formData.append("spoken_text", voiceText);

                try {
                    const resp = await fetch("/demo/execute", {
                        method: "POST",
                        body: formData
                    });

                    const rawText = await resp.text();
                    let displayContent = "";
                    
                    try {
                        const parsedJson = JSON.parse(rawText);
                        if (parsedJson.detail) {
                            displayContent = typeof parsedJson.detail === 'string' ? parsedJson.detail : JSON.stringify(parsedJson.detail, null, 2);
                        } else {
                            displayContent = JSON.stringify(parsedJson, null, 2);
                        }
                    } catch (e) {
                        displayContent = rawText;
                    }

                    if (resp.status === 200) {
                        resBox.innerHTML = `<span class="badge-200">✓ SUCCESS (1/1 Daily Demo Limit Used)</span>\n\n` + displayContent;
                    } else {
                        resBox.innerHTML = `<span class="badge-429">✕ Response [${resp.status}]</span>\n\n` + displayContent;
                    }
                } catch (err) {
                    resBox.innerHTML = `<span class="badge-429">Network Error</span>\n\n` + err.message;
                } finally {
                    btn.disabled = false;
                    btn.innerText = "Run Feature Demo";
                }
            }
        </script>
    </body>
    </html>
    """

@router.post("/execute")
async def execute_demo_feature(
    request: Request,
    feature_name: str = Form(...),
    context: Optional[str] = Form(None),
    question: Optional[str] = Form(None),
    spoken_text: Optional[str] = Form(None)
):
    client_ip = get_client_ip(request)

    try:
        check_and_enforce_ip_daily_limit(client_ip, feature_name)
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"detail": e.detail})
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"Database Rate Limit Error: {str(e)}"})

    try:
        if feature_name == "qa_workspace":
            if not context or not question:
                return JSONResponse(status_code=400, content={"detail": "Context and Question are required."})
            if len(context) > 3500 or len(question) > 300:
                return JSONResponse(status_code=400, content={"detail": "Text exceeds demo character limit."})

            system_prompt = (
                "You are a strict academic evaluator. Answer the question using ONLY the context provided. "
                "Follow format: QUESTION: [q] | ANSWER: [ans] | PAGES: N/A"
            )
            resp = await groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"}
                ],
                model=GROQ_TEXT_MODEL,
                temperature=0.0
            )
            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "feature": "AI Q&A Workspace",
                    "result": resp.choices[0].message.content.strip()
                }
            )

        elif feature_name == "voice_evaluator":
            if not context or not question or not spoken_text:
                return JSONResponse(status_code=400, content={"detail": "Context, Question, and Spoken Text are required."})

            prompt = (
                f"You are an oral exam evaluator.\n"
                f"Reference Context: {context}\n"
                f"Question: {question}\n"
                f"Student Spoken Response: {spoken_text}\n\n"
                f"Evaluate semantic accuracy and return strictly JSON: {{\"percentage\": 85, \"verdict\": \"correct\" or \"retry\"}}"
            )
            resp = await groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Output valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                model=GROQ_TEXT_MODEL,
                response_format={"type": "json_object"},
                temperature=0.0
            )
            evaluation_data = json.loads(resp.choices[0].message.content.strip())
            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "feature": "Voice Recall Evaluator",
                    "evaluation": evaluation_data
                }
            )

        else:
            return JSONResponse(status_code=400, content={"detail": "Unknown feature selected."})

    except Exception as general_err:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"detail": f"Execution Engine Error: {str(general_err)}"}
        )