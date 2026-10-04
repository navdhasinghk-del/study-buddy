Study Buddy Vector Document Engine

A high performance document ingestion and semantic question answering system. The engine extracts text from uploaded PDF materials, generates vector embeddings, stores them in Supabase using pgvector, and retrieves exact answers paired with verified source page citations.

Video Demos
Get Answer Mode Walkthrough:
https://youtube.com/shorts/Pm_sFVeCBPQ

Practice Mode Walkthrough:
https://youtube.com/shorts/g4Gaorr_7Lk

System Architecture Flow
Upload PDF Document -> Text and Structure Extraction -> Semantic Chunking -> Vector Embedding Pipeline -> Supabase pgvector Store -> Cosine Similarity Match against Query -> Synthesized Output with Page Citations

Tech Stack
Language and Runtime: Python 3.10 plus
Database and Vector Engine: Supabase PostgreSQL with pgvector
API Framework: FastAPI and Uvicorn
Deployment Infrastructure: Cloudflare Workers and Render

Key Capabilities
Automatic PDF text and layout extraction
Vector storage with fast similarity queries
Direct source page citations for every response
Dynamic practice mode generation from ingested material
Secure API routing with built in rate limiting
Lightweight backend architecture with zero runtime bloat

Local Setup Guide

1 Clone the repository
git clone https://github.com/navdhasinghk-del/study-buddy-.git
cd study-buddy-

2 Set environment variables in a dot env file
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
GEMINI_API_KEY=your_gemini_api_key
PORT=8000

3 Install required dependencies
pip install -r requirements.txt

4 Start the API service
uvicorn main:app --reload --port 8000

Contact and Inquiries
For custom API builds, database pipelines, and direct project work:
WhatsApp: +91 7509756798
