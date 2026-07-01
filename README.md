# SHL Assessment Recommender

Conversational FastAPI agent that recommends SHL Individual Test Solutions
through dialogue. Handles clarify, recommend, refine, compare, and scope
enforcement. All URLs come from the SHL catalog — no hallucination possible.

## Quick start (5 steps)

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the SHL catalog data
```bash
curl -o data/assessments.json "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
```

### 3. Build catalog.json
```bash
python data/clean_catalog.py
```
Expected output: `Done. Wrote 3XX assessments to data/catalog.json`

### 4. Run the tests
```bash
python -m pytest tests/test_api.py -v
```
Expected: all tests pass.

### 5. Start the server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Test the API

```bash
# Health check
curl http://localhost:8000/health

# Vague query (should clarify)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"I need an assessment"}]}'

# Multi-turn with seniority
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages":[
      {"role":"user","content":"Hiring a Java developer who works with stakeholders"},
      {"role":"assistant","content":"What level is this for?"},
      {"role":"user","content":"Mid-level, around 4 years"}
    ]
  }'

# Compare two assessments
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is the difference between OPQ and GSA?"}]}'
```

## API Schema

### POST /chat
```json
{
  "messages": [
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

Response:
```json
{
  "reply": "Here are 5 assessments...",
  "recommendations": [
    {"name": "Core Java (Advanced Level) (New)", "url": "https://www.shl.com/...", "test_type": "K"}
  ],
  "end_of_conversation": false
}
```

### GET /health
```json
{"status": "ok"}
```

## Optional: LLM reply polishing (Groq)

To enable natural-language rephrasing of replies (doesn't affect recommendations):

```bash
export LLM_REPHRASE_ENABLED=true
export GROQ_API_KEY=your_key_here
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Get a free Groq API key at https://console.groq.com

## Deploy to Render (free tier)

1. Push this repo to GitHub
2. Go to https://render.com → New Web Service → connect your repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Add env var: `PYTHONUNBUFFERED=1`
6. Deploy

Note: Make sure `data/catalog.json` is committed to the repo before deploying.
