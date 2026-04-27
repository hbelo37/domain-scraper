from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

from scraper import fetch_pages, build_context, build_prompt, parse_json

app = FastAPI()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


class ScrapeRequest(BaseModel):
    domain: str


@app.post("/scrape")
async def scrape(req: ScrapeRequest):
    domain = req.domain.replace("https://", "").replace("http://", "").rstrip("/")

    if not GROQ_API_KEY:
        raise HTTPException(500, "GROQ_API_KEY not set in .env")

    pages = await fetch_pages(domain)
    if not pages:
        raise HTTPException(422, f"Could not fetch {domain} — site may be Cloudflare-protected")

    ctx    = build_context(pages)
    prompt = build_prompt(ctx)

    from groq import Groq
    try:
        res = Groq(api_key=GROQ_API_KEY).chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000,
        )
        raw = res.choices[0].message.content.strip()
    except Exception as e:
        err = str(e)
        if "401" in err or "invalid_api_key" in err:
            raise HTTPException(401, "Invalid Groq API key")
        elif "429" in err:
            raise HTTPException(429, "Rate limit hit — wait 60s and retry")
        raise HTTPException(500, err)

    result = parse_json(raw)
    if not result:
        raise HTTPException(500, f"LLM returned unparseable response: {raw[:200]}")

    return result


@app.get("/health")
def health():
    return {"status": "ok", "groq_key_set": bool(GROQ_API_KEY)}


app.mount("/", StaticFiles(directory="static", html=True), name="static")