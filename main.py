from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

# Import the top-level scrape() function — this handles all 3 layers internally
from scraper import scrape as run_scrape

app = FastAPI()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


class ScrapeRequest(BaseModel):
    domain: str


@app.post("/scrape")
async def scrape(req: ScrapeRequest):
    domain = req.domain.replace("https://", "").replace("http://", "").rstrip("/")

    if not domain:
        raise HTTPException(400, "Domain is required")

    if not GROQ_API_KEY:
        raise HTTPException(500, "GROQ_API_KEY not set — add it in Render environment variables")

    try:
        result = await run_scrape(domain, use_llm=True)
    except ValueError as e:
        # LLM errors (invalid key, rate limit) raised from scraper
        raise HTTPException(500, str(e))
    except Exception as e:
        raise HTTPException(500, f"Unexpected error: {str(e)[:200]}")

    if not result:
        raise HTTPException(422, f"Could not extract data for {domain}. Site may be fully protected.")

    return result


@app.get("/health")
def health():
    return {"status": "ok", "groq_key_set": bool(GROQ_API_KEY)}


# Serve frontend — must be last
app.mount("/", StaticFiles(directory="static", html=True), name="static")