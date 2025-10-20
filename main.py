import os, json, re
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
import httpx

# ---------- Env ----------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SHARED_SECRET  = os.getenv("SHARED_SECRET")  # optional per-request header auth

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- App ----------
app = FastAPI(title="ListingMatcher Core API", version="3.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---------- Models ----------
class Criteria(BaseModel):
    location: str
    min_price: Optional[int] = None
    max_price: Optional[int] = None
    beds_min: Optional[int] = None
    baths_min: Optional[float] = None
    property_types: Optional[List[str]] = None
    keywords: Optional[str] = None

class RequestBody(BaseModel):
    limit: Optional[int] = 8
    language: Optional[str] = "fr"
    criteria: Criteria

class Listing(BaseModel):
    mls: Optional[str] = None
    url: Optional[str] = None
    address: Optional[str] = None
    price_cad: Optional[int] = None
    beds: Optional[int] = None
    baths: Optional[float] = None
    type: Optional[str] = None
    note: Optional[str] = None

class MatchResponse(BaseModel):
    reply: str
    properties: List[Listing] = []

# ---------- Helpers ----------
MLS_PATTERNS = [
    re.compile(r"\b\d{7}\b"),              # Centris: 7 digits
    re.compile(r"\b[A-Z0-9-]{6,12}\b", re.I) # REALTOR.ca: alphanumeric
]

def is_valid_mls(value: Optional[str]) -> bool:
    if not value or not isinstance(value, str):
        return False
    return any(p.search(value) for p in MLS_PATTERNS)

async def web_search(query: str, num: int) -> List[Dict[str, Any]]:
    if not SERPER_API_KEY:
        # still return an item so the model can reason about lack of results
        return []
    async with httpx.AsyncClient(timeout=20) as s:
        r = await s.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": num}
        )
    data = r.json()
    organic = data.get("organic") or []
    return [{"title": o.get("title"), "url": o.get("link"), "snippet": o.get("snippet")} for o in organic[:num]]

def extract_properties(text: str) -> List[Listing]:
    # find "properties": [ ... ] in the model output
    m = re.search(r'"properties"\s*:\s*(\[[\s\S]*?\])', text)
    if not m:
        return []
    try:
        raw = json.loads(m.group(1))
    except Exception:
        return []
    out: List[Listing] = []
    for item in raw:
        # coerce bad MLS to None
        mls = item.get("mls")
        if not is_valid_mls(mls):
            item["mls"] = None
        out.append(Listing(**{k: item.get(k) for k in ["mls","url","address","price_cad","beds","baths","type","note"]}))
    return out

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/listings/run", response_model=MatchResponse)
async def listings_run(request: Request):
    # optional lightweight auth
    if SHARED_SECRET:
        if request.headers.get("x-api-key") != SHARED_SECRET:
            raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    payload = RequestBody(**body)
    c = payload.criteria
    limit = max(1, min(payload.limit or 8, 12))
    lang = (payload.language or "fr").lower()

    # 1) Search query
    q = (
        f"site:centris.ca OR site:realtor.ca OR site:remax-quebec.com "
        f"OR site:royallepage.ca OR site:duproprio.com "
        f"{c.location} {c.keywords or ''} "
        f"{(str(c.min_price)+' to '+str(c.max_price)) if c.min_price or c.max_price else ''} "
        f"{' '.join(c.property_types or [])}"
    ).strip()
    results = await web_search(q, num=max(limit, 8))

    # format search results for the model
    search_text = "\n\n".join([f"{i+1}. {r['title']}\n{r['url']}\n{r['snippet']}" for i, r in enumerate(results)])

    # 2) System + User prompts
    system_prompt = (
        "You are ListingMatcher, a bilingual (French first, then English) assistant for Québec real estate.\n"
        "- Use the provided search results to pick ONLY CURRENT residential listings in Québec.\n"
        "- Never invent MLS numbers; if you cannot find one, set mls: null.\n"
        "- Return 5–12 items max, dedupe by MLS or exact address.\n"
        "- Fields: mls, url, address, price_cad, beds, baths, type, note.\n"
        "- END your response with a JSON object containing only the key \"properties\" (no extra commentary after the JSON)."
    )

    user_prompt = (
        f"Critères:\n"
        f"- Localisation: {c.location}\n"
        f"- Budget: {c.min_price or 'N/A'} à {c.max_price or 'N/A'} CAD\n"
        f"- Chambres min: {c.beds_min or 'N/A'}\n"
        f"- Salles de bain min: {c.baths_min or 'N/A'}\n"
        f"- Types: {', '.join(c.property_types or []) or 'Indifférent'}\n"
        f"- Mots-clés: {c.keywords or 'Aucun'}\n\n"
        f"Résultats de recherche:\n{search_text}\n\n"
        f"Réponds en français d'abord, puis une courte version en anglais.\n"
        f"Ensuite, fournis STRICTEMENT le JSON 'properties' (aucun texte après le JSON). "
        f"Limite: {limit} éléments."
    )

    # 3) Call OpenAI
    try:
        resp = client.responses.create(
            model="gpt-4.1",
            input=[
                {"role": "system", "content": [{"type":"input_text","text": system_prompt}]},
                {"role": "user",   "content": [{"type":"input_text","text": user_prompt}]},
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI error: {e}")

    text = getattr(resp, "output_text", "") or ""
    props = extract_properties(text)

    # final safety pass: enforce count & MLS validity already handled
    props = props[:limit]

    if not text:
        raise HTTPException(status_code=502, detail="Empty model output")
    return MatchResponse(reply=text, properties=props)
