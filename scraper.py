#!/usr/bin/env python3
"""
Company Intelligence Scraper
3-layer fetch: jina.ai reader → direct httpx → LLM inference fallback

Install: pip install httpx beautifulsoup4 groq python-dotenv
"""

import asyncio, sys, json, re, os
import httpx
from bs4 import BeautifulSoup

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL        = "llama-3.3-70b-versatile"

HEADERS = {
    "User-Agent":                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept":                    "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language":           "en-US,en;q=0.9",
    "Accept-Encoding":           "gzip, deflate, br",
    "Sec-Fetch-Dest":            "document",
    "Sec-Fetch-Mode":            "navigate",
    "Sec-Fetch-Site":            "none",
    "Sec-Fetch-User":            "?1",
    "Upgrade-Insecure-Requests": "1",
    "Cache-Control":             "max-age=0",
}

JINA_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/plain",
    "X-No-Cache": "true",
}


# ─────────────────────────────────────────────────────────────────
# LAYER 1: JINA.AI READER  (bypasses Cloudflare)
# ─────────────────────────────────────────────────────────────────

async def fetch_via_jina(client: httpx.AsyncClient, url: str) -> str:
    """
    Jina.ai reader converts any URL to clean markdown.
    Free, no API key needed, bypasses Cloudflare.
    https://jina.ai/reader/
    """
    try:
        jina_url = f"https://r.jina.ai/{url}"
        r = await client.get(jina_url, headers=JINA_HEADERS, timeout=20)
        if r.status_code == 200 and len(r.text) > 200:
            return r.text
    except Exception:
        pass
    return ""


# ─────────────────────────────────────────────────────────────────
# LAYER 2: DIRECT FETCH  (fallback for non-Cloudflare sites)
# ─────────────────────────────────────────────────────────────────

async def fetch_direct(client: httpx.AsyncClient, url: str) -> str:
    try:
        r = await client.get(url, headers=HEADERS, timeout=12)
        if r.status_code == 200 and len(r.text) > 200:
            return r.text
    except Exception:
        pass
    return ""


# ─────────────────────────────────────────────────────────────────
# SMART FETCH — tries jina first, falls back to direct
# ─────────────────────────────────────────────────────────────────

async def smart_fetch(client: httpx.AsyncClient, label: str, url: str) -> tuple:
    # Try jina first — works on Cloudflare-protected sites
    content = await fetch_via_jina(client, url)
    if content:
        return label, content, "jina"

    # Fall back to direct httpx
    content = await fetch_direct(client, url)
    if content:
        return label, content, "direct"

    return label, "", "failed"


async def fetch_pages(domain: str) -> tuple:
    """
    Returns (pages dict, fetch_method).
    Tries key pages, uses whatever fetch method works per page.
    """
    base = f"https://{domain}"

    urls = {
        "home":         base,
        "about":        f"{base}/about",
        "products":     f"{base}/products",
        "platform":     f"{base}/platform",
        "solutions":    f"{base}/solutions",
        "customers":    f"{base}/customers",
        "integrations": f"{base}/integrations",
        "pricing":      f"{base}/pricing",
    }

    async with httpx.AsyncClient(follow_redirects=True) as client:
        results = await asyncio.gather(*[
            smart_fetch(client, k, v) for k, v in urls.items()
        ])

    pages = {}
    methods = []
    for label, html, method in results:
        if html:
            pages[label] = html
            methods.append(method)

    primary_method = "jina" if methods.count("jina") >= methods.count("direct") else "direct"
    return pages, primary_method


# ─────────────────────────────────────────────────────────────────
# STRUCTURED EXTRACTION
# ─────────────────────────────────────────────────────────────────

def _clean_soup(html: str) -> BeautifulSoup:
    s = BeautifulSoup(html, "html.parser")
    for tag in s(["script", "style", "noscript", "svg"]):
        tag.decompose()
    return s


def _is_markdown(html: str) -> bool:
    """Jina returns markdown, not HTML. Detect which we got."""
    return not html.strip().startswith("<")


def _extract_from_markdown(text: str) -> dict:
    """Extract signals directly from jina markdown output."""
    lines = text.split("\n")

    headings, paragraphs = [], []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Markdown headings
        if line.startswith("# ") or line.startswith("## "):
            h = re.sub(r'^#+\s*', '', line).strip()
            if h and len(h) > 5 and len(h) < 150:
                headings.append(h)
        # Paragraphs — substantial lines that aren't nav/links
        elif len(line) > 50 and not line.startswith("[") and not line.startswith("!"):
            skip = ["cookie", "privacy", "terms", "©", "all rights reserved"]
            if not any(s in line.lower() for s in skip):
                paragraphs.append(line[:350])

    return {
        "headings":   headings[:20],
        "paragraphs": paragraphs[:10],
    }


def _meta(soup: BeautifulSoup) -> dict:
    out = {}
    for m in soup.find_all("meta"):
        prop    = m.get("property", "") or m.get("name", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if   prop == "og:site_name":  out["site_name"] = content
        elif prop == "og:title":      out["og_title"]  = content
        elif prop in ("og:description", "description") and "og_desc" not in out:
            out["og_desc"] = content
    return out


def _headings(soup: BeautifulSoup, limit=12) -> list:
    noise = {"navigation menu", "menu", "skip to content",
             "saved searches", "provide feedback", "site-wide links"}
    seen, out = set(), []
    for h in soup.find_all(["h1", "h2"])[:limit * 3]:
        t = " ".join(h.get_text(" ", strip=True).split())
        if t and len(t) > 5 and t.lower() not in noise and t not in seen:
            seen.add(t); out.append(t)
        if len(out) >= limit:
            break
    return out


def _paragraphs(soup: BeautifulSoup, limit=6) -> list:
    skip = ["cookie", "privacy policy", "terms of service", "all rights reserved", "©"]
    out = []
    for p in soup.find_all("p"):
        t = " ".join(p.get_text(" ", strip=True).split())
        if 40 < len(t) < 350 and not any(s in t.lower() for s in skip):
            out.append(t)
        if len(out) >= limit:
            break
    return out


def _nav(soup: BeautifulSoup) -> list:
    seen, out = set(), []
    for nav in soup.find_all("nav"):
        for a in nav.find_all("a"):
            t = " ".join(a.get_text(" ", strip=True).split())
            if t and 2 < len(t) < 50 and t not in seen:
                seen.add(t); out.append(t)
    return out[:30]


def _logos(soup: BeautifulSoup) -> list:
    names = []
    skip  = ["headshot", "profile", "avatar", ".png", ".jpg", "photo", "logo"]
    for section in soup.find_all(["section", "div"]):
        cls = " ".join(section.get("class", [])).lower()
        if any(k in cls for k in ["customer", "client", "logo", "trusted", "partner"]):
            for img in section.find_all("img"):
                alt = img.get("alt", "").strip()
                if alt and len(alt) < 40 and not any(s in alt.lower() for s in skip):
                    names.append(alt)
    return list(dict.fromkeys(names))[:15]


KNOWN_TOOLS = [
    "Salesforce", "HubSpot", "Slack", "Stripe", "Zapier", "Jira",
    "Asana", "Monday", "Notion", "Datadog", "Twilio", "Okta",
    "AWS", "Azure", "Google Cloud", "Marketo", "Outreach",
    "Segment", "Zendesk", "Intercom", "Gong", "Salesloft",
    "Shopify", "QuickBooks", "Zoom", "Teams", "Figma", "Snowflake",
]

def _integrations(text: str) -> list:
    return [t for t in KNOWN_TOOLS if t.lower() in text.lower()]


def build_context(pages: dict) -> dict:
    ctx = dict(site_name="", og_title="", og_desc="",
               headings=[], paragraphs=[], nav_items=[],
               customers=[], integrations=[])

    home = pages.get("home", "")
    if home:
        if _is_markdown(home):
            # Jina markdown path
            extracted = _extract_from_markdown(home)
            ctx["headings"]   = extracted["headings"]
            ctx["paragraphs"] = extracted["paragraphs"]
            # Try to get title from first heading
            if extracted["headings"]:
                ctx["og_title"] = extracted["headings"][0]
        else:
            # Direct HTML path
            s    = _clean_soup(home)
            meta = _meta(s)
            ctx.update({
                "site_name":  meta.get("site_name", ""),
                "og_title":   meta.get("og_title",  ""),
                "og_desc":    meta.get("og_desc",   ""),
                "nav_items":  _nav(s),
                "headings":   _headings(s),
                "paragraphs": _paragraphs(s),
                "customers":  _logos(s),
            })
        ctx["integrations"] = _integrations(home)

    seen_h = set(ctx["headings"])
    seen_c = set(ctx["customers"])

    for key in ("about", "products", "platform", "solutions", "customers", "integrations"):
        html = pages.get(key, "")
        if not html:
            continue

        if _is_markdown(html):
            extracted = _extract_from_markdown(html)
            new_h = [h for h in extracted["headings"] if h not in seen_h]
            ctx["headings"]   += new_h; seen_h.update(new_h)
            ctx["paragraphs"] += extracted["paragraphs"][:3]
        else:
            s     = _clean_soup(html)
            new_h = [h for h in _headings(s, 8) if h not in seen_h]
            new_c = [c for c in _logos(s)        if c not in seen_c]
            ctx["headings"]   += new_h; seen_h.update(new_h)
            ctx["paragraphs"] += _paragraphs(s, 3)
            ctx["customers"]  += new_c; seen_c.update(new_c)

        ctx["integrations"] += [i for i in _integrations(html)
                                 if i not in ctx["integrations"]]

    ctx["headings"]     = ctx["headings"][:20]
    ctx["paragraphs"]   = ctx["paragraphs"][:10]
    ctx["customers"]    = ctx["customers"][:20]
    ctx["integrations"] = ctx["integrations"][:20]
    return ctx


# ─────────────────────────────────────────────────────────────────
# LAYER 3: LLM INFERENCE FALLBACK
# ─────────────────────────────────────────────────────────────────

INFERENCE_PROMPT = """\
You are a B2B market intelligence analyst with knowledge of major companies.

The domain is: {domain}

No website content could be extracted (site is likely protected).
Use your existing knowledge of this company to fill in what you know.
Only include information you are confident about.
Return "" or [] for anything you are not sure of.

Return ONLY valid JSON, no markdown:

{{
  "company_name": "Brand name only",
  "description": "2 sentences: what they do and who they serve",
  "icp": {{
    "personas":     ["Job title"],
    "company_size": ["SMB", "Mid-Market", "Enterprise"],
    "industries":   ["Industry"],
    "summary":      "2-3 sentence ICP description as flowing prose"
  }},
  "offerings": [
    {{
      "name": "Product name",
      "category": "Category",
      "description": "1 sentence",
      "value_proposition": "1 sentence benefit",
      "pain_points": ["Pain point"]
    }}
  ],
  "value_proposition": "1 sentence for the whole company",
  "partners": ["Known integration partner"],
  "existing_customers": [],
  "pricing_tier": "e.g. Freemium, Enterprise, Contact Sales"
}}
"""

EXTRACTION_PROMPT = """\
You are a B2B market intelligence analyst.

Use ONLY the data below. Do not invent anything not present.
Return "" or [] when evidence is missing.

SITE NAME:      {site_name}
OG TITLE:       {og_title}
OG DESCRIPTION: {og_desc}

NAV ITEMS:
{nav_items}

PAGE HEADINGS:
{headings}

KEY PARAGRAPHS:
{paragraphs}

CUSTOMER LOGOS:
{customers}

INTEGRATIONS FOUND:
{integrations}

Return ONLY valid JSON, no markdown:

{{
  "company_name": "Brand name only — strip taglines",

  "description": "2 sentences max. What they do and who they serve.",

  "icp": {{
    "personas":     ["Job title e.g. VP Sales"],
    "company_size": ["SMB", "Mid-Market", "Enterprise"],
    "industries":   ["Industry e.g. SaaS"],
    "summary":      "2-3 sentences as flowing prose. Must weave together industry, company size, and personas. Example: Targets mid-market B2B SaaS companies with 200+ employees. Primary buyers are VP of Sales and Revenue Operations leaders."
  }},

  "offerings": [
    {{
      "name":              "Exact product name e.g. Marketing Hub",
      "category":          "Short category",
      "description":       "1 sentence: what this product does",
      "value_proposition": "1 sentence: core benefit",
      "pain_points":       ["Pain point"]
    }}
  ],

  "value_proposition": "1 sentence for the whole company",
  "partners":          ["Integration or partner name"],
  "existing_customers":["Customer name from logo wall"],
  "pricing_tier":      "e.g. Freemium, Enterprise, Contact Sales"
}}

STRICT RULES FOR OFFERINGS:
VALID: Named software modules or platforms — "Marketing Hub", "SalesOS", "Copilot"
INVALID: Nav labels, CTAs, generic words — "Careers", "Get started", "Platform", "AI"
If no clearly named products found, return offerings: []
Max 6 offerings.
"""

def _fmt(lst: list) -> str:
    return "\n".join(f"  • {x}" for x in lst) if lst else "  (none found)"


def build_prompt(ctx: dict, domain: str, use_inference: bool = False) -> str:
    if use_inference:
        return INFERENCE_PROMPT.format(domain=domain)

    return EXTRACTION_PROMPT.format(
        site_name    = ctx["site_name"]  or "(not found)",
        og_title     = ctx["og_title"]   or "(not found)",
        og_desc      = ctx["og_desc"]    or "(not found)",
        nav_items    = _fmt(ctx["nav_items"]),
        headings     = _fmt(ctx["headings"]),
        paragraphs   = _fmt(ctx["paragraphs"]),
        customers    = _fmt(ctx["customers"]),
        integrations = _fmt(ctx["integrations"]),
    )


def call_llm(prompt: str) -> str:
    from groq import Groq
    try:
        res = Groq(api_key=GROQ_API_KEY).chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000,
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        err = str(e)
        if "401" in err or "invalid_api_key" in err:
            raise ValueError("Invalid Groq API key")
        elif "429" in err:
            raise ValueError("Rate limited — wait 60s")
        raise ValueError(str(e)[:100])


def parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except:
                pass
    return {}


def _has_content(ctx: dict) -> bool:
    """Check if we extracted meaningful content from pages."""
    return bool(
        ctx.get("og_desc") or
        len(ctx.get("headings", [])) > 2 or
        len(ctx.get("paragraphs", [])) > 1
    )


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

async def scrape(domain: str, use_llm: bool = True) -> dict:
    domain = domain.replace("https://", "").replace("http://", "").rstrip("/")

    print(f"\n🚀  {domain}")
    print("─" * 55)

    # Phase 1 — fetch
    print("📥  Fetching pages...", end=" ", flush=True)
    pages, fetch_method = await fetch_pages(domain)

    if not pages:
        print(f"❌  No pages fetched")
        if use_llm and GROQ_API_KEY:
            print("🧠  Falling back to LLM inference from domain knowledge...")
            raw    = call_llm(build_prompt({}, domain, use_inference=True))
            result = parse_json(raw)
            if result:
                result["_source"] = "llm_inference"
                return result
        return {}

    print(f"✅  {list(pages.keys())} via {fetch_method}")

    # Build context
    ctx = build_context(pages)
    has_content = _has_content(ctx)

    print(f"\n📐  Signals (via {fetch_method}):")
    print(f"    site_name    : {ctx['site_name'] or '—'}")
    print(f"    og_desc      : {(ctx['og_desc'][:70] + '…') if ctx['og_desc'] else '—'}")
    print(f"    headings     : {len(ctx['headings'])}")
    print(f"    paragraphs   : {len(ctx['paragraphs'])}")
    print(f"    customers    : {len(ctx['customers'])}")
    print(f"    integrations : {ctx['integrations']}")
    print(f"    has_content  : {has_content}")

    if not use_llm or not GROQ_API_KEY:
        return ctx

    # Phase 2 — LLM
    if has_content:
        print("\n🤖  LLM extraction from scraped content...", end=" ", flush=True)
        prompt = build_prompt(ctx, domain, use_inference=False)
    else:
        # Pages fetched but content is sparse — use LLM knowledge
        print("\n🧠  Content sparse — LLM inference from domain knowledge...", end=" ", flush=True)
        prompt = build_prompt(ctx, domain, use_inference=True)

    raw    = call_llm(prompt)
    result = parse_json(raw)

    if not result:
        print(f"❌  Unparseable LLM response")
        return ctx

    print("✅")
    result["_source"] = "scraped" if has_content else "llm_inference"
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scraper.py <domain>")
        print("       python scraper.py <domain> --no-llm")
        sys.exit(1)

    result = asyncio.run(scrape(
        domain  = sys.argv[1],
        use_llm = "--no-llm" not in sys.argv,
    ))

    print("\n📊  RESULT:\n")
    print(json.dumps(result, indent=2))