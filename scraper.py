#!/usr/bin/env python3
"""
Company Intelligence Scraper
Single file. One LLM call. Clean output.

Install: pip install httpx beautifulsoup4 groq
Usage:
    python scraper.py hubspot.com
    python scraper.py hubspot.com --no-llm
"""

import asyncio, sys, json, re, os
import httpx
from bs4 import BeautifulSoup

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────
# FETCH
# ─────────────────────────────────────────────────────────────────

async def _fetch(client: httpx.AsyncClient, label: str, url: str):
    try:
        r = await client.get(url, timeout=12)
        if r.status_code == 200:
            return label, r.text
    except Exception:
        pass
    return label, ""


async def fetch_pages(domain: str) -> dict:
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
    async with httpx.AsyncClient(headers=HEADERS, follow_redirects=True) as client:
        results = await asyncio.gather(*[_fetch(client, k, v) for k, v in urls.items()])
    return {label: html for label, html in results if html}


# ─────────────────────────────────────────────────────────────────
# PHASE 1 — STRUCTURED EXTRACTION (no LLM)
# ─────────────────────────────────────────────────────────────────

def _clean_soup(html: str) -> BeautifulSoup:
    s = BeautifulSoup(html, "html.parser")
    for tag in s(["script", "style", "noscript", "svg"]):
        tag.decompose()
    return s


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

def _integrations(html: str) -> list:
    return [t for t in KNOWN_TOOLS if t.lower() in html.lower()]


def build_context(pages: dict) -> dict:
    ctx = dict(site_name="", og_title="", og_desc="",
               headings=[], paragraphs=[], nav_items=[],
               customers=[], integrations=[])

    home = pages.get("home", "")
    if home:
        s    = _clean_soup(home)
        meta = _meta(s)
        ctx.update({
            "site_name":    meta.get("site_name", ""),
            "og_title":     meta.get("og_title",  ""),
            "og_desc":      meta.get("og_desc",   ""),
            "nav_items":    _nav(s),
            "headings":     _headings(s),
            "paragraphs":   _paragraphs(s),
            "customers":    _logos(s),
            "integrations": _integrations(home),
        })

    seen_h = set(ctx["headings"])
    seen_c = set(ctx["customers"])

    for key in ("about", "products", "platform", "solutions", "customers", "integrations"):
        html = pages.get(key, "")
        if not html:
            continue
        s     = _clean_soup(html)
        new_h = [h for h in _headings(s, 8)  if h not in seen_h]
        new_c = [c for c in _logos(s)         if c not in seen_c]
        ctx["headings"]     += new_h;  seen_h.update(new_h)
        ctx["paragraphs"]   += _paragraphs(s, 3)
        ctx["customers"]    += new_c;  seen_c.update(new_c)
        ctx["integrations"] += [i for i in _integrations(html) if i not in ctx["integrations"]]

    ctx["headings"]     = ctx["headings"][:20]
    ctx["paragraphs"]   = ctx["paragraphs"][:10]
    ctx["customers"]    = ctx["customers"][:20]
    ctx["integrations"] = ctx["integrations"][:20]
    return ctx


# ─────────────────────────────────────────────────────────────────
# PHASE 2 — SINGLE LLM CALL
# ─────────────────────────────────────────────────────────────────
# One call extracts everything. This is faster, cheaper, and more
# consistent than calling the LLM once per product.

PROMPT = """\
You are a B2B market intelligence analyst.

Use ONLY the data below. Do not invent anything not present.
Return "" or [] when evidence is missing.

━━━ WEBSITE DATA ━━━

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

━━━ OUTPUT ━━━

Return ONLY valid JSON, no markdown, no explanation:

{{
  "company_name": "Brand name only — strip taglines e.g. 'HubSpot' not 'HubSpot | CRM'",

  "description": "2 sentences max. What they do and who they serve. No fluff.",

  "icp": {{
    "personas":     ["Job title e.g. VP Sales", "Marketing Manager"],
    "company_size": ["SMB", "Mid-Market", "Enterprise"],
    "industries":   ["SaaS", "Financial Services"]
  }},

  "offerings": [
    {{
      "name":               "Exact product name e.g. Marketing Hub",
      "category":           "Short category e.g. Marketing Automation",
      "description":        "1 sentence: what this product does",
      "value_proposition":  "1 sentence: the core benefit it delivers",
      "pain_points":        ["Pain point it solves"]
    }}
  ],

  "value_proposition": "1 sentence for the whole company",

  "partners":            ["Integration or partner name"],

  "existing_customers":  ["Customer name from logo wall"],

  "pricing_tier":        "e.g. Freemium, Mid-Market, Enterprise, Contact Sales"
}}

━━━ STRICT RULES FOR OFFERINGS ━━━

VALID products: Named software modules or platforms this company sells.
  Examples: "Marketing Hub", "Sales Hub", "Copilot", "SalesOS", "CRM", "Analytics"

INVALID — do NOT include these as products:
  - Nav labels:     "By Team Size", "Careers", "Resources", "Blog"
  - CTAs:           "Get started", "Book a demo", "Build pipeline", "Automate marketing"
  - Generic words:  "Artificial Intelligence", "Platform", "Solutions", "Features"
  - Company pages:  "About", "Pricing", "Contact", "Partners"

If you cannot find clearly named products, return offerings: []
Max 6 offerings.
"""

def _fmt(lst: list) -> str:
    return "\n".join(f"  • {x}" for x in lst) if lst else "  (none found)"


def build_prompt(ctx: dict) -> str:
    return PROMPT.format(
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
            model       = MODEL,
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0.1,
            max_tokens  = 2000,
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        err = str(e)
        if "401" in err or "invalid_api_key" in err:
            print("\n❌  Invalid Groq API key → https://console.groq.com/keys")
        elif "429" in err:
            print("\n❌  Rate limited — wait 60s and retry")
        else:
            print(f"\n❌  Groq error: {err[:100]}")
        return ""


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


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

async def scrape(domain: str, use_llm: bool = True) -> dict:
    domain = domain.replace("https://", "").replace("http://", "").rstrip("/")

    print(f"\n🚀  {domain}")
    print("─" * 55)

    # Phase 1
    print("📥  Fetching pages...", end=" ", flush=True)
    pages = await fetch_pages(domain)

    if not pages:
        print("❌\n\n  Site blocked (Cloudflare Bot Management).")
        print("  These sites require Playwright or a residential proxy.")
        return {}

    print(f"✅  {list(pages.keys())}")

    ctx = build_context(pages)
    print(f"\n📐  Phase 1 signals:")
    print(f"    site_name    : {ctx['site_name'] or '—'}")
    print(f"    og_desc      : {(ctx['og_desc'][:70] + '…') if ctx['og_desc'] else '—'}")
    print(f"    nav_items    : {len(ctx['nav_items'])}")
    print(f"    headings     : {len(ctx['headings'])}")
    print(f"    customers    : {len(ctx['customers'])}")
    print(f"    integrations : {ctx['integrations']}")

    if not use_llm or not GROQ_API_KEY:
        print("\n⚠️   Phase 1 only (set GROQ_API_KEY to enable LLM)")
        return ctx

    # Phase 2 — single LLM call
    print("\n🤖  Phase 2: LLM extraction...", end=" ", flush=True)
    raw    = call_llm(build_prompt(ctx))

    if not raw:
        return ctx

    result = parse_json(raw)

    if not result:
        print(f"❌  Unparseable response\n    Raw: {raw[:200]}")
        return ctx

    print("✅")
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