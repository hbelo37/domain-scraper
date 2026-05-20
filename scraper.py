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
    "Accept":     "text/plain",
    "X-No-Cache": "true",
}


# ─────────────────────────────────────────────────────────────────
# FETCH — 3 layers
# ─────────────────────────────────────────────────────────────────

async def _fetch_jina(client, url):
    try:
        r = await client.get(f"https://r.jina.ai/{url}", headers=JINA_HEADERS, timeout=20)
        if r.status_code == 200 and len(r.text) > 200:
            return r.text, "jina"
    except Exception:
        pass
    return "", ""


async def _fetch_direct(client, url):
    try:
        r = await client.get(url, headers=HEADERS, timeout=12)
        if r.status_code == 200 and len(r.text) > 200:
            return r.text, "direct"
    except Exception:
        pass
    return "", ""


async def _smart_fetch(client, label, url):
    content, method = await _fetch_jina(client, url)
    if content:
        return label, content, method
    content, method = await _fetch_direct(client, url)
    if content:
        return label, content, method
    return label, "", "failed"


async def fetch_pages(domain):
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
        results = await asyncio.gather(*[_smart_fetch(client, k, v) for k, v in urls.items()])

    pages, methods = {}, []
    for label, html, method in results:
        if html:
            pages[label] = html
            methods.append(method)

    primary = "jina" if methods.count("jina") >= methods.count("direct") else "direct"
    return pages, primary


# ─────────────────────────────────────────────────────────────────
# EXTRACTION HELPERS
# ─────────────────────────────────────────────────────────────────

def _is_markdown(html):
    return not html.strip().startswith("<")


def _clean_soup(html):
    s = BeautifulSoup(html, "html.parser")
    for tag in s(["script", "style", "noscript", "svg"]): tag.decompose()
    return s


def _meta(soup):
    out = {}
    for m in soup.find_all("meta"):
        prop    = m.get("property", "") or m.get("name", "")
        content = (m.get("content") or "").strip()
        if not content: continue
        if   prop == "og:site_name":  out["site_name"] = content
        elif prop == "og:title":      out["og_title"]  = content
        elif prop in ("og:description", "description") and "og_desc" not in out:
            out["og_desc"] = content
    return out


def _headings(soup, limit=15):
    noise = {
        "navigation menu", "menu", "skip to content", "saved searches",
        "provide feedback", "site-wide links",
        "search code, repositories, users, issues, pull requests...",
    }
    seen, out = set(), []
    for h in soup.find_all(["h1", "h2", "h3"])[:limit * 3]:
        t = " ".join(h.get_text(" ", strip=True).split())
        if t and 5 < len(t) < 120 and t.lower() not in noise and t not in seen:
            seen.add(t); out.append(t)
        if len(out) >= limit: break
    return out


def _paragraphs(soup, limit=8):
    """Extract paragraphs — longer limit, richer content for LLM."""
    skip = ["cookie", "privacy policy", "terms of service", "all rights reserved", "©"]
    out = []
    for p in soup.find_all("p"):
        t = " ".join(p.get_text(" ", strip=True).split())
        if 40 < len(t) < 600 and not any(s in t.lower() for s in skip):
            out.append(t)
        if len(out) >= limit: break
    return out


def _nav_products(soup):
    """
    Extract product names from nav dropdown items.
    Nav li items follow pattern: "ProductName short description of what it does"
    The description after the product name is the key signal it's a product — not a page link.
    """
    products = []
    skip_starters = {
        "why", "about", "blog", "pricing", "contact", "careers", "login",
        "sign", "get", "try", "free", "demo", "resources", "docs",
        "documentation", "support", "help", "changelog", "community",
        "events", "customers", "partners", "enterprise", "education",
        "terms", "privacy", "sitemap", "newsletter", "subscribe",
        "healthcare", "financial", "manufacturing", "government",
        "startups", "nonprofits", "topics", "trending", "collections",
        "shop", "view", "see", "explore", "find", "read",
    }
    action_verbs = {
        "write", "build", "manage", "automate", "deploy", "track",
        "review", "integrate", "find", "secure", "create", "run",
        "connect", "scale", "monitor", "analyze", "optimize", "send",
        "collaborate", "visualize", "sync", "capture", "record",
        "schedule", "design", "test", "debug", "ship", "launch",
    }
    seen = set()
    for nav in soup.find_all("nav"):
        for li in nav.find_all("li"):
            text = " ".join(li.get_text(" ", strip=True).split())
            words = text.split()
            if not (4 <= len(words) <= 14): continue
            first = words[0].lower()
            if first in skip_starters: continue
            if text.lower().startswith(("view all", "see all", "learn more", "get started")): continue
            # Must contain an action verb — confirms it's describing a product capability
            if not any(v in text.lower() for v in action_verbs): continue
            # Extract product name: words before the first lowercase word
            name_words = []
            for w in words:
                if w[0].isupper() or w[0].isdigit():
                    name_words.append(w)
                else:
                    break
            product_name = " ".join(name_words).strip()
            if len(product_name) < 3 or product_name in seen: continue
            seen.add(product_name)
            tagline = " ".join(words[len(name_words):])
            products.append({"name": product_name, "tagline": tagline})
    return products[:8]


def _logos(soup):
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


def _pain_sections(soup):
    """Extract text from sections that describe problems/challenges."""
    pain_signals = [
        "struggle", "challenge", "problem", "difficult", "pain",
        "without", "manual", "slow", "inefficient", "broken",
        "before", "the challenge", "the problem",
    ]
    out = []
    for elem in soup.find_all(["p", "li", "h2", "h3"]):
        t = " ".join(elem.get_text(" ", strip=True).split())
        if 30 < len(t) < 400 and any(s in t.lower() for s in pain_signals):
            out.append(t)
        if len(out) >= 6: break
    return out


def _value_sections(soup):
    """Extract text from benefit/outcome sections."""
    value_signals = [
        "increase", "reduce", "save", "improve", "faster", "easier",
        "eliminate", "automate", "streamline", "10x", "2x", "50%",
        "hours", "revenue", "pipeline", "convert", "close", "grow",
    ]
    out = []
    for elem in soup.find_all(["p", "li", "h2", "h3"]):
        t = " ".join(elem.get_text(" ", strip=True).split())
        if 20 < len(t) < 300 and any(s in t.lower() for s in value_signals):
            out.append(t)
        if len(out) >= 6: break
    return out


KNOWN_TOOLS = [
    "Salesforce", "HubSpot", "Slack", "Stripe", "Zapier", "Jira",
    "Asana", "Monday", "Notion", "Datadog", "Twilio", "Okta",
    "AWS", "Azure", "Google Cloud", "Marketo", "Outreach",
    "Segment", "Zendesk", "Intercom", "Gong", "Salesloft",
    "Shopify", "QuickBooks", "Zoom", "Teams", "Figma", "Snowflake",
]

def _integrations(text):
    return [t for t in KNOWN_TOOLS if t.lower() in text.lower()]


def _extract_from_markdown(text):
    """Extract structured signals from jina markdown."""
    lines = text.split("\n")
    headings, paragraphs = [], []
    for line in lines:
        line = line.strip()
        if not line: continue
        if re.match(r'^#{1,3}\s', line):
            h = re.sub(r'^#+\s*', '', line).strip()
            if h and 5 < len(h) < 120:
                headings.append(h)
        elif len(line) > 50 and not line.startswith("[") and not line.startswith("!"):
            skip = ["cookie", "privacy", "terms", "©", "all rights reserved"]
            if not any(s in line.lower() for s in skip):
                paragraphs.append(line[:500])
    return {"headings": headings[:20], "paragraphs": paragraphs[:10]}


def build_context(pages):
    ctx = dict(
        site_name="", og_title="", og_desc="",
        headings=[], paragraphs=[], nav_products=[],
        nav_items=[], customers=[], integrations=[],
        pain_signals=[], value_signals=[],
    )

    home = pages.get("home", "")
    if home:
        if _is_markdown(home):
            extracted = _extract_from_markdown(home)
            ctx["headings"]   = extracted["headings"]
            ctx["paragraphs"] = extracted["paragraphs"]
            if extracted["headings"]:
                ctx["og_title"] = extracted["headings"][0]
        else:
            s = _clean_soup(home); meta = _meta(s)
            ctx.update({
                "site_name":    meta.get("site_name", ""),
                "og_title":     meta.get("og_title",  ""),
                "og_desc":      meta.get("og_desc",   ""),
                "nav_products": _nav_products(s),   # ← structured product extraction
                "headings":     _headings(s),
                "paragraphs":   _paragraphs(s),
                "customers":    _logos(s),
                "pain_signals": _pain_sections(s),
                "value_signals":_value_sections(s),
            })
        ctx["integrations"] = _integrations(home)

    seen_h = set(ctx["headings"])
    seen_c = set(ctx["customers"])

    for key in ("about", "products", "platform", "solutions", "customers", "integrations"):
        html = pages.get(key, "")
        if not html: continue

        if _is_markdown(html):
            ex = _extract_from_markdown(html)
            new_h = [h for h in ex["headings"] if h not in seen_h]
            ctx["headings"]   += new_h; seen_h.update(new_h)
            ctx["paragraphs"] += ex["paragraphs"][:4]
        else:
            s = _clean_soup(html)
            new_h = [h for h in _headings(s, 10) if h not in seen_h]
            new_c = [c for c in _logos(s)         if c not in seen_c]
            ctx["headings"]    += new_h; seen_h.update(new_h)
            ctx["paragraphs"]  += _paragraphs(s, 4)
            ctx["customers"]   += new_c; seen_c.update(new_c)
            ctx["pain_signals"]  += _pain_sections(s)
            ctx["value_signals"] += _value_sections(s)
            # Pick up more nav products from subpages
            extra_np = _nav_products(s)
            existing_names = {p["name"] for p in ctx["nav_products"]}
            ctx["nav_products"] += [p for p in extra_np if p["name"] not in existing_names]

        ctx["integrations"] += [i for i in _integrations(html) if i not in ctx["integrations"]]

    ctx["headings"]      = ctx["headings"][:20]
    ctx["paragraphs"]    = ctx["paragraphs"][:12]
    ctx["customers"]     = ctx["customers"][:20]
    ctx["integrations"]  = ctx["integrations"][:20]
    ctx["pain_signals"]  = list(dict.fromkeys(ctx["pain_signals"]))[:8]
    ctx["value_signals"] = list(dict.fromkeys(ctx["value_signals"]))[:8]
    ctx["nav_products"]  = ctx["nav_products"][:8]
    return ctx


# ─────────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """\
You are a B2B market intelligence analyst. Extract structured business intelligence from the website data below.

STRICT RULES — read before filling any field:
- Use ONLY information present in the data. Do not invent.
- Return "" or [] only when there is genuinely no evidence.
- Every description must be specific to THIS company — never generic.

━━━ WEBSITE DATA ━━━

SITE NAME:      {site_name}
OG TITLE:       {og_title}
OG DESCRIPTION: {og_desc}

PRODUCTS FOUND IN NAV (name + what it does):
{nav_products}

PAGE HEADINGS:
{headings}

KEY PARAGRAPHS:
{paragraphs}

PAIN/PROBLEM SIGNALS FROM PAGE:
{pain_signals}

VALUE/BENEFIT SIGNALS FROM PAGE:
{value_signals}

CUSTOMER LOGOS:
{customers}

INTEGRATIONS FOUND:
{integrations}

━━━ OUTPUT ━━━

Return ONLY valid JSON, no markdown, no explanation:

{{
  "company_name": "Brand name only. Strip everything after | or –. Never include taglines.",

  "description": "2-3 sentences. Must answer: (1) what they build/provide, (2) who uses it, (3) what outcome it delivers. Write like a business analyst, not a marketer. BAD: 'A platform that helps teams work better.' GOOD: 'GitHub is a cloud-based platform for version control and software collaboration used by 100M+ developers and enterprises. Teams use it to host code, review changes, automate CI/CD pipelines, and ship software faster.'",

  "icp": {{
    "personas":     ["Real job titles only — e.g. VP of Sales, DevOps Engineer, Head of Marketing"],
    "company_size": ["Use standard tiers: Startup, SMB, Mid-Market, Enterprise"],
    "industries":   ["Specific verticals — e.g. B2B SaaS, Financial Services, E-commerce, Healthcare"],
    "summary":      "3 sentences as flowing prose. Sentence 1: what type of company they target (size + industry). Sentence 2: the key roles/personas that use the product. Sentence 3: the business context or trigger that makes them a buyer. BAD: 'Targets companies of all sizes.' GOOD: 'Primarily serves mid-market and enterprise B2B software companies with 100–5000 employees, particularly in financial services, healthcare, and technology. The primary buyers are VP of Sales, Revenue Operations Managers, and Sales Development Representatives. Companies typically adopt the platform when scaling outbound sales beyond what spreadsheets and manual research can support.'"
  }},

  "offerings": [
    {{
      "name":              "Exact product/module name visible in nav or headings. Never generic.",
      "category":          "1–3 word category: e.g. Sales Intelligence, CI/CD Automation, CRM",
      "description":       "1–2 sentences specific to this product. What it does, not what the company does. BAD: 'A tool that helps users.' GOOD: 'Copilot is an AI pair programmer that suggests code completions, writes functions from comments, and explains unfamiliar code inline in the editor.'",
      "value_proposition": "1 sentence. A specific measurable or tangible outcome. Must contain a verb and a result. BAD: 'Improves productivity.' GOOD: 'Reduces time spent on code review by surfacing relevant context and automating boilerplate, so engineers spend more time on logic that matters.'",
      "pain_points":       ["Each point is a specific problem this product solves. BAD: 'Manual processes.' GOOD: 'Engineers waste hours context-switching between documentation and their editor.'"]
    }}
  ],

  "value_proposition": "1–2 sentences for the whole company. Specific outcome + for whom. BAD: 'We help teams move faster.' GOOD: 'GitHub gives development teams a unified platform to write, review, secure, and deploy code — cutting the time from idea to production while keeping every change auditable and reversible.'",

  "partners":           ["Integration or technology partner names"],
  "existing_customers": ["Company names only from logo wall — no individuals"],
  "pricing_tier":       "One of: Free, Freemium, Subscription, Tiered, Contact Sales, Usage-Based"
}}

━━━ OFFERINGS RULES ━━━

USE the PRODUCTS FOUND IN NAV section as your primary source for offering names.
Each item there is in format: "Product Name | what it does" — use the name directly.

VALID offerings: Named products with a dedicated nav entry or page section.
  GOOD: "Copilot", "Marketing Hub", "SalesOS", "Actions", "Advanced Security"
  BAD:  "Careers", "Blog", "Get started", "Platform", "AI", "By Team Size", "Solutions"

If NAV PRODUCTS section has items — use those. Do not invent offerings not present in the data.
Max 6 offerings. If fewer than 6 are clearly identifiable, return only those.
"""

INFERENCE_PROMPT = """\
You are a B2B market intelligence analyst with deep knowledge of technology companies.

Domain: {domain}

Website content could not be extracted. Use your knowledge of this company.
Only include what you are confident about — return "" or [] for uncertain fields.

Return ONLY valid JSON, no markdown:

{{
  "company_name": "Official brand name",
  "description": "2-3 specific sentences: what they build, who uses it, what outcome it delivers",
  "icp": {{
    "personas":     ["Real job titles"],
    "company_size": ["Startup/SMB/Mid-Market/Enterprise"],
    "industries":   ["Specific verticals"],
    "summary":      "3 sentences: target company profile, key buyer roles, buying trigger"
  }},
  "offerings": [
    {{
      "name": "Product name",
      "category": "Category",
      "description": "1-2 specific sentences about what this product does",
      "value_proposition": "Specific outcome it delivers",
      "pain_points": ["Specific problem it solves"]
    }}
  ],
  "value_proposition": "Specific outcome the company delivers and for whom",
  "partners": ["Known integration partners"],
  "existing_customers": [],
  "pricing_tier": "Free/Freemium/Subscription/Tiered/Contact Sales/Usage-Based"
}}
"""


def _fmt(lst):
    return "\n".join(f"  • {x}" for x in lst) if lst else "  (none found)"

def _fmt_nav_products(products):
    if not products:
        return "  (none found)"
    return "\n".join(f"  • {p['name']} | {p['tagline']}" for p in products)


def build_prompt(ctx, domain, use_inference=False):
    if use_inference:
        return INFERENCE_PROMPT.format(domain=domain)

    return EXTRACTION_PROMPT.format(
        site_name     = ctx["site_name"]     or "(not found)",
        og_title      = ctx["og_title"]      or "(not found)",
        og_desc       = ctx["og_desc"]       or "(not found)",
        nav_products  = _fmt_nav_products(ctx.get("nav_products", [])),
        headings      = _fmt(ctx["headings"]),
        paragraphs    = _fmt(ctx["paragraphs"]),
        pain_signals  = _fmt(ctx.get("pain_signals", [])),
        value_signals = _fmt(ctx.get("value_signals", [])),
        customers     = _fmt(ctx["customers"]),
        integrations  = _fmt(ctx["integrations"]),
    )


# ─────────────────────────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────────────────────────

def call_llm(prompt):
    from groq import Groq
    try:
        res = Groq(api_key=GROQ_API_KEY).chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2500,
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        err = str(e)
        if "401" in err or "invalid_api_key" in err:
            raise ValueError("Invalid Groq API key")
        elif "429" in err:
            raise ValueError("Rate limited — wait 60s")
        raise ValueError(str(e)[:100])


def parse_json(text):
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try: return json.loads(m.group())
            except: pass
    return {}


def _has_content(ctx):
    return bool(
        ctx.get("og_desc") or
        len(ctx.get("headings", [])) > 2 or
        len(ctx.get("paragraphs", [])) > 1 or
        len(ctx.get("nav_products", [])) > 0
    )


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

async def scrape(domain, use_llm=True):
    domain = domain.replace("https://", "").replace("http://", "").rstrip("/")

    print(f"\n🚀  {domain}")
    print("─" * 55)

    print("📥  Fetching pages...", end=" ", flush=True)
    pages, fetch_method = await fetch_pages(domain)

    if not pages:
        print("❌  No pages fetched")
        if use_llm and GROQ_API_KEY:
            print("🧠  LLM inference fallback...")
            result = parse_json(call_llm(build_prompt({}, domain, use_inference=True)))
            if result:
                result["_source"] = "llm_inference"
                return result
        return {}

    print(f"✅  {list(pages.keys())} via {fetch_method}")

    ctx = build_context(pages)
    has_content = _has_content(ctx)

    print(f"\n📐  Signals:")
    print(f"    site_name    : {ctx['site_name'] or '—'}")
    print(f"    og_desc      : {(ctx['og_desc'][:70] + '…') if ctx['og_desc'] else '—'}")
    print(f"    nav_products : {[p['name'] for p in ctx.get('nav_products',[])]}")
    print(f"    headings     : {len(ctx['headings'])}")
    print(f"    paragraphs   : {len(ctx['paragraphs'])}")
    print(f"    pain_signals : {len(ctx.get('pain_signals',[]))}")
    print(f"    value_signals: {len(ctx.get('value_signals',[]))}")
    print(f"    customers    : {len(ctx['customers'])}")
    print(f"    integrations : {ctx['integrations']}")

    if not use_llm or not GROQ_API_KEY:
        return ctx

    use_inference = not has_content
    mode = "inference" if use_inference else "extraction"
    print(f"\n🤖  LLM {mode}...", end=" ", flush=True)

    result = parse_json(call_llm(build_prompt(ctx, domain, use_inference=use_inference)))

    if not result:
        print("❌  Unparseable response")
        return ctx

    print("✅")
    result["_source"] = "llm_inference" if use_inference else "scraped"
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