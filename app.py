# linkedin_company_finder.py
from __future__ import annotations

import concurrent.futures
import json
import os
import re
import time
from typing import Any, Dict, List, Tuple
from urllib.parse import urlsplit

import pandas as pd
import requests
import streamlit as st
from openai import OpenAI, OpenAIError
from requests.exceptions import HTTPError, RequestException, Timeout

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
SCRAPINGDOG_URL   = "https://api.scrapingdog.com/linkedin"
SCRAPINGDOG_API_KEY: str | None = None

OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL      = "gpt-4o-mini"
LLM_TIMEOUT       = 20
EXPERIENCE_LIMIT  = 3
SCRAPINGDOG_DELAY = 1.1   # seconds between calls

client: OpenAI | None = None           # set after user types key

# ─────────────────────────────────────────────────────────────────────────────
# Helpers – run with timeout
# ─────────────────────────────────────────────────────────────────────────────
def _with_timeout(fn, seconds: int):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(fn)
        try:
            return fut.result(timeout=seconds)
        except concurrent.futures.TimeoutError:
            fut.cancel()
            raise Timeout(f"LLM call exceeded {seconds}s")

# ─────────────────────────────────────────────────────────────────────────────
# LLM helper
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You must output exactly ONE item:

• a LinkedIn company URL that appears verbatim in the records, with no query
  parameters or fragments, e.g. https://www.linkedin.com/company/example
    –OR–
• the single word: None

No other text. The company must clearly work in web-design, web-development,
software development, SaaS, or an adjacent tech field.
""".strip()

def build_user_prompt(exps: List[Dict[str, Any]]) -> str:
    """Return the raw JSON-string of the 3 experience objects."""
    return json.dumps(exps, ensure_ascii=False)  # <-- no pretty-printing

def call_llm(user_prompt: str) -> str:
    if client is None:
        raise OpenAIError("OpenAI client not initialised.")
    r = client.chat.completions.create(
        model       = OPENAI_MODEL,
        messages    = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature = 0.0,
        max_tokens  = 32,
    )
    return r.choices[0].message.content.strip()

def llm_pick_company_url(exps: List[Dict[str, Any]]) -> Tuple[str | None, str, str]:
    """
    Returns:
        cleaned_url | None,
        raw_reply,
        user_prompt (the JSON string we sent)
    """
    if not exps:
        return None, "(no experiences)", "[]"

    user_prompt = build_user_prompt(exps)

    try:
        raw_reply = _with_timeout(lambda: call_llm(user_prompt), LLM_TIMEOUT)
    except Exception as exc:
        return None, f"(OpenAI error: {exc})", user_prompt

    cleaned = (
        raw_reply.split("?", 1)[0].split("#", 1)[0].rstrip("/").strip()
        if raw_reply.lower() != "none" else None
    )
    return cleaned, raw_reply, user_prompt

# ─────────────────────────────────────────────────────────────────────────────
# ScrapingDog helper
# ─────────────────────────────────────────────────────────────────────────────
def call_scrapingdog(api_key: str, link_type: str, link_id: str) -> Dict[str, Any]:
    params = {
        "api_key": api_key,
        "type"   : link_type,
        "linkId" : link_id,
        "private": "false",
    }
    try:
        resp = requests.get(SCRAPINGDOG_URL, params=params, timeout=30)
    except (Timeout, RequestException) as exc:
        raise RuntimeError(f"ScrapingDog network error: {exc}")

    if resp.status_code == 401:
        raise RuntimeError("ScrapingDog 401 – bad key")
    if resp.status_code == 429:
        raise RuntimeError("ScrapingDog 429 – rate limit")

    resp.raise_for_status()
    data = resp.json()
    return data[0] if isinstance(data, list) and data else data

# ─────────────────────────────────────────────────────────────────────────────
# Website extraction
# ─────────────────────────────────────────────────────────────────────────────
_WEBSITE_KEYS = ("website", "website_url", "url", "homepage", "site")

def _looks_like_site(s: str | None) -> bool:
    if not isinstance(s, str):
        return False
    s = s.strip()
    return (
        s.startswith(("http://", "https://", "www."))
        and not any(dom in s for dom in ("linkedin.com", "licdn.com"))
        and not re.search(r"\.(png|jpe?g|gif|svg)$", s, re.I)
    )

def _normalise_site(s: str) -> str:
    s = s.strip().rstrip("/")
    if s.startswith(("http://", "https://")):
        return s
    if s.startswith("www."):
        return f"https://{s}"
    return s

def extract_company_website(raw: Any) -> str | None:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            pass

    if isinstance(raw, dict):
        for k in _WEBSITE_KEYS:
            v = raw.get(k)
            if _looks_like_site(v):
                return _normalise_site(v)

    def _crawl(o):
        if isinstance(o, str) and _looks_like_site(o):
            return _normalise_site(o)
        if isinstance(o, dict):
            for v in o.values():
                r = _crawl(v)
                if r:
                    return r
        if isinstance(o, list):
            for v in o:
                r = _crawl(v)
                if r:
                    return r
        return None

    return _crawl(raw)

# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline for one profile
# ─────────────────────────────────────────────────────────────────────────────
def serialise(obj: Any) -> str:
    if isinstance(obj, (dict, list)):
        return json.dumps(obj, ensure_ascii=False)
    return str(obj)

def get_company_info(profile_url: str, sd_key: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"profile_url": profile_url}

    try:
        parts = [p for p in urlsplit(profile_url).path.split("/") if p]
        if len(parts) < 2 or parts[0] != "in":
            raise ValueError("Invalid profile URL")
        profile_id = parts[1]

        person = call_scrapingdog(sd_key, "profile", profile_id)
        time.sleep(SCRAPINGDOG_DELAY)

    except Exception as exc:
        out["error"] = f"Profile scrape error: {exc}"
        return out

    exps = person.get("experience", [])[:EXPERIENCE_LIMIT]
    url_clean, raw_reply, user_prompt = llm_pick_company_url(exps)

    out.update({
        "experiences_json"    : serialise(exps),
        "llm_system_prompt"   : SYSTEM_PROMPT,
        "llm_user_prompt"     : user_prompt,
        "llm_response_raw"    : raw_reply,
        "cleaned_llm_url"     : url_clean or "",
    })

    if not url_clean:
        out["error"] = "LLM returned None or unusable URL"
        return out

    def _clean(u: str) -> str:
        return u.split("?", 1)[0].split("#", 1)[0].rstrip("/").lower()

    chosen_exp = next(
        (e for e in exps if _clean(str(e.get("company_url", ""))) == url_clean.lower()),
        None
    )
    out["chosen_experience_json"] = serialise(chosen_exp) if chosen_exp else ""

    try:
        segs = [p for p in urlsplit(url_clean).path.split("/") if p]
        cid  = segs[segs.index("company") + 1]
        comp = call_scrapingdog(sd_key, "company", cid)
        time.sleep(SCRAPINGDOG_DELAY)

        site = extract_company_website(comp)
        out.update({
            "company_linkedin"   : url_clean,
            "company_json"       : serialise(comp),
            "company_website"    : site or "",
            "error"              : None if site else "Website not found in company JSON",
        })

    except Exception as exc:
        out.update({
            "company_json"    : "{}",
            "company_website" : "",
            "error"           : f"Company scrape error: {exc}",
        })

    return out

# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="🔗 LinkedIn → Company Website Finder", layout="wide")

with st.sidebar:
    st.markdown("## 🔑 API Keys")
    SCRAPINGDOG_API_KEY = st.text_input("ScrapingDog API Key", type="password", value=SCRAPINGDOG_API_KEY or "")
    OPENAI_API_KEY      = st.text_input("OpenAI API Key",      type="password", value=OPENAI_API_KEY or "")
    if OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY)
    st.markdown("---")
    st.info("Enter keys, then choose a mode.")

if not SCRAPINGDOG_API_KEY or client is None:
    st.stop()

single_tab, batch_tab = st.tabs(["🔍 Single Profile", "🗃️ Batch CSV"])

# ── Single profile ───────────────────────────────────────────────────────────
with single_tab:
    st.header("Single Profile Lookup")
    p_url = st.text_input("LinkedIn Profile URL", placeholder="https://www.linkedin.com/in/username")

    if st.button("Fetch"):
        if not p_url:
            st.error("Please enter a URL.")
        else:
            with st.spinner("Processing …"):
                info = get_company_info(p_url.strip(), SCRAPINGDOG_API_KEY)
            if info.get("error"):
                st.error(info["error"])
            st.json(info)

# ── Batch CSV ────────────────────────────────────────────────────────────────
with batch_tab:
    st.header("Batch CSV Lookup")
    up_file = st.file_uploader(
        "Upload CSV",
        type="csv",
        help="Select the column that contains the LinkedIn profile URLs."
    )

    if up_file is not None:
        try:
            df_in = pd.read_csv(up_file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        if df_in.empty:
            st.error("The uploaded CSV is empty.")
            st.stop()

        candidates = [c for c in df_in.columns if re.search(r"url|linkedin", c, re.I)]
        default_idx = df_in.columns.get_loc(candidates[0]) if candidates else 0

        url_col = st.selectbox(
            "Column containing LinkedIn profile URLs",
            df_in.columns,
            index=default_idx
        )

        max_rows = st.number_input(
            "Rows to process",
            min_value=1,
            max_value=len(df_in),
            value=min(1000, len(df_in)),
        )

        if st.button("Run Batch"):
            results: List[Dict[str, Any]] = []
            progress = st.progress(0.0)
            total = int(max_rows)
            start = time.time()

            with st.spinner(f"Processing {total} profiles …"):
                for i, url in enumerate(df_in[url_col].head(total), start=1):
                    info = get_company_info(str(url).strip(), SCRAPINGDOG_API_KEY)
                    results.append(info)
                    progress.progress(i / total)
                    time.sleep(0.2)  # small delay

            progress.empty()
            elapsed = time.time() - start

            df_full = pd.DataFrame(results)
            successes = df_full[df_full["error"].isna()]

            st.success(
                f"Finished in {elapsed:.1f}s – "
                f"{len(successes)} successes, {len(df_full) - len(successes)} errors."
            )

            if not successes.empty:
                st.download_button(
                    "⬇️ Download Successes CSV",
                    data=successes.to_csv(index=False).encode("utf-8"),
                    file_name="linkedin_company_websites.csv",
                    mime="text/csv",
                )

            st.download_button(
                "⬇️ Download Detailed Debug CSV",
                data=df_full.to_csv(index=False).encode("utf-8"),
                file_name="linkedin_debug_full.csv",
                mime="text/csv",
            )

            st.markdown("#### Debug preview (first 10 rows)")
            st.dataframe(df_full.head(10), use_container_width=True)
