# linkedin_experience_scraper.py
# Upload CSV → choose URL column & slice → scrape LinkedIn experiences →
# GPT-4.1* labels every employer as web-design-agency yes/no → download enriched CSV

from __future__ import annotations

import concurrent.futures, json, re, time, requests
from pathlib import Path
from typing import Any

import openai
import pandas as pd
import streamlit as st
from requests.exceptions import HTTPError, Timeout, RequestException

# ─────────────────────────────────────────────────────────────────────────────
# Default prompt (shown as initial value in the sidebar text-area)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_SYSTEM_PROMPT = """You will receive a JSON array of job-experience objects scraped from LinkedIn.

Task
────
• Evaluate every employer and decide **at most one** company that is clearly a web-design / web-development agency (or broader digital agency that sells website work also).
• If such a strongest match exists, mark that single company with "isWebDesignAgency": "yes" and mark **all others** "no".
• If none qualify, mark **all** companies "no".
• ↺ Exactly one "yes" *or* zero "yes"; **never more than one**.

Output
──────
Return a JSON array **in the same order you received**. Each element must contain only:
{
  "companyName":        "<string>",
  "companyLinkedIn":    "<url or empty string>",
  "isWebDesignAgency":  "yes" | "no",
  "reason":             "<1–3 short lines explaining why>"
}

Provide no extra keys and no commentary before or after the array."""
# (Triple-quoted so embedded line-breaks become real “\n” characters.)

# ─────────────────────────────────────────────────────────────────────────────
# UI – sidebar for credentials, model & prompt
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("🔑 Functionalities")
    sd_key_input = st.text_input(
        "ScrapingDog API key",
        type="password",
        value=st.session_state.get("scrapingdog_key", ""),
    )
    oa_key_input = st.text_input(
        "OpenAI API key",
        type="password",
        value=st.session_state.get("openai_key", ""),
    )

    # st.divider()
    # st.subheader("⚙️ OpenAI model")
    model_options = ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"]
    default_model = st.session_state.get("openai_model", "gpt-4.1-nano")
    model_choice = st.selectbox(
        "Choose model",
        model_options,
        index=model_options.index(default_model),
    )

    # st.divider()
    # st.subheader("📝 System prompt")
    with st.popover("📝 System prompt (click to edit)", use_container_width= True):
        prompt_input = st.text_area(
            "Edit or paste your system prompt (new lines are preserved)",
            value=st.session_state.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
            height=320,
            key="system_prompt_input",
        )

    if st.button("💾 Save credentials", type="primary", use_container_width= True):
        st.session_state["scrapingdog_key"] = sd_key_input.strip()
        st.session_state["openai_key"] = oa_key_input.strip()
        st.session_state["openai_model"] = model_choice
        st.session_state["system_prompt"] = prompt_input
        st.success("Saved! You can close the sidebar.")

# Block remainder of app until mandatory keys exist
if not st.session_state.get("scrapingdog_key") or not st.session_state.get("openai_key"):
    st.warning("Enter both API keys in the sidebar, then press **Save credentials**.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS (pulled from session_state)
# ─────────────────────────────────────────────────────────────────────────────
SCRAPINGDOG_API_KEY: str = st.session_state["scrapingdog_key"]
SCRAPINGDOG_URL = "https://api.scrapingdog.com/linkedin/"
THREADS = 6
PER_REQUEST_DELAY = 1.1
TIMEOUT_SECONDS = 25

OPENAI_API_KEY: str = st.session_state["openai_key"]
OPENAI_MODEL: str = st.session_state["openai_model"]
OPENAI_TEMP = 0
OPENAI_TIMEOUT = 45

openai.api_key = OPENAI_API_KEY

# Use the user-supplied (or default) prompt
SYSTEM_PROMPT = st.session_state.get("system_prompt", DEFAULT_SYSTEM_PROMPT)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
_linkid_rx = re.compile(r"linkedin\.com\/in\/([^\/\?\s]+)", re.I)


def extract_linkid(url: str) -> str:
    m = _linkid_rx.search(url)
    if not m:
        raise ValueError("Couldn’t detect linkId in URL")
    return m.group(1).strip()


def call_llm(profile_payload: dict[str, Any]) -> str:
    """Send payload to OpenAI and return JSON string (or 'LLM_ERROR: …')."""
    try:
        resp = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(profile_payload, ensure_ascii=False),
                },
            ],
            temperature=OPENAI_TEMP,
            timeout=OPENAI_TIMEOUT,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM_ERROR: {e!s}"


def scrape_and_classify(linkedin_url: str) -> dict[str, str]:
    """Scrape LinkedIn → run LLM → return dict for DataFrame row."""
    link_id = extract_linkid(linkedin_url)
    params = {"api_key": SCRAPINGDOG_API_KEY, "type": "profile", "linkId": link_id}

    try:
        r = requests.get(SCRAPINGDOG_URL, params=params, timeout=TIMEOUT_SECONDS)
        r.raise_for_status()
        raw = r.json()
        raw_profile = raw[0] if isinstance(raw, list) and raw else raw
        experiences = raw_profile.get("experience", [])
        description = raw_profile.get("description", "")
        profile_input = {"experiences": experiences, "description": description}
    except (HTTPError, Timeout, RequestException, ValueError) as exc:
        experiences = {"error": str(exc)}
    finally:
        time.sleep(PER_REQUEST_DELAY)

    llm_result = (
        call_llm(profile_input)
        if isinstance(experiences, list)
        else "SKIPPED_LLM_DUE_TO_SCRAPE_ERROR"
    )

    agency_url = ""
    try:
        parsed = json.loads(llm_result) if llm_result.strip().startswith("[") else []
        if isinstance(parsed, list):
            for item in parsed:
                if item.get("isWebDesignAgency", "").lower() == "yes":
                    agency_url = item.get("companyLinkedIn", "") or ""
                    break
    except json.JSONDecodeError:
        pass

    slug = ""
    if agency_url:
        m = re.search(r"https://www\.linkedin\.com/company/([^/?]+)", agency_url)
        if m:
            agency_url = f"https://www.linkedin.com/company/{m.group(1)}"
            slug = m.group(1)

    company_website = ""
    if slug:
        try:
            params2 = {
                "api_key": SCRAPINGDOG_API_KEY,
                "type": "company",
                "linkId": slug,
                "private": "false",
            }
            r2 = requests.get(SCRAPINGDOG_URL, params=params2, timeout=TIMEOUT_SECONDS)
            r2.raise_for_status()
            comp_json = r2.json()
            comp_obj = (
                comp_json[0] if isinstance(comp_json, list) and comp_json else comp_json
            )
            company_website = comp_obj.get("website", "") or ""
        except (HTTPError, Timeout, RequestException, ValueError):
            company_website = ""
        finally:
            time.sleep(PER_REQUEST_DELAY)

    return {
        "profile_linkedin_url": linkedin_url,
        "experiences": json.dumps(experiences, ensure_ascii=False),
        "web_design_agencies": llm_result,
        "web_design_company_linkedin_url": agency_url,
        "company_website": company_website,
    }


def process_rows(df: pd.DataFrame, url_col: str, start: int, end: int) -> pd.DataFrame:
    slice_df = df.iloc[start - 1 : end].copy()
    urls = slice_df[url_col].astype(str).tolist()

    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as ex:
        futures = {ex.submit(scrape_and_classify, u): u for u in urls}
        progress = st.progress(0.0)
        for idx, fut in enumerate(concurrent.futures.as_completed(futures), start=1):
            progress.progress(idx / len(urls))
            results.append(fut.result())

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config("LinkedIn Experience Scraper + Classifier", "🔗")
st.title("🔗 LinkedIn Experience Scraper + Web-Design-Agency Classifier")

uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.success(f"Loaded {len(df):,} rows.")

    

        # 🔄 Entire body UI rewritten as two-column layout
    left_col, right_col = st.columns(2)

    # ——— LEFT  : all configuration & action button ———
    with left_col:
        st.markdown("### 🔧 Configuration")

        url_col = st.selectbox(
            "Column with LinkedIn profile URLs", df.columns.tolist()
        )

        r1, r2 = st.columns(2)
        with r1:
            start_row = st.number_input(
                "Start row (1-based)", 1, len(df), 1, 1, format="%i"
            )
        with r2:
            end_row = st.number_input(
                "End row (inclusive)", 1, len(df), len(df), 1, format="%i"
            )

        run_btn = st.button(
            f"🚀 Scrape & Classify rows {start_row}-{end_row}",
            type="primary",
            help="Consumes ScrapingDog + OpenAI credits.",
            use_container_width=True,
        )

    # ——— RIGHT : live preview (dataset first, results later) ———
    with right_col:
        st.markdown("### 📄 Dataset preview")
        st.dataframe(df.head())

    # ——— When the user clicks the button ———
    if run_btn:
        if start_row > end_row:
            st.warning("Start row must be ≤ end row.")
            st.stop()

        with st.spinner("Working…"):
            result_df = process_rows(df, url_col, start_row, end_row)

        st.success(f"Done! Parsed {len(result_df)} profiles.")

        with right_col:
            st.markdown("### 📑 Output preview")
            st.dataframe(result_df.head())

            out_path = Path("linkedin_experiences_classified.csv")
            result_df.to_csv(out_path, index=False, encoding="utf-8")
            st.download_button(
                label="⬇️ Download CSV with LLM column",
                data=out_path.read_bytes(),
                mime="text/csv",
                file_name=out_path.name,
                use_container_width=True,
            )

