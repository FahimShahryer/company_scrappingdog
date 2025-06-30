import streamlit as st
import pandas as pd
import requests
from requests.exceptions import HTTPError, Timeout, RequestException
from urllib.parse import urlsplit, urlunsplit
from google import genai
import json

# ———————————
# CONFIGURATION
# ———————————
SCRAPINGDOG_ENDPOINT = "https://api.scrapingdog.com/linkedin"
SCRAPINGDOG_API_KEY = None  # set via sidebar

GEMINI_API_KEY   = "AIzaSyA9kZU2GHmqc4N5BSbUg9bXof5mwGdgvS8"
GEMINI_MODEL     = "gemini-2.0-flash"
EXPERIENCE_LIMIT = 3  # top N current roles

# Initialize Gemini client
genai_client = genai.Client(api_key=GEMINI_API_KEY)

# ———————————
# SCRAPINGDOG CALL
# ———————————
def call_scrapingdog(link_type: str, link_id: str) -> dict:
    """
    Fetch JSON from ScrapingDog; raises on error.
    """
    params = {"api_key": SCRAPINGDOG_API_KEY, "type": link_type,
              "linkId": link_id, "private": "false"}
    try:
        resp = requests.get(SCRAPINGDOG_ENDPOINT, params=params, timeout=10)
        if resp.status_code == 401:
            raise HTTPError("401 Unauthorized – invalid API key")
        if resp.status_code == 429:
            raise HTTPError("429 Too Many Requests – rate limited")
        resp.raise_for_status()
        return resp.json()
    except Timeout:
        raise Timeout("ScrapingDog request timed out.")
    except HTTPError:
        raise
    except Exception as e:
        raise RequestException(f"ScrapingDog network error: {e}")

# ———————————
# LLM LAYER
# ———————————
def call_gemini_select_company(exps: list[dict]) -> tuple[dict | None, str]:
    """
    Ask Gemini to pick the best matching company; returns (choice, raw response).
    """
    if not exps:
        return None, ""

    system_prompt = (
        "You are an assistant that, given current employers and details, "
        "chooses one that best fits 'web design agency, web development company, or software development company'. "
        "If none match, reply 'None'."
    )
    lines = []
    for i, e in enumerate(exps, 1):
        lines.append(
            f"{i}) Position: {e.get('position','')}\n"
            f"   Company: {e.get('company_name','')}\n"
            f"   LinkedIn URL: {e.get('company_url','')}\n"
            f"   Location: {e.get('location','')}\n"
            f"   Summary: {e.get('summary','')}\n"
        )
    user_prompt = (
        "Here are the current companies:\n\n" +
        "\n".join(lines) +
        f"\nReply with the number (1–{len(exps)}) of the best match, or 'None'."
    )
    prompt = system_prompt + "\n\n" + user_prompt

    try:
        response = genai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        answer = response.text.strip()
    except Exception as e:
        return None, f"LLM error: {e}"

    if answer.lower().startswith("none"):
        return None, answer
    try:
        idx = int(answer.split()[0]) - 1
        if 0 <= idx < len(exps):
            return exps[idx], answer
    except Exception:
        pass
    return None, answer

# ———————————
# HELPERS
# ———————————
def extract_current_experiences(exps: list[dict]) -> list[dict]:
    """Return up to EXPERIENCE_LIMIT of roles where ends_at=='Present'."""
    current = [e for e in exps if e.get('ends_at') == 'Present']
    return current[:EXPERIENCE_LIMIT]


def fetch_company_website(company_url: str) -> str:
    """Given a LinkedIn company URL, fetch website via ScrapingDog."""
    try:
        segs = [s for s in urlsplit(company_url).path.split('/') if s]
        idx = segs.index('company')
        cid = segs[idx+1]
    except Exception:
        return ''
    try:
        data = call_scrapingdog('company', cid)
        comp = data[0] if isinstance(data, list) else data
        return comp.get('website','') or ''
    except Exception:
        return ''


def get_company_info(profile_url: str) -> dict:
    """Fetch profile, select company via LLM, fetch its website, return flat dict."""
    # parse profile ID
    try:
        segs = [s for s in urlsplit(profile_url).path.split('/') if s]
        if len(segs) < 2 or segs[0] != 'in':
            raise ValueError
        pid = segs[1]
    except Exception:
        return {'error':'Invalid LinkedIn profile URL.'}

    # fetch profile
    try:
        pdata = call_scrapingdog('profile', pid)
        prof = pdata[0] if isinstance(pdata,list) and pdata else pdata
    except Exception as e:
        return {'error':f'Profile fetch error: {e}'}

    # extract name
    first = prof.get('first_name') or ''
    last  = prof.get('last_name')  or ''
    if not (first and last):
        parts = prof.get('fullName','').split()
        if parts:
            first = first or parts[0]
            last  = last  or parts[-1]

    # experiences & debug
    exps = prof.get('experience') or []
    current = extract_current_experiences(exps)
    candidates_json = json.dumps(current, ensure_ascii=False)
    chosen, llm_resp = call_gemini_select_company(current)

    # fetch website if chosen
    if chosen:
        website = fetch_company_website(chosen.get('company_url',''))
        pos = chosen.get('position','')
        cname = chosen.get('company_name','')
        clink = chosen.get('company_url','')
        loc = chosen.get('location','')
        summary = chosen.get('summary','')
    else:
        website = 'No web design agency found'
        pos = cname = clink = loc = summary = ''

    return {
        'first_name': first,
        'last_name': last,
        'profile_url': profile_url,
        'position': pos,
        'company_name': cname,
        'company_linkedin': clink,
        'location': loc,
        'summary': summary,
        'company_website': website,
        # debug
        'candidates': candidates_json,
        'llm_response': llm_resp,
        'model_used': GEMINI_MODEL
    }

# ———————————
# STREAMLIT UI
# ———————————
st.set_page_config(page_title="🔗 LinkedIn → Company Website Finder", layout="wide")

with st.sidebar:
    st.markdown("## 🔑 API Configuration")
    SCRAPINGDOG_API_KEY = st.text_input("ScrapingDog API Key", type="password",
        help="Enter your ScrapingDog API key.")
    st.markdown("---")
    st.info("After entering your key, choose Single or Batch entry.")
if not SCRAPINGDOG_API_KEY:
    st.warning("⚠️ ScrapingDog API key required.")
    st.stop()

tab1, tab2 = st.tabs(["🔍 Single Entry", "🗃️ Batch Entry"])

# Single
with tab1:
    st.header("Single Profile Lookup")
    c1,c2 = st.columns([1,2],gap="large")
    with c1:
        url = st.text_input("LinkedIn Profile URL", placeholder="https://www.linkedin.com/in/username")
        go = st.button("Fetch Company Info")
    with c2:
        if go:
            if not url.strip():
                st.error("Please enter a LinkedIn profile URL.")
            else:
                with st.spinner("Processing…"):
                    info = get_company_info(url.strip())
                if info.get('error'):
                    st.error(info['error'])
                else:
                    data_tab, debug_tab = st.tabs(["Data","Debug Report"])
                    with data_tab:
                        st.success("✅ Data retrieved")
                        st.write(info)
                    with debug_tab:
                        st.markdown("#### Debug Report")
                        st.write("Candidates JSON:", info['candidates'])
                        st.write("LLM Response:", info['llm_response'])
                        st.write("Model Used:", info['model_used'])

# Batch
with tab2:
    st.header("Batch CSV Lookup")
    left,right = st.columns([1,2],gap="large")
    df=None; run=False
    with left:
        upload = st.file_uploader("Upload CSV",type="csv")
        if upload:
            try: df=pd.read_csv(upload)
            except Exception as e:
                st.error(f"CSV read error: {e}")
                df=None
        if df is not None:
            col=st.selectbox("LinkedIn URL column",df.columns)
            sample=df[col].dropna().astype(str).head(5)
            if not sample.str.startswith('http').any(): st.error("No valid URLs.")
            else: run=st.button("Run Batch")
    with right:
        if df is not None:
            st.dataframe(df.head(),use_container_width=True)
        if df is not None and run:
            res=[]
            for _,row in df.iterrows():
                u=row[col].strip()
                if not u.lower().startswith('http'): continue
                info=get_company_info(u)
                if info.get('error'): continue
                res.append(info)
            if not res: st.warning("No valid data.")
            else:
                out=pd.DataFrame(res)
                rtab,dbtab=st.tabs(["Results","Debug Report"])
                with rtab:
                    st.dataframe(out[[ 'first_name','last_name','profile_url',
                     'position','company_name','company_linkedin','location',
                     'summary','company_website']])
                with dbtab:
                    st.dataframe(out[[ 'profile_url','candidates',
                     'llm_response','model_used']])
