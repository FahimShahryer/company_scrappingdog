import streamlit as st
import pandas as pd
import requests
from requests.exceptions import HTTPError, Timeout, RequestException
from urllib.parse import urlsplit, urlunsplit

st.set_page_config(
    page_title="🔗 LinkedIn → Company Website Finder",
    layout="wide"
)

# — SIDEBAR: API Key —
with st.sidebar:
    st.markdown("## 🔑 API Configuration")
    api_key = st.text_input(
        "ScrapingDog API Key",
        type="password",
        help="Find this in your ScrapingDog dashboard."
    )
    st.markdown("---")
    st.info("Add your key above, then select Single or Batch entry.")

if not api_key:
    st.warning("⚠️ Please enter your ScrapingDog API key in the sidebar.")
    st.stop()


def call_scrapingdog(link_type: str, link_id: str) -> dict:
    """
    Call the ScrapingDog API for either 'profile' or 'company'.
    Returns JSON dict on success, or raises HTTPError / ValueError.
    """
    endpoint = "https://api.scrapingdog.com/linkedin"
    params = {
        "api_key": api_key,
        "type": link_type,
        "linkId": link_id,
        "private": "false"
    }
    try:
        resp = requests.get(endpoint, params=params, timeout=10)
        # Handle specific HTTP errors
        if resp.status_code == 401:
            raise HTTPError("401 Unauthorized – check your API key", response=resp)
        if resp.status_code == 429:
            raise HTTPError("429 Too Many Requests – rate limit reached", response=resp)
        resp.raise_for_status()
        return resp.json()
    except Timeout:
        raise Timeout("The request timed out. Please try again later.")
    except HTTPError as http_err:
        raise HTTPError(str(http_err))
    except ValueError:
        raise ValueError("Invalid JSON response from ScrapingDog.")
    except RequestException as e:
        raise RequestException(f"Network error: {e}")


def get_company_info(profile_url: str) -> dict:
    """
    Given a LinkedIn profile URL, returns a dict with:
      first_name, last_name, profile_url,
      company_name, company_linkedin_url, company_website.
    On any error, returns {'error': <message>}.
    """
    # Validate and parse profile URL
    try:
        parts = urlsplit(profile_url)
        segs = [s for s in parts.path.split('/') if s]
        if len(segs) < 2 or segs[0] != 'in':
            raise ValueError
        profile_id = segs[1]
    except Exception:
        return {"error": "Invalid LinkedIn profile URL format."}

    # Fetch profile data
    try:
        pdata = call_scrapingdog('profile', profile_id)
        prof = pdata[0] if isinstance(pdata, list) and pdata else pdata
    except Exception as e:
        return {"error": f"Profile fetch error: {e}"}

    # Extract name
    first = prof.get('firstName') or ''
    last = prof.get('lastName') or ''
    if not (first and last):
        full = prof.get('fullName', '')
        parts = full.split()
        first = first or (parts[0] if parts else '')
        last = last or (parts[-1] if len(parts) > 1 else '')

    # Extract raw company link
    raw = prof.get('description', {}).get('description1_link')
    if not raw:
        return {"error": "No current company link found in profile."}

    # Normalize company page URL
    sp = urlsplit(raw)
    comp_page = urlunsplit((sp.scheme, sp.netloc, sp.path, '', ''))

    # Parse company ID
    try:
        csegs = [s for s in urlsplit(comp_page).path.split('/') if s]
        idx = csegs.index('company')
        company_id = csegs[idx + 1]
    except Exception:
        return {"error": "Could not extract company identifier from URL."}

    # Fetch company data
    try:
        cdata = call_scrapingdog('company', company_id)
        comp = cdata[0] if isinstance(cdata, list) and cdata else cdata
    except Exception as e:
        return {"error": f"Company fetch error: {e}"}

    return {
        'first_name': first,
        'last_name': last,
        'profile_url': profile_url,
        'company_name': comp.get('company_name', ''),
        'company_linkedin_url': f"https://www.linkedin.com/company/{company_id}",
        'company_website': comp.get('website', '')
    }

# — TABS —
tab1, tab2 = st.tabs(["🔍 Single Entry", "🗃️ Batch Entry"])

with tab1:
    st.header("Single Profile Lookup")
    c1, c2 = st.columns([1, 2], gap="large")

    with c1:
        profile_url = st.text_input(
            "LinkedIn Profile URL",
            placeholder="https://www.linkedin.com/in/username"
        )
        fetch = st.button("Fetch Company Info")

    with c2:
        if fetch:
            if not profile_url:
                st.error("Please enter a LinkedIn profile URL.")
            else:
                with st.spinner("Fetching data…"):
                    info = get_company_info(profile_url)
                if info.get('error'):
                    st.error(info['error'])
                else:
                    st.success("✅ Data retrieved")
                    st.markdown("**👤 Person**")
                    st.write("First Name:", info['first_name'])
                    st.write("Last Name:", info['last_name'])
                    st.write("Profile:", info['profile_url'])
                    st.markdown("**🏢 Company**")
                    st.write("Name:", info['company_name'])
                    st.write("LinkedIn:", info['company_linkedin_url'])
                    st.write("Website:", info['company_website'])

with tab2:
    st.header("Batch CSV Lookup")
    left, right = st.columns([1, 2], gap="large")

    with left:
        uploaded = st.file_uploader(
            "Upload CSV",
            type="csv",
            help="Must contain a column of LinkedIn profile URLs."
        )
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                if df.empty:
                    st.error("Uploaded CSV is empty.")
                    uploaded = None
            except Exception:
                st.error("Failed to read CSV. Check file format.")
                uploaded = None

        if uploaded:
            linkedin_col = st.selectbox(
                "Select LinkedIn URL column",
                options=df.columns
            )
            # Validate selected column for URLs
            sample = df[linkedin_col].dropna().astype(str).head(5)
            valid_count = sample.str.startswith('http').sum()
            if valid_count == 0:
                st.error("Selected column contains no valid URLs. Please choose a different column.")
                run_batch = False
            else:
                max_rows = st.number_input(
                    "Rows to process",
                    min_value=1,
                    max_value=len(df),
                    value=min(20, len(df))
                )
                run_batch = st.button("Run Batch")
        else:
            run_batch = False

    with right:
        if uploaded:
            st.markdown("#### Input Preview")
            st.dataframe(df.head(), use_container_width=True)

        if uploaded and run_batch:
            results = []
            with st.spinner(f"Processing up to {max_rows} rows…"):
                for _, row in df.head(max_rows).iterrows():
                    url = row.get(linkedin_col, '')
                    if not isinstance(url, str) or not url.startswith('http'):
                        continue
                    try:
                        info = get_company_info(url)
                    except Exception:
                        continue
                    if info.get('error') or not info.get('company_website'):
                        continue
                    results.append(info)

            if not results:
                st.warning("No valid data found to process.")
            else:
                out_df = pd.DataFrame(results)
                st.markdown("#### Results")
                st.dataframe(out_df, use_container_width=True)
                csv_bytes = out_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Download Results CSV",
                    data=csv_bytes,
                    file_name="linkedin_company_websites.csv",
                    mime="text/csv"
                )
