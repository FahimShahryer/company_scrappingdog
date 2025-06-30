#!/usr/bin/env python3
import requests, re
from requests.exceptions import HTTPError, Timeout, RequestException

API_KEY   = "685ec4c35257ed02ae0575b4"
COMPANY   = "https://www.linkedin.com/company/skyhook-internet-marketing"
TIMEOUT   = 25
BASE_URL  = "https://api.scrapingdog.com/linkedin"

slug = re.search(r"https://www\.linkedin\.com/company/([^/?]+)", COMPANY).group(1)
params = {"api_key": API_KEY, "type": "company", "linkId": slug}

try:
    data = requests.get(BASE_URL, params=params, timeout=TIMEOUT).json()
    obj  = data[0] if isinstance(data, list) and data else data
    print(obj.get("website", "") or "(blank)")
except (HTTPError, Timeout, RequestException, ValueError) as e:
    print("❌", e)
