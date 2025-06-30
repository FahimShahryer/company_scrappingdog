#!/usr/bin/env python3
"""
test_llm.py – quick check for GPT-4.1 mini response format.

• Uses the same system prompt as linkedin_experience_scraper.py
• Sends either:
    1) a built-in toy “experiences” list (default), or
    2) a path to a JSON file that contains an array of experience objects.

Example
-------
$ python test_llm.py                             # uses toy sample
$ python test_llm.py my_experiences.json         # sends your own file
"""

import json, sys, os, pathlib, openai, textwrap

# ─────────────────────────  EDIT AS NEEDED  ──────────────────────────
OPENAI_API_KEY = (
    "sk-proj-7Qh2v2-y-AhZVEwYuIQzs46LlA6Hv8rULYG1TuOJ7JC4ttaVGDSUx1Qpbc8WWwC31"
    "ynS_GJLcPT3BlbkFJ7BGx05ky7mfDpC4V_uOXV9hKYRPHkRs-R4Utm9kF30wjUQxz61JnJDi_c"
    "gYMbe8-x_ugka3pQA"
)
MODEL   = "gpt-4.1-mini"
TEMP    = 0
TIMEOUT = 45
# ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are a concise analyst. A JSON array of job experiences follows.
    Identify every company that is **clearly** a web-design, web-development,
    digital-studio or software-agency.

    Return **only** a JSON array. Each element must be an object:
      {
        "companyName": "<name>",
        "reason": "<≈3 sentences why>"
      }
    Use max 4 short lines in each reason.
    If none match, respond with [] and nothing else."""
)

TOY_EXPERIENCES = [
    {
        "companyName": "PixelForge Studios",
        "title": "Front-End Developer",
        "description": "Designed marketing sites for SaaS clients…",
    },
    {
        "companyName": "Acme Bank",
        "title": "Software Engineer",
        "description": "Supported internal banking platform…",
    },
    {
        "companyName": "BrightIdea Web Solutions",
        "title": "UI/UX Designer",
        "description": "Created responsive websites for SMEs…",
    },
]

def load_experiences_from_file(path: str):
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("JSON root must be a list/array.")
    return data

def ask_llm(experiences):
    openai.api_key = OPENAI_API_KEY
    resp = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": json.dumps(experiences, ensure_ascii=False)},
        ],
        temperature=TEMP,
        timeout=TIMEOUT,
    )
    return resp.choices[0].message.content.strip()

def main():
    if len(sys.argv) > 1:
        experiences = load_experiences_from_file(sys.argv[1])
        print(f"Loaded {len(experiences)} experiences from {sys.argv[1]}\n")
    else:
        experiences = TOY_EXPERIENCES
        print("Using built-in toy experiences\n")

    print("Sending to LLM…\n")
    answer_text = ask_llm(experiences)
    print("Raw LLM reply:\n", answer_text, "\n")

    # Optional: attempt to parse & pretty-print JSON
    try:
        parsed = json.loads(answer_text)
        print("Parsed JSON:\n", json.dumps(parsed, indent=2, ensure_ascii=False))
    except json.JSONDecodeError as e:
        print("⚠️  Could not parse reply as JSON:", e)

if __name__ == "__main__":
    main()
