"""
Check that the LLM planner works with your settings:   python -m analyst.llm_check

Reads .env like the app does, makes ONE real planning call per configured model, and reports what
happened in plain words (key missing, key rejected, model retired, rate limit, or working).
"""
from __future__ import annotations

import os
import sys
import time
from datetime import date

from .config import load_env
from .llm import LLMPlanner, ProviderUnavailable, plan_from_json

QUESTION = "Why did success rate drop last week?"


def _explain(err: str) -> str:
    e = err.lower()
    if "not set" in e:
        return "no API key found: put GEMINI_API_KEY in .env (copy .env.example) and run again"
    if "http 400" in e and "api key" in e or "http 401" in e or "http 403" in e:
        return "the key was rejected: check it was copied completely and is enabled for the Gemini API"
    if "http 404" in e:
        return "this model is not available to your key (retired or not enabled); set LLM_MODEL in .env"
    if "http 429" in e:
        return "rate limit or quota reached: wait a minute (free tier = 15 requests/minute) and run again"
    if "http 5" in e:
        return "Google is overloaded right now: the app falls back to the backup model or the rule planner"
    if "network error" in e:
        return "could not reach Google: check your internet connection, proxy or firewall"
    return err


def main() -> int:
    loaded = load_env()
    provider = os.getenv("LLM_PROVIDER", "none")
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    print(f".env loaded   : {', '.join(loaded) if loaded else 'nothing new (no .env, or values already set)'}")
    print(f"provider      : {provider}")
    print(f"API key       : {'set (' + str(len(key)) + ' characters, ends ...' + key[-4:] + ')' if key else 'NOT SET'}")
    if provider.lower() not in ("gemini", "openai"):
        print("RESULT        : LLM is OFF. Set LLM_PROVIDER=gemini and GEMINI_API_KEY in .env, then run again.")
        return 1
    p = LLMPlanner(date(2026, 8, 31), date(2026, 10, 25), ["Coastal Bank"], ["ecommerce"])
    print(f"models        : {' -> '.join(p.models)}")
    ok_any = False
    for model in p.models:
        t0 = time.perf_counter()
        try:
            text, ti, to = p._complete(f"Question: {QUESTION}", False, model)
            ms = (time.perf_counter() - t0) * 1000
            plan = plan_from_json(text)
            print(f"  {model:<24} OK in {ms:,.0f} ms, {ti}+{to} tokens -> planned intent '{plan.intent}', "
                  f"metric '{plan.metric}'")
            ok_any = True
        except (ProviderUnavailable, RuntimeError, ValueError, KeyError) as e:
            print(f"  {model:<24} FAILED: {_explain(str(e))}")
            print(f"  {'':<24} (raw: {str(e)[:160]})")
    print("RESULT        : " + ("LLM is WORKING. Start the app with python run_app.py; the sidebar should show the model."
                                if ok_any else "LLM is NOT working (see the reasons above). The app will use the rule planner."))
    return 0 if ok_any else 1


if __name__ == "__main__":
    sys.exit(main())
