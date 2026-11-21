"""Load settings from a .env file in the project folder, so the LLM key and options don't have to be
typed into the terminal before every start. Values already set in the environment win. No dependency.

.env format (see .env.example):   NAME=value   # comments and blank lines allowed, quotes optional
.env is in .gitignore: keys never get committed.
"""
from __future__ import annotations

import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_env(path: str | None = None) -> list[str]:
    """Set variables from .env that are not already set. Returns the names it set (never values)."""
    path = path or os.path.join(ROOT, ".env")
    if not os.path.exists(path):
        return []
    loaded = []
    with open(path, encoding="utf-8-sig") as f:  # utf-8-sig: Notepad on Windows may add a BOM
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            name = name.strip().removeprefix("export ").strip()
            value = value.split(" #", 1)[0].strip().strip('"').strip("'")
            if name and name not in os.environ:
                os.environ[name] = value
                loaded.append(name)
    return loaded
