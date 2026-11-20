"""
Authentication: the role comes from who the user IS, never from a dropdown.

Two modes (AUTH_MODE):

  password (default)  Users in a JSON file (USERS_FILE, default config/users.demo.json) with salted
                      PBKDF2-SHA256 password hashes and a role. Add or change users with
                      `python -m analyst.auth add <username> <role>`.
  header              Behind an SSO reverse proxy (e.g. oauth2-proxy / an OIDC gateway) that has
                      already authenticated the user and sets a trusted header (AUTH_HEADER, default
                      X-Forwarded-Email). The role is looked up from the users file by that identity.
                      Only safe when the app is reachable exclusively through that proxy.

Roles are never taken from user input; an identity that is not in the users file gets no access.
"""
from __future__ import annotations

import getpass
import hashlib
import hmac
import json
import os
import secrets
import sys
from dataclasses import dataclass

from .rbac import ROLES

DEFAULT_USERS_FILE = os.path.join(os.path.dirname(__file__), "..", "config", "users.demo.json")
ITERATIONS = 200_000


@dataclass(frozen=True)
class User:
    username: str
    role: str
    display_name: str = ""


def _users_file() -> str:
    return os.getenv("USERS_FILE", DEFAULT_USERS_FILE)


def load_users(path: str | None = None) -> dict[str, dict]:
    path = path or _users_file()
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {u["username"].lower(): u for u in data.get("users", [])}


def hash_password(password: str, salt: str | None = None, iterations: int = ITERATIONS) -> dict:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), iterations).hex()
    return {"salt": salt, "iterations": iterations, "hash": digest}


def verify_password(username: str, password: str, path: str | None = None) -> User | None:
    rec = load_users(path).get((username or "").strip().lower())
    # always run the KDF, so response time does not reveal whether the username exists
    salt = rec["salt"] if rec else "00" * 16
    iters = int(rec.get("iterations", ITERATIONS)) if rec else ITERATIONS
    candidate = hash_password(password or "", salt, iters)["hash"]
    if not rec or not hmac.compare_digest(candidate, rec["hash"]) or rec.get("role") not in ROLES:
        return None
    return User(rec["username"], rec["role"], rec.get("display_name", rec["username"]))


def user_from_headers(headers, path: str | None = None) -> User | None:
    """Header mode: identity asserted by the SSO proxy; role from the users file, never the request."""
    header = os.getenv("AUTH_HEADER", "X-Forwarded-Email")
    ident = (headers.get(header) or "").strip().lower() if headers else ""
    rec = load_users(path).get(ident)
    if not ident or not rec or rec.get("role") not in ROLES:
        return None
    return User(rec["username"], rec["role"], rec.get("display_name", rec["username"]))


def auth_mode() -> str:
    return os.getenv("AUTH_MODE", "password").lower()


def _cli(argv: list[str]) -> int:
    """python -m analyst.auth add <username> <role> [--file path]  (prompts for the password)"""
    if len(argv) < 3 or argv[0] != "add" or argv[2] not in ROLES:
        print(f"usage: python -m analyst.auth add <username> <role> [--file path]\nroles: {', '.join(ROLES)}")
        return 2
    path = argv[argv.index("--file") + 1] if "--file" in argv else _users_file()
    data = {"users": list(load_users(path).values())}
    pw = getpass.getpass("password: ")
    entry = {"username": argv[1], "role": argv[2], "display_name": argv[1], **hash_password(pw)}
    data["users"] = [u for u in data["users"] if u["username"].lower() != argv[1].lower()] + [entry]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"saved {argv[1]} ({argv[2]}) to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
