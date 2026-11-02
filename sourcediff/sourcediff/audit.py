"""SHA-256 helpers for the audit chain. Stdlib only.

Each history row stores  row_hash = SHA256(prev_row_hash + canonical(core fields)).
Change or delete any old row and every later hash stops matching, so the edit is detectable.
This is tamper-EVIDENT, not tamper-proof: someone who rewrites the whole chain can still do it.
Anchoring the newest hash somewhere you do not control (a printed receipt, a public post) closes that gap.
"""
from __future__ import annotations

import hashlib
import json

GENESIS = "0" * 64


def canonical(obj) -> str:
    """One stable text form of a JSON-able object (sorted keys, no spaces)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chain_hash(prev_hash: str, core: dict) -> str:
    return sha256_text(prev_hash + canonical(core))
