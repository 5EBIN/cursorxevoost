"""Auth guards for the two server-to-server contracts.

- ``require_api_key``  -> SERVER.md: ``X-API-Key: <SERVER_SHARED_SECRET>``
- ``require_bearer``   -> KB-ADAPTER.md: ``Authorization: Bearer <KB_SERVICE_API_KEY>``

No user auth here — the main backend / ProfSidekick already authenticated the user.
The secret never reaches the browser; only server-to-server callers hold it.
"""

from __future__ import annotations

from fastapi import Header, HTTPException

import config


def require_api_key(x_api_key: str = Header(None)) -> None:
    if x_api_key != config.SERVER_SHARED_SECRET:
        raise HTTPException(status_code=401, detail="bad or missing X-API-Key")


def require_bearer(authorization: str = Header(None)) -> None:
    token = (authorization or "").removeprefix("Bearer ").strip()
    if token != config.KB_SERVICE_API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")
