"""Central configuration for the secondary server + KB adapter.

Reads from the environment (.env). Defaults to 'devsecret' so the smoke tests in
SERVER.md / KB-ADAPTER.md work out of the box — override these in backend/.env
for any real deployment.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# Shared secret for server-to-server calls from the main backend (SERVER.md, X-API-Key).
SERVER_SHARED_SECRET = os.getenv("SERVER_SHARED_SECRET", "devsecret")

# Bearer token for the External Real Estate KB contract (KB-ADAPTER.md, Authorization: Bearer).
KB_SERVICE_API_KEY = os.getenv("KB_SERVICE_API_KEY", "devsecret")

# The single shared Abu Dhabi market collection exposed via the KB contract.
MARKET_COLLECTION_ID = os.getenv("MARKET_COLLECTION_ID", "abudhabi-market-v1")

# Optional smart path (LangChain agent) for /query — OFF by default to stay under the 5s timeout.
KB_SMART_PATH = os.getenv("KB_SMART_PATH", "false").strip().lower() in {"1", "true", "yes"}

VERSION = "1.0.0"
