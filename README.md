# myOS Real Estate — Intelligence Microservice

A FastAPI microservice for the **Abu Dhabi AI PropTech Challenge**. It turns six synthetic
datasets + one real OpenStreetMap dataset into a clean JSON + PDF intelligence API:

- **Deterministic rankings** — land potential & investor↔parcel matching (fast, pandas, no LLM).
- **District analytics** — price trends, yield, demand, amenities (chart-ready bundles).
- **Semantic search (RAG)** — vector search over real OSM amenity text.
- **LLM copilot** — a LangChain tool-calling agent (via OpenRouter) that routes + explains.
- **KB adapter** — a contract-conformant `/v1/agents/.../query` for ProfSidekick.
- **PDF reports** — polished downloadable reports (ReportLab).

> Design rule: **the maths is deterministic Python; the LLM only explains and routes.**

---

## Quick start

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env            # macOS/Linux: cp .env.example .env
uvicorn main:app --reload --port 8000
```

Open <http://localhost:8000/docs> for the interactive Swagger UI.

No `data/` folder? No problem — the loader falls back to the public Hugging Face mirror, so the
service runs anywhere.

---

## Endpoints at a glance

| Method | Path | Auth | Returns |
|---|---|---|---|
| GET | `/health` | — | status + row counts |
| GET | `/districts` | — | district reference + aggregates |
| POST | `/land/rank` | — | ranked parcels + score breakdown |
| POST | `/investment/match` | — | investor + ranked matches + fit breakdown |
| GET | `/analytics/{district}` | — | analytics bundle (charts) |
| POST | `/copilot` | — | `{answer, tools_used, evidence}` (LLM) |
| POST | `/search` | — | semantic hits + map coords (RAG) |
| GET | `/listings` | — | listings with coords |
| POST | `/admin/reindex` | — | (re)build the vector index |
| POST | `/report/pdf` | `X-API-Key` | PDF report bytes |
| POST | `/v1/agents` | `Bearer` | provision (idempotent) |
| POST | `/v1/agents/{cid}/documents` | `Bearer` | import proxy (no-op) |
| POST | `/v1/agents/{cid}/query` | `Bearer` | KB prose chunks |
| POST | `/v1/agents/{cid}/report/pdf` | `Bearer` | PDF report bytes |

---

## Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | — | enables `/copilot` (OpenAI-compatible via OpenRouter) |
| `OPENROUTER_MODEL` | `anthropic/claude-sonnet-4.5` | copilot model slug |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter base URL |
| `KB_SERVICE_API_KEY` | `devsecret` | Bearer token for `/v1/agents/*` |
| `SERVER_SHARED_SECRET` | `devsecret` | X-API-Key for `/report/pdf` |
| `MARKET_COLLECTION_ID` | `abudhabi-market-v1` | the single shared KB collection id |
| `KB_SMART_PATH` | `false` | use the LLM agent for `/query` (off — keeps under 5s) |
| `OPENAI_API_KEY` | — | optional: OpenAI embeddings for RAG (else local model) |

> **Change the secrets** for any real deployment — the `devsecret` defaults are public.

---

## Smoke test

```bash
curl localhost:8000/health
curl -X POST localhost:8000/land/rank -H "Content-Type: application/json" -d '{"status":"vacant","top_n":5}'
curl -X POST localhost:8000/v1/agents/abudhabi-market-v1/query \
  -H "Authorization: Bearer devsecret" -H "Content-Type: application/json" \
  -d '{"query":"Which vacant districts have high potential but few amenities?","top_k":5}'
```

---

## Deploy (Railway)

The repo ships `railway.json`, `Procfile`, and `.python-version`. Railway injects `$PORT`; the
app binds `0.0.0.0:$PORT`. Set the env vars above in Railway → Variables. Healthcheck is `/health`.

---

## Project layout

```
backend/
├── main.py          # FastAPI app + all routes
├── data_loader.py   # loads 7 CSVs once + cached district aggregates (HF fallback)
├── scoring.py       # deterministic land/investment scoring + analytics
├── context.py       # column docs, district facts, scoring rubric, system prompt
├── tools.py         # LangChain tools wrapping the data
├── agent.py         # LangChain tool-calling agent (OpenRouter) for /copilot
├── rag.py           # Chroma vector index + semantic search over OSM amenities
├── pipeline.py      # fast intent router + prose-chunk renderer for the KB contract
├── reports/pdf.py   # ReportLab PDF report builder
├── auth.py          # X-API-Key + Bearer guards
├── config.py        # secrets, collection id, feature flags
├── schemas.py       # Pydantic request models
├── requirements.txt
├── railway.json · Procfile · .python-version
└── .env.example
```

See **MD_FINAL.md** for the full architecture, data model, scoring formulas, and design decisions.
