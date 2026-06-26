# MD_FINAL.md — myOS Real Estate Intelligence Microservice

A complete, detailed reference for the backend microservice built for the **Abu Dhabi AI
PropTech Challenge** (Cursor × eVoost AI, Hub71). This document explains *what* the service
does, *how* every layer works, the *data* it stands on, the *formulas* behind every score, the
*contracts* it speaks, and the *design decisions* (and deliberate divergences) behind it all.

---

## 1. What this service is

One Python (FastAPI) service that converts the challenge datasets into an intelligence API:

1. Loads all **7 datasets once at startup** and holds them in memory (pandas).
2. Exposes **deterministic ranking + analytics** endpoints (fast, no LLM).
3. Exposes a **semantic search** endpoint (RAG over real OSM amenity text).
4. Exposes an **LLM copilot** (LangChain agent via OpenRouter) that routes to tools and explains.
5. Conforms to an **external KB contract** (`/v1/agents/.../query`) for ProfSidekick, backed by
   the same pipeline.
6. Generates **PDF reports** on demand.

**Guiding principle:** the maths is deterministic Python; the LLM only explains and routes. This
keeps the demo robust (judges score what runs) while keeping the AI clearly central (it reasons,
routes, and narrates).

---

## 2. Architecture & request flow

```
┌──────────────┐     ┌──────────────────────┐     ┌───────────────────────────┐
│  FRONTEND    │ ──► │  MAIN BACKEND /       │ ──► │  THIS MICROSERVICE        │
│ chat + avatar│ ◄── │  ProfSidekick (brain) │ ◄── │  rankings · analytics ·   │
│  (OpenAI)    │     │  OpenAI orchestration │     │  RAG · KB chunks · PDF    │
└──────────────┘     └──────────────────────┘     └───────────────────────────┘
   user-facing            decides WHAT to ask            answers WITH data
```

- The **frontend/avatar** talks to the main backend (or directly to the public endpoints here).
- The **main backend / ProfSidekick** is the single orchestrating "brain" (OpenAI function calling).
- **This service** is a stateless data + document API. It never runs the user's chat loop; it
  takes structured requests and returns JSON or a PDF.

### Module map

| File | Responsibility |
|---|---|
| `main.py` | FastAPI app, CORS, all route definitions |
| `data_loader.py` | load CSVs once, cache district-level aggregates, HF fallback |
| `scoring.py` | deterministic land/investment scoring + district analytics |
| `context.py` | hardcoded domain context injected into LLM prompts |
| `tools.py` | LangChain `@tool` wrappers over the data |
| `agent.py` | LangChain tool-calling agent (OpenRouter) for `/copilot` |
| `rag.py` | Chroma vector index + semantic search over OSM amenities |
| `pipeline.py` | fast intent router + `to_chunks`/`prose` for the KB contract |
| `reports/pdf.py` | ReportLab PDF report builder |
| `auth.py` | `X-API-Key` and `Bearer` auth guards |
| `config.py` | secrets, collection id, feature flags |
| `schemas.py` | Pydantic request models |

---

## 3. Data model

Seven datasets, all joining on `district`. Loaded once into `data_loader.DATA`.

| Dataset | Rows | Key columns used |
|---|---|---|
| `districts.csv` | 20 | `district, area_type, profile, base_sale_aed_sqm, gross_yield_pct, infrastructure_score, latitude, longitude` |
| `sample_parcels.csv` | 600 | `parcel_id, district, zone, land_use, parcel_size_sqm, current_status, infrastructure_score, development_potential_score, estimated_value_aed, recommended_use` |
| `sample_transactions.csv` | 5,000 | `district, asset_type, transaction_value_aed, size_sqm, price_per_sqm, date, buyer_type` |
| `sample_investors.csv` | 200 | `investor_id, investor_type, preferred_sector, preferred_district, capital_range_aed, risk_profile, investment_horizon` |
| `sample_communities.csv` | 90 | `district, population_estimate, occupancy_rate, service_demand_index, mobility_score, resident_experience_score` |
| `sample_listings.csv` | 6,000 | `district, listing_type, property_type, price_aed, size_sqm, latitude, longitude` |
| `osm_amenities.csv` | 3,155 | `amenity_id, category, subtype, name, latitude, longitude, district` — **REAL** OSM data |

> **Column-name note:** the real CSV headers differ from some early drafts (e.g. parcels use
> `zone`/`land_use`/`parcel_size_sqm`; districts use `base_sale_aed_sqm`/`gross_yield_pct`/
> `latitude`/`longitude`; amenities/listings use `latitude`/`longitude`; amenity type is
> `category`). The code uses the **real** columns above.

### Cached aggregates

On import, `data_loader._build_district_aggregates()` computes **one row per district** with:
mean transaction `price_per_sqm`, transaction count, mean deal value, reference base price /
yield / infrastructure, total amenity count, and community metrics (population, occupancy,
service demand, mobility, resident experience). Reused by scoring, analytics, and the pipeline.

### Local-or-cloud loading

`data_loader._source()` resolves `../data/<file>` relative to the repo; if missing, it falls
back to `https://huggingface.co/datasets/eVoost/abu-dhabi-ai-proptech-challenge/resolve/main/`.
So the service runs with or without the `data/` folder (important on Railway, where only the
backend is deployed).

---

## 4. Scoring (deterministic — `scoring.py`)

### 4.1 Land potential score (per parcel)

Each component is min-max normalised to [0,1] **across the candidate set**, then:

```
score = 0.40 · development_potential_norm
      + 0.30 · infrastructure_norm
      + 0.20 · amenity_density_norm      (OSM amenity count in the parcel's district)
      + 0.10 · district_yield_norm       (gross_yield_pct)
score is scaled ×100 for readability.
```

Opportunity queries filter to `current_status == "vacant"`. The response includes the full
per-component **breakdown** (`raw`, `norm`, `weight`) so every ranking is explainable.

### 4.2 Investment fit score (investor × parcel)

Multiplicative, then ×100 (capped at 100):

```
fit = capital_match · sector_match · risk_match · district_pref_bonus
```

- **capital_match**: `1.0` if parcel value is inside the investor's `capital_range_aed`,
  `0.6` if below the floor, `0.2` if above the ceiling.
- **sector_match**: `1.0` exact `land_use == preferred_sector`, `0.7` compatible
  (via `SECTOR_LAND_USE` map), `0.25` otherwise.
- **risk_match**: from `RISK_STATUS_BONUS[risk_profile][current_status]` (0.5–1.0), then nudged
  by development potential — aggressive mandates reward high potential, conservative the reverse.
- **district_pref_bonus**: `1.15` if `district == preferred_district`, else `1.0`.

Returns the investor profile + top-N parcels with the per-factor breakdown.

### 4.3 District analytics

`district_analytics(district)` returns a chart-ready bundle: market reference (base price, yield,
infrastructure, centroid), transactions (count, avg price/sqm, avg deal value, **monthly price
trend**, momentum %, price-by-asset-type), community (population, occupancy, service demand,
mobility, resident experience), and amenities (total + counts by category).

---

## 5. RAG layer (`rag.py`)

**Hybrid rule:** numbers → pandas; meaning/text → vectors. Only genuinely textual data is embedded.

- **Corpus:** the 3,155 **real** OSM amenities, embedded as `"{name} — {subtype} ({category}) in {district}"`.
- **Store:** Chroma, persisted at `backend/vectorstore/` (gitignored), cosine space.
- **Embeddings:** OpenAI `text-embedding-3-small` if `OPENAI_API_KEY` is set; otherwise a local,
  keyless ONNX model (`all-MiniLM-L6-v2`) that runs fully offline ("demo insurance").
- **Build:** lazy on first use (`ensure_index`), or via `POST /admin/reindex`. Persisted, so
  restarts are instant.
- **Query:** `search(query, k, district?)` returns hits with `text`, metadata (`district`,
  `category`, `subtype`, `name`, `lat`, `lng`) and a cosine `score`. District filter is applied
  as a metadata `where` clause.

---

## 6. LLM copilot (`agent.py` + `tools.py` + `context.py`)

- **Model transport:** OpenRouter (OpenAI-compatible) via `langchain-openai`'s `ChatOpenAI`,
  configured by `OPENROUTER_API_KEY` / `OPENROUTER_MODEL` / `OPENROUTER_BASE_URL`.
- **Tools** (`tools.py`, each returns compact JSON, rows capped at 20):
  `rank_parcels`, `match_investor`, `district_analytics`, `transaction_trends`, `service_gap`,
  `query_listings`. They wrap `scoring.py`/`rag.py` — **data only**, no narration.
- **Agent:** `create_tool_calling_agent` + `AgentExecutor(return_intermediate_steps=True,
  max_iterations=6)`. The system prompt (`context.py`) grounds it with `COLUMN_DOCS`,
  `DISTRICT_FACTS`, `SCORING_RUBRIC`, and rules ("never invent numbers; cite district + metric").
- **Response:** `{answer, tools_used, evidence}` — the `tools_used`/`evidence` triplet shows
  *which datasets the AI chose to read*, a strong "use of AI" demo signal.
- **Graceful degrade:** the executor is built lazily; with no key, `/copilot` returns a friendly
  message and every other endpoint keeps working.

---

## 7. KB adapter (`pipeline.py` + `/v1/agents/*`)

Conforms to the **External Real Estate KB** contract, but backs `/query` with the live pipeline
instead of a static vector lookup. ProfSidekick sees a standard KB; inside, it's the intelligence engine.

### 7.1 Fast path (default)

`pipeline.run(query, top_k)`:
1. `classify(query)` — a cheap **keyword** intent router (no LLM): `land | investment | district | semantic`.
2. Dispatch:
   - `land` → `scoring.land_score` (extracts a district name if present)
   - `investment` → `scoring.investment_fit` (extracts `INV-xxx`; falls back to semantic if none)
   - `district` → `scoring.district_analytics` (needs a district)
   - `semantic`/fallback → `rag.search`
3. Normalise rows → `to_chunks`.

No LLM in the hot path → stays well under ProfSidekick's **5s timeout**, scores are real, output
is reliable. The optional **smart path** (full LangChain agent) is gated behind `KB_SMART_PATH`
and intended only for offline/PDF use.

### 7.2 Chunk rendering (`to_chunks` / `prose`)

Each result becomes a contract chunk:

```json
{ "title": "...", "content": "<plain prose, ≤1200 chars>", "score": 0.82,
  "metadata": { "type": "market_insight", "district": "...", "kind": "land_rank|investment_match|analytics|amenity" } }
```

- `content` is **plain sentences** — never JSON/HTML/bullets (the contract injects it verbatim).
- Capped to **≤5 chunks, ≤1,200 chars** each.
- `metadata.type` is always `market_insight` (the only enum value that fits city analytics).

### 7.3 Contract endpoints

| Endpoint | Behaviour |
|---|---|
| `POST /v1/agents` | Provision — idempotent, resolves to `MARKET_COLLECTION_ID`, `status:"ready"` |
| `POST /v1/agents/{cid}/documents` | Import proxy — accepted no-op (chunks generated live, no storage) |
| `POST /v1/agents/{cid}/query` | Runs the pipeline → chunks. Unknown collection → `chunks:[]` (never 404). **Fails open** (200 + `[]`) on internal error |
| `POST /v1/agents/{cid}/report/pdf` | PDF extension (off-contract), Bearer-guarded |

### 7.4 Deliberate divergences from the contract

- LangChain lives in the **query** path (smart path only), not ingest.
- `score` is a relevance/confidence value, not always cosine; `min_score` is advisory.
- **One shared collection** (`abudhabi-market-v1`) instead of one-per-agent; isolation check kept
  for shape conformance.
- **No storage** — `/documents` and `/export` are intentionally not implemented (v1). Chunks are
  generated per query.

---

## 8. PDF reports (`reports/pdf.py`)

- `build_report(report_type, params) -> bytes` reuses `scoring.py` and renders with **ReportLab**.
- **Why ReportLab, not WeasyPrint:** WeasyPrint needs GTK/Pango/Cairo system libraries (painful
  on Windows / minimal containers). ReportLab is pure Python, zero system deps — the documented
  safe choice. Output is a branded A4 doc: accent header, styled tables, footer with timestamp
  and "Data: synthetic + © OpenStreetMap contributors".
- **Types:** `district` (analytics bundle), `land` (top vacant parcels + breakdown),
  `investment` (an investor's top matches).
- Served as `application/pdf` with `Content-Disposition: attachment`.

---

## 9. Security & auth (`auth.py` + `config.py`)

Two **shared-secret** server-to-server guards (no user auth — the caller already authenticated
the user; secrets stay server-side, never in the browser):

| Guard | Header | Env var | Routes |
|---|---|---|---|
| `require_api_key` | `X-API-Key` | `SERVER_SHARED_SECRET` | `POST /report/pdf` |
| `require_bearer` | `Authorization: Bearer` | `KB_SERVICE_API_KEY` | all `/v1/agents/*` |

Public/analytics endpoints are intentionally unauthenticated for the avatar frontend. Both
secrets **default to `devsecret`** so smoke tests work out of the box — **override them in
production** (Railway → Variables). CORS is open (`*`) for the public endpoints.

---

## 10. Full endpoint reference

| Method | Path | Auth | Body / Query | Returns |
|---|---|---|---|---|
| GET | `/health` | — | — | `{status, version, rows_loaded}` |
| GET | `/districts` | — | — | district reference + aggregates |
| POST | `/land/rank` | — | `{district?, status?, top_n}` | ranked parcels + breakdown |
| POST | `/investment/match` | — | `{investor_id, top_n}` | investor + ranked matches + breakdown |
| GET | `/analytics/{district}` | — | — | analytics bundle |
| POST | `/copilot` | — | `{question}` | `{answer, tools_used, evidence}` |
| POST | `/search` | — | `{query, district?, k?}` | semantic hits + coords |
| GET | `/listings` | — | `?district&listing_type&max_price` | listings + coords |
| POST | `/admin/reindex` | — | `?force=true` | index build status |
| POST | `/report/pdf` | X-API-Key | `{report_type, params}` | PDF bytes |
| POST | `/v1/agents` | Bearer | `{name?, collection_id?, config?}` | provisioned collection |
| POST | `/v1/agents/{cid}/documents` | Bearer | `{documents:[...]}` | import accepted (no-op) |
| POST | `/v1/agents/{cid}/query` | Bearer | `{query, top_k?, min_score?, scope_assertion?}` | KB chunks |
| POST | `/v1/agents/{cid}/report/pdf` | Bearer | `{report_type, params}` | PDF bytes |

---

## 11. Configuration reference

| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | — | enables `/copilot` |
| `OPENROUTER_MODEL` | `anthropic/claude-sonnet-4.5` | copilot model |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter base URL |
| `KB_SERVICE_API_KEY` | `devsecret` | Bearer token for `/v1/agents/*` |
| `SERVER_SHARED_SECRET` | `devsecret` | X-API-Key for `/report/pdf` |
| `MARKET_COLLECTION_ID` | `abudhabi-market-v1` | the single KB collection id |
| `KB_SMART_PATH` | `false` | enable LLM agent for `/query` |
| `OPENAI_API_KEY` | — | optional OpenAI embeddings for RAG |
| `PORT` | `8000` | bind port (set by Railway) |

---

## 12. Deployment (Railway)

- **Builder:** Railpack (Python). `railway.json` sets the start command
  `uvicorn main:app --host 0.0.0.0 --port $PORT`, healthcheck `/health`, restart-on-failure.
- **Procfile** mirrors the start command; `.python-version` pins **3.11** (avoids 3.13 wheel
  surprises for chromadb/onnxruntime).
- **Port:** the app binds `$PORT`; set a `PORT` variable (or let Railway auto-detect) to match the
  generated domain's target port.
- **Cold-start notes:** no `data/` folder → CSVs download from Hugging Face on boot; first
  `/search` downloads the embedding model (~80 MB) and builds the index. Set `OPENAI_API_KEY` to
  move embeddings off-box if memory is tight on small plans.

---

## 13. Build order (how it was assembled)

1. `data_loader.py` + `/health` → confirm CSVs load.
2. `scoring.py` + `/land/rank` + `/investment/match` → deterministic rankings.
3. `context.py` + `tools.py` + `agent.py` + `/copilot` → LLM layer.
4. `/analytics/{district}` + `/listings` → charts & map.
5. `rag.py` + `/search` → semantic layer.
6. `auth.py` + `reports/pdf.py` + `/report/pdf` → secondary server + PDF.
7. `pipeline.py` + `/v1/agents/*` → KB contract adapter.
8. Railway config → deploy.

---

## 14. Repository note

This microservice lives in `backend/` on the `backend` branch of `terka2610/myos_real_estate`,
and is also published standalone (backend files at repo root) to `5EBIN/cursorxevoost` for the
Railway deployment. Datasets are synthetic except `osm_amenities.csv`, which is real
OpenStreetMap data (© OpenStreetMap contributors, ODbL).
