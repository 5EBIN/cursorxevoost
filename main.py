"""FastAPI app exposing the Land, Investment and Decision agents as JSON.

Local:   uvicorn main:app --reload --port 8000   (then open /docs)
Railway: binds to 0.0.0.0:$PORT (see __main__ + railway.json / Procfile).
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import Depends, FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

import auth
import config
import data_loader
import scoring
from schemas import CopilotReq, LandReq, MatchReq, QueryReq, ReportReq, SearchReq

app = FastAPI(
    title="myOS Real Estate API",
    description="Land / Investment / Decision intelligence over the Abu Dhabi datasets.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "version": config.VERSION, "rows_loaded": data_loader.row_counts()}


@app.get("/districts")
def districts():
    """District reference table + cached aggregates."""
    agg = data_loader.district_aggregates()
    return {"count": int(len(agg)), "districts": agg.to_dict(orient="records")}


@app.post("/land/rank")
def land_rank(req: LandReq):
    return scoring.land_score(req.district, req.status, req.top_n)


@app.post("/investment/match")
def match(req: MatchReq):
    return scoring.investment_fit(req.investor_id, req.top_n)


@app.get("/analytics/{district}")
def analytics(district: str):
    return scoring.district_analytics(district)


@app.post("/copilot")
def copilot(req: CopilotReq):
    import agent  # imported lazily so the API boots without langchain configured

    return agent.ask(req.question)


@app.post("/search")
def search(req: SearchReq):
    """Semantic (RAG) search over textual amenity data — meaning, not exact fields."""
    import rag  # imported lazily so the API boots without chromadb loaded

    return rag.search(req.query, k=req.k, district=req.district)


@app.post("/admin/reindex")
def reindex(force: bool = True):
    """(Re)build the vector index. Run once after first deploy; persisted afterwards."""
    import rag

    return rag.build_index(force=force)


@app.get("/listings")
def listings(
    district: str = Query(...),
    listing_type: str = Query(...),
    max_price: Optional[int] = Query(None),
):
    df = data_loader.get("listings").copy()
    df = df[df["district"].str.lower() == district.lower()]
    df = df[df["listing_type"].str.lower() == listing_type.lower()]
    if max_price is not None:
        df = df[df["price_aed"] <= max_price]

    records = [
        {
            "listing_id": r["listing_id"],
            "district": r["district"],
            "community": r["community"],
            "listing_type": r["listing_type"],
            "property_type": r["property_type"],
            "bedrooms": int(r["bedrooms"]),
            "price_aed": int(r["price_aed"]),
            "size_sqm": float(r["size_sqm"]),
            "lat": float(r["latitude"]),
            "lng": float(r["longitude"]),
        }
        for _, r in df.head(200).iterrows()
    ]
    return {"count": len(records), "listings": records}


# --- Secondary server: PDF reports (SERVER.md, X-API-Key) --------------------

@app.post("/report/pdf", dependencies=[Depends(auth.require_api_key)])
def report_pdf(req: ReportReq):
    """Generate a PDF report (district | land | investment). Server-to-server only."""
    from reports.pdf import build_report

    pdf = build_report(req.report_type, req.params)
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{req.report_type}_report.pdf"'},
    )


# --- KB adapter: External Real Estate KB contract (KB-ADAPTER.md, Bearer) -----

@app.post("/v1/agents/{collection_id}/query", dependencies=[Depends(auth.require_bearer)])
def kb_query(collection_id: str, req: QueryReq):
    """Contract-conformant query. Backed by the fast deterministic pipeline + RAG.

    Fails open: any internal error returns 200 with chunks: [] so ProfSidekick keeps
    its existing knowledge instead of seeing a 5xx.
    """
    # Isolation: only the one shared market collection is known. Never 404.
    if collection_id != config.MARKET_COLLECTION_ID:
        return {"collection_id": collection_id, "query": req.query, "chunks": []}

    try:
        import pipeline

        top_k = min(req.top_k, 5)
        results = pipeline.run(req.query, top_k=top_k)
        chunks = pipeline.to_chunks(results, max_chunks=top_k)
    except Exception:
        chunks = []

    return {"collection_id": collection_id, "query": req.query, "chunks": chunks}


@app.post("/v1/agents/{collection_id}/report/pdf", dependencies=[Depends(auth.require_bearer)])
def kb_report_pdf(collection_id: str, req: ReportReq):
    """PDF extension (off-contract): separate tool path, returns bytes not chunks."""
    from reports.pdf import build_report

    pdf = build_report(req.report_type, req.params)
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{req.report_type}_report.pdf"'},
    )


if __name__ == "__main__":
    # Railway (and any PaaS) injects the port to bind on via $PORT.
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
