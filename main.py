"""FastAPI app exposing the Land, Investment and Decision agents as JSON.

Run: uvicorn main:app --reload --port 8000  (then open /docs)
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

import data_loader
import scoring
from schemas import CopilotReq, LandReq, MatchReq, SearchReq

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
    return {"status": "ok", "rows_loaded": data_loader.row_counts()}


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
