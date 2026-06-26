"""LangChain tools wrapping the DataFrames for the /copilot agent.

Each tool takes simple args and returns a compact JSON string (rows capped) so
the agent gets data only — it composes the narrative itself.
"""

from __future__ import annotations

import json
from typing import Optional

import pandas as pd
from langchain_core.tools import tool

import data_loader
import scoring

_MAX_ROWS = 20


def _dump(obj) -> str:
    return json.dumps(obj, default=str)


@tool
def rank_parcels(district: Optional[str] = None, status: Optional[str] = "vacant", top_n: int = 10) -> str:
    """Rank land parcels by development potential score. Filter by district name
    and/or current_status ('vacant', 'under_development', 'developed', 'reserved').
    Returns ranked parcels with an explainable score breakdown."""
    result = scoring.land_score(district=district, status=status, top_n=min(top_n, _MAX_ROWS))
    return _dump(result)


@tool
def match_investor(investor_id: str, top_n: int = 5) -> str:
    """Find the best-fitting parcels for a given investor_id (e.g. 'INV-001').
    Returns ranked parcels with a per-factor fit breakdown (capital, sector, risk, district)."""
    result = scoring.investment_fit(investor_id=investor_id, top_n=min(top_n, _MAX_ROWS))
    return _dump(result)


@tool
def district_analytics(district: str) -> str:
    """Full analytics bundle for one district: avg price/sqm, transaction count &
    momentum, yield, population, service_demand_index, and amenity counts by type."""
    return _dump(scoring.district_analytics(district))


@tool
def transaction_trends(district: Optional[str] = None, asset_type: Optional[str] = None) -> str:
    """Transaction time-series summary (price/sqm momentum and YoY-style change).
    Optionally filter by district and/or asset_type ('apartment', 'villa', ...)."""
    tx = data_loader.get("transactions").copy()
    if district:
        tx = tx[tx["district"].str.lower() == district.lower()]
    if asset_type:
        tx = tx[tx["asset_type"].str.lower() == asset_type.lower()]
    if tx.empty:
        return _dump({"count": 0, "trend": []})

    tx["date"] = pd.to_datetime(tx["date"])
    tx["year"] = tx["date"].dt.year
    yearly = tx.groupby("year")["price_per_sqm"].mean().sort_index()
    monthly = tx.set_index("date").resample("ME")["price_per_sqm"].mean().dropna()

    yoy = None
    if len(yearly) >= 2 and yearly.iloc[-2] > 0:
        yoy = round(float((yearly.iloc[-1] - yearly.iloc[-2]) / yearly.iloc[-2] * 100), 1)
    momentum = None
    if len(monthly) >= 2 and monthly.iloc[0] > 0:
        momentum = round(float((monthly.iloc[-1] - monthly.iloc[0]) / monthly.iloc[0] * 100), 1)

    return _dump(
        {
            "filters": {"district": district, "asset_type": asset_type},
            "count": int(len(tx)),
            "avg_price_per_sqm": round(float(tx["price_per_sqm"].mean()), 0),
            "yoy_change_pct": yoy,
            "momentum_pct": momentum,
            "yearly_avg_price_per_sqm": {int(y): round(float(v), 0) for y, v in yearly.items()},
        }
    )


@tool
def service_gap(district: Optional[str] = None) -> str:
    """Identify under-served districts by comparing community service demand
    against OSM amenity supply. Returns districts ranked by demand-per-amenity gap."""
    agg = data_loader.district_aggregates().copy()
    agg = agg[agg["avg_service_demand"] > 0]
    # Higher demand with fewer amenities = bigger gap.
    agg["amenities_per_1k"] = agg["amenity_count"] / (agg["population"] / 1000).replace(0, pd.NA)
    agg["gap_score"] = agg["avg_service_demand"] / (agg["amenity_count"] + 1)

    if district:
        agg = agg[agg["district"].str.lower() == district.lower()]

    agg = agg.sort_values("gap_score", ascending=False).head(_MAX_ROWS)
    rows = [
        {
            "district": r["district"],
            "service_demand_index": round(float(r["avg_service_demand"]), 1),
            "amenity_count": int(r["amenity_count"]),
            "population": int(r["population"]),
            "gap_score": round(float(r["gap_score"]), 2),
        }
        for _, r in agg.iterrows()
    ]
    return _dump({"count": len(rows), "under_served": rows})


@tool
def query_listings(district: str, listing_type: str, max_price: Optional[int] = None) -> str:
    """Return filtered residential listings with coordinates (for maps).
    listing_type is 'rent' or 'buy'/'sale'; max_price filters price_aed."""
    listings = data_loader.get("listings").copy()
    listings = listings[listings["district"].str.lower() == district.lower()]
    listings = listings[listings["listing_type"].str.lower() == listing_type.lower()]
    if max_price is not None:
        listings = listings[listings["price_aed"] <= max_price]

    listings = listings.head(_MAX_ROWS)
    rows = [
        {
            "listing_id": r["listing_id"],
            "district": r["district"],
            "listing_type": r["listing_type"],
            "property_type": r["property_type"],
            "price_aed": int(r["price_aed"]),
            "size_sqm": float(r["size_sqm"]),
            "lat": float(r["latitude"]),
            "lng": float(r["longitude"]),
        }
        for _, r in listings.iterrows()
    ]
    return _dump({"count": len(rows), "listings": rows})


ALL_TOOLS = [
    rank_parcels,
    match_investor,
    district_analytics,
    transaction_trends,
    service_gap,
    query_listings,
]
