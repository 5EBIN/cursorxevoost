"""Deterministic ranking + analytics. No LLM in the hot path.

All maths is plain pandas/Python so the demo is fast and reproducible. The LLM
layer (tools.py / agent.py) only routes to and explains these functions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

import data_loader

# Sector -> compatible land uses (lifted from the investment-matching example).
SECTOR_LAND_USE: Dict[str, set[str]] = {
    "residential": {"residential", "mixed_use"},
    "commercial": {"commercial", "mixed_use"},
    "hospitality": {"hospitality", "mixed_use"},
    "mixed_use": {"mixed_use", "residential", "commercial"},
    "logistics": {"industrial"},
    "industrial": {"industrial"},
    "community": {"community", "residential"},
}

# Risk profile -> tolerance for each parcel status (0-15), lifted from the example.
RISK_STATUS_BONUS: Dict[str, Dict[str, int]] = {
    "conservative": {"developed": 15, "under_development": 5, "vacant": 0, "reserved": 0},
    "balanced": {"developed": 5, "under_development": 10, "vacant": 8, "reserved": 3},
    "aggressive": {"developed": 0, "under_development": 8, "vacant": 15, "reserved": 10},
}


def _normalise(series: pd.Series) -> pd.Series:
    """Min-max normalise to [0, 1]; constant series -> all ones."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(1.0, index=series.index)
    return (series - lo) / (hi - lo)


# --- Land potential ---------------------------------------------------------

def land_score(
    district: Optional[str] = None,
    status: Optional[str] = "vacant",
    top_n: int = 10,
) -> Dict[str, Any]:
    """Rank parcels by land potential with an explainable component breakdown."""
    parcels = data_loader.get("parcels").copy()
    agg = data_loader.district_aggregates()

    if district:
        parcels = parcels[parcels["district"].str.lower() == district.lower()]
    if status:
        parcels = parcels[parcels["current_status"].str.lower() == status.lower()]

    if parcels.empty:
        return {"count": 0, "results": [], "filters": {"district": district, "status": status}}

    # District-level signals merged onto each parcel.
    district_signals = agg[["district", "yield_pct", "amenity_count"]]
    parcels = parcels.merge(district_signals, on="district", how="left").fillna(0)

    dev_norm = _normalise(parcels["development_potential_score"])
    infra_norm = _normalise(parcels["infrastructure_score"])
    amenity_norm = _normalise(parcels["amenity_count"])
    yield_norm = _normalise(parcels["yield_pct"])

    parcels["score"] = (
        0.40 * dev_norm
        + 0.30 * infra_norm
        + 0.20 * amenity_norm
        + 0.10 * yield_norm
    ) * 100

    parcels["_dev_norm"] = dev_norm
    parcels["_infra_norm"] = infra_norm
    parcels["_amenity_norm"] = amenity_norm
    parcels["_yield_norm"] = yield_norm

    ranked = parcels.sort_values("score", ascending=False).head(top_n)

    results: List[Dict[str, Any]] = []
    for _, p in ranked.iterrows():
        results.append(
            {
                "parcel_id": p["parcel_id"],
                "district": p["district"],
                "land_use": p["land_use"],
                "current_status": p["current_status"],
                "parcel_size_sqm": int(p["parcel_size_sqm"]),
                "estimated_value_aed": int(p["estimated_value_aed"]),
                "recommended_use": p["recommended_use"],
                "score": round(float(p["score"]), 1),
                "breakdown": {
                    "development_potential": {
                        "raw": float(p["development_potential_score"]),
                        "norm": round(float(p["_dev_norm"]), 3),
                        "weight": 0.40,
                    },
                    "infrastructure": {
                        "raw": float(p["infrastructure_score"]),
                        "norm": round(float(p["_infra_norm"]), 3),
                        "weight": 0.30,
                    },
                    "amenity_density": {
                        "raw": float(p["amenity_count"]),
                        "norm": round(float(p["_amenity_norm"]), 3),
                        "weight": 0.20,
                    },
                    "district_yield": {
                        "raw": float(p["yield_pct"]),
                        "norm": round(float(p["_yield_norm"]), 3),
                        "weight": 0.10,
                    },
                },
            }
        )

    return {
        "count": len(results),
        "filters": {"district": district, "status": status},
        "results": results,
    }


# --- Investment fit ---------------------------------------------------------

def parse_capital_range_aed(capital_range: str) -> tuple[float, float]:
    """'15M-60M' -> (15_000_000, 60_000_000); '600M-2.5B' -> (..., 2.5e9)."""

    def to_num(token: str) -> float:
        token = token.strip().upper()
        mult = 1_000_000_000 if token.endswith("B") else 1_000_000
        return float(token.rstrip("MB")) * mult

    parts = str(capital_range).split("-")
    if len(parts) == 1:
        hi = to_num(parts[0])
        return 0.0, hi
    return to_num(parts[0]), to_num(parts[-1])


def _capital_match(value: float, lo: float, hi: float) -> float:
    if lo <= value <= hi:
        return 1.0
    if value < lo:
        return 0.6  # affordable but below mandate floor
    return 0.2  # above mandate ceiling


def _sector_match(land_use: str, preferred_sector: str) -> float:
    if land_use == preferred_sector:
        return 1.0
    if land_use in SECTOR_LAND_USE.get(preferred_sector, set()):
        return 0.7
    return 0.25


def _risk_match(risk_profile: str, status: str, dev_potential: float) -> float:
    bonus = RISK_STATUS_BONUS.get(risk_profile, {}).get(status, 0)
    base = 0.5 + (bonus / 15.0) * 0.5  # 0.5 .. 1.0 from status tolerance
    # Aggressive mandates reward high development potential; conservative the reverse.
    if risk_profile == "aggressive":
        base *= 0.8 + 0.2 * (dev_potential / 100.0)
    elif risk_profile == "conservative":
        base *= 0.8 + 0.2 * (1 - dev_potential / 100.0)
    return base


def investment_fit(investor_id: str, top_n: int = 5) -> Dict[str, Any]:
    """Rank parcels for one investor with a per-factor fit breakdown."""
    investors = data_loader.get("investors")
    parcels = data_loader.get("parcels").copy()

    match = investors[investors["investor_id"].str.lower() == investor_id.lower()]
    if match.empty:
        return {"error": f"investor_id '{investor_id}' not found", "results": []}
    investor = match.iloc[0]

    lo, hi = parse_capital_range_aed(investor["capital_range_aed"])
    pref_sector = investor["preferred_sector"]
    pref_district = investor["preferred_district"]
    risk = investor["risk_profile"]

    results: List[Dict[str, Any]] = []
    for _, p in parcels.iterrows():
        capital_match = _capital_match(float(p["estimated_value_aed"]), lo, hi)
        sector_match = _sector_match(p["land_use"], pref_sector)
        risk_match = _risk_match(risk, p["current_status"], float(p["development_potential_score"]))
        district_bonus = 1.15 if p["district"] == pref_district else 1.0

        fit = capital_match * sector_match * risk_match * district_bonus * 100
        results.append(
            {
                "parcel_id": p["parcel_id"],
                "district": p["district"],
                "land_use": p["land_use"],
                "current_status": p["current_status"],
                "estimated_value_aed": int(p["estimated_value_aed"]),
                "fit_score": round(min(fit, 100.0), 1),
                "breakdown": {
                    "capital_match": round(capital_match, 3),
                    "sector_match": round(sector_match, 3),
                    "risk_match": round(risk_match, 3),
                    "district_pref_bonus": round(district_bonus, 3),
                },
            }
        )

    results.sort(key=lambda r: r["fit_score"], reverse=True)
    return {
        "investor": {
            "investor_id": investor["investor_id"],
            "investor_type": investor["investor_type"],
            "preferred_sector": pref_sector,
            "preferred_district": pref_district,
            "capital_range_aed": investor["capital_range_aed"],
            "risk_profile": risk,
            "investment_horizon": investor["investment_horizon"],
        },
        "count": min(top_n, len(results)),
        "results": results[:top_n],
    }


# --- District analytics -----------------------------------------------------

def district_analytics(district: str) -> Dict[str, Any]:
    """Full analytics bundle for one district (for charts)."""
    agg = data_loader.district_aggregates()
    row = agg[agg["district"].str.lower() == district.lower()]
    if row.empty:
        return {"error": f"district '{district}' not found"}
    d = row.iloc[0]
    name = d["district"]

    tx = data_loader.get("transactions")
    tx_d = tx[tx["district"] == name].copy()

    # Monthly price/sqm trend + simple momentum (last vs first month).
    trend: List[Dict[str, Any]] = []
    momentum_pct = None
    if not tx_d.empty:
        tx_d["month"] = pd.to_datetime(tx_d["date"]).dt.to_period("M").astype(str)
        monthly = tx_d.groupby("month")["price_per_sqm"].mean().sort_index()
        trend = [{"month": m, "avg_price_per_sqm": round(float(v), 0)} for m, v in monthly.items()]
        if len(monthly) >= 2 and monthly.iloc[0] > 0:
            momentum_pct = round(float((monthly.iloc[-1] - monthly.iloc[0]) / monthly.iloc[0] * 100), 1)

    by_asset = (
        tx_d.groupby("asset_type")["price_per_sqm"].mean().round(0).to_dict()
        if not tx_d.empty
        else {}
    )

    amenities = data_loader.get("amenities")
    amenities_d = amenities[amenities["district"] == name]
    amenity_by_type = amenities_d["category"].value_counts().to_dict()

    return {
        "district": name,
        "reference": {
            "area_type": d.get("area_type"),
            "profile": d.get("profile"),
            "base_price_sqm": float(d["base_price_sqm"]),
            "yield_pct": float(d["yield_pct"]),
            "infrastructure_score": float(d["infrastructure_score"]),
            "centroid_lat": float(d["centroid_lat"]),
            "centroid_lng": float(d["centroid_lng"]),
        },
        "transactions": {
            "count": int(d["transaction_count"]),
            "avg_price_per_sqm": round(float(d["avg_price_per_sqm"]), 0),
            "avg_transaction_value_aed": round(float(d["avg_transaction_value_aed"]), 0),
            "momentum_pct": momentum_pct,
            "monthly_trend": trend,
            "avg_price_per_sqm_by_asset": {k: float(v) for k, v in by_asset.items()},
        },
        "community": {
            "population": int(d["population"]),
            "avg_occupancy_rate": round(float(d["avg_occupancy_rate"]), 3),
            "service_demand_index": round(float(d["avg_service_demand"]), 1),
            "mobility_score": round(float(d["avg_mobility"]), 1),
            "resident_experience_score": round(float(d["avg_resident_experience"]), 1),
        },
        "amenities": {
            "total": int(d["amenity_count"]),
            "by_type": {k: int(v) for k, v in amenity_by_type.items()},
        },
    }
