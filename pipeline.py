"""KB pipeline — fast deterministic router that powers /v1/agents/{cid}/query.

Per KB-ADAPTER.md: the default (fast) path uses a cheap keyword intent router — no
LLM — so we stay well under ProfSidekick's 5s query timeout, scores stay real, and
output is reliable. Results are normalised into uniform rows, then ``to_chunks``
renders them into the contract's prose-chunk shape.

The optional smart path (full LangChain agent) is gated behind config.KB_SMART_PATH
and is intended for offline/PDF use, never the live chat hook.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import data_loader
import rag
import scoring

ALLOWED_TYPE = "market_insight"  # the only contract enum value that fits city analytics (C6)

_LAND_WORDS = {"land", "parcel", "parcels", "plot", "plots", "develop", "development",
               "vacant", "build", "buildable", "site"}
_INVEST_WORDS = {"investor", "invest", "investment", "capital", "fund", "mandate",
                 "portfolio", "deal", "deals", "buyer"}
_DISTRICT_WORDS = {"analytics", "market", "price", "prices", "trend", "trends", "yield",
                   "population", "demand", "transactions", "performing", "performance"}
_SEMANTIC_WORDS = {"near", "family", "quiet", "cultural", "wellness", "amenity",
                   "amenities", "park", "parks", "school", "schools", "clinic",
                   "hospital", "cafe", "restaurant", "friendly", "walkable", "green"}


def _districts() -> List[str]:
    return data_loader.get("districts")["district"].tolist()


def _find_district(query: str) -> Optional[str]:
    q = query.lower()
    # Longest name first so "Al Reem Island" wins over a substring match.
    for name in sorted(_districts(), key=len, reverse=True):
        if name.lower() in q:
            return name
    return None


def _find_investor_id(query: str) -> Optional[str]:
    m = re.search(r"INV[-_ ]?(\d+)", query, flags=re.IGNORECASE)
    if not m:
        return None
    return f"INV-{int(m.group(1)):03d}"


def classify(query: str) -> str:
    """Cheap keyword router -> land | investment | district | semantic."""
    words = set(re.findall(r"[a-z]+", query.lower()))
    score = {
        "investment": len(words & _INVEST_WORDS),
        "land": len(words & _LAND_WORDS),
        "district": len(words & _DISTRICT_WORDS),
        "semantic": len(words & _SEMANTIC_WORDS),
    }
    best = max(score, key=score.get)
    if score[best] == 0:
        return "semantic"  # fallback
    return best


# --- normalisers: structured rows -> uniform pipeline rows -------------------

def _aed(value: float) -> str:
    return f"AED {value:,.0f}"


def _land_rows(query: str, top_k: int) -> List[Dict[str, Any]]:
    district = _find_district(query)
    res = scoring.land_score(district=district, status="vacant", top_n=top_k)
    rows: List[Dict[str, Any]] = []
    for r in res.get("results", []):
        b = r["breakdown"]
        rows.append(
            {
                "kind": "land_rank",
                "label": f"{r['parcel_id']} — {r['district']} ({r['land_use']})",
                "score": round(r["score"] / 100.0, 4),
                "district": r["district"],
                "prose": (
                    f"Parcel {r['parcel_id']} in {r['district']} ranks high for vacant-land "
                    f"development potential (score {r['score']/100:.2f}): development potential "
                    f"{b['development_potential']['raw']:.0f}/100, infrastructure "
                    f"{b['infrastructure']['raw']:.0f}/100, surrounding amenity density "
                    f"{b['amenity_density']['raw']:.0f}, and a district yield of "
                    f"{b['district_yield']['raw']:.1f}%. Recommended use: "
                    f"{str(r['recommended_use']).replace('_', ' ')}. Estimated value "
                    f"{_aed(r['estimated_value_aed'])}."
                ),
            }
        )
    return rows


def _investment_rows(query: str, top_k: int) -> List[Dict[str, Any]]:
    investor_id = _find_investor_id(query)
    if not investor_id:
        return []  # caller falls back to semantic
    res = scoring.investment_fit(investor_id=investor_id, top_n=top_k)
    inv = res.get("investor", {})
    rows: List[Dict[str, Any]] = []
    for r in res.get("results", []):
        b = r["breakdown"]
        rows.append(
            {
                "kind": "investment_match",
                "label": f"{r['parcel_id']} — {r['district']} for {investor_id}",
                "score": round(r["fit_score"] / 100.0, 4),
                "district": r["district"],
                "prose": (
                    f"For investor {investor_id} ({inv.get('investor_type')}, "
                    f"{inv.get('preferred_sector')} focus, {inv.get('risk_profile')} risk), "
                    f"parcel {r['parcel_id']} in {r['district']} ({r['land_use']}) is a strong "
                    f"match (fit {r['fit_score']/100:.2f}): capital match {b['capital_match']:.2f}, "
                    f"sector match {b['sector_match']:.2f}, risk match {b['risk_match']:.2f}, "
                    f"district-preference factor {b['district_pref_bonus']:.2f}. Estimated value "
                    f"{_aed(r['estimated_value_aed'])}."
                ),
            }
        )
    return rows


def _district_rows(query: str) -> List[Dict[str, Any]]:
    district = _find_district(query)
    if not district:
        return []
    d = scoring.district_analytics(district)
    if "error" in d:
        return []
    ref, tx, com, am = d["reference"], d["transactions"], d["community"], d["amenities"]
    top_amenities = ", ".join(
        f"{k} ({v})" for k, v in list(am.get("by_type", {}).items())[:4]
    )
    momentum = tx.get("momentum_pct")
    momentum_txt = f"{momentum:+.1f}% price momentum" if momentum is not None else "stable pricing"
    return [
        {
            "kind": "analytics",
            "label": f"{district} market analytics",
            "score": 1.0,
            "district": district,
            "prose": (
                f"{district} ({ref.get('profile')}, {ref.get('area_type')}): average transaction "
                f"price {_aed(tx['avg_price_per_sqm'])}/sqm across {tx['count']} transactions "
                f"({momentum_txt}), reference yield {ref['yield_pct']:.1f}%, infrastructure score "
                f"{ref['infrastructure_score']:.0f}/100. Population {com['population']:,} with a "
                f"service-demand index of {com['service_demand_index']:.0f}/100 and resident "
                f"experience {com['resident_experience_score']:.0f}/100. Amenities: {am['total']} "
                f"mapped ({top_amenities})."
            ),
        }
    ]


def _semantic_rows(query: str, top_k: int) -> List[Dict[str, Any]]:
    district = _find_district(query)
    res = rag.search(query, k=top_k, district=district)
    rows: List[Dict[str, Any]] = []
    for r in res.get("results", []):
        rows.append(
            {
                "kind": "amenity",
                "label": str(r.get("name") or r.get("text", ""))[:200],
                "score": round(float(r.get("score", 0.0)), 4),
                "district": r.get("district"),
                "prose": (
                    f"{r.get('name', 'A place')} is a {r.get('subtype') or r.get('category')} "
                    f"in {r.get('district')} (category: {r.get('category')}), located at "
                    f"{r.get('lat')}, {r.get('lng')}. Surfaced by semantic relevance to the query."
                ),
            }
        )
    return rows


def run(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Route a free-text query to the right deterministic/RAG path (fast, no LLM)."""
    top_k = max(1, min(top_k, 5))
    intent = classify(query)

    if intent == "land":
        rows = _land_rows(query, top_k)
    elif intent == "investment":
        rows = _investment_rows(query, top_k)
    elif intent == "district":
        rows = _district_rows(query)
    else:
        rows = _semantic_rows(query, top_k)

    # Graceful fallback to semantic search if the chosen path produced nothing.
    if not rows and intent != "semantic":
        rows = _semantic_rows(query, top_k)

    return rows[:top_k]


def to_chunks(results: List[Dict[str, Any]], max_chunks: int = 5) -> List[Dict[str, Any]]:
    """Render pipeline rows into contract chunks: plain prose, capped, typed."""
    chunks: List[Dict[str, Any]] = []
    for r in results[:max_chunks]:
        chunks.append(
            {
                "title": str(r.get("label", ""))[:200],
                "content": str(r.get("prose", ""))[:1200],  # plain sentences, no JSON
                "score": float(r.get("score", 0.0)),
                "metadata": {
                    "type": ALLOWED_TYPE,
                    "district": r.get("district"),
                    "kind": r.get("kind"),
                },
            }
        )
    return chunks
