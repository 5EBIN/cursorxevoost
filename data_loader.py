"""Load all challenge CSVs once into memory and expose them + cached aggregates.

Resolves the repo-root ``data/`` folder relative to this file. If a file is
missing locally it falls back to the public Hugging Face mirror (same pattern as
the example agents).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
HF_BASE = "https://huggingface.co/datasets/eVoost/abu-dhabi-ai-proptech-challenge/resolve/main/"

# logical name -> (filename, parse_dates)
_FILES: Dict[str, tuple[str, list[str] | None]] = {
    "districts": ("districts.csv", None),
    "parcels": ("sample_parcels.csv", None),
    "transactions": ("sample_transactions.csv", ["date"]),
    "investors": ("sample_investors.csv", None),
    "communities": ("sample_communities.csv", None),
    "listings": ("sample_listings.csv", None),
    "amenities": ("osm_amenities.csv", None),
}


def _source(filename: str) -> str:
    """Local CSV when running inside the starter kit; Hugging Face otherwise."""
    local = DATA_DIR / filename
    return str(local) if local.exists() else HF_BASE + filename


def _load_all() -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for name, (filename, parse_dates) in _FILES.items():
        frames[name] = pd.read_csv(_source(filename), parse_dates=parse_dates)
    return frames


# Module-level cache: load everything exactly once on import.
DATA: Dict[str, pd.DataFrame] = _load_all()


def _build_district_aggregates() -> pd.DataFrame:
    """District-level aggregates reused by scoring and the copilot.

    One row per district with: mean transaction price/sqm, transaction count,
    mean transaction value, district reference yield/base price/infrastructure,
    amenity count, community demand metrics.
    """
    districts = DATA["districts"].copy()
    tx = DATA["transactions"]
    amenities = DATA["amenities"]
    communities = DATA["communities"]

    tx_agg = (
        tx.groupby("district")
        .agg(
            avg_price_per_sqm=("price_per_sqm", "mean"),
            transaction_count=("transaction_id", "count"),
            avg_transaction_value_aed=("transaction_value_aed", "mean"),
        )
        .reset_index()
    )

    amenity_agg = (
        amenities.groupby("district")
        .size()
        .reset_index(name="amenity_count")
    )

    community_agg = (
        communities.groupby("district")
        .agg(
            population=("population_estimate", "sum"),
            avg_occupancy_rate=("occupancy_rate", "mean"),
            avg_service_demand=("service_demand_index", "mean"),
            avg_mobility=("mobility_score", "mean"),
            avg_resident_experience=("resident_experience_score", "mean"),
        )
        .reset_index()
    )

    agg = districts.rename(
        columns={
            "base_sale_aed_sqm": "base_price_sqm",
            "gross_yield_pct": "yield_pct",
            "latitude": "centroid_lat",
            "longitude": "centroid_lng",
        }
    )
    for other in (tx_agg, amenity_agg, community_agg):
        agg = agg.merge(other, on="district", how="left")

    return agg.fillna(0)


DISTRICT_AGG: pd.DataFrame = _build_district_aggregates()


def get(name: str) -> pd.DataFrame:
    """Return a loaded DataFrame by logical name (e.g. ``parcels``)."""
    if name not in DATA:
        raise KeyError(f"Unknown dataset '{name}'. Known: {list(DATA)}")
    return DATA[name]


def district_aggregates() -> pd.DataFrame:
    """Cached one-row-per-district aggregate table."""
    return DISTRICT_AGG


def row_counts() -> Dict[str, int]:
    """Row count per loaded dataset (used by /health)."""
    return {name: int(len(df)) for name, df in DATA.items()}
