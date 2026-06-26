"""Hardcoded domain context injected into LLM prompts.

Keeps the model grounded in the real datasets and Abu Dhabi geography so it
explains numbers instead of inventing them.
"""

COLUMN_DOCS = """\
districts.csv
  district: district name (join key across all datasets)
  area_type: island | mainland | suburban | coastal etc.
  profile: market positioning (premium | mid_market | affordable ...)
  base_sale_aed_sqm: reference sale price per sqm for the district (AED)
  gross_yield_pct: reference gross rental yield (%)
  infrastructure_score: 0-100 infrastructure quality
  latitude/longitude: district centroid coordinates
  established_year: year the district was established

sample_parcels.csv
  parcel_id: unique parcel id
  district: district name
  zone: zoning code (e.g. Z-RES-01)
  land_use: residential | commercial | mixed_use | hospitality | industrial | community
  parcel_size_sqm: parcel area in sqm
  current_status: vacant | under_development | developed | reserved
  infrastructure_score: 0-100 infrastructure around the parcel
  development_potential_score: 0-100 modeled development upside
  estimated_value_aed: estimated parcel value (AED)
  recommended_use: model-suggested best use

sample_transactions.csv
  transaction_id, date, district, asset_type, transaction_value_aed,
  size_sqm, price_per_sqm, buyer_type

sample_investors.csv
  investor_id, investor_type, preferred_sector, preferred_district,
  capital_range_aed (e.g. '15M-60M'), risk_profile (conservative|balanced|aggressive),
  investment_horizon, strategic_fit_score

sample_communities.csv
  community_id, district, population_estimate, occupancy_rate,
  service_demand_index (0-100 unmet demand), mobility_score,
  resident_experience_score, optimization_opportunity

sample_listings.csv
  listing_id, district, community, listing_type (rent|buy/sale),
  property_type, bedrooms, bathrooms, size_sqm, price_aed,
  price_per_sqm_aed, furnished, amenities, latitude, longitude, status

osm_amenities.csv (REAL OpenStreetMap data)
  amenity_id, category (amenity type), subtype, name, latitude, longitude, district
"""

DISTRICT_FACTS = """\
- Saadiyat Island: premium cultural district (Louvre, museums); high-end, lower yield.
- Yas Island: tourism & entertainment hub (theme parks, F1); hospitality-led.
- Al Reem Island: dense mid-market residential, strong rental yields.
- Al Reef: affordable suburban community, high yield, family-oriented.
- Masdar City: sustainable/affordable tech community, high yield.
- Al Bateen: established prime coastal residential.
- Zayed City (Khalifa City area): growing residential with rising service demand.
- Al Maryah Island: financial/commercial core (ADGM), premium commercial.
- Al Raha Beach: waterfront mid-to-premium residential.
- Corniche / Markaziya: dense urban core, mature market.
Other districts span mainland and suburban areas; treat base_sale_aed_sqm and
gross_yield_pct from districts.csv as the source of truth for pricing/yield.
"""

SCORING_RUBRIC = """\
LAND POTENTIAL SCORE (per parcel, 0-100), all components normalised 0-1 across the candidate set:
  score = 0.40 * development_potential_score
        + 0.30 * infrastructure_score
        + 0.20 * amenity_density      (count of OSM amenities in the parcel's district)
        + 0.10 * district_yield       (gross_yield_pct from districts.csv)
  Opportunity queries filter to current_status == 'vacant'.

INVESTMENT FIT SCORE (investor x parcel, 0-100), multiplicative factors:
  fit = capital_match   (parcel estimated_value_aed inside investor capital_range_aed)
      * sector_match    (parcel land_use vs investor preferred_sector)
      * risk_match      (current_status & development_potential vs risk_profile)
      * district_pref_bonus (small boost if district == preferred_district)
"""

SYSTEM_PROMPT = (
    "You are a property-intelligence analyst for Abu Dhabi. "
    "Use only the provided tools and data. Never invent numbers. "
    "Cite the district and metric behind each claim. "
    "When you present rankings or analytics, briefly explain the drivers using the scoring rubric.\n\n"
    "DISTRICT FACTS:\n" + DISTRICT_FACTS + "\n\n"
    "SCORING RUBRIC:\n" + SCORING_RUBRIC
)
