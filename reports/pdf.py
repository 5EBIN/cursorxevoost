"""PDF report generation (SERVER.md /report/pdf).

Reuses the deterministic logic in scoring.py and renders a polished PDF.

PDF backend choice: **ReportLab** (pure Python, zero system deps). SERVER.md prefers
Jinja2 -> WeasyPrint for the nicest output, but WeasyPrint needs GTK/Pango/Cairo
system libraries that are painful on Windows; ReportLab is the documented safe
fallback and installs cleanly everywhere.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any, Dict, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

import scoring

BRAND = "myOS Real Estate"
SUBBRAND = "Abu Dhabi AI PropTech"
FOOTER_NOTE = "Data: synthetic + \u00a9 OpenStreetMap contributors"

_ACCENT = colors.HexColor("#0f4c81")
_LIGHT = colors.HexColor("#eef3f8")
_GREY = colors.HexColor("#666666")


def _styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontSize=20, textColor=_ACCENT, spaceAfter=2),
        "subtitle": ParagraphStyle("subtitle", parent=base["Normal"], fontSize=10, textColor=_GREY, spaceAfter=12),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=13, textColor=_ACCENT, spaceBefore=10, spaceAfter=4),
        "body": ParagraphStyle("body", parent=base["Normal"], fontSize=10, leading=14),
        "cell": ParagraphStyle("cell", parent=base["Normal"], fontSize=9, leading=12),
    }


def _aed(value: Any) -> str:
    try:
        return f"AED {float(value):,.0f}"
    except (TypeError, ValueError):
        return str(value)


def _footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(_LIGHT)
    canvas.line(18 * mm, 15 * mm, A4[0] - 18 * mm, 15 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(_GREY)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    canvas.drawString(18 * mm, 10 * mm, f"{FOOTER_NOTE}  ·  Generated {ts}")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def _header(story: List, styles: Dict[str, ParagraphStyle], title: str) -> None:
    story.append(Paragraph(f"{BRAND}", styles["title"]))
    story.append(Paragraph(f"{SUBBRAND} — {title}", styles["subtitle"]))


def _table(rows: List[List[Any]], col_widths: List[float], header: bool = True) -> Table:
    t = Table(rows, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _LIGHT]),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, _ACCENT),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), _ACCENT),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    t.setStyle(TableStyle(style))
    return t


# --- per-report stories ------------------------------------------------------

def _error_story(styles, message: str) -> List:
    story: List = []
    _header(story, styles, "Report")
    story.append(Paragraph(message, styles["body"]))
    return story


def _district_story(styles, params: Dict[str, Any]) -> List:
    district = params.get("district", "")
    data = scoring.district_analytics(district)
    if "error" in data:
        return _error_story(styles, f"District not found: {district}")

    ref, tx, com, am = data["reference"], data["transactions"], data["community"], data["amenities"]
    story: List = []
    _header(story, styles, f"District Report — {data['district']}")

    story.append(Paragraph("Market reference", styles["h2"]))
    story.append(_table(
        [["Metric", "Value"],
         ["Profile", f"{ref.get('profile')} · {ref.get('area_type')}"],
         ["Base sale price", f"{_aed(ref['base_price_sqm'])}/sqm"],
         ["Reference yield", f"{ref['yield_pct']:.1f}%"],
         ["Infrastructure score", f"{ref['infrastructure_score']:.0f}/100"]],
        [60 * mm, 100 * mm]))

    story.append(Paragraph("Transactions", styles["h2"]))
    momentum = tx.get("momentum_pct")
    story.append(_table(
        [["Metric", "Value"],
         ["Transactions", f"{tx['count']:,}"],
         ["Avg price/sqm", f"{_aed(tx['avg_price_per_sqm'])}"],
         ["Avg deal value", f"{_aed(tx['avg_transaction_value_aed'])}"],
         ["Price momentum", f"{momentum:+.1f}%" if momentum is not None else "n/a"]],
        [60 * mm, 100 * mm]))

    story.append(Paragraph("Community & demand", styles["h2"]))
    story.append(_table(
        [["Metric", "Value"],
         ["Population", f"{com['population']:,}"],
         ["Occupancy rate", f"{com['avg_occupancy_rate']*100:.0f}%"],
         ["Service demand index", f"{com['service_demand_index']:.0f}/100"],
         ["Mobility score", f"{com['mobility_score']:.0f}/100"],
         ["Resident experience", f"{com['resident_experience_score']:.0f}/100"]],
        [60 * mm, 100 * mm]))

    story.append(Paragraph("Amenities (OpenStreetMap)", styles["h2"]))
    by_type = am.get("by_type", {})
    rows = [["Category", "Count"]] + [[k, str(v)] for k, v in by_type.items()]
    rows.append(["Total", str(am.get("total", 0))])
    story.append(_table(rows, [100 * mm, 60 * mm]))
    return story


def _land_story(styles, params: Dict[str, Any]) -> List:
    data = scoring.land_score(
        params.get("district"), params.get("status", "vacant"), params.get("top_n", 10)
    )
    story: List = []
    _header(story, styles, "Land Potential Report")
    flt = data.get("filters", {})
    story.append(Paragraph(
        f"Filters — district: {flt.get('district') or 'all'} · status: {flt.get('status') or 'any'} · "
        f"{data.get('count', 0)} parcels ranked.", styles["body"]))

    rows = [["#", "Parcel", "District", "Use", "Score", "Est. value"]]
    for i, r in enumerate(data.get("results", []), 1):
        rows.append([
            str(i), r["parcel_id"], r["district"],
            str(r["land_use"]), f"{r['score']:.1f}", _aed(r["estimated_value_aed"]),
        ])
    story.append(_table(rows, [10 * mm, 28 * mm, 38 * mm, 34 * mm, 18 * mm, 34 * mm]))

    if data.get("results"):
        top = data["results"][0]
        b = top["breakdown"]
        story.append(Paragraph("Top parcel — score breakdown", styles["h2"]))
        story.append(_table(
            [["Component", "Raw", "Normalised", "Weight"],
             ["Development potential", f"{b['development_potential']['raw']:.0f}", f"{b['development_potential']['norm']:.2f}", "0.40"],
             ["Infrastructure", f"{b['infrastructure']['raw']:.0f}", f"{b['infrastructure']['norm']:.2f}", "0.30"],
             ["Amenity density", f"{b['amenity_density']['raw']:.0f}", f"{b['amenity_density']['norm']:.2f}", "0.20"],
             ["District yield", f"{b['district_yield']['raw']:.1f}%", f"{b['district_yield']['norm']:.2f}", "0.10"]],
            [60 * mm, 30 * mm, 35 * mm, 25 * mm]))
    return story


def _investment_story(styles, params: Dict[str, Any]) -> List:
    data = scoring.investment_fit(params["investor_id"], params.get("top_n", 5))
    story: List = []
    _header(story, styles, "Investment Match Report")
    if "error" in data:
        story.append(Paragraph(data["error"], styles["body"]))
        return story

    inv = data["investor"]
    story.append(Paragraph(
        f"Investor {inv['investor_id']} — {inv['investor_type']} · sector {inv['preferred_sector']} · "
        f"district {inv['preferred_district']} · capital {inv['capital_range_aed']} · "
        f"risk {inv['risk_profile']} · horizon {inv['investment_horizon']}.", styles["body"]))

    rows = [["#", "Parcel", "District", "Land use", "Fit", "Est. value"]]
    for i, r in enumerate(data.get("results", []), 1):
        rows.append([
            str(i), r["parcel_id"], r["district"], str(r["land_use"]),
            f"{r['fit_score']:.1f}", _aed(r["estimated_value_aed"]),
        ])
    story.append(_table(rows, [10 * mm, 28 * mm, 36 * mm, 34 * mm, 18 * mm, 34 * mm]))
    return story


_BUILDERS = {
    "district": _district_story,
    "land": _land_story,
    "investment": _investment_story,
}


def build_report(report_type: str, params: Dict[str, Any]) -> bytes:
    """Build a PDF report and return its bytes."""
    styles = _styles()
    builder = _BUILDERS.get(report_type)
    if builder is None:
        raise ValueError(f"unknown report_type: {report_type!r}")

    story = builder(styles, params or {})
    story.insert(2, Spacer(1, 4 * mm))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm, topMargin=18 * mm, bottomMargin=20 * mm,
        title=f"{BRAND} {report_type} report",
    )
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()
