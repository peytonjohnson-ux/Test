"""
Clinical Decision Support System – Immunosuppressant Drug Interaction Checker
Flask web application.

Run:
    pip install flask
    python cdss/app.py

Then open http://localhost:5000
"""

from __future__ import annotations
import json
from flask import Flask, render_template, request, jsonify
from drug_db import (
    lookup_interactions,
    all_immunosuppressants,
    all_co_therapies,
    canonical,
    SEVERITY_ORDER,
)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["JSON_SORT_KEYS"] = False


# ── Helpers ───────────────────────────────────────────────────────────────────

SEVERITY_CSS = {
    "Contraindicated": "severity-contraindicated",
    "Major":           "severity-major",
    "Moderate":        "severity-moderate",
    "Minor":           "severity-minor",
    "None":            "severity-none",
}

CONFIDENCE_CSS = {
    "High":   "confidence-high",
    "Medium": "confidence-medium",
    "Low":    "confidence-low",
}


def _enrich(results: list[dict]) -> list[dict]:
    """Attach CSS class names to each result dict."""
    for r in results:
        r["severity_css"]    = SEVERITY_CSS.get(r["severity"], "")
        r["confidence_css"]  = CONFIDENCE_CSS.get(r["confidence"], "")
        r["severity_rank"]   = SEVERITY_ORDER.get(r["severity"], 0)
    return results


def _overall_severity(results: list[dict]) -> str:
    """Return the highest severity across all results."""
    if not results:
        return "None"
    return max(results, key=lambda r: SEVERITY_ORDER.get(r["severity"], 0))["severity"]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template(
        "index.html",
        immunosuppressants=sorted(all_immunosuppressants()),
        co_therapies=sorted(all_co_therapies()),
    )


@app.route("/check", methods=["POST"])
def check():
    data = request.get_json(force=True)
    immunosuppressant = data.get("immunosuppressant", "").strip()
    co_therapies_raw  = data.get("co_therapies", [])

    if not immunosuppressant:
        return jsonify({"error": "immunosuppressant is required"}), 400
    if not co_therapies_raw:
        return jsonify({"error": "at least one co-therapy is required"}), 400

    co_therapies = [c.strip() for c in co_therapies_raw if c.strip()]
    results = lookup_interactions(immunosuppressant, co_therapies)
    results = _enrich(results)
    overall = _overall_severity(results)

    return jsonify({
        "immunosuppressant": canonical(immunosuppressant),
        "overall_severity":  overall,
        "overall_severity_css": SEVERITY_CSS.get(overall, ""),
        "results": results,
    })


@app.route("/drugs")
def drugs():
    """Return all known drug names (for autocomplete)."""
    return jsonify({
        "immunosuppressants": sorted(all_immunosuppressants()),
        "co_therapies":       sorted(all_co_therapies()),
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
