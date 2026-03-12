"""
remedy_engine.py
----------------
Generates actionable remedies and rerouting suggestions based on
forecast results and active traffic incidents.

Usage (called by forecast_engine.py):
    from remedy_engine import generate_remedies, suggest_rerouting
"""

from __future__ import annotations

from typing import List

from models_forecast import (
    AlertLevel,
    ForecastHour,
    Remedy,
    ReroutingSuggestion,
    TrafficAlert,
    TrafficIncident,
)


# ---------------------------------------------------------------------------
# Zone definitions (mirrors graph_engine ZONE_POPULATION)
# ---------------------------------------------------------------------------
ZONES = ["Downtown", "Suburbs", "Industrial", "Residential", "Highway Corridor"]

ELEVATED_ZONES   = ["Suburbs", "Highway Corridor"]
LOW_LYING_ZONES  = ["Downtown", "Industrial"]
PEAK_ZONES       = ["Downtown", "Highway Corridor"]


# ---------------------------------------------------------------------------
# Remedy catalog  —  keyed by risk type
# ---------------------------------------------------------------------------

_FLOOD_REMEDIES = [
    Remedy(
        action="Activate storm-drain pre-clearing crews in low-lying zones",
        priority=1,
        target_zones=LOW_LYING_ZONES,
        risk_type="flood",
        estimated_impact="Reduces localised flooding risk by up to 40%",
    ),
    Remedy(
        action="Redirect traffic from low-lying zones (Downtown, Industrial) to elevated routes (Suburbs, Highway Corridor)",
        priority=1,
        target_zones=LOW_LYING_ZONES + ELEVATED_ZONES,
        risk_type="flood",
        estimated_impact="Prevents vehicles from becoming stranded in flood-prone areas",
    ),
    Remedy(
        action="Deploy portable flood barriers at critical underpasses and intersections",
        priority=2,
        target_zones=LOW_LYING_ZONES,
        risk_type="flood",
        estimated_impact="Protects key arterial routes from becoming impassable",
    ),
    Remedy(
        action="Issue public advisory: avoid low-lying roads and use public transit where possible",
        priority=2,
        target_zones=ZONES,
        risk_type="flood",
        estimated_impact="Reduces traffic volume in at-risk areas by 15-25%",
    ),
    Remedy(
        action="Pre-position emergency rescue boats and amphibious vehicles in flood-prone zones",
        priority=3,
        target_zones=LOW_LYING_ZONES,
        risk_type="flood",
        estimated_impact="Reduces emergency response time from 45 min to 10 min in flooded areas",
    ),
]

_CONGESTION_REMEDIES = [
    Remedy(
        action="Open contraflow lanes on major arterials to increase outbound capacity",
        priority=1,
        target_zones=PEAK_ZONES,
        risk_type="congestion",
        estimated_impact="Increases throughput by 30-50% on affected corridors",
    ),
    Remedy(
        action="Activate dynamic speed limits and ramp metering on Highway Corridor",
        priority=1,
        target_zones=["Highway Corridor"],
        risk_type="congestion",
        estimated_impact="Smooths traffic flow and reduces stop-and-go by 25%",
    ),
    Remedy(
        action="Divert non-essential traffic via alternate corridors (Suburbs, Residential)",
        priority=2,
        target_zones=["Suburbs", "Residential"],
        risk_type="congestion",
        estimated_impact="Distributes load and reduces Downtown congestion by 20%",
    ),
    Remedy(
        action="Issue advisory for flexible work-from-home hours before predicted peak congestion",
        priority=2,
        target_zones=ZONES,
        risk_type="congestion",
        estimated_impact="Can reduce peak-hour traffic demand by 10-15%",
    ),
    Remedy(
        action="Increase public transit frequency on high-demand routes",
        priority=3,
        target_zones=PEAK_ZONES,
        risk_type="congestion",
        estimated_impact="Shifts 5-10% of commuters from private vehicles to transit",
    ),
]

_POWER_OUTAGE_REMEDIES = [
    Remedy(
        action="Pre-deploy mobile traffic signal units at critical intersections",
        priority=1,
        target_zones=PEAK_ZONES,
        risk_type="power_outage",
        estimated_impact="Prevents intersection gridlock during signal failure",
    ),
    Remedy(
        action="Activate emergency backup power to traffic management centres and key junctions",
        priority=1,
        target_zones=ZONES,
        risk_type="power_outage",
        estimated_impact="Maintains signal operation for 4-8 hours during outage",
    ),
    Remedy(
        action="Deploy traffic police to top-20 highest-volume intersections",
        priority=2,
        target_zones=PEAK_ZONES,
        risk_type="power_outage",
        estimated_impact="Manual control prevents accidents at signal-less intersections",
    ),
    Remedy(
        action="Issue public advisory: expect signal outages, reduce speed, and yield at all dark intersections",
        priority=2,
        target_zones=ZONES,
        risk_type="power_outage",
        estimated_impact="Reduces accident risk by 30% during outage events",
    ),
]

_GENERAL_REMEDIES = [
    Remedy(
        action="Activate city-wide Emergency Traffic Management Plan",
        priority=1,
        target_zones=ZONES,
        risk_type="general",
        estimated_impact="Coordinated response across all agencies reduces impact duration by 40%",
    ),
    Remedy(
        action="Issue multi-channel public alerts (SMS, radio, social media, highway signs)",
        priority=1,
        target_zones=ZONES,
        risk_type="general",
        estimated_impact="Reaches 85% of commuters within 15 minutes",
    ),
    Remedy(
        action="Pre-position emergency response vehicles at zone boundaries",
        priority=2,
        target_zones=ZONES,
        risk_type="general",
        estimated_impact="Reduces average emergency response time by 35%",
    ),
]


# ---------------------------------------------------------------------------
# Remedy selection logic
# ---------------------------------------------------------------------------

def generate_remedies(
    timeline: List[ForecastHour],
    alerts: List[TrafficAlert],
) -> List[Remedy]:
    """
    Select relevant remedies based on the forecast timeline and generated alerts.

    Returns a de-duplicated, priority-sorted list of remedies.
    """
    if not alerts:
        return []

    needed_types: set[str] = set()
    max_level = AlertLevel.GREEN

    for alert in alerts:
        needed_types.add(alert.alert_type)
        if alert.alert_level.value > max_level.value:
            max_level = alert.alert_level

    remedies: List[Remedy] = []

    if "flood" in needed_types:
        remedies.extend(_FLOOD_REMEDIES)
    if "congestion" in needed_types:
        remedies.extend(_CONGESTION_REMEDIES)
    if "power_outage" in needed_types:
        remedies.extend(_POWER_OUTAGE_REMEDIES)

    # For ORANGE+ alerts, always include general remedies
    if max_level in (AlertLevel.ORANGE, AlertLevel.RED):
        remedies.extend(_GENERAL_REMEDIES)

    # For multi_hazard, include all categories
    if "multi_hazard" in needed_types:
        all_remedies = _FLOOD_REMEDIES + _CONGESTION_REMEDIES + _POWER_OUTAGE_REMEDIES + _GENERAL_REMEDIES
        existing_actions = {r.action for r in remedies}
        for r in all_remedies:
            if r.action not in existing_actions:
                remedies.append(r)

    # De-duplicate by action text
    seen = set()
    unique: List[Remedy] = []
    for r in remedies:
        if r.action not in seen:
            seen.add(r.action)
            unique.append(r)

    # Sort by priority (1 = highest)
    unique.sort(key=lambda r: r.priority)

    return unique


# ---------------------------------------------------------------------------
# Rerouting suggestions
# ---------------------------------------------------------------------------

_REROUTING_MAP = {
    "incident": {
        "Downtown":         ["Suburbs via Ring Road", "Highway Corridor via Bypass"],
        "Industrial":       ["Highway Corridor via Industrial Bypass", "Suburbs"],
        "Highway Corridor": ["Downtown via Arterial Roads", "Residential via Service Road"],
        "Suburbs":          ["Residential", "Downtown via Alternate Route"],
        "Residential":      ["Suburbs", "Highway Corridor"],
    },
    "construction": {
        "Downtown":         ["Suburbs via Ring Road"],
        "Industrial":       ["Highway Corridor via Alternate Entry"],
        "Highway Corridor": ["Service Road parallel route"],
        "Suburbs":          ["Residential feeder roads"],
        "Residential":      ["Suburbs connector"],
    },
    "congestion": {
        "Downtown":         ["Suburbs via Ring Road", "Highway Corridor via Express Lane"],
        "Highway Corridor": ["Residential via Service Road", "Downtown Arterials"],
    },
}


def suggest_rerouting(incidents: List[TrafficIncident]) -> List[ReroutingSuggestion]:
    """
    Given active incidents, suggest rerouting away from affected zones.
    Maps incident lat/lon to the nearest zone and provides alternatives.
    """
    if not incidents:
        return []

    suggestions: List[ReroutingSuggestion] = []
    seen_zones: set[str] = set()

    for inc in incidents:
        zone = _assign_zone(inc.lat, inc.lng)
        inc_type = inc.type or "incident"

        if zone in seen_zones:
            continue
        seen_zones.add(zone)

        routes = _REROUTING_MAP.get(inc_type, _REROUTING_MAP.get("incident", {}))
        alt_routes = routes.get(zone, ["Use alternate route — check live navigation"])

        reason_parts = []
        if inc.short_desc:
            reason_parts.append(inc.short_desc)
        else:
            reason_parts.append(f"Active {inc_type} reported")
        if inc.severity and inc.severity >= 3:
            reason_parts.append("(high severity)")

        suggestions.append(ReroutingSuggestion(
            avoid_zone=zone,
            reason=" ".join(reason_parts),
            alternative_routes=alt_routes,
            incident_id=inc.id,
        ))

    return suggestions


def _assign_zone(lat: float | None, lng: float | None) -> str:
    """
    Rough zone assignment based on coordinates.
    In a real system this would use geo-fencing or a spatial index.
    Here we use a simple quadrant-based heuristic.
    """
    if lat is None or lng is None:
        return "Downtown"  # default

    # Simple hash-based distribution for demo purposes
    lat_frac = abs(lat) % 1
    lng_frac = abs(lng) % 1

    if lat_frac < 0.2:
        return "Downtown"
    elif lat_frac < 0.4:
        return "Industrial"
    elif lat_frac < 0.6:
        return "Suburbs"
    elif lat_frac < 0.8:
        return "Highway Corridor"
    else:
        return "Residential"
