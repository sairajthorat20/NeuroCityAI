"""
traffic_incidents_service.py
----------------------------
Fetches live traffic incidents from the MapQuest Traffic API.
Requires a free API key from https://developer.mapquest.com

Usage:
    from traffic_incidents_service import fetch_traffic_incidents
    resp = await fetch_traffic_incidents(lat=21.97, lon=78.98, radius_km=25)
"""

from __future__ import annotations

import httpx
from typing import List

from config import MAPQUEST_API_KEY, MAPQUEST_INCIDENTS_URL, MAPQUEST_FILTERS
from models_forecast import TrafficIncident, TrafficIncidentsResponse


# MapQuest incident type codes → human labels
_INCIDENT_TYPE_MAP = {
    1: "construction",
    2: "event",
    3: "congestion",
    4: "incident",
}


async def fetch_traffic_incidents(
    lat: float,
    lon: float,
    radius_km: float = 25.0,
    timeout: float = 15.0,
) -> TrafficIncidentsResponse:
    """
    Fetch live traffic incidents within a bounding box around (lat, lon).

    The bounding box is computed from *radius_km* using a rough degree
    approximation (1° ≈ 111 km).

    Parameters
    ----------
    lat, lon : float
        Center coordinates.
    radius_km : float
        Radius in kilometres for the bounding box.
    timeout : float
        HTTP timeout seconds.

    Returns
    -------
    TrafficIncidentsResponse
    """
    if not MAPQUEST_API_KEY:
        return TrafficIncidentsResponse(
            latitude=lat,
            longitude=lon,
            radius_km=radius_km,
            incidents=[],
            total_count=0,
            source="MapQuest",
            warning="MAPQUEST_API_KEY not set — no live incident data available. "
                    "Get a free key at https://developer.mapquest.com",
        )

    # Approximate bounding box
    delta_lat = radius_km / 111.0
    delta_lon = radius_km / (111.0 * max(abs(_cos_deg(lat)), 0.01))

    north = lat + delta_lat
    south = lat - delta_lat
    east  = lon + delta_lon
    west  = lon - delta_lon

    # MapQuest expects: boundingBox=north,west,south,east
    bbox = f"{north},{west},{south},{east}"

    params = {
        "key": MAPQUEST_API_KEY,
        "boundingBox": bbox,
        "filters": MAPQUEST_FILTERS,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(MAPQUEST_INCIDENTS_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as exc:
        print(f"[IncidentService] MapQuest request failed: {exc}")
        return TrafficIncidentsResponse(
            latitude=lat, longitude=lon, radius_km=radius_km,
            incidents=[], total_count=0,
            source="MapQuest", warning=f"API request failed: {exc}",
        )
    except Exception as exc:
        print(f"[IncidentService] Unexpected error: {exc}")
        return TrafficIncidentsResponse(
            latitude=lat, longitude=lon, radius_km=radius_km,
            incidents=[], total_count=0,
            source="MapQuest", warning=f"Unexpected error: {exc}",
        )

    # Parse incidents from MapQuest response
    raw_incidents = data.get("incidents", [])
    incidents: List[TrafficIncident] = []

    for raw in raw_incidents:
        inc = TrafficIncident(
            id=str(raw.get("id", "")),
            type=_INCIDENT_TYPE_MAP.get(raw.get("type"), "unknown"),
            severity=raw.get("severity"),
            short_desc=raw.get("shortDesc", "").strip(),
            full_desc=raw.get("fullDesc", "").strip(),
            lat=raw.get("lat"),
            lng=raw.get("lng"),
            start_time=raw.get("startTime"),
            end_time=raw.get("endTime"),
            impacting=raw.get("impacting"),
            delay_seconds=raw.get("delayFromTypical"),
            distance_km=raw.get("distance"),
        )
        incidents.append(inc)

    return TrafficIncidentsResponse(
        latitude=lat,
        longitude=lon,
        radius_km=radius_km,
        incidents=incidents,
        total_count=len(incidents),
        source="MapQuest",
    )


def _cos_deg(degrees: float) -> float:
    """Cosine of an angle in degrees (for longitude scaling)."""
    import math
    return math.cos(math.radians(degrees))
