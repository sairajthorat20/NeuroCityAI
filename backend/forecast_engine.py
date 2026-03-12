"""
forecast_engine.py
------------------
Core forecasting logic for the Traffic Forecaster EWS.

Combines:
  • Live weather forecast (Open-Meteo)
  • Live traffic incidents (MapQuest)
  • Pre-trained ML models (RealDataMLModel)

to produce an hourly traffic risk timeline with alerts.

Usage:
    from forecast_engine import generate_traffic_forecast
    result = await generate_traffic_forecast(lat=21.97, lon=78.98, hours=48)
"""

from __future__ import annotations

import datetime
from typing import List, Optional

from config import ALERT_THRESHOLDS
from models_forecast import (
    AlertLevel,
    ForecastHour,
    RiskScores,
    TrafficAlert,
    TrafficForecastResponse,
    TrafficIncident,
    WeatherDataPoint,
)
from weather_service import fetch_weather_forecast
from traffic_incidents_service import fetch_traffic_incidents
from remedy_engine import generate_remedies, suggest_rerouting

# ---------------------------------------------------------------------------
# Module-level reference to the pre-trained ML model.
# Set by main.py at startup via `set_ml_model()`.
# ---------------------------------------------------------------------------
_ml_model = None


def set_ml_model(model) -> None:
    """Register the loaded RealDataMLModel instance for the forecast engine."""
    global _ml_model
    _ml_model = model


# ---------------------------------------------------------------------------
# Alert classification
# ---------------------------------------------------------------------------

def classify_alert_level(risk_score: float) -> AlertLevel:
    """Map a 0-1 risk score to an AlertLevel."""
    for level_name in ("RED", "ORANGE", "YELLOW", "GREEN"):
        lo, hi = ALERT_THRESHOLDS[level_name]
        if lo <= risk_score <= hi:
            return AlertLevel(level_name)
    return AlertLevel.GREEN


# ---------------------------------------------------------------------------
# Normalisation helpers (weather → ML feature space)
# ---------------------------------------------------------------------------

# Approximate max values used during training (from train_real_data.py dataset)
_PRECIP_MAX = 50.0   # mm/h
_WIND_MAX   = 60.0   # km/h
_TEMP_RANGE = (0.0, 50.0)


def _normalise_weather(dp: WeatherDataPoint) -> dict:
    """Convert a WeatherDataPoint into the feature dict expected by RealDataMLModel."""
    precip = dp.precipitation_mm or 0.0
    rain   = dp.rain_mm or 0.0
    wind   = dp.wind_speed_kmh or 0.0
    temp   = dp.temperature_c or 28.0
    humid  = dp.humidity_pct or 55.0
    cloud  = dp.cloud_cover_pct or 30.0
    soil   = dp.soil_moisture or 0.25
    gusts  = dp.wind_gusts_kmh or 0.0

    rainfall_intensity = min(max(precip, rain) / _PRECIP_MAX, 1.0)

    # Estimate flow_norm and speed_norm from weather severity
    weather_severity = 0.3 * rainfall_intensity + 0.3 * min(wind / _WIND_MAX, 1.0) + 0.2 * (cloud / 100.0) + 0.2 * max(0, (temp - 38) / 12.0)
    weather_severity = min(max(weather_severity, 0.0), 1.0)

    flow_norm  = 0.5 + 0.3 * weather_severity   # more severe → higher vehicle density
    occupancy  = 0.4 + 0.3 * weather_severity
    speed_norm = max(0.1, 1.0 - weather_severity)

    # Parse hour/month from time string
    hour = 12
    month = 6
    is_weekend = 0
    try:
        dt = datetime.datetime.fromisoformat(dp.time)
        hour = dt.hour
        month = dt.month
        is_weekend = int(dt.weekday() >= 5)
    except Exception:
        pass

    return {
        "rainfall_intensity": rainfall_intensity,
        "flow_norm":          flow_norm,
        "occupancy":          occupancy,
        "speed_norm":         speed_norm,
        "temperature_avg":    temp,
        "humidity_avg":       humid,
        "wind_avg":           wind,
        "soil_moist_avg":     soil,
        "cloud_avg":          cloud,
        "disaster_context":   0.35,   # default; could be enhanced with EM-DAT
        "is_peak_hour":       int(hour in {8, 9, 17, 18, 19}),
        "is_weekend":         is_weekend,
        "month":              month,
    }


# ---------------------------------------------------------------------------
# Alert generation
# ---------------------------------------------------------------------------

def _generate_alerts(timeline: List[ForecastHour]) -> List[TrafficAlert]:
    """Scan the hourly timeline and generate consolidated alerts for periods above YELLOW."""
    alerts: List[TrafficAlert] = []
    current_alert = None

    for fh in timeline:
        if fh.alert_level in (AlertLevel.YELLOW, AlertLevel.ORANGE, AlertLevel.RED):
            # Determine dominant risk type
            rs = fh.risk_scores
            dominant = "multi_hazard"
            max_risk = 0
            for risk_type, score in [("congestion", rs.congestion), ("flood", rs.flood), ("power_outage", rs.power_outage)]:
                if score > max_risk:
                    max_risk = score
                    dominant = risk_type

            if current_alert and current_alert.alert_type == dominant:
                # Extend existing alert window
                current_alert.time_end = fh.time
                current_alert.risk_score = max(current_alert.risk_score, rs.overall)
                if fh.alert_level.value > current_alert.alert_level.value:
                    current_alert.alert_level = fh.alert_level
            else:
                # Close previous alert and start a new one
                if current_alert:
                    alerts.append(current_alert)
                current_alert = TrafficAlert(
                    alert_level=fh.alert_level,
                    alert_type=dominant,
                    time_start=fh.time,
                    time_end=fh.time,
                    description=_alert_description(dominant, fh.alert_level, fh.contributing_factors),
                    risk_score=rs.overall,
                )
        else:
            if current_alert:
                alerts.append(current_alert)
                current_alert = None

    if current_alert:
        alerts.append(current_alert)

    return alerts


def _alert_description(risk_type: str, level: AlertLevel, factors: List[str]) -> str:
    """Generate a human-readable alert description."""
    level_labels = {
        AlertLevel.YELLOW: "Watch",
        AlertLevel.ORANGE: "Warning",
        AlertLevel.RED:    "CRITICAL",
    }
    severity = level_labels.get(level, "Advisory")

    type_labels = {
        "congestion":    "Traffic Congestion",
        "flood":         "Flood Risk / Road Flooding",
        "power_outage":  "Power Outage / Signal Failure",
        "multi_hazard":  "Multiple Hazards",
    }
    type_label = type_labels.get(risk_type, risk_type.replace("_", " ").title())

    desc = f"{severity}: {type_label} expected."
    if factors:
        desc += f" Contributing factors: {', '.join(factors[:4])}."
    return desc


def _contributing_factors(dp: WeatherDataPoint, rs: RiskScores) -> List[str]:
    """Identify the top contributing factors for a given hour."""
    factors = []
    precip = dp.precipitation_mm or 0.0
    wind = dp.wind_speed_kmh or 0.0
    temp = dp.temperature_c or 28.0

    if precip > 5.0:
        factors.append(f"Heavy rainfall ({precip:.1f}mm)")
    elif precip > 2.0:
        factors.append(f"Moderate rain ({precip:.1f}mm)")
    if wind > 40:
        factors.append(f"High winds ({wind:.0f}km/h)")
    elif wind > 25:
        factors.append(f"Strong winds ({wind:.0f}km/h)")
    if temp > 40:
        factors.append(f"Extreme heat ({temp:.0f}°C)")
    if rs.flood > 0.5:
        factors.append("Elevated flood risk")
    if rs.power_outage > 0.5:
        factors.append("Power outage risk")

    try:
        dt = datetime.datetime.fromisoformat(dp.time)
        if dt.hour in {8, 9, 17, 18, 19}:
            factors.append("Peak traffic hour")
    except Exception:
        pass

    return factors


# ---------------------------------------------------------------------------
# Incident risk amplification
# ---------------------------------------------------------------------------

def _incident_risk_boost(incidents: List[TrafficIncident]) -> float:
    """
    Calculate an additive risk boost (0-0.3) from active incidents.
    More / more severe incidents → higher boost.
    """
    if not incidents:
        return 0.0

    boost = 0.0
    for inc in incidents:
        sev = inc.severity or 1
        if inc.impacting:
            boost += 0.02 * sev
        else:
            boost += 0.01 * sev
    return min(boost, 0.30)


# ---------------------------------------------------------------------------
# Main forecast generator
# ---------------------------------------------------------------------------

async def generate_traffic_forecast(
    lat: float,
    lon: float,
    hours: int = 48,
    radius_km: float = 25.0,
) -> TrafficForecastResponse:
    """
    Generate a full traffic risk forecast combining weather, incidents, and ML.

    Parameters
    ----------
    lat, lon : float
        Location coordinates.
    hours : int
        Forecast window in hours (default 48).
    radius_km : float
        Radius for incident search (default 25 km).

    Returns
    -------
    TrafficForecastResponse
    """
    # 1. Fetch live data concurrently
    import asyncio
    weather_task = fetch_weather_forecast(lat, lon, hours)
    incidents_task = fetch_traffic_incidents(lat, lon, radius_km)
    weather_forecast, incidents_resp = await asyncio.gather(weather_task, incidents_task)

    incidents = incidents_resp.incidents
    incident_boost = _incident_risk_boost(incidents)

    # 2. Build hourly timeline
    timeline: List[ForecastHour] = []

    for dp in weather_forecast.hours:
        features = _normalise_weather(dp)
        rs = _predict_risks(features, incident_boost)
        factors = _contributing_factors(dp, rs)
        level = classify_alert_level(rs.overall)

        fh = ForecastHour(
            time=dp.time,
            weather=dp,
            risk_scores=rs,
            alert_level=level,
            contributing_factors=factors,
        )
        timeline.append(fh)

    # 3. Generate alerts from timeline
    alerts = _generate_alerts(timeline)

    # 4. Generate remedies based on the highest-risk hours
    remedies = generate_remedies(timeline, alerts)

    # 5. Suggest rerouting based on active incidents
    rerouting = suggest_rerouting(incidents)

    # 6. Build summary
    risk_scores = [fh.risk_scores.overall for fh in timeline]
    max_risk = max(risk_scores) if risk_scores else 0.0
    avg_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 0.0
    peak_hour = timeline[risk_scores.index(max_risk)].time if risk_scores else "N/A"

    summary = {
        "max_risk_score":    round(max_risk, 4),
        "avg_risk_score":    round(avg_risk, 4),
        "peak_risk_hour":    peak_hour,
        "total_alerts":      len(alerts),
        "active_incidents":  len(incidents),
        "red_hours":         sum(1 for fh in timeline if fh.alert_level == AlertLevel.RED),
        "orange_hours":      sum(1 for fh in timeline if fh.alert_level == AlertLevel.ORANGE),
        "yellow_hours":      sum(1 for fh in timeline if fh.alert_level == AlertLevel.YELLOW),
        "green_hours":       sum(1 for fh in timeline if fh.alert_level == AlertLevel.GREEN),
        "weather_source":    weather_forecast.source,
        "incidents_source":  incidents_resp.source,
        "incidents_warning": incidents_resp.warning,
    }

    return TrafficForecastResponse(
        latitude=lat,
        longitude=lon,
        forecast_hours=hours,
        generated_at=datetime.datetime.utcnow().isoformat(),
        timeline=timeline,
        alerts=alerts,
        remedies=remedies,
        rerouting=rerouting,
        active_incidents=incidents,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# ML prediction wrapper
# ---------------------------------------------------------------------------

def _predict_risks(features: dict, incident_boost: float = 0.0) -> RiskScores:
    """
    Run the pre-trained ML models to estimate risk probabilities.
    Falls back to heuristic estimates if no model is loaded.
    """
    if _ml_model is not None:
        try:
            congestion = _ml_model.predict_congestion_probability(
                rainfall_intensity=features["rainfall_intensity"],
                flow_norm=features["flow_norm"],
                occupancy=features["occupancy"],
                speed_norm=features["speed_norm"],
                hour=12,  # will be derived from is_peak_hour
                month=features["month"],
                is_weekend=features["is_weekend"],
                temperature_avg=features["temperature_avg"],
                humidity_avg=features["humidity_avg"],
                wind_avg=features["wind_avg"],
                soil_moist_avg=features["soil_moist_avg"],
                cloud_avg=features["cloud_avg"],
                disaster_context=features["disaster_context"],
            )
            flood = _ml_model.predict_flood_probability(
                rainfall_intensity=features["rainfall_intensity"],
                flow_norm=features["flow_norm"],
                occupancy=features["occupancy"],
                speed_norm=features["speed_norm"],
                month=features["month"],
                temperature_avg=features["temperature_avg"],
                humidity_avg=features["humidity_avg"],
                wind_avg=features["wind_avg"],
                soil_moist_avg=features["soil_moist_avg"],
                cloud_avg=features["cloud_avg"],
                disaster_context=features["disaster_context"],
            )
            power_outage = _ml_model.predict_power_outage_probability(
                wind_avg=features["wind_avg"],
                temperature_avg=features["temperature_avg"],
                rainfall_intensity=features["rainfall_intensity"],
                flow_norm=features["flow_norm"],
                occupancy=features["occupancy"],
                speed_norm=features["speed_norm"],
                humidity_avg=features["humidity_avg"],
                soil_moist_avg=features["soil_moist_avg"],
                cloud_avg=features["cloud_avg"],
                disaster_context=features["disaster_context"],
            )
            disaster_sev = _ml_model.predict_disaster_severity(
                disaster_context=features["disaster_context"],
                rainfall_intensity=features["rainfall_intensity"],
                temperature_avg=features["temperature_avg"],
            )
        except Exception as exc:
            print(f"[ForecastEngine] ML prediction failed: {exc}")
            congestion, flood, power_outage, disaster_sev = _heuristic_risks(features)
    else:
        congestion, flood, power_outage, disaster_sev = _heuristic_risks(features)

    # Composite overall risk score
    overall = (
        0.40 * congestion +
        0.30 * flood +
        0.20 * power_outage +
        0.10 * disaster_sev
    )
    overall = min(overall + incident_boost, 1.0)

    return RiskScores(
        congestion=round(congestion, 4),
        flood=round(flood, 4),
        power_outage=round(power_outage, 4),
        disaster_severity=round(disaster_sev, 4),
        overall=round(overall, 4),
    )


def _heuristic_risks(features: dict):
    """Simple heuristic risk estimation when no ML model is available."""
    rain = features.get("rainfall_intensity", 0.0)
    wind = features.get("wind_avg", 0.0) / 60.0  # normalise
    temp = features.get("temperature_avg", 28.0)
    peak = features.get("is_peak_hour", 0)

    congestion   = min(0.2 + 0.4 * rain + 0.2 * peak + 0.1 * wind, 1.0)
    flood        = min(0.5 * rain + 0.2 * (rain > 0.3) + 0.1 * wind, 1.0)
    power_outage = min(0.3 * wind + 0.2 * (temp > 38) * ((temp - 38) / 12), 1.0)
    disaster_sev = min(0.3 * rain + 0.2 * wind + 0.1 * max(0, (temp - 35) / 15), 1.0)

    return congestion, flood, power_outage, disaster_sev
