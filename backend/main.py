"""
main.py
-------
UrbanPulse AI — Traffic Forecaster / Early Warning System (EWS)

Combines pre-trained ML models with live weather data (Open-Meteo)
and live traffic incidents (MapQuest) to produce hourly traffic risk
forecasts, alerts, and actionable remedies.

Run with:
    uvicorn main:app --reload --port 8000

Original endpoints (preserved):
    GET  /health
    GET  /ml/status
    GET  /data/status

NEW EWS endpoints:
    GET  /forecast              — Full traffic forecast with alerts & remedies
    GET  /forecast/weather      — Raw weather forecast for a location
    GET  /forecast/incidents    — Live traffic incidents for a location
    GET  /forecast/alerts       — Just the generated alerts
    GET  /forecast/remedies     — Suggested remedies for current conditions
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import datetime

from config import DEFAULT_LAT, DEFAULT_LON, FORECAST_HOURS, DEFAULT_RADIUS_KM, REAL_MODELS_DIR
from models_forecast import (
    WeatherForecast,
    TrafficIncidentsResponse,
    TrafficForecastResponse,
    TrafficAlert,
    Remedy,
    ReroutingSuggestion,
)
from real_data_ml_model import RealDataMLModel
from forecast_engine import generate_traffic_forecast, set_ml_model
from weather_service import fetch_weather_forecast
from traffic_incidents_service import fetch_traffic_incidents
from database import init_db
from auth import router as auth_router
from chat import router as chat_router

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="UrbanPulse AI — Traffic Forecaster EWS",
    description=(
        "City-scale traffic risk forecasting and early warning system. "
        "Combines live weather forecasts (Open-Meteo), real-time traffic "
        "incidents (MapQuest), and pre-trained ML models to predict "
        "traffic congestion, flood risk, power outages, and more. "
        "Provides actionable remedies and rerouting suggestions."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register feature routers
app.include_router(auth_router)
app.include_router(chat_router)

# ---------------------------------------------------------------------------
# Startup: load pre-trained ML models
# ---------------------------------------------------------------------------
_real_model: Optional[RealDataMLModel] = None


@app.on_event("startup")
async def startup_event():
    global _real_model
    # Initialise SQLite database (creates tables if not present)
    init_db()
    print("[UrbanPulse EWS] Database initialised.")
    try:
        print("[UrbanPulse EWS] Loading pre-trained ML models...")
        _real_model = RealDataMLModel(models_dir=REAL_MODELS_DIR)
        _real_model.load(verbose=True)
        set_ml_model(_real_model)
        print("[UrbanPulse EWS] ML models loaded and registered with forecast engine.")
    except Exception as e:
        print(f"[UrbanPulse EWS] ML model load failed ({e}). Forecast will use heuristic fallback.")


# ═══════════════════════════════════════════════════════════════════════════
# EWS ENDPOINTS — Traffic Forecasting
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/forecast", response_model=TrafficForecastResponse, tags=["Forecast EWS"])
async def traffic_forecast(
    lat: float = Query(default=DEFAULT_LAT, description="Latitude"),
    lon: float = Query(default=DEFAULT_LON, description="Longitude"),
    hours: int = Query(default=FORECAST_HOURS, ge=1, le=384, description="Forecast hours (1-384)"),
    radius_km: float = Query(default=DEFAULT_RADIUS_KM, ge=1, le=100, description="Incident search radius in km"),
):
    """
    🚦 **Main EWS Endpoint** — Full traffic risk forecast.

    Returns an hourly timeline of traffic risk scores (congestion, flood,
    power outage), generated alerts, actionable remedies with priority
    rankings, rerouting suggestions, and active traffic incidents.

    **Data sources:**
    - Weather forecast: Open-Meteo (free, no key)
    - Traffic incidents: MapQuest (set MAPQUEST_API_KEY env var)
    - Risk models: Pre-trained ML (RandomForest / GradientBoosting)

    **Example:**
        GET /forecast?lat=21.97&lon=78.98&hours=24&radius_km=25
    """
    return await generate_traffic_forecast(lat, lon, hours, radius_km)


@app.get("/forecast/weather", response_model=WeatherForecast, tags=["Forecast EWS"])
async def forecast_weather(
    lat: float = Query(default=DEFAULT_LAT, description="Latitude"),
    lon: float = Query(default=DEFAULT_LON, description="Longitude"),
    hours: int = Query(default=FORECAST_HOURS, ge=1, le=384, description="Forecast hours"),
):
    """
    🌦️ **Weather Forecast** — Raw hourly weather data from Open-Meteo.

    Returns temperature, precipitation, wind speed, cloud cover,
    humidity, soil moisture, and WMO weather codes.

    **Example:**
        GET /forecast/weather?lat=19.076&lon=72.877&hours=48
    """
    return await fetch_weather_forecast(lat, lon, hours)


@app.get("/forecast/incidents", response_model=TrafficIncidentsResponse, tags=["Forecast EWS"])
async def forecast_incidents(
    lat: float = Query(default=DEFAULT_LAT, description="Latitude"),
    lon: float = Query(default=DEFAULT_LON, description="Longitude"),
    radius_km: float = Query(default=DEFAULT_RADIUS_KM, ge=1, le=100, description="Search radius in km"),
):
    """
    🚧 **Live Traffic Incidents** — From MapQuest Traffic API.

    Returns construction, accidents, congestion events, and road closures
    within the specified radius. Requires MAPQUEST_API_KEY env var.

    **Example:**
        GET /forecast/incidents?lat=19.076&lon=72.877&radius_km=30
    """
    return await fetch_traffic_incidents(lat, lon, radius_km)


@app.get("/forecast/alerts", tags=["Forecast EWS"])
async def forecast_alerts(
    lat: float = Query(default=DEFAULT_LAT, description="Latitude"),
    lon: float = Query(default=DEFAULT_LON, description="Longitude"),
    hours: int = Query(default=FORECAST_HOURS, ge=1, le=384, description="Forecast hours"),
):
    """
    ⚠️ **Forecast Alerts Only** — Extracted alerts from the full forecast.

    Returns just the generated alerts (YELLOW/ORANGE/RED) without the
    full hourly timeline, for lightweight monitoring dashboards.

    **Example:**
        GET /forecast/alerts?lat=21.97&lon=78.98&hours=24
    """
    result = await generate_traffic_forecast(lat, lon, hours)
    return {
        "latitude": lat,
        "longitude": lon,
        "forecast_hours": hours,
        "generated_at": result.generated_at,
        "total_alerts": len(result.alerts),
        "alerts": result.alerts,
        "summary": {
            "max_risk_score": result.summary.get("max_risk_score"),
            "red_hours": result.summary.get("red_hours"),
            "orange_hours": result.summary.get("orange_hours"),
            "yellow_hours": result.summary.get("yellow_hours"),
        },
    }


@app.get("/forecast/remedies", tags=["Forecast EWS"])
async def forecast_remedies(
    lat: float = Query(default=DEFAULT_LAT, description="Latitude"),
    lon: float = Query(default=DEFAULT_LON, description="Longitude"),
    hours: int = Query(default=24, ge=1, le=384, description="Forecast hours to analyse"),
):
    """
    💊 **Suggested Remedies** — Actionable recommendations.

    Analyses the forecast and returns prioritised remedies to prevent
    or mitigate predicted traffic events, plus rerouting suggestions
    based on any active incidents.

    **Example:**
        GET /forecast/remedies?lat=21.97&lon=78.98&hours=24
    """
    result = await generate_traffic_forecast(lat, lon, hours)
    return {
        "latitude": lat,
        "longitude": lon,
        "generated_at": result.generated_at,
        "total_remedies": len(result.remedies),
        "remedies": result.remedies,
        "rerouting": result.rerouting,
        "alert_summary": {
            "total_alerts": len(result.alerts),
            "max_alert_level": max((a.alert_level for a in result.alerts), default="GREEN"),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["System"])
def health_check():
    """System health check."""
    return {
        "status": "ok",
        "engine": "UrbanPulse AI — Traffic Forecaster EWS",
        "version": "2.0.0",
        "ml_models_loaded": _real_model is not None and _real_model.using_real_data,
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }


@app.get("/ml/status", tags=["ML"])
def ml_status():
    """ML model status and accuracy metrics."""
    if _real_model and _real_model.using_real_data:
        return {
            "trained": True,
            "models": ["congestion", "flood", "power_outage", "disaster_severity"],
            "accuracy_metrics": _real_model.metrics,
            "source": "Real Dataset (traffic.csv + open-meteo + EM-DAT)",
        }
    return {
        "trained": False,
        "message": "No ML models loaded. Forecast uses heuristic fallback.",
    }


@app.get("/data/status", tags=["System"])
def data_status():
    """
    Shows which data sources power the system.
    """
    ml_info = {}
    if _real_model and _real_model.using_real_data:
        ml_info = {
            "ml_source": "Real Dataset (traffic.csv + open-meteo + EM-DAT)",
            "models_dir": str(REAL_MODELS_DIR),
            "metrics": _real_model.metrics,
        }
    else:
        ml_info = {"ml_source": "Heuristic fallback (no ML models)"}

    return {
        **ml_info,
        "weather_api": "Open-Meteo (free, no key required)",
        "incidents_api": "MapQuest Traffic API (free tier, key required)",
        "forecast_engine": "weather + incidents + ML → hourly risk timeline",
    }
