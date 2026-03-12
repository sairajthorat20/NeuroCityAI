"""
config.py
---------
Centralized configuration for the UrbanPulse Traffic Forecaster EWS.

Environment variables:
    MAPQUEST_API_KEY    — Free-tier key from developer.mapquest.com
    DEFAULT_LAT         — Default latitude  (default: 21.97  — MP, India)
    DEFAULT_LON         — Default longitude (default: 78.98  — MP, India)
    FORECAST_HOURS      — Default forecast window (default: 48)
    REAL_MODELS_DIR     — Pre-trained model directory (default: current dir)
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------
MAPQUEST_API_KEY: str = "cIMVqp020N8YleZjnTvhspg6lX1yBhGT"

# ---------------------------------------------------------------------------
# Open-Meteo (no key required)
# ---------------------------------------------------------------------------
OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"

OPEN_METEO_HOURLY_PARAMS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "weather_code",
    "cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m",
    "soil_moisture_0_to_7cm",
    "surface_pressure",
]

# ---------------------------------------------------------------------------
# MapQuest Traffic API
# ---------------------------------------------------------------------------
MAPQUEST_INCIDENTS_URL = "https://www.mapquestapi.com/traffic/v2/incidents"
MAPQUEST_FILTERS = "construction,incidents,congestion,event"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_LAT: float = float(os.environ.get("DEFAULT_LAT", "21.97"))
DEFAULT_LON: float = float(os.environ.get("DEFAULT_LON", "78.98"))
FORECAST_HOURS: int = int(os.environ.get("FORECAST_HOURS", "48"))
DEFAULT_RADIUS_KM: float = 25.0

# ---------------------------------------------------------------------------
# Pre-trained models directory (same dir as this file by default)
# ---------------------------------------------------------------------------
REAL_MODELS_DIR: str = os.environ.get(
    "REAL_MODELS_DIR", str(Path(__file__).resolve().parent)
)

# ---------------------------------------------------------------------------
# Alert thresholds  (risk_score 0-1 → alert level)
# ---------------------------------------------------------------------------
ALERT_THRESHOLDS = {
    "GREEN":  (0.0,  0.30),
    "YELLOW": (0.30, 0.55),
    "ORANGE": (0.55, 0.75),
    "RED":    (0.75, 1.00),
}
