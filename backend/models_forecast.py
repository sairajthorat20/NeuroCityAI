"""
models_forecast.py
------------------
Pydantic models for the Traffic Forecaster EWS responses.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Dict
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AlertLevel(str, Enum):
    GREEN  = "GREEN"
    YELLOW = "YELLOW"
    ORANGE = "ORANGE"
    RED    = "RED"


class IncidentType(str, Enum):
    CONSTRUCTION = "construction"
    EVENT        = "event"
    CONGESTION   = "congestion"
    INCIDENT     = "incident"


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------

class WeatherDataPoint(BaseModel):
    """Single hourly weather observation / forecast."""
    time: str = Field(..., description="ISO-8601 timestamp")
    temperature_c: Optional[float] = Field(None, description="Temperature in °C")
    humidity_pct: Optional[float] = Field(None, description="Relative humidity %")
    precipitation_mm: Optional[float] = Field(None, description="Precipitation in mm")
    rain_mm: Optional[float] = Field(None, description="Rain in mm")
    weather_code: Optional[int] = Field(None, description="WMO weather code")
    cloud_cover_pct: Optional[float] = Field(None, description="Cloud cover %")
    wind_speed_kmh: Optional[float] = Field(None, description="Wind speed km/h")
    wind_gusts_kmh: Optional[float] = Field(None, description="Wind gusts km/h")
    soil_moisture: Optional[float] = Field(None, description="Soil moisture 0-7cm m³/m³")
    surface_pressure_hpa: Optional[float] = Field(None, description="Surface pressure hPa")


class WeatherForecast(BaseModel):
    """Collection of hourly weather data points for a location."""
    latitude: float
    longitude: float
    timezone: Optional[str] = None
    hours: List[WeatherDataPoint] = []
    source: str = "Open-Meteo"


# ---------------------------------------------------------------------------
# Traffic Incidents
# ---------------------------------------------------------------------------

class TrafficIncident(BaseModel):
    """Single traffic incident from MapQuest."""
    id: Optional[str] = None
    type: Optional[str] = Field(None, description="construction | event | congestion | incident")
    severity: Optional[int] = Field(None, description="Severity 0-4")
    short_desc: Optional[str] = None
    full_desc: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    impacting: Optional[bool] = None
    delay_seconds: Optional[float] = Field(None, description="Delay from typical in seconds")
    distance_km: Optional[float] = None


class TrafficIncidentsResponse(BaseModel):
    """Live traffic incidents for a bounding box."""
    latitude: float
    longitude: float
    radius_km: float
    incidents: List[TrafficIncident] = []
    total_count: int = 0
    source: str = "MapQuest"
    warning: Optional[str] = None


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------

class RiskScores(BaseModel):
    """ML-derived risk probabilities for a single hour."""
    congestion: float = Field(0.0, description="Traffic congestion probability 0-1")
    flood: float = Field(0.0, description="Flood risk probability 0-1")
    power_outage: float = Field(0.0, description="Power outage probability 0-1")
    disaster_severity: float = Field(0.0, description="Overall disaster severity 0-1")
    overall: float = Field(0.0, description="Composite risk score 0-1")


class ForecastHour(BaseModel):
    """Traffic risk forecast for a single hour."""
    time: str = Field(..., description="ISO-8601 timestamp")
    weather: WeatherDataPoint
    risk_scores: RiskScores
    alert_level: AlertLevel = AlertLevel.GREEN
    contributing_factors: List[str] = []


class TrafficAlert(BaseModel):
    """An alert generated when risk exceeds a threshold."""
    alert_level: AlertLevel
    alert_type: str = Field(..., description="congestion | flood | power_outage | multi_hazard")
    time_start: str
    time_end: Optional[str] = None
    description: str
    affected_zones: List[str] = []
    risk_score: float = 0.0


class Remedy(BaseModel):
    """Actionable recommendation to mitigate a forecasted risk."""
    action: str
    priority: int = Field(..., ge=1, le=5, description="1=highest priority, 5=lowest")
    target_zones: List[str] = []
    risk_type: str = Field(..., description="congestion | flood | power_outage | general")
    estimated_impact: str = Field("", description="Expected effect of implementing this remedy")


class ReroutingSuggestion(BaseModel):
    """Suggestion to reroute traffic away from affected zones."""
    avoid_zone: str
    reason: str
    alternative_routes: List[str] = []
    incident_id: Optional[str] = None


class TrafficForecastResponse(BaseModel):
    """Full traffic forecast response — the main EWS output."""
    latitude: float
    longitude: float
    forecast_hours: int
    generated_at: str
    timeline: List[ForecastHour] = []
    alerts: List[TrafficAlert] = []
    remedies: List[Remedy] = []
    rerouting: List[ReroutingSuggestion] = []
    active_incidents: List[TrafficIncident] = []
    summary: Dict[str, object] = Field(default_factory=dict, description="High-level summary stats")
