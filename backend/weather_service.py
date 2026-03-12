"""
weather_service.py
------------------
Fetches live hourly weather forecasts from the Open-Meteo API.
Free, no API key required.  https://open-meteo.com/en/docs

Usage:
    from weather_service import fetch_weather_forecast
    forecast = await fetch_weather_forecast(lat=21.97, lon=78.98, hours=48)
"""

from __future__ import annotations

import httpx
from typing import Optional

from config import OPEN_METEO_BASE_URL, OPEN_METEO_HOURLY_PARAMS
from models_forecast import WeatherDataPoint, WeatherForecast


async def fetch_weather_forecast(
    lat: float,
    lon: float,
    hours: int = 48,
    timeout: float = 15.0,
) -> WeatherForecast:
    """
    Call Open-Meteo and return a WeatherForecast for the next *hours* hours.

    Parameters
    ----------
    lat, lon : float
        Geographic coordinates.
    hours : int
        Number of forecast hours to include (max ~384 = 16 days).
    timeout : float
        HTTP request timeout in seconds.

    Returns
    -------
    WeatherForecast
        Typed forecast with a list of hourly WeatherDataPoints.
    """
    forecast_days = max(1, min((hours + 23) // 24, 16))

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(OPEN_METEO_HOURLY_PARAMS),
        "forecast_days": forecast_days,
        "timezone": "auto",
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(OPEN_METEO_BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as exc:
        print(f"[WeatherService] Open-Meteo request failed: {exc}")
        return WeatherForecast(latitude=lat, longitude=lon, hours=[], source="Open-Meteo (unavailable)")
    except Exception as exc:
        print(f"[WeatherService] Unexpected error: {exc}")
        return WeatherForecast(latitude=lat, longitude=lon, hours=[], source="Open-Meteo (error)")

    # Parse the hourly arrays from the response
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])

    data_points: list[WeatherDataPoint] = []
    for i, t in enumerate(times[:hours]):
        dp = WeatherDataPoint(
            time=t,
            temperature_c=_safe_idx(hourly.get("temperature_2m"), i),
            humidity_pct=_safe_idx(hourly.get("relative_humidity_2m"), i),
            precipitation_mm=_safe_idx(hourly.get("precipitation"), i),
            rain_mm=_safe_idx(hourly.get("rain"), i),
            weather_code=_safe_idx(hourly.get("weather_code"), i),
            cloud_cover_pct=_safe_idx(hourly.get("cloud_cover"), i),
            wind_speed_kmh=_safe_idx(hourly.get("wind_speed_10m"), i),
            wind_gusts_kmh=_safe_idx(hourly.get("wind_gusts_10m"), i),
            soil_moisture=_safe_idx(hourly.get("soil_moisture_0_to_7cm"), i),
            surface_pressure_hpa=_safe_idx(hourly.get("surface_pressure"), i),
        )
        data_points.append(dp)

    return WeatherForecast(
        latitude=lat,
        longitude=lon,
        timezone=data.get("timezone"),
        hours=data_points,
        source="Open-Meteo",
    )


def _safe_idx(arr: Optional[list], idx: int):
    """Return arr[idx] if arr is a non-empty list and idx is valid, else None."""
    if arr is None:
        return None
    if idx < 0 or idx >= len(arr):
        return None
    return arr[idx]
