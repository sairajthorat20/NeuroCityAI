"""
chat.py
-------
Chatbot router for NeuroCity.
POST /chat — sends message to OpenAI (GPT-4o-mini) with live city context.
Falls back to a smart local response engine if OPENAI_API_KEY is not set.

Set the key via environment variable:
    set OPENAI_API_KEY=sk-...  (Windows CMD)
    $env:OPENAI_API_KEY="sk-..."  (PowerShell)
"""

import os
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/chat", tags=["Chatbot"])

OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    # Optional live city context passed from the frontend
    context: Optional[dict] = None


class ChatResponse(BaseModel):
    reply: str
    model: str  # "gpt-4o-mini" or "local-fallback"


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------
def build_system_prompt(context: dict | None) -> str:
    base = (
        "You are NeuroCity AI, a smart urban intelligence assistant for Pune, India. "
        "You help citizens with real-time information about traffic, flood risk, road closures, "
        "weather, power outages, and city events. "
        "Keep responses concise (2-4 sentences), factual, and actionable."
    )
    if context:
        risk = context.get("overall_risk", 0)
        flood = context.get("flood_risk", 0)
        congestion = context.get("congestion", 0)
        weather = context.get("weather_slogan", "")
        alerts = context.get("active_alerts", [])

        base += (
            f"\n\nCURRENT PUNE CITY STATUS (live data):"
            f"\n- Overall risk: {risk:.0%}"
            f"\n- Flood risk: {flood:.0%}"
            f"\n- Traffic congestion: {congestion:.0%}"
        )
        if weather:
            base += f"\n- Weather: {weather}"
        if alerts:
            base += f"\n- Active alerts: {'; '.join(str(a) for a in alerts[:3])}"
    return base


# ---------------------------------------------------------------------------
# Smart local fallback (works without any API key)
# ---------------------------------------------------------------------------
def smart_fallback(message: str, context: dict | None) -> str:
    q = message.lower()
    ctx = context or {}
    risk = ctx.get("overall_risk", 0)
    flood = ctx.get("flood_risk", 0)
    congestion = ctx.get("congestion", 0)
    weather = ctx.get("weather_slogan", "conditions appear normal")
    alerts = ctx.get("active_alerts", [])

    if any(w in q for w in ["flood", "water", "rain", "waterlog"]):
        level = "HIGH" if flood > 0.6 else "MODERATE" if flood > 0.3 else "LOW"
        return (
            f"Flood risk in Pune is currently {level} ({flood:.0%}). "
            f"{weather} "
            f"{'Avoid low-lying roads and underpasses.' if flood > 0.3 else 'No major waterlogging expected.'}"
        )

    if any(w in q for w in ["traffic", "congestion", "jam", "road", "drive", "commute"]):
        level = "severe" if congestion > 0.6 else "moderate" if congestion > 0.3 else "light"
        return (
            f"Traffic is {level} ({congestion:.0%}) across Pune right now. "
            f"{'Consider alternate routes or delay travel by 1-2 hours.' if congestion > 0.4 else 'Roads are moving well.'}"
        )

    if any(w in q for w in ["weather", "forecast", "temperature", "rain", "wind"]):
        return f"Current Pune conditions: {weather}"

    if any(w in q for w in ["safe", "clear", "okay", "normal", "status"]):
        level = "HIGH RISK" if risk > 0.6 else "MODERATE" if risk > 0.3 else "SAFE"
        note = alerts[0] if alerts else "No active alerts."
        return f"City status: {level} (overall risk {risk:.0%}). {note}"

    if any(w in q for w in ["power", "electricity", "outage", "light", "msedcl"]):
        return "No widespread power outage reports currently. Check msedcl.com for scheduled maintenance in your area."

    if any(w in q for w in ["route", "navigate", "get to", "direction"]):
        return (
            "For routing, avoid areas with active alerts. "
            "Currently, Shivajinagar–Hadapsar corridor and NH-48 may have delays. "
            "Consider Ring Road or Katraj bypass for east-west travel."
        )

    # General overview
    return (
        f"Pune city overview — Overall risk: {risk:.0%}, "
        f"Congestion: {congestion:.0%}, Flood: {flood:.0%}. "
        f"{weather} Ask me about traffic, floods, weather, routing, or power."
    )


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------
@router.post("", response_model=ChatResponse)
async def chat(body: ChatRequest):
    """Send a message to NeuroCity AI and get a response."""
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # ── Try OpenAI if API key is configured ──
    if OPENAI_API_KEY:
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=OPENAI_API_KEY)
            response = await client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": build_system_prompt(body.context)},
                    {"role": "user", "content": body.message},
                ],
                max_tokens=220,
                temperature=0.7,
            )
            reply = response.choices[0].message.content.strip()
            return {"reply": reply, "model": OPENAI_MODEL}
        except Exception as e:
            print(f"[NeuroCity Chat] OpenAI error: {e} — using local fallback")

    # ── Smart local fallback ──
    reply = smart_fallback(body.message, body.context)
    return {"reply": reply, "model": "local-fallback"}
