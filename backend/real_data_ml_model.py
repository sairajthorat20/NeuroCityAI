"""
real_data_ml_model.py
---------------------
Loads the pre-trained pickle models (from train_real_data.py) and plugs them
into the UrbanPulse AI backend, replacing both the synthetic fallback AND the
IUTF model with real dataset-trained classifiers.

Datasets used for training:
  • traffic.csv               – 14,592 hourly vehicle counts, 4 junctions, 2015-2017
  • open-meteo-*.xlsx         – 91,325 hourly weather readings, MP India, 2000-2009
  • public_emdat_project.csv  – 795 India disaster events, 2000-2024 (EM-DAT)

Features (13 total):
  rainfall_intensity    normalised hourly precipitation     (weather)
  flow_norm             normalised total vehicle flow       (traffic)
  occupancy             flow × flood_risk_rate proxy        (derived)
  speed_norm            1 - flow_norm                       (derived)
  temperature_avg       seasonal mean temperature °C        (weather)
  humidity_avg          seasonal mean relative humidity %   (weather)
  wind_avg              seasonal mean wind speed km/h       (weather)
  soil_moist_avg        seasonal mean soil moisture         (weather)
  cloud_avg             seasonal mean cloud cover %         (weather)
  disaster_context      annual EM-DAT severity score 0-1   (EM-DAT)
  is_peak_hour          1 if hour in {8,9,17,18,19}         (temporal)
  is_weekend            1 if Saturday or Sunday             (temporal)
  month                 calendar month 1-12                 (temporal)

Usage:
    from backend.real_data_ml_model import RealDataMLModel, get_real_model

    model = get_real_model(models_dir="/tmp/models")
    prob  = model.predict_congestion_probability(
                rainfall_intensity=0.4,
                flow_norm=0.7,
                occupancy=0.6,
                speed_norm=0.3,
            )
    edge_probs = model.get_edge_probabilities()
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Default feature values (seasonal medians for MP, India)
# Used when optional features are not supplied at inference time.
# ---------------------------------------------------------------------------
FEATURE_DEFAULTS = {
    "rainfall_intensity": 0.0,
    "flow_norm":          0.5,
    "occupancy":          0.4,
    "speed_norm":         0.5,
    "temperature_avg":    28.0,
    "humidity_avg":       55.0,
    "wind_avg":           10.0,
    "soil_moist_avg":     0.25,
    "cloud_avg":          30.0,
    "disaster_context":   0.35,
    "is_peak_hour":       0,
    "is_weekend":         0,
    "month":              6,
}

FEATURE_ORDER = list(FEATURE_DEFAULTS.keys())  # must match training order


class RealDataMLModel:
    """
    Wraps four pre-trained sklearn pipelines for disaster prediction.

    Models
    ------
    congestion_model.pkl    RandomForest   → traffic congestion
    flood_model.pkl         GradientBoost  → flood risk
    power_outage_model.pkl  RandomForest   → power outage risk
    disaster_sev_model.pkl  GradientBoost  → high-disaster-year flag

    Parameters
    ----------
    models_dir : str | Path
        Directory containing the .pkl files and edge_probabilities.json.
        Default: /tmp/models  (written by train_real_data.py)
    """

    def __init__(self, models_dir: str | Path = "/tmp/models"):
        self.models_dir = Path(models_dir)
        self._models:     Dict[str, object] = {}
        self._edge_probs: Dict[Tuple[str, str], float] = {}
        self._metrics:    Dict[str, dict] = {}
        self._loaded:     bool = False

    # ------------------------------------------------------------------
    def load(self, verbose: bool = True) -> "RealDataMLModel":
        """Load all .pkl models and the edge probability JSON."""
        model_files = {
            "congestion":   "congestion_model.pkl",
            "flood":        "flood_model.pkl",
            "power_outage": "power_outage_model.pkl",
            "disaster_sev": "disaster_sev_model.pkl",
        }

        for name, fname in model_files.items():
            path = self.models_dir / fname
            if not path.exists():
                raise FileNotFoundError(
                    f"Model file not found: {path}\n"
                    f"Run train_real_data.py first to generate the .pkl files."
                )
            with open(path, "rb") as f:
                self._models[name] = pickle.load(f)
            if verbose:
                print(f"[RealDataMLModel] Loaded {name} <- {fname}")

        # Load empirical edge probabilities
        ep_path = self.models_dir / "edge_probabilities.json"
        if ep_path.exists():
            with open(ep_path) as f:
                raw = json.load(f)
            self._edge_probs = {
                tuple(k.split("|||")): v for k, v in raw.items()
            }
            if verbose:
                print(f"[RealDataMLModel] Loaded {len(self._edge_probs)} edge probabilities")

        # Load training summary metrics if available
        summary_path = self.models_dir / "training_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            self._metrics = summary.get("models", {})

        self._loaded = True
        return self

    # ------------------------------------------------------------------
    def _require_loaded(self):
        if not self._loaded:
            raise RuntimeError(
                "Models not loaded. Call .load() first, or use get_real_model()."
            )

    def _predict(self, model_name: str, feature_dict: dict) -> float:
        """Build feature vector from dict and return positive-class probability."""
        self._require_loaded()
        vec = np.array(
            [feature_dict.get(f, FEATURE_DEFAULTS[f]) for f in FEATURE_ORDER],
            dtype=np.float32,
        ).reshape(1, -1)
        return float(self._models[model_name].predict_proba(vec)[0][1])

    # ------------------------------------------------------------------
    # Public inference API  (mirrors iutf_ml_model.py interface)
    # ------------------------------------------------------------------

    def predict_congestion_probability(
        self,
        rainfall_intensity: float,
        flow_norm: float,
        occupancy: float,
        speed_norm: float,
        hour: int = 12,
        month: int = 6,
        is_weekend: int = 0,
        **kwargs,
    ) -> float:
        """
        Predict traffic congestion probability.

        Core parameters (all 0-1 unless noted):
            rainfall_intensity  normalised precipitation
            flow_norm           normalised vehicle flow
            occupancy           road occupancy fraction
            speed_norm          normalised speed (1 = free flow)
            hour                hour of day (0-23), used to set is_peak_hour
            month               calendar month (1-12)
            is_weekend          1 if weekend
        """
        return round(self._predict("congestion", {
            "rainfall_intensity": rainfall_intensity,
            "flow_norm":          flow_norm,
            "occupancy":          occupancy,
            "speed_norm":         speed_norm,
            "is_peak_hour":       int(hour in {8, 9, 17, 18, 19}),
            "is_weekend":         is_weekend,
            "month":              month,
            **{k: kwargs.get(k, FEATURE_DEFAULTS[k]) for k in [
                "temperature_avg","humidity_avg","wind_avg",
                "soil_moist_avg","cloud_avg","disaster_context"
            ]},
        }), 4)

    def predict_flood_probability(
        self,
        rainfall_intensity: float,
        flow_norm: float = 0.5,
        occupancy: float = 0.3,
        speed_norm: float = 0.5,
        month: int = 7,
        **kwargs,
    ) -> float:
        """Predict flood risk probability (driven primarily by rainfall_intensity)."""
        return round(self._predict("flood", {
            "rainfall_intensity": rainfall_intensity,
            "flow_norm":          flow_norm,
            "occupancy":          occupancy,
            "speed_norm":         speed_norm,
            "month":              month,
            **{k: kwargs.get(k, FEATURE_DEFAULTS[k]) for k in [
                "temperature_avg","humidity_avg","wind_avg",
                "soil_moist_avg","cloud_avg","disaster_context","is_peak_hour","is_weekend"
            ]},
        }), 4)

    def predict_power_outage_probability(
        self,
        wind_avg: float,
        temperature_avg: float,
        rainfall_intensity: float = 0.0,
        flow_norm: float = 0.5,
        **kwargs,
    ) -> float:
        """Predict power outage probability (driven by wind + temperature)."""
        return round(self._predict("power_outage", {
            "wind_avg":           wind_avg,
            "temperature_avg":    temperature_avg,
            "rainfall_intensity": rainfall_intensity,
            "flow_norm":          flow_norm,
            **{k: kwargs.get(k, FEATURE_DEFAULTS[k]) for k in [
                "occupancy","speed_norm","humidity_avg","soil_moist_avg",
                "cloud_avg","disaster_context","is_peak_hour","is_weekend","month"
            ]},
        }), 4)

    def predict_disaster_severity(
        self,
        disaster_context: float,
        rainfall_intensity: float = 0.0,
        temperature_avg: float = 28.0,
        **kwargs,
    ) -> float:
        """
        Predict probability that current conditions constitute a high-severity
        disaster year (based on EM-DAT India annual severity index).
        """
        return round(self._predict("disaster_sev", {
            "disaster_context":   disaster_context,
            "rainfall_intensity": rainfall_intensity,
            "temperature_avg":    temperature_avg,
            **{k: kwargs.get(k, FEATURE_DEFAULTS[k]) for k in [
                "flow_norm","occupancy","speed_norm","humidity_avg",
                "wind_avg","soil_moist_avg","cloud_avg","is_peak_hour","is_weekend","month"
            ]},
        }), 4)

    # ------------------------------------------------------------------
    # Graph edge probabilities
    # ------------------------------------------------------------------

    def get_edge_probabilities(self) -> Dict[Tuple[str, str], float]:
        """
        Return empirical (src, dst) → probability dict for the causal graph.
        These were computed directly from the real dataset statistics
        in train_real_data.py, Step 5.
        """
        self._require_loaded()
        return dict(self._edge_probs)

    def estimate_edge_probabilities(self, severity: float) -> Dict[str, float]:
        """
        Convenience shim matching the original ml_model.py interface.
        Returns node-name → probability for the graph engine's use_ml path.
        """
        self._require_loaded()
        flood = self.predict_flood_probability(rainfall_intensity=severity, month=7)
        cong  = self.predict_congestion_probability(
            rainfall_intensity=severity,
            flow_norm=0.5 + 0.3 * severity,
            occupancy=0.4 + 0.3 * severity,
            speed_norm=max(0.1, 1.0 - severity),
        )
        power = self.predict_power_outage_probability(
            wind_avg=severity * 40.0,
            temperature_avg=20.0 + severity * 20.0,
            rainfall_intensity=severity,
        )
        return {
            "Flooding":           flood,
            "Traffic Congestion": cong,
            "Power Grid Failure": power,
        }

    @property
    def metrics(self) -> Dict[str, dict]:
        return self._metrics

    @property
    def using_real_data(self) -> bool:
        return self._loaded


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------
_global_real_model: Optional[RealDataMLModel] = None


def get_real_model(models_dir: str = "/tmp/models") -> RealDataMLModel:
    """Load and cache the pre-trained real-data model (singleton)."""
    global _global_real_model
    if _global_real_model is None:
        _global_real_model = RealDataMLModel(models_dir=models_dir)
        _global_real_model.load(verbose=True)
    return _global_real_model
