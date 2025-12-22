"""
Water/Streamflow Domain - USGS river flow prediction with hydrological regimes.

Regimes: low_flow, below_normal, normal, high_flow, flood
Methods: Persistence, MA7, MA30, Seasonal365, FloodThreshold
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from .base import DomainEnvironment, DomainMethod


WATER_REGIMES = ["low_flow", "below_normal", "normal", "high_flow", "flood"]
WATER_METHODS = ["Persistence", "MA7", "MA30", "Seasonal365", "FloodThreshold"]


class WaterMethod(DomainMethod):
    """Streamflow prediction method."""

    def __init__(self, name: str, optimal_regimes: List[str]):
        self.name = name
        self.optimal_regimes = optimal_regimes
        self._history = []
        self._flood_threshold = None

    def predict(self, current_flow: float, history: np.ndarray) -> float:
        """Predict next streamflow value."""
        if self.name == "Persistence":
            return current_flow
        elif self.name == "MA7":
            if len(history) >= 7:
                return np.mean(history[-7:])
            return current_flow
        elif self.name == "MA30":
            if len(history) >= 30:
                return np.mean(history[-30:])
            return current_flow
        elif self.name == "Seasonal365":
            if len(history) >= 365:
                return history[-365]
            elif len(history) >= 30:
                return history[-30]
            return current_flow
        elif self.name == "FloodThreshold":
            # If above threshold, predict recession
            if self._flood_threshold is None and len(history) > 0:
                self._flood_threshold = np.percentile(history, 95)

            if self._flood_threshold and current_flow > self._flood_threshold:
                # Flood recession model
                return current_flow * 0.85
            return current_flow
        return current_flow

    def execute(self, observation: np.ndarray) -> Dict:
        """Execute method on observation."""
        flow = observation[0] if len(observation) > 0 else 5000

        prediction = self.predict(flow, np.array(self._history))
        self._history.append(flow)
        if len(self._history) > 400:
            self._history = self._history[-365:]

        # Signal based on predicted change
        if flow > 0:
            pct_change = (prediction - flow) / flow
        else:
            pct_change = 0
        signal = np.clip(pct_change * 5, -1, 1)  # Scale for sensitivity
        return {"signal": signal, "prediction": prediction, "confidence": 0.5}


def load_water_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """Load USGS streamflow data from CSV."""
    if data_path is None:
        data_path = Path(__file__).parent.parent.parent / "data" / "water" / "usgs_streamflow.csv"

    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def create_water_environment(
    n_bars: int = 2000,
    gauge: str = "Colorado_River_CO",
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, WaterMethod]]:
    """
    Create streamflow prediction environment from real data.

    State: [flow_cfs, flow_ma7, flow_ma30, month, day_of_year]
    """
    df = load_water_data()

    # Filter by gauge
    gauge_df = df[df['gauge'] == gauge].copy()
    if len(gauge_df) == 0:
        gauge = df['gauge'].iloc[0]
        gauge_df = df[df['gauge'] == gauge].copy()

    gauge_df = gauge_df.sort_values('date').reset_index(drop=True)

    # Sample if needed
    if len(gauge_df) > n_bars:
        if seed is not None:
            np.random.seed(seed)
        start_idx = np.random.randint(0, len(gauge_df) - n_bars)
        gauge_df = gauge_df.iloc[start_idx:start_idx + n_bars].reset_index(drop=True)

    # Extract state features
    state_df = pd.DataFrame({
        'flow_cfs': gauge_df['flow_cfs'],
        'flow_ma7': gauge_df['flow_ma7'],
        'flow_ma30': gauge_df['flow_ma30'],
        'month': gauge_df['month'],
        'day_of_year': gauge_df['day_of_year'],
    })

    regimes = pd.Series(gauge_df['regime'].values)

    # Create methods
    methods = {
        "Persistence": WaterMethod("Persistence", ["normal", "below_normal"]),
        "MA7": WaterMethod("MA7", ["normal", "high_flow"]),
        "MA30": WaterMethod("MA30", ["normal", "low_flow"]),
        "Seasonal365": WaterMethod("Seasonal365", ["normal", "low_flow"]),
        "FloodThreshold": WaterMethod("FloodThreshold", ["high_flow", "flood"]),
    }

    return state_df, regimes, methods


class WaterDomain:
    """Wrapper for water/streamflow domain environment."""

    def __init__(self, n_bars: int = 2000, gauge: str = "Colorado_River_CO", seed: int = None):
        self.df, self.regimes, self.methods = create_water_environment(
            n_bars=n_bars, gauge=gauge, seed=seed
        )

    @property
    def regime_names(self):
        return WATER_REGIMES

    @property
    def method_names(self):
        return WATER_METHODS
