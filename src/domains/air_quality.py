"""
Air Quality Domain - EPA PM2.5/AQI prediction with quality regimes.

Regimes: good, moderate, unhealthy_sensitive, unhealthy
Methods: Persistence, MA(7), Seasonal(365), Trend
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from .base import DomainEnvironment, DomainMethod


AIR_QUALITY_REGIMES = ["good", "moderate", "unhealthy_sensitive", "unhealthy"]
AIR_QUALITY_METHODS = ["Persistence", "MA7", "Seasonal365", "Trend"]


class AirQualityMethod(DomainMethod):
    """Air quality prediction method."""

    def __init__(self, name: str, optimal_regimes: List[str]):
        self.name = name
        self.optimal_regimes = optimal_regimes
        self._history = []

    def predict(self, current_value: float, history: np.ndarray) -> float:
        """Predict next PM2.5 value."""
        if self.name == "Persistence":
            return current_value
        elif self.name == "MA7":
            if len(history) >= 7:
                return np.mean(history[-7:])
            return current_value
        elif self.name == "Seasonal365":
            if len(history) >= 365:
                return history[-365]
            elif len(history) >= 7:
                return history[-7]  # Fallback to weekly
            return current_value
        elif self.name == "Trend":
            if len(history) >= 7:
                recent = history[-7:]
                slope = (recent[-1] - recent[0]) / 7
                return current_value + slope
            return current_value
        return current_value

    def execute(self, observation: np.ndarray) -> Dict:
        """Execute method on observation."""
        pm25 = observation[0] if len(observation) > 0 else 20.0
        prediction = self.predict(pm25, np.array(self._history))
        self._history.append(pm25)
        if len(self._history) > 400:
            self._history = self._history[-365:]

        # Convert to signal based on prediction accuracy expectation
        signal = np.clip((prediction - pm25) / 10, -1, 1)
        return {"signal": signal, "prediction": prediction, "confidence": 0.5}


def load_air_quality_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """Load EPA air quality data from CSV."""
    if data_path is None:
        data_path = Path(__file__).parent.parent.parent / "data" / "air_quality" / "epa_daily_aqi.csv"

    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def create_air_quality_environment(
    n_bars: int = 2000,
    city: str = "Los_Angeles",
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, AirQualityMethod]]:
    """
    Create air quality prediction environment from real data.

    State: [pm25, pm25_lag1, pm25_ma7, day_of_week, month]
    """
    df = load_air_quality_data()

    # Filter by city
    city_df = df[df['city'] == city].copy()
    if len(city_df) == 0:
        # Use first available city
        city = df['city'].iloc[0]
        city_df = df[df['city'] == city].copy()

    city_df = city_df.sort_values('date').reset_index(drop=True)

    # Sample if needed
    if len(city_df) > n_bars:
        if seed is not None:
            np.random.seed(seed)
        start_idx = np.random.randint(0, len(city_df) - n_bars)
        city_df = city_df.iloc[start_idx:start_idx + n_bars].reset_index(drop=True)

    # Extract state features
    state_df = pd.DataFrame({
        'pm25': city_df['pm25'],
        'pm25_lag1': city_df['pm25_lag1'],
        'pm25_ma7': city_df['pm25_ma7'],
        'day_of_week': city_df['day_of_week'],
        'month': city_df['month'],
    })

    regimes = pd.Series(city_df['regime'].values)

    # Create methods with optimal regime mappings
    methods = {
        "Persistence": AirQualityMethod("Persistence", ["good", "moderate"]),
        "MA7": AirQualityMethod("MA7", ["moderate", "unhealthy_sensitive"]),
        "Seasonal365": AirQualityMethod("Seasonal365", ["good"]),
        "Trend": AirQualityMethod("Trend", ["unhealthy_sensitive", "unhealthy"]),
    }

    return state_df, regimes, methods


class AirQualityDomain:
    """Wrapper for air quality domain environment."""

    def __init__(self, n_bars: int = 2000, city: str = "Los_Angeles", seed: int = None):
        self.df, self.regimes, self.methods = create_air_quality_environment(
            n_bars=n_bars, city=city, seed=seed
        )

    @property
    def regime_names(self):
        return AIR_QUALITY_REGIMES

    @property
    def method_names(self):
        return AIR_QUALITY_METHODS
