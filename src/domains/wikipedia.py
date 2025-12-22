"""
Wikipedia Domain - Page view prediction with traffic regimes.

Regimes: normal, trending, viral, declining
Methods: Persistence, MA7, EventDecay, WeeklyPattern
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from .base import DomainEnvironment, DomainMethod


WIKIPEDIA_REGIMES = ["normal", "trending", "viral", "declining"]
WIKIPEDIA_METHODS = ["Persistence", "MA7", "EventDecay", "WeeklyPattern"]


class WikipediaMethod(DomainMethod):
    """Wikipedia pageview prediction method."""

    def __init__(self, name: str, optimal_regimes: List[str]):
        self.name = name
        self.optimal_regimes = optimal_regimes
        self._history = []
        self._weekly_history = []

    def predict(self, current_value: float, history: np.ndarray, day_of_week: int = 0) -> float:
        """Predict next pageview count."""
        if self.name == "Persistence":
            return current_value
        elif self.name == "MA7":
            if len(history) >= 7:
                return np.mean(history[-7:])
            return current_value
        elif self.name == "EventDecay":
            # Assume spikes decay exponentially
            if len(history) >= 3:
                recent = history[-3:]
                if recent[-1] > recent.mean() * 1.5:  # Spike detected
                    decay_rate = 0.7
                    return current_value * decay_rate
            return current_value
        elif self.name == "WeeklyPattern":
            # Use same day last week
            if len(history) >= 7:
                return history[-7]
            return current_value
        return current_value

    def execute(self, observation: np.ndarray) -> Dict:
        """Execute method on observation."""
        views = observation[0] if len(observation) > 0 else 10000
        dow = int(observation[1]) if len(observation) > 1 else 0

        prediction = self.predict(views, np.array(self._history), dow)
        self._history.append(views)
        if len(self._history) > 30:
            self._history = self._history[-30:]

        # Signal based on predicted change
        if views > 0:
            pct_change = (prediction - views) / views
        else:
            pct_change = 0
        signal = np.clip(pct_change, -1, 1)
        return {"signal": signal, "prediction": prediction, "confidence": 0.5}


def load_wikipedia_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """Load Wikipedia pageview data from CSV."""
    if data_path is None:
        data_path = Path(__file__).parent.parent.parent / "data" / "wikipedia" / "daily_pageviews.csv"

    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def create_wikipedia_environment(
    n_bars: int = 2000,
    article: str = "Python_(programming_language)",
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, WikipediaMethod]]:
    """
    Create Wikipedia pageview prediction environment from real data.

    State: [views, day_of_week, views_ma7, views_std7]
    """
    df = load_wikipedia_data()

    # Filter by article
    article_df = df[df['article'] == article].copy()
    if len(article_df) == 0:
        article = df['article'].iloc[0]
        article_df = df[df['article'] == article].copy()

    article_df = article_df.sort_values('date').reset_index(drop=True)

    # Sample if needed
    if len(article_df) > n_bars:
        if seed is not None:
            np.random.seed(seed)
        start_idx = np.random.randint(0, len(article_df) - n_bars)
        article_df = article_df.iloc[start_idx:start_idx + n_bars].reset_index(drop=True)

    # Extract state features
    state_df = pd.DataFrame({
        'views': article_df['views'],
        'day_of_week': article_df['day_of_week'],
        'views_ma7': article_df['views_ma7'],
        'views_std7': article_df['views_std7'],
    })

    regimes = pd.Series(article_df['regime'].values)

    # Create methods
    methods = {
        "Persistence": WikipediaMethod("Persistence", ["normal", "declining"]),
        "MA7": WikipediaMethod("MA7", ["normal"]),
        "EventDecay": WikipediaMethod("EventDecay", ["viral", "trending"]),
        "WeeklyPattern": WikipediaMethod("WeeklyPattern", ["normal", "declining"]),
    }

    return state_df, regimes, methods


class WikipediaDomain:
    """Wrapper for Wikipedia domain environment."""

    def __init__(self, n_bars: int = 2000, article: str = "Python_(programming_language)", seed: int = None):
        self.df, self.regimes, self.methods = create_wikipedia_environment(
            n_bars=n_bars, article=article, seed=seed
        )

    @property
    def regime_names(self):
        return WIKIPEDIA_REGIMES

    @property
    def method_names(self):
        return WIKIPEDIA_METHODS
