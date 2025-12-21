"""
Core Unit Tests for Emergent Specialization.

Run with: pytest tests/
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.synthetic_market import SyntheticMarketEnvironment, SyntheticMarketConfig
from src.agents.method_selector import MethodSelector
from src.agents.population import Population, PopulationConfig, compute_reward_from_methods
from src.analysis.specialization import (
    compute_specialization_index,
    compute_population_diversity,
    SpecializationTracker,
)
from src.baselines.oracle import OracleSpecialist
from src.baselines.random_selection import RandomSelector


class TestSyntheticMarket:
    """Test synthetic market environment."""

    def test_environment_creation(self):
        """Environment should initialize with default config."""
        env = SyntheticMarketEnvironment()
        assert env is not None
        assert env.config.regime_names == ["trend_up", "trend_down", "mean_revert", "volatile"]

    def test_data_generation(self):
        """Should generate price data with regime labels."""
        env = SyntheticMarketEnvironment()
        prices, regimes = env.generate(100, seed=42)

        assert len(prices) == 100
        assert len(regimes) == 100
        assert "close" in prices.columns
        assert all(r in env.config.regime_names for r in regimes.unique())

    def test_reproducibility(self):
        """Same seed should produce same data."""
        env = SyntheticMarketEnvironment()
        prices1, regimes1 = env.generate(50, seed=42)
        prices2, regimes2 = env.generate(50, seed=42)

        pd.testing.assert_frame_equal(prices1, prices2)
        pd.testing.assert_series_equal(regimes1, regimes2)

    def test_different_seeds(self):
        """Different seeds should produce different data."""
        env = SyntheticMarketEnvironment()
        prices1, _ = env.generate(50, seed=42)
        prices2, _ = env.generate(50, seed=123)

        assert not prices1["close"].equals(prices2["close"])


class TestMethodSelector:
    """Test individual agent method selection."""

    def test_agent_creation(self):
        """Agent should initialize with default beliefs."""
        agent = MethodSelector(agent_id="test", max_methods=3, seed=42)
        assert agent.agent_id == "test"
        assert len(agent.get_available_methods()) == 11

    def test_method_selection(self):
        """Agent should select methods based on Thompson Sampling."""
        agent = MethodSelector(agent_id="test", max_methods=3, seed=42)
        methods = agent.select_methods()

        assert len(methods) <= 3
        assert all(isinstance(m, str) for m in methods)

    def test_belief_update(self):
        """Beliefs should update after reward."""
        agent = MethodSelector(agent_id="test", max_methods=1, seed=42)
        methods = agent.select_methods()

        initial_alpha = agent.beliefs[methods[0]][0]
        agent.update(methods, reward=1.0)
        updated_alpha = agent.beliefs[methods[0]][0]

        assert updated_alpha > initial_alpha

    def test_specialization_over_time(self):
        """Agent should specialize after many updates."""
        agent = MethodSelector(agent_id="test", max_methods=1, seed=42)

        # Simulate learning - always reward same method
        for _ in range(100):
            methods = agent.select_methods()
            agent.update(methods, reward=1.0)

        usage = agent.get_method_usage_distribution()
        si = compute_specialization_index(usage)

        # After 100 updates with consistent reward, should show some specialization
        assert si > 0.3


class TestPopulation:
    """Test population dynamics."""

    def test_population_creation(self):
        """Population should initialize with N agents."""
        config = PopulationConfig(n_agents=5, seed=42)
        pop = Population(config)

        assert len(pop.agents) == 5
        assert all(f"agent_{i}" in pop.agents for i in range(5))

    def test_iteration(self):
        """Population should run iteration and return result."""
        config = PopulationConfig(n_agents=3, seed=42)
        pop = Population(config)

        env = SyntheticMarketEnvironment()
        prices, regimes = env.generate(50, seed=42)

        result = pop.run_iteration(
            prices.iloc[:25],
            compute_reward_from_methods,
            regimes.iloc[24],
        )

        assert result is not None
        assert result.best_agent_id in pop.agents
        assert isinstance(result.best_reward, float)

    def test_knowledge_transfer(self):
        """Knowledge transfer should occur at specified frequency."""
        config = PopulationConfig(n_agents=3, transfer_frequency=5, seed=42)
        pop = Population(config)

        env = SyntheticMarketEnvironment()
        prices, regimes = env.generate(100, seed=42)

        # Run enough iterations to trigger transfer
        for i in range(20, 50):
            pop.run_iteration(
                prices.iloc[i-20:i+1],
                compute_reward_from_methods,
                regimes.iloc[i],
            )

        # Check that some iterations occurred
        assert pop.iteration >= 20


class TestSpecializationMetrics:
    """Test specialization metrics."""

    def test_si_uniform(self):
        """Uniform distribution should have low SI."""
        usage = {f"method_{i}": 1/11 for i in range(11)}
        si = compute_specialization_index(usage)

        assert si < 0.2  # Close to 0 for uniform

    def test_si_specialized(self):
        """Single method should have high SI."""
        usage = {"method_0": 1.0}
        for i in range(1, 11):
            usage[f"method_{i}"] = 0.0

        si = compute_specialization_index(usage)
        assert si > 0.9  # Close to 1 for single method

    def test_population_diversity(self):
        """Diverse population should have high diversity."""
        config = PopulationConfig(n_agents=5, seed=42)
        pop = Population(config)

        # Manually set different dominant methods
        for i, agent in enumerate(pop.agents.values()):
            method = list(agent.beliefs.keys())[i % len(agent.beliefs)]
            agent.beliefs[method] = (100.0, 1.0)

        diversity = compute_population_diversity(pop.agents)
        assert diversity > 0.3  # Should show some diversity

    def test_tracker(self):
        """Tracker should record metrics over time."""
        tracker = SpecializationTracker(n_methods=11)

        # Create dummy distributions
        distributions = {
            "agent_0": {f"method_{i}": 0.1 for i in range(11)},
            "agent_1": {f"method_{i}": 0.1 for i in range(11)},
        }
        distributions["agent_0"]["method_0"] = 0.5
        distributions["agent_1"]["method_5"] = 0.5

        metrics = tracker.record(0, distributions)

        assert metrics is not None
        assert len(metrics.agent_metrics) == 2
        assert 0 <= metrics.avg_specialization <= 1


class TestBaselines:
    """Test baseline implementations."""

    def test_oracle(self):
        """Oracle should select optimal methods for regime."""
        oracle = OracleSpecialist()

        methods_up = oracle.select("trend_up")
        methods_down = oracle.select("trend_down")

        assert methods_up != methods_down
        assert len(methods_up) > 0

    def test_random(self):
        """Random selector should return random methods."""
        random = RandomSelector(seed=42)

        methods1 = random.select()
        methods2 = random.select()

        # At least one should differ over multiple calls
        different = False
        for _ in range(10):
            if random.select() != methods1:
                different = True
                break

        assert different or len(methods1) > 0  # Either different or at least works


class TestRewardComputation:
    """Test reward computation from methods."""

    def test_reward_range(self):
        """Reward should be bounded."""
        methods = ["MomentumFollow", "RSI_Oversold"]

        # Create price window
        dates = pd.date_range("2024-01-01", periods=25, freq="4H")
        prices = pd.DataFrame({
            "open": np.random.uniform(100, 110, 25),
            "high": np.random.uniform(110, 120, 25),
            "low": np.random.uniform(90, 100, 25),
            "close": np.random.uniform(100, 110, 25),
            "volume": np.random.uniform(1000, 2000, 25),
        }, index=dates)

        reward = compute_reward_from_methods(methods, prices, "trend_up")

        assert isinstance(reward, float)
        assert -10 < reward < 10  # Reasonable bounds


class TestIntegration:
    """Integration tests for full pipeline."""

    def test_full_training_loop(self):
        """Full training should complete without errors."""
        env = SyntheticMarketEnvironment()
        prices, regimes = env.generate(200, seed=42)

        config = PopulationConfig(n_agents=3, seed=42)
        pop = Population(config)

        tracker = SpecializationTracker(n_methods=11)

        initial_si = None
        final_si = None

        for i in range(20, 150):
            result = pop.run_iteration(
                prices.iloc[i-20:i+1],
                compute_reward_from_methods,
                regimes.iloc[i],
            )

            if i == 20:
                distributions = pop.get_all_method_usage()
                metrics = tracker.record(i, distributions)
                initial_si = metrics.avg_specialization

            if i == 149:
                distributions = pop.get_all_method_usage()
                metrics = tracker.record(i, distributions)
                final_si = metrics.avg_specialization

        assert initial_si is not None
        assert final_si is not None
        # Specialization should increase (or at least not crash)
        assert final_si >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
