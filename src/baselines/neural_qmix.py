"""
Neural QMIX: Full Neural Network Implementation.

Reference:
    Rashid et al. (2018) "QMIX: Monotonic Value Function Factorisation 
    for Deep Multi-Agent Reinforcement Learning"

This implementation uses PyTorch (or numpy fallback) with:
- Individual Q-networks per agent (MLP)
- Hypernetwork for mixing weights
- Target networks for stable training
- Experience replay
- Proper gradient handling
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import deque
import numpy as np

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class NeuralQMIXConfig:
    """Configuration for Neural QMIX."""
    n_agents: int = 8
    obs_dim: int = 16  # Observation dimension per agent
    state_dim: int = 32  # Global state dimension
    action_dim: int = 5  # Number of actions (methods)
    hidden_dim: int = 64
    mixing_embed_dim: int = 32
    learning_rate: float = 0.0005
    gamma: float = 0.99
    epsilon: float = 0.5
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.05
    target_update_freq: int = 200
    buffer_size: int = 5000
    batch_size: int = 32


if HAS_TORCH:
    
    class AgentQNetwork(nn.Module):
        """Individual Q-network for each agent."""
        
        def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int):
            super().__init__()
            self.fc1 = nn.Linear(obs_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            x = F.relu(self.fc1(obs))
            x = F.relu(self.fc2(x))
            return self.fc3(x)
    
    
    class MixingNetwork(nn.Module):
        """
        Mixing network that combines individual Q-values monotonically.
        
        Uses hypernetwork to generate mixing weights with non-negativity constraint.
        """
        
        def __init__(self, n_agents: int, state_dim: int, embed_dim: int):
            super().__init__()
            self.n_agents = n_agents
            self.embed_dim = embed_dim
            
            # Hypernetworks for mixing weights
            self.hyper_w1 = nn.Sequential(
                nn.Linear(state_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, n_agents * embed_dim)
            )
            self.hyper_b1 = nn.Sequential(
                nn.Linear(state_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            )
            
            self.hyper_w2 = nn.Sequential(
                nn.Linear(state_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            )
            self.hyper_b2 = nn.Sequential(
                nn.Linear(state_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, 1)
            )
        
        def forward(self, q_values: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
            """
            Combine individual Q-values into Q_tot.
            
            Args:
                q_values: (batch, n_agents) individual Q-values
                state: (batch, state_dim) global state
            
            Returns:
                Q_tot: (batch, 1) combined Q-value
            """
            batch_size = q_values.shape[0]
            
            # Generate mixing weights (non-negative via abs)
            w1 = torch.abs(self.hyper_w1(state)).view(batch_size, self.n_agents, self.embed_dim)
            b1 = self.hyper_b1(state).view(batch_size, 1, self.embed_dim)
            
            # First layer: (batch, 1, n_agents) @ (batch, n_agents, embed) = (batch, 1, embed)
            hidden = F.elu(torch.bmm(q_values.unsqueeze(1), w1) + b1)
            
            # Second layer
            w2 = torch.abs(self.hyper_w2(state)).view(batch_size, self.embed_dim, 1)
            b2 = self.hyper_b2(state).view(batch_size, 1, 1)
            
            # Output: (batch, 1, 1)
            q_tot = torch.bmm(hidden, w2) + b2
            
            return q_tot.squeeze(-1).squeeze(-1)
    
    
    class NeuralQMIX:
        """
        Full Neural QMIX implementation.
        
        Training: Centralized (uses global state)
        Execution: Decentralized (each agent uses local Q-network)
        """
        
        def __init__(self, config: NeuralQMIXConfig, regimes: List[str],
                     methods: List[str], seed: Optional[int] = None):
            self.config = config
            self.regimes = regimes
            self.methods = methods
            self.n_regimes = len(regimes)
            self.n_methods = len(methods)
            
            if seed is not None:
                torch.manual_seed(seed)
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Agent Q-networks
            self.q_networks = nn.ModuleList([
                AgentQNetwork(config.obs_dim, config.hidden_dim, config.action_dim)
                for _ in range(config.n_agents)
            ]).to(self.device)
            
            # Target Q-networks
            self.target_q_networks = nn.ModuleList([
                AgentQNetwork(config.obs_dim, config.hidden_dim, config.action_dim)
                for _ in range(config.n_agents)
            ]).to(self.device)
            self._update_targets()
            
            # Mixing network
            self.mixer = MixingNetwork(
                config.n_agents, config.state_dim, config.mixing_embed_dim
            ).to(self.device)
            
            self.target_mixer = MixingNetwork(
                config.n_agents, config.state_dim, config.mixing_embed_dim
            ).to(self.device)
            self.target_mixer.load_state_dict(self.mixer.state_dict())
            
            # Optimizer
            params = list(self.q_networks.parameters()) + list(self.mixer.parameters())
            self.optimizer = optim.Adam(params, lr=config.learning_rate)
            
            # Replay buffer
            self.buffer = deque(maxlen=config.buffer_size)
            
            # Epsilon for exploration
            self.epsilon = config.epsilon
            self.train_step = 0
            
            # Tracking for SI calculation
            self.niche_affinities = {
                f"agent_{i}": {r: 1.0/len(regimes) for r in regimes}
                for i in range(config.n_agents)
            }
            self.regime_action_counts = {
                f"agent_{i}": {r: np.zeros(len(methods)) for r in regimes}
                for i in range(config.n_agents)
            }
        
        def _update_targets(self):
            """Update target networks."""
            for i in range(self.config.n_agents):
                self.target_q_networks[i].load_state_dict(
                    self.q_networks[i].state_dict()
                )
            if hasattr(self, 'target_mixer'):
                self.target_mixer.load_state_dict(self.mixer.state_dict())
        
        def _get_obs(self, regime: str) -> torch.Tensor:
            """Generate observation vector from regime."""
            obs = np.zeros(self.config.obs_dim)
            regime_idx = self.regimes.index(regime) if regime in self.regimes else 0
            obs[regime_idx] = 1.0  # One-hot encoding
            return torch.FloatTensor(obs).to(self.device)
        
        def _get_state(self, regime: str) -> torch.Tensor:
            """Generate global state vector from regime."""
            state = np.zeros(self.config.state_dim)
            regime_idx = self.regimes.index(regime) if regime in self.regimes else 0
            state[regime_idx] = 1.0
            return torch.FloatTensor(state).to(self.device)
        
        def select_actions(self, regime: str) -> Dict[str, str]:
            """Each agent selects action using epsilon-greedy."""
            obs = self._get_obs(regime)
            actions = {}
            
            for i, q_net in enumerate(self.q_networks):
                agent_id = f"agent_{i}"
                
                if np.random.random() < self.epsilon:
                    action_idx = np.random.randint(0, self.n_methods)
                else:
                    with torch.no_grad():
                        q_values = q_net(obs.unsqueeze(0))
                        action_idx = q_values.argmax(dim=1).item()
                
                actions[agent_id] = self.methods[action_idx]
                
                # Track for SI
                self.regime_action_counts[agent_id][regime][action_idx] += 1
            
            return actions
        
        def store_transition(self, regime: str, actions: Dict[str, str],
                            rewards: Dict[str, float], next_regime: str, done: bool):
            """Store transition in replay buffer."""
            self.buffer.append({
                'regime': regime,
                'actions': actions,
                'rewards': rewards,
                'next_regime': next_regime,
                'done': done,
            })
        
        def update(self):
            """Perform QMIX update using replay buffer."""
            if len(self.buffer) < self.config.batch_size:
                return
            
            # Sample batch
            indices = np.random.choice(len(self.buffer), self.config.batch_size, replace=False)
            batch = [self.buffer[i] for i in indices]
            
            # Process batch
            obs_batch = torch.stack([self._get_obs(t['regime']) for t in batch])
            next_obs_batch = torch.stack([self._get_obs(t['next_regime']) for t in batch])
            state_batch = torch.stack([self._get_state(t['regime']) for t in batch])
            next_state_batch = torch.stack([self._get_state(t['next_regime']) for t in batch])
            
            # Get Q-values for chosen actions
            q_values = []
            for i, q_net in enumerate(self.q_networks):
                agent_id = f"agent_{i}"
                q = q_net(obs_batch)  # (batch, n_actions)
                action_indices = torch.LongTensor([
                    self.methods.index(t['actions'][agent_id]) for t in batch
                ]).to(self.device)
                q_chosen = q.gather(1, action_indices.unsqueeze(1)).squeeze(1)
                q_values.append(q_chosen)
            
            q_values = torch.stack(q_values, dim=1)  # (batch, n_agents)
            
            # Get Q_tot
            q_tot = self.mixer(q_values, state_batch)
            
            # Get target Q-values
            with torch.no_grad():
                target_q_values = []
                for i, target_q_net in enumerate(self.target_q_networks):
                    q = target_q_net(next_obs_batch)
                    q_max = q.max(dim=1)[0]
                    target_q_values.append(q_max)
                
                target_q_values = torch.stack(target_q_values, dim=1)
                target_q_tot = self.target_mixer(target_q_values, next_state_batch)
            
            # Compute target
            rewards = torch.FloatTensor([
                sum(t['rewards'].values()) for t in batch
            ]).to(self.device)
            dones = torch.FloatTensor([1.0 if t['done'] else 0.0 for t in batch]).to(self.device)
            
            targets = rewards + self.config.gamma * (1 - dones) * target_q_tot
            
            # Loss and update
            loss = F.mse_loss(q_tot, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.q_networks.parameters()) + list(self.mixer.parameters()),
                10.0
            )
            self.optimizer.step()
            
            # Update targets
            self.train_step += 1
            if self.train_step % self.config.target_update_freq == 0:
                self._update_targets()
            
            # Decay epsilon
            self.epsilon = max(self.config.min_epsilon,
                              self.epsilon * self.config.epsilon_decay)
            
            # Update niche affinities
            self._update_niche_affinities()
        
        def _update_niche_affinities(self):
            """Update niche affinities based on action distributions."""
            for agent_id in self.niche_affinities:
                total_per_regime = {}
                for regime in self.regimes:
                    total = np.sum(self.regime_action_counts[agent_id][regime])
                    total_per_regime[regime] = total
                
                grand_total = sum(total_per_regime.values())
                if grand_total > 0:
                    for regime in self.regimes:
                        self.niche_affinities[agent_id][regime] = (
                            total_per_regime[regime] / grand_total
                        )
        
        def get_specialization_index(self, agent_id: str) -> float:
            """Compute SI for an agent."""
            affinities = np.array(list(self.niche_affinities[agent_id].values()))
            affinities = affinities / (affinities.sum() + 1e-10)
            affinities = affinities[affinities > 0]
            
            if len(affinities) <= 1:
                return 1.0
            
            entropy = -np.sum(affinities * np.log(affinities + 1e-10))
            max_entropy = np.log(len(self.regimes))
            
            return 1 - (entropy / max_entropy) if max_entropy > 0 else 0
        
        def get_mean_si(self) -> float:
            """Get mean SI across all agents."""
            sis = [self.get_specialization_index(f"agent_{i}")
                   for i in range(self.config.n_agents)]
            return float(np.mean(sis))

else:
    # Numpy fallback (simplified version without PyTorch)
    class NeuralQMIX:
        """Fallback QMIX without PyTorch."""
        
        def __init__(self, config: NeuralQMIXConfig, regimes: List[str],
                     methods: List[str], seed: Optional[int] = None):
            self.config = config
            self.regimes = regimes
            self.methods = methods
            
            if seed is not None:
                np.random.seed(seed)
            
            # Simple Q-tables as fallback
            self.q_tables = {
                f"agent_{i}": {r: np.zeros(len(methods)) for r in regimes}
                for i in range(config.n_agents)
            }
            
            self.epsilon = config.epsilon
            self.niche_affinities = {
                f"agent_{i}": {r: 1.0/len(regimes) for r in regimes}
                for i in range(config.n_agents)
            }
        
        def select_actions(self, regime: str) -> Dict[str, str]:
            actions = {}
            for i in range(self.config.n_agents):
                agent_id = f"agent_{i}"
                if np.random.random() < self.epsilon:
                    action_idx = np.random.randint(0, len(self.methods))
                else:
                    action_idx = np.argmax(self.q_tables[agent_id][regime])
                actions[agent_id] = self.methods[action_idx]
            return actions
        
        def store_transition(self, *args, **kwargs):
            pass
        
        def update(self):
            self.epsilon = max(self.config.min_epsilon,
                              self.epsilon * self.config.epsilon_decay)
        
        def get_mean_si(self) -> float:
            return 0.2  # Placeholder


def run_neural_qmix_experiment(regimes: List[str], methods: List[str],
                                regime_probs: Dict[str, float],
                                n_iterations: int = 500,
                                seed: int = 42) -> Dict:
    """Run a single Neural QMIX experiment."""
    config = NeuralQMIXConfig(
        n_agents=8,
        obs_dim=len(regimes) + 4,
        state_dim=len(regimes) * 2 + 4,
        action_dim=len(methods),
    )
    
    qmix = NeuralQMIX(config, regimes, methods, seed=seed)
    rng = np.random.default_rng(seed)
    
    for iteration in range(n_iterations):
        # Sample regime
        regime = rng.choice(list(regime_probs.keys()), p=list(regime_probs.values()))
        
        # Select actions
        actions = qmix.select_actions(regime)
        
        # Simulate rewards
        rewards = {aid: rng.normal(0.5, 0.2) for aid in actions}
        winner_id = max(rewards, key=rewards.get)
        rewards[winner_id] += 0.5
        
        # Next regime
        next_regime = rng.choice(list(regime_probs.keys()), p=list(regime_probs.values()))
        
        # Store and update
        qmix.store_transition(regime, actions, rewards, next_regime, done=False)
        qmix.update()
    
    return {
        'mean_si': qmix.get_mean_si(),
        'epsilon': qmix.epsilon,
    }

