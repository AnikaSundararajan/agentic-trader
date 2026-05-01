"""
Sell Agent: PPO-trained neural network that decides whether to exit each open position.
Runs after stop loss has already fired. Never merged with the buy agent.

Input: augmented state vector = market features + position-specific features
Output: binary action per position (hold / exit)

Position-specific features appended to the base state vector:
  - unrealized_pnl_pct: (current_price - entry_price) / entry_price
  - days_held_norm: days_held / MAX_HOLD_DAYS (normalized to [0,1])
  - days_held_sq: days_held_norm^2 (convex urgency near max hold)
  - trailing_stop_active: 1 if trailing stop is armed, else 0
  - drawdown_from_peak: (peak_price - current_price) / peak_price
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Bernoulli
from pathlib import Path

MAX_HOLD_DAYS = 120
N_POSITION_FEATURES = 5
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints" / "sell"


def build_sell_state(
    base_state: np.ndarray,
    entry_price: float,
    current_price: float,
    days_held: int,
    peak_price: float,
    trailing_active: bool,
) -> np.ndarray:
    """
    Augment the market state vector with position-specific context.
    Returns a new array of shape (len(base_state) + N_POSITION_FEATURES,).
    """
    unrealized_pnl = (current_price - entry_price) / entry_price
    days_norm = days_held / MAX_HOLD_DAYS
    days_sq = days_norm ** 2
    trailing_flag = float(trailing_active)
    drawdown_from_peak = (peak_price - current_price) / peak_price if peak_price > 0 else 0.0

    position_features = np.array(
        [unrealized_pnl, days_norm, days_sq, trailing_flag, drawdown_from_peak],
        dtype=np.float32,
    )
    return np.concatenate([base_state, position_features])


class SellAgentNetwork(nn.Module):
    """
    Actor-critic MLP for the sell decision.
    Input:  augmented state vector (n_features + N_POSITION_FEATURES,)
    Output: action_logit (scalar), value (scalar)

    Uses a slightly deeper network than the buy agent — sell decisions
    require integrating both market context and position history.
    """

    def __init__(self, n_features: int, hidden_sizes: list[int] = [256, 128, 64]):
        super().__init__()
        in_size = n_features + N_POSITION_FEATURES

        layers = []
        for h in hidden_sizes:
            layers += [nn.Linear(in_size, h), nn.LayerNorm(h), nn.ReLU()]
            in_size = h
        self.shared = nn.Sequential(*layers)

        self.policy_head = nn.Linear(in_size, 1)
        self.value_head = nn.Linear(in_size, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Bias policy head toward holding (negative init → lower sell probability early)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.constant_(self.policy_head.bias, -1.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        return self.policy_head(h).squeeze(-1), self.value_head(h).squeeze(-1)

    def get_action(
        self, x: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logit, value = self.forward(x)
        dist = Bernoulli(logits=logit)
        action = (logit > 0).float() if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logit, value = self.forward(x)
        dist = Bernoulli(logits=logit)
        return dist.log_prob(action), dist.entropy(), value


class SellAgent:
    """
    Wrapper around SellAgentNetwork with checkpoint handling.
    """

    def __init__(self, n_base_features: int, device: str = "cpu"):
        self.device = torch.device(device)
        self.n_base_features = n_base_features
        self.n_features = n_base_features + N_POSITION_FEATURES
        self.network = SellAgentNetwork(n_base_features).to(self.device)
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    def act(
        self,
        position_states: list[tuple[int, np.ndarray]],
        deterministic: bool = False,
    ) -> list[int]:
        """
        position_states: list of (permno, augmented_state_vector)
        Returns list of permnos to sell.
        """
        if not position_states:
            return []

        permnos = [p for p, _ in position_states]
        states = np.stack([s for _, s in position_states])
        x = torch.tensor(states, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            actions, _, _ = self.network.get_action(x, deterministic=deterministic)

        sell_mask = actions.cpu().numpy().astype(bool)
        return [p for p, sell in zip(permnos, sell_mask) if sell]

    def act_with_info(
        self, position_states: list[tuple[int, np.ndarray]]
    ) -> tuple[list[int], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (sell_permnos, actions, log_probs, values) for training rollouts."""
        if not position_states:
            empty = torch.tensor([], device=self.device)
            return [], empty, empty, empty

        permnos = [p for p, _ in position_states]
        states = np.stack([s for _, s in position_states])
        x = torch.tensor(states, dtype=torch.float32, device=self.device)

        actions, log_probs, values = self.network.get_action(x)
        sell_mask = actions.detach().cpu().numpy().astype(bool)
        sell_permnos = [p for p, sell in zip(permnos, sell_mask) if sell]
        return sell_permnos, actions, log_probs, values

    def save(self, path: Path | None = None):
        path = path or CHECKPOINT_DIR / "latest.pt"
        torch.save({
            "network": self.network.state_dict(),
            "n_base_features": self.n_base_features,
        }, path)

    def load(self, path: Path | None = None):
        path = path or CHECKPOINT_DIR / "latest.pt"
        ckpt = torch.load(path, map_location=self.device)
        self.network.load_state_dict(ckpt["network"])

    def save_best(self, sharpe: float, episode: int):
        path = CHECKPOINT_DIR / f"best_sharpe{sharpe:.3f}_ep{episode}.pt"
        torch.save({
            "network": self.network.state_dict(),
            "n_base_features": self.n_base_features,
            "sharpe": sharpe,
            "episode": episode,
        }, path)
