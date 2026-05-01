"""
Buy Agent: PPO-trained neural network that decides which Donchian breakout
candidates to enter. Input is the state vector from the feature store.
Output is a binary action per candidate (buy / skip).

Architecture: MLP with LayerNorm. Separate policy and value heads (actor-critic).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Bernoulli
from pathlib import Path

CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints" / "buy"


class BuyAgentNetwork(nn.Module):
    """
    Actor-critic MLP for the buy decision.
    Input:  state vector (n_features,)
    Output: action_logit (scalar), value (scalar)
    """

    def __init__(self, n_features: int, hidden_sizes: list[int] = [256, 128, 64]):
        super().__init__()

        layers = []
        in_size = n_features
        for h in hidden_sizes:
            layers += [nn.Linear(in_size, h), nn.LayerNorm(h), nn.ReLU()]
            in_size = h
        self.shared = nn.Sequential(*layers)

        self.policy_head = nn.Linear(in_size, 1)   # logit for Bernoulli(buy)
        self.value_head = nn.Linear(in_size, 1)    # state value estimate

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Policy head: smaller init to start near uniform
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_logit, value). x shape: (batch, n_features)."""
        h = self.shared(x)
        return self.policy_head(h).squeeze(-1), self.value_head(h).squeeze(-1)

    def get_action(
        self, x: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample or select action for a batch of candidates.
        Returns (action, log_prob, value).
        action: 0 = skip, 1 = buy
        """
        logit, value = self.forward(x)
        dist = Bernoulli(logits=logit)
        if deterministic:
            action = (logit > 0).float()
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log_prob and entropy for given actions (used in PPO update).
        Returns (log_prob, entropy, value).
        """
        logit, value = self.forward(x)
        dist = Bernoulli(logits=logit)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, value


class BuyAgent:
    """
    Wrapper around BuyAgentNetwork with checkpoint save/load.
    Handles device placement and numpy↔tensor conversion.
    """

    def __init__(self, n_features: int, device: str = "cpu"):
        self.device = torch.device(device)
        self.network = BuyAgentNetwork(n_features).to(self.device)
        self.n_features = n_features
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    def act(
        self, candidates: list[tuple[int, np.ndarray]], deterministic: bool = False
    ) -> list[int]:
        """
        Given a list of (permno, state_vector) candidates, return list of permnos to buy.
        Used at inference time (no gradient tracking).
        """
        if not candidates:
            return []

        permnos = [p for p, _ in candidates]
        states = np.stack([s for _, s in candidates])
        x = torch.tensor(states, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            actions, _, _ = self.network.get_action(x, deterministic=deterministic)

        buy_mask = actions.cpu().numpy().astype(bool)
        return [p for p, buy in zip(permnos, buy_mask) if buy]

    def act_with_info(
        self, candidates: list[tuple[int, np.ndarray]]
    ) -> tuple[list[int], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (buy_permnos, actions, log_probs, values) for use in training rollouts.
        Gradients are tracked.
        """
        if not candidates:
            empty = torch.tensor([], device=self.device)
            return [], empty, empty, empty

        permnos = [p for p, _ in candidates]
        states = np.stack([s for _, s in candidates])
        x = torch.tensor(states, dtype=torch.float32, device=self.device)

        actions, log_probs, values = self.network.get_action(x)
        buy_mask = actions.detach().cpu().numpy().astype(bool)
        buy_permnos = [p for p, buy in zip(permnos, buy_mask) if buy]
        return buy_permnos, actions, log_probs, values

    def save(self, path: Path | None = None):
        path = path or CHECKPOINT_DIR / "latest.pt"
        torch.save({"network": self.network.state_dict(), "n_features": self.n_features}, path)

    def load(self, path: Path | None = None):
        path = path or CHECKPOINT_DIR / "latest.pt"
        ckpt = torch.load(path, map_location=self.device)
        self.network.load_state_dict(ckpt["network"])

    def save_best(self, sharpe: float, episode: int):
        path = CHECKPOINT_DIR / f"best_sharpe{sharpe:.3f}_ep{episode}.pt"
        torch.save({
            "network": self.network.state_dict(),
            "n_features": self.n_features,
            "sharpe": sharpe,
            "episode": episode,
        }, path)
