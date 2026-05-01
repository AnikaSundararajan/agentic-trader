"""
PPO training loop for the Buy agent.
Run: python -m training.train_buy

Uses MockTradingEnvironment by default. Swap to TradingEnvironment for real training.
Checkpoints every 100 episodes; keeps best by validation Sharpe.
Logs every trade to logs/buy_trades.csv.
"""

import random
import csv
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from dataclasses import dataclass
from collections import deque

from data.mock_environment import MockTradingEnvironment
from agents.buy_agent import BuyAgent
from agents.stop_loss import StopLossManager

# Uncomment to train on real WRDS data:
# from data.environment import TradingEnvironment

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    n_episodes: int = 2000
    n_steps_per_update: int = 512     # collect this many steps before each PPO update
    n_epochs: int = 4                 # PPO update epochs per batch
    batch_size: int = 64
    gamma: float = 0.99               # discount factor
    gae_lambda: float = 0.95          # GAE lambda
    clip_eps: float = 0.2             # PPO clip epsilon
    entropy_coef: float = 0.01        # entropy bonus (keeps exploration alive)
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    val_every: int = 100              # validate every N episodes
    checkpoint_every: int = 100


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    def __init__(self):
        self.states: list[np.ndarray] = []
        self.actions: list[float] = []
        self.log_probs: list[float] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.dones: list[bool] = []

    def add(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(float(action))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.dones.append(bool(done))

    def compute_returns_and_advantages(self, last_value: float, gamma: float, gae_lambda: float):
        """GAE advantage estimation."""
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)
        gae = 0.0

        values = np.array(self.values + [last_value], dtype=np.float32)
        for t in reversed(range(n)):
            delta = self.rewards[t] + gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        self.advantages = advantages
        self.returns = returns

    def to_tensors(self, device):
        states = torch.tensor(np.stack(self.states), dtype=torch.float32, device=device)
        actions = torch.tensor(self.actions, dtype=torch.float32, device=device)
        log_probs = torch.tensor(self.log_probs, dtype=torch.float32, device=device)
        advantages = torch.tensor(self.advantages, dtype=torch.float32, device=device)
        returns = torch.tensor(self.returns, dtype=torch.float32, device=device)
        return states, actions, log_probs, advantages, returns

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)


# ---------------------------------------------------------------------------
# Trade logger
# ---------------------------------------------------------------------------

class TradeLogger:
    def __init__(self, path: Path):
        self.path = path
        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            "episode", "permno", "entry_date", "exit_date",
            "entry_price", "exit_price", "gross_return", "net_return",
            "exit_reason", "days_held",
        ])

    def log(self, episode: int, trade):
        self._writer.writerow([
            episode,
            getattr(trade, "permno", ""),
            getattr(trade, "entry_date", ""),
            getattr(trade, "exit_date", ""),
            getattr(trade, "entry_price", ""),
            getattr(trade, "exit_price", ""),
            getattr(trade, "gross_return", ""),
            getattr(trade, "net_return", ""),
            getattr(trade, "exit_reason", ""),
            getattr(trade, "days_held", ""),
        ])

    def close(self):
        self._file.close()


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(agent: BuyAgent, optimizer, buffer: RolloutBuffer, cfg: PPOConfig):
    states, actions, old_log_probs, advantages, returns = buffer.to_tensors(agent.device)

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    total_loss = 0.0
    for _ in range(cfg.n_epochs):
        # Mini-batch shuffling
        indices = torch.randperm(len(states))
        for start in range(0, len(states), cfg.batch_size):
            idx = indices[start:start + cfg.batch_size]
            s, a, olp, adv, ret = states[idx], actions[idx], old_log_probs[idx], advantages[idx], returns[idx]

            log_prob, entropy, value = agent.network.evaluate(s, a)

            ratio = torch.exp(log_prob - olp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (ret - value).pow(2).mean()
            entropy_loss = -entropy.mean()

            loss = policy_loss + cfg.value_coef * value_loss + cfg.entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.network.parameters(), cfg.max_grad_norm)
            optimizer.step()

            total_loss += loss.item()

    return total_loss


# ---------------------------------------------------------------------------
# Reward shaping for buy decisions
# ---------------------------------------------------------------------------

def shape_buy_reward(n_buys: int, n_candidates: int, step_reward: float) -> float:
    """
    Augment environment reward with buy-selectivity signal.
    Penalize buying everything; reward skipping bad signals.
    Target: 1-3 buys from 10-20 candidates.
    """
    if n_candidates == 0:
        return step_reward

    buy_rate = n_buys / n_candidates

    if buy_rate > 0.5:
        # Buying too many — penalize proportionally
        step_reward -= (buy_rate - 0.5) * 0.1
    elif n_buys == 0 and n_candidates > 0:
        # Skipping everything is fine when cautious, small neutral nudge
        step_reward += 0.001

    return step_reward


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(agent: BuyAgent, n_val_episodes: int = 50, seed_offset: int = 9999) -> float:
    """Run agent greedily on val episodes. Returns mean Sharpe ratio proxy."""
    agent.network.eval()
    returns_list = []

    for ep in range(n_val_episodes):
        env = MockTradingEnvironment(n_episodes=200, seed=seed_offset + ep)
        candidates = env.reset()
        ep_returns = []

        while True:
            buy_permnos = agent.act(candidates, deterministic=True)
            candidates, reward, done, info = env.step(buy_permnos, sell_action=[])
            ep_returns.append(reward)
            if done:
                break

        if len(ep_returns) > 1:
            mean_r = np.mean(ep_returns)
            std_r = np.std(ep_returns) + 1e-8
            returns_list.append(mean_r / std_r)

    agent.network.train()
    return float(np.mean(returns_list)) if returns_list else 0.0


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: PPOConfig | None = None):
    cfg = cfg or PPOConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = MockTradingEnvironment(n_episodes=cfg.n_steps_per_update * 4, seed=42)
    candidates = env.reset()
    n_features = candidates[0][1].shape[0] if candidates else 75

    agent = BuyAgent(n_features=n_features, device=device)
    optimizer = optim.Adam(agent.network.parameters(), lr=cfg.lr)
    buffer = RolloutBuffer()
    stop_loss_mgr = StopLossManager()
    trade_logger = TradeLogger(LOG_DIR / "buy_trades.csv")

    best_sharpe = -np.inf
    reward_history: deque = deque(maxlen=100)
    step_count = 0

    print(f"Training Buy Agent | device={device} | features={n_features}")

    for episode in range(1, cfg.n_episodes + 1):
        env = MockTradingEnvironment(n_episodes=cfg.n_steps_per_update * 2, seed=episode)
        candidates = env.reset()
        ep_reward = 0.0
        done = False

        while not done and len(buffer) < cfg.n_steps_per_update:
            if not candidates:
                candidates, reward, done, info = env.step([], [])
                continue

            # Stop loss check on open positions (fires before agent)
            stop_exits = stop_loss_mgr.check(
                {pos.permno: env._prices.get(pos.permno, pos.entry_price) for pos in env.positions}
            )

            # Buy agent decision
            buy_permnos, actions, log_probs, values = agent.act_with_info(candidates)

            # Remove stop-loss exits from buy consideration (already exiting)
            buy_permnos = [p for p in buy_permnos if p not in stop_exits]

            n_candidates = len(candidates)
            candidates, env_reward, done, info = env.step(buy_permnos, stop_exits)

            shaped_reward = shape_buy_reward(len(buy_permnos), n_candidates, env_reward)
            ep_reward += shaped_reward

            # Store one transition per candidate in the rollout buffer
            for i, (permno, state) in enumerate(
                [(p, s) for p, s in [(p, s) for p, s in candidates[:len(actions)]]]
                if len(actions) > 0 else []
            ):
                if i < len(actions):
                    buffer.add(
                        state=state,
                        action=actions[i].item(),
                        log_prob=log_probs[i].item(),
                        reward=shaped_reward / max(len(candidates), 1),
                        value=values[i].item(),
                        done=done,
                    )

            step_count += 1

            # Register new buys with stop loss manager
            for pos in env.positions:
                if pos.permno not in stop_loss_mgr._states:
                    atr = 0.02 * env._prices.get(pos.permno, 100)  # mock: 2% ATR proxy
                    stop_loss_mgr.register(pos.permno, pos.entry_price, atr=atr, beta=1.0)

            for permno in stop_exits:
                stop_loss_mgr.remove(permno)

        # PPO update
        if len(buffer) > 0:
            with torch.no_grad():
                _, last_value = agent.network(
                    torch.zeros(1, n_features, device=agent.device)
                )
            buffer.compute_returns_and_advantages(
                last_value=last_value.item(), gamma=cfg.gamma, gae_lambda=cfg.gae_lambda
            )
            ppo_update(agent, optimizer, buffer, cfg)
            buffer.clear()

        reward_history.append(ep_reward)

        if episode % 10 == 0:
            print(
                f"Ep {episode:4d} | reward={ep_reward:+.4f} "
                f"| avg100={np.mean(reward_history):+.4f} "
                f"| positions={info.get('n_positions', 0)}"
            )

        # Checkpoint + validation
        if episode % cfg.checkpoint_every == 0:
            agent.save()
            sharpe = validate(agent)
            print(f"  [Val] Sharpe proxy: {sharpe:.4f}")
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                agent.save_best(sharpe, episode)
                print(f"  [Best] New best Sharpe: {sharpe:.4f}")

    trade_logger.close()
    agent.save()
    print(f"\nTraining complete. Best val Sharpe: {best_sharpe:.4f}")
    return agent


if __name__ == "__main__":
    train()
