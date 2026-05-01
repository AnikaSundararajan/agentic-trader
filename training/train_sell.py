"""
PPO training loop for the Sell agent.
Run: python -m training.train_sell

The Sell agent is trained jointly with the Buy agent's decisions frozen:
a pre-trained (or random) Buy agent fills positions, then the Sell agent
learns when to exit them. This prevents the two agents interfering during training.

Swap MockTradingEnvironment for TradingEnvironment for real data training.
"""

import random
import csv
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from collections import deque
from dataclasses import dataclass

from data.mock_environment import MockTradingEnvironment
from agents.buy_agent import BuyAgent
from agents.sell_agent import SellAgent, build_sell_state
from agents.stop_loss import StopLossManager

# Uncomment for real training:
# from data.environment import TradingEnvironment

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

MAX_HOLD_DAYS = 120


@dataclass
class SellPPOConfig:
    n_episodes: int = 2000
    n_steps_per_update: int = 512
    n_epochs: int = 4
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.005     # lower than buy: sell agent should be more decisive
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    val_every: int = 100
    checkpoint_every: int = 100
    patience_penalty_days: int = 5  # penalize sells within this many days of entry


# ---------------------------------------------------------------------------
# Rollout buffer (same structure as buy training)
# ---------------------------------------------------------------------------

class SellRolloutBuffer:
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
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        values = np.array(self.values + [last_value], dtype=np.float32)
        for t in reversed(range(n)):
            delta = self.rewards[t] + gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * gae
            advantages[t] = gae
        self.advantages = advantages
        self.returns = advantages + np.array(self.values, dtype=np.float32)

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
# Reward shaping for sell decisions
# ---------------------------------------------------------------------------

def shape_sell_reward(
    env_reward: float,
    sell_permnos: list[int],
    positions: list,
    cfg: SellPPOConfig,
) -> float:
    """
    Augment environment reward to encourage patient holding.
    - Selling within patience_penalty_days → penalty
    - Holding a losing position past max hold → heavy penalty (forced exit)
    """
    reward = env_reward

    held_map = {pos.permno: pos.days_held for pos in positions}
    for permno in sell_permnos:
        days = held_map.get(permno, cfg.patience_penalty_days + 1)
        if days < cfg.patience_penalty_days:
            # Penalize premature sells
            reward -= 0.05 * (cfg.patience_penalty_days - days)

    return reward


# ---------------------------------------------------------------------------
# PPO update (identical logic to buy, shared for clarity)
# ---------------------------------------------------------------------------

def ppo_update(agent: SellAgent, optimizer, buffer: SellRolloutBuffer, cfg: SellPPOConfig):
    states, actions, old_log_probs, advantages, returns = buffer.to_tensors(agent.device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(cfg.n_epochs):
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


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(sell_agent: SellAgent, buy_agent: BuyAgent, n_val_episodes: int = 50) -> float:
    sell_agent.network.eval()
    sharpe_list = []

    for ep in range(n_val_episodes):
        env = MockTradingEnvironment(n_episodes=200, seed=99000 + ep)
        candidates = env.reset()
        ep_rewards = []
        stop_mgr = StopLossManager()

        while True:
            # Buy agent fills positions (frozen)
            buy_permnos = buy_agent.act(candidates, deterministic=True)
            for p in buy_permnos:
                price = env._prices.get(p, 100)
                stop_mgr.register(p, price, atr=price * 0.02, beta=1.0)

            # Stop loss
            stop_exits = stop_mgr.check({pos.permno: env._prices.get(pos.permno, 100) for pos in env.positions})

            # Sell agent decisions on open positions
            position_states = _build_position_states(env, sell_agent.n_base_features)
            sell_permnos = sell_agent.act(position_states, deterministic=True)
            sell_permnos = list(set(sell_permnos) - set(stop_exits))

            all_exits = list(set(stop_exits + sell_permnos))
            for p in all_exits:
                stop_mgr.remove(p)

            candidates, reward, done, info = env.step(buy_permnos, all_exits)
            ep_rewards.append(reward)
            if done:
                break

        if len(ep_rewards) > 1:
            m, s = np.mean(ep_rewards), np.std(ep_rewards) + 1e-8
            sharpe_list.append(m / s)

    sell_agent.network.train()
    return float(np.mean(sharpe_list)) if sharpe_list else 0.0


# ---------------------------------------------------------------------------
# Helper: build sell agent input states for all open positions
# ---------------------------------------------------------------------------

def _build_position_states(
    env: MockTradingEnvironment,
    n_base_features: int,
) -> list[tuple[int, np.ndarray]]:
    """Build augmented state vectors for each open position."""
    result = []
    for pos in env.positions:
        current_price = env._prices.get(pos.permno, pos.entry_price)
        peak_price = max(current_price, pos.entry_price)  # mock: no real peak tracking
        base_state = np.zeros(n_base_features, dtype=np.float32)  # placeholder in mock

        aug_state = build_sell_state(
            base_state=base_state,
            entry_price=pos.entry_price,
            current_price=current_price,
            days_held=pos.days_held,
            peak_price=peak_price,
            trailing_active=False,
        )
        result.append((pos.permno, aug_state))
    return result


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: SellPPOConfig | None = None, buy_agent: BuyAgent | None = None):
    cfg = cfg or SellPPOConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Spin up a mock env to determine feature size
    _probe_env = MockTradingEnvironment(n_episodes=10, seed=0)
    probe_candidates = _probe_env.reset()
    n_base_features = probe_candidates[0][1].shape[0] if probe_candidates else 75

    # Buy agent (frozen during sell training)
    if buy_agent is None:
        buy_agent = BuyAgent(n_features=n_base_features, device=device)
        buy_ckpt = Path(__file__).parent.parent / "checkpoints" / "buy" / "latest.pt"
        if buy_ckpt.exists():
            buy_agent.load(buy_ckpt)
            print(f"Loaded buy agent from {buy_ckpt}")
        else:
            print("No buy agent checkpoint found — using random buy agent")
    buy_agent.network.eval()
    for p in buy_agent.network.parameters():
        p.requires_grad = False

    sell_agent = SellAgent(n_base_features=n_base_features, device=device)
    optimizer = optim.Adam(sell_agent.network.parameters(), lr=cfg.lr)
    buffer = SellRolloutBuffer()
    reward_history: deque = deque(maxlen=100)
    best_sharpe = -np.inf

    # Trade log
    log_file = open(LOG_DIR / "sell_trades.csv", "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["episode", "permno", "days_held", "net_return", "exit_reason"])

    print(f"Training Sell Agent | device={device} | base_features={n_base_features}")

    for episode in range(1, cfg.n_episodes + 1):
        env = MockTradingEnvironment(n_episodes=cfg.n_steps_per_update * 2, seed=episode + 10000)
        candidates = env.reset()
        stop_mgr = StopLossManager()
        ep_reward = 0.0
        done = False

        while not done and len(buffer) < cfg.n_steps_per_update:
            # Buy agent fills positions (no gradient)
            with torch.no_grad():
                buy_permnos = buy_agent.act(candidates)

            for p in buy_permnos:
                price = env._prices.get(p, 100.0)
                stop_mgr.register(p, price, atr=price * 0.02, beta=1.0)

            # Stop loss
            stop_exits = stop_mgr.check(
                {pos.permno: env._prices.get(pos.permno, pos.entry_price) for pos in env.positions}
            )

            # Sell agent decisions
            position_states = _build_position_states(env, n_base_features)
            position_states_no_stop = [
                (p, s) for p, s in position_states if p not in stop_exits
            ]

            if position_states_no_stop:
                sell_permnos, actions, log_probs, values = sell_agent.act_with_info(position_states_no_stop)
            else:
                sell_permnos, actions, log_probs, values = [], torch.tensor([]), torch.tensor([]), torch.tensor([])

            all_exits = list(set(stop_exits + sell_permnos))

            shaped_reward = shape_sell_reward(0.0, sell_permnos, env.positions, cfg)

            candidates, env_reward, done, info = env.step(buy_permnos, all_exits)
            shaped_reward += env_reward
            ep_reward += shaped_reward

            for p in all_exits:
                stop_mgr.remove(p)

            # Add to buffer
            for i, (permno, state) in enumerate(position_states_no_stop):
                if i < len(actions):
                    buffer.add(
                        state=state,
                        action=actions[i].item(),
                        log_prob=log_probs[i].item(),
                        reward=shaped_reward / max(len(position_states_no_stop), 1),
                        value=values[i].item(),
                        done=done,
                    )

            # Max-hold forced exits log
            for pos in list(env.positions):
                if pos.days_held >= MAX_HOLD_DAYS:
                    log_writer.writerow([episode, pos.permno, pos.days_held, 0.0, "max_hold"])

        # PPO update
        if len(buffer) > 0:
            with torch.no_grad():
                zero_state = torch.zeros(1, n_base_features + 5, device=sell_agent.device)
                _, last_value = sell_agent.network(zero_state)
            buffer.compute_returns_and_advantages(last_value.item(), cfg.gamma, cfg.gae_lambda)
            ppo_update(sell_agent, optimizer, buffer, cfg)
            buffer.clear()

        reward_history.append(ep_reward)

        if episode % 10 == 0:
            print(
                f"Ep {episode:4d} | reward={ep_reward:+.4f} "
                f"| avg100={np.mean(reward_history):+.4f} "
                f"| positions={info.get('n_positions', 0)}"
            )

        if episode % cfg.checkpoint_every == 0:
            sell_agent.save()
            sharpe = validate(sell_agent, buy_agent)
            print(f"  [Val] Sharpe proxy: {sharpe:.4f}")
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                sell_agent.save_best(sharpe, episode)
                print(f"  [Best] New best Sharpe: {sharpe:.4f}")

    log_file.close()
    sell_agent.save()
    print(f"\nTraining complete. Best val Sharpe: {best_sharpe:.4f}")
    return sell_agent


if __name__ == "__main__":
    train()
