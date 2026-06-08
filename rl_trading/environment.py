"""B3 trading RL environment."""
import gymnasium as gym
import numpy as np
from typing import Tuple, Dict

class B3TradingEnv(gym.Env):
    def __init__(self, prices: np.ndarray, features: np.ndarray,
                 initial_cash: float = 100_000.0, transaction_cost: float = 0.001):
        super().__init__()
        self.prices = prices
        self.features = features
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.window = 60
        self.n_features = features.shape[1]
        # Action: 0=hold, 1=buy25%, 2=buy50%, 3=sell25%, 4=sell50%, 5=sell_all
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
            shape=(self.window * self.n_features + 3,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None):
        self.step_idx = self.window
        self.cash = self.initial_cash
        self.shares = 0.0
        self.portfolio_history = [self.initial_cash]
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        market_features = self.features[self.step_idx - self.window:self.step_idx].flatten()
        portfolio_state = np.array([self.cash / self.initial_cash,
            self.shares * self.prices[self.step_idx] / self.initial_cash,
            (self.cash + self.shares * self.prices[self.step_idx]) / self.initial_cash])
        return np.concatenate([market_features, portfolio_state]).astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        price = self.prices[self.step_idx]
        portfolio_before = self.cash + self.shares * price
        if action == 1: shares_to_buy = (self.cash * 0.25) / price * (1 - self.transaction_cost); self.shares += shares_to_buy; self.cash -= shares_to_buy * price
        elif action == 2: shares_to_buy = (self.cash * 0.50) / price * (1 - self.transaction_cost); self.shares += shares_to_buy; self.cash -= shares_to_buy * price
        elif action == 3: shares_sold = self.shares * 0.25; self.cash += shares_sold * price * (1 - self.transaction_cost); self.shares -= shares_sold
        elif action == 4: shares_sold = self.shares * 0.50; self.cash += shares_sold * price * (1 - self.transaction_cost); self.shares -= shares_sold
        elif action == 5: self.cash += self.shares * price * (1 - self.transaction_cost); self.shares = 0
        self.step_idx += 1
        portfolio_after = self.cash + self.shares * self.prices[self.step_idx]
        self.portfolio_history.append(portfolio_after)
        reward = (portfolio_after - portfolio_before) / portfolio_before
        done = self.step_idx >= len(self.prices) - 1
        return self._get_obs(), reward, done, False, {"portfolio_value": portfolio_after}
