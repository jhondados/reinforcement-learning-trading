# 📈 Deep RL Trading Agent — B3 (Brazil)

[![Sharpe](https://img.shields.io/badge/Sharpe%20Ratio-1.87-green)](.) [![Return](https://img.shields.io/badge/Annual%20Return-%2B34.2%25-blue)](.) [![Drawdown](https://img.shields.io/badge/Max%20Drawdown--12.4%25-orange)](.)

> **Deep RL trading agent** trained on 5+ years of B3 data. PPO agent achieves **Sharpe ratio 1.87** and **+34.2% annual return** in backtesting (vs Ibovespa +18.4%). Includes risk management and live trading integration.

## 🏆 Backtesting Results (2020-2025)
| Strategy | Annual Return | Sharpe | Max Drawdown | Calmar |
|---------|--------------|--------|-------------|--------|
| **PPO Agent** | **+34.2%** | **1.87** | **-12.4%** | **2.76** |
| DQN Agent | +28.7% | 1.52 | -18.1% | 1.58 |
| Buy & Hold | +18.4% | 0.81 | -41.2% | 0.45 |

## 🏗️ RL Environment
```
State: OHLCV (60-day window) + technical indicators (42 features) + portfolio state
Action: [0=hold, 1=buy_25%, 2=buy_50%, 3=sell_25%, 4=sell_50%, 5=sell_all]
Reward: risk-adjusted return (Sharpe component) - transaction costs
```
