import numpy as np
import pandas as pd

from src.strategy import TradingStrategy

# Quiet spread for the lookback, a sharp move away from the mean, then reversion
QUIET = [0.0, 0.1, -0.1, 0.1, -0.1, 0.1]


def run(spread_values):
    spread = pd.Series(spread_values, index=pd.bdate_range('2024-01-01', periods=len(spread_values)))
    prices = pd.DataFrame({'A': spread_values, 'B': np.zeros(len(spread_values))}, index=spread.index)
    strategy = TradingStrategy(entry_threshold=2.0, exit_threshold=0.5,
                               transaction_cost=0.001, lookback_period=5)
    signals = strategy.generate_signals(spread)
    returns = strategy.calculate_returns(spread, signals, np.array([1.0, -1.0]), prices)
    return signals, returns


def test_long_entry_then_rising_spread_is_profitable():
    signals, returns = run(QUIET + [-2.0, -1.0, 0.0, 0.0])

    assert signals.iloc[6] == 1  # long the spread after it drops below its mean
    assert returns.iloc[6] > 0   # spread rises from -2.0 to -1.0
    assert returns.iloc[7] > 0   # and again from -1.0 to 0.0
    assert returns.sum() > 0


def test_short_entry_then_falling_spread_is_profitable():
    signals, returns = run(QUIET + [2.0, 1.0, 0.0, 0.0])

    assert signals.iloc[6] == -1
    assert returns.iloc[6] > 0
    assert returns.iloc[7] > 0
    assert returns.sum() > 0


def test_costs_scale_with_gross_exposure():
    # Flat spread after the entry, so the entry-day return is only the cost
    values = QUIET + [-2.0, -2.0]
    spread = pd.Series(values, index=pd.bdate_range('2024-01-01', periods=len(values)))
    prices = pd.DataFrame({'A': values, 'B': np.zeros(len(values))}, index=spread.index)
    strategy = TradingStrategy(transaction_cost=0.001, lookback_period=5)
    signals = strategy.generate_signals(spread)

    small = strategy.calculate_returns(spread, signals, np.array([1.0, -1.0]), prices)
    large = strategy.calculate_returns(spread, signals, np.array([2.0, -1.0]), prices)

    assert signals.iloc[6] == 1
    assert np.isclose(small.iloc[6], -0.002)
    assert np.isclose(large.iloc[6], -0.003)


def test_warmup_lets_the_first_evaluated_day_trade():
    values = QUIET + [-2.0, -1.0, 0.0, 0.0]
    spread = pd.Series(values, index=pd.bdate_range('2024-01-01', periods=len(values)))
    prices = pd.DataFrame({'A': values, 'B': np.zeros(len(values))}, index=spread.index)
    strategy = TradingStrategy(transaction_cost=0.001, lookback_period=5)
    weights = np.array([1.0, -1.0])

    with_warmup = strategy.backtest(spread, weights, prices, warmup=6)
    without_warmup = strategy.backtest(spread.iloc[6:], weights, prices.iloc[6:])

    assert with_warmup['total_return'] > 0
    assert without_warmup['total_return'] == 0
