"""
Run and compare weight optimizers on the same objective and budget.
"""

import argparse
import numpy as np
import pandas as pd
from typing import Dict, Iterable, Optional, Tuple
from src.bayesian_opt import BayesianOptimizer, create_objective_function
from src.cointegration import calculate_spread
from src.optimizers import CMAESOptimizer, TuRBOOptimizer
from src.strategy import TradingStrategy


OPTIMIZERS = ('bo', 'cmaes', 'turbo')


class OptimizationRunner:
    """
    Optimize weights on a training period with each method, then evaluate
    the resulting weights on a separate test period.

    Every optimizer maximizes the same objective (create_objective_function on
    the training prices) with the same number of evaluations and the same
    weight bounds as BayesianOptimizer.
    """

    def __init__(
        self,
        train_prices: pd.DataFrame,
        test_prices: pd.DataFrame,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        transaction_cost: float = 0.001,
        metric: str = 'sharpe',
        n_trials: int = 50,
        seed: int = 42
    ):
        """
        Parameters:
        -----------
        train_prices : pd.DataFrame
            Log prices used for optimization
        test_prices : pd.DataFrame
            Log prices used for evaluation (should follow train_prices in time)
        entry_threshold, exit_threshold, transaction_cost : float
            Trading strategy parameters, used for both periods
        metric : str
            Objective metric: 'sharpe', 'return' or 'profit_factor'
        n_trials : int
            Objective evaluations per optimizer
        seed : int
            Random seed for CMA-ES and TuRBO
        """
        if list(train_prices.columns) != list(test_prices.columns):
            raise ValueError("Train and test prices must have the same assets")

        self.train_prices = train_prices
        self.test_prices = test_prices
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.transaction_cost = transaction_cost
        self.metric = metric
        self.n_trials = n_trials
        self.seed = seed
        self.n_assets = train_prices.shape[1]
        self.results: Dict[str, Dict] = {}

    def _objective(self):
        return create_objective_function(
            self.train_prices,
            entry_threshold=self.entry_threshold,
            exit_threshold=self.exit_threshold,
            transaction_cost=self.transaction_cost,
            metric=self.metric
        )

    def optimize(self, name: str) -> Tuple[np.ndarray, float, int]:
        """
        Run one optimizer on the training period.

        Returns:
        --------
        Tuple[np.ndarray, float, int]
            (best_weights, best_train_score, n_evaluations) tuple
        """
        objective = self._objective()

        if name == 'bo':
            n_evals = 0

            def counted(weights):
                nonlocal n_evals
                n_evals += 1
                return objective(weights)

            optimizer = BayesianOptimizer(
                objective_func=counted,
                n_assets=self.n_assets,
                n_trials=self.n_trials,
                n_random_starts=min(10, self.n_trials)
            )
            weights, score = optimizer.optimize()
            return weights, score, n_evals

        if name == 'cmaes':
            optimizer = CMAESOptimizer(objective, self.n_assets, self.n_trials, seed=self.seed)
        elif name == 'turbo':
            optimizer = TuRBOOptimizer(objective, self.n_assets, self.n_trials, seed=self.seed)
        else:
            raise ValueError(f"Unknown optimizer: {name}. Choose from {OPTIMIZERS}")

        weights, score = optimizer.optimize()
        return weights, score, optimizer.n_evals

    def evaluate(self, weights: np.ndarray) -> Dict:
        """
        Backtest weights on the test period.

        The end of the training period warms up the rolling statistics, so the
        strategy can trade from the first test day and metrics cover the whole
        test period.
        """
        strategy = TradingStrategy(
            entry_threshold=self.entry_threshold,
            exit_threshold=self.exit_threshold,
            transaction_cost=self.transaction_cost
        )
        warmup = min(strategy.lookback_period, len(self.train_prices))
        prices = pd.concat([self.train_prices.iloc[len(self.train_prices) - warmup:], self.test_prices])
        spread = calculate_spread(prices, weights)
        return strategy.backtest(spread, weights, prices, warmup=warmup)

    def run(self, optimizers: Iterable[str] = OPTIMIZERS, verbose: bool = True) -> pd.DataFrame:
        """
        Optimize on the training period and evaluate on the test period.

        Returns:
        --------
        pd.DataFrame
            One row per optimizer with the training score and test metrics
        """
        rows = []
        for name in optimizers:
            if verbose:
                print(f"Running {name}...")

            weights, train_score, n_evals = self.optimize(name)
            test_metrics = self.evaluate(weights)
            self.results[name] = {
                'weights': weights,
                'train_score': train_score,
                'n_evals': n_evals,
                'test_metrics': test_metrics
            }

            rows.append({
                'optimizer': name,
                'n_evals': n_evals,
                f'train_{self.metric}': train_score,
                'test_sharpe_ratio': test_metrics['sharpe_ratio'],
                'test_total_return': test_metrics['total_return'],
                'test_max_drawdown': test_metrics['max_drawdown'],
                'test_profit_factor': test_metrics['profit_factor'],
                'test_num_trades': test_metrics['num_trades'],
                'weights': np.round(weights, 4).tolist()
            })

        return pd.DataFrame(rows).set_index('optimizer')


def train_test_split(prices: pd.DataFrame, test_size: float = 0.25) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: the test period is the last test_size of the data."""
    split = int(len(prices) * (1 - test_size))
    return prices.iloc[:split], prices.iloc[split:]


def main(argv: Optional[list] = None):
    from src.data_utils import fetch_price_data, prepare_data

    parser = argparse.ArgumentParser(description='Compare weight optimizers for basket trading')
    parser.add_argument('--assets', nargs='+', required=True, help='Asset symbols')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--optimizers', nargs='+', default=list(OPTIMIZERS), choices=OPTIMIZERS)
    parser.add_argument('--n-trials', type=int, default=50, help='Evaluations per optimizer (default: 50)')
    parser.add_argument('--metric', default='sharpe', choices=['sharpe', 'return', 'profit_factor'])
    parser.add_argument('--test-size', type=float, default=0.25,
                        help='Fraction of data held out for testing (default: 0.25)')
    parser.add_argument('--entry-threshold', type=float, default=2.0)
    parser.add_argument('--exit-threshold', type=float, default=0.5)
    parser.add_argument('--transaction-cost', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)

    prices = fetch_price_data(args.assets, args.start_date, args.end_date)
    log_prices, _ = prepare_data(prices)
    train, test = train_test_split(log_prices, args.test_size)
    print(f"Train: {train.index[0].date()} to {train.index[-1].date()} ({len(train)} days)")
    print(f"Test:  {test.index[0].date()} to {test.index[-1].date()} ({len(test)} days)")

    runner = OptimizationRunner(
        train, test,
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        transaction_cost=args.transaction_cost,
        metric=args.metric,
        n_trials=args.n_trials,
        seed=args.seed
    )
    summary = runner.run(args.optimizers)

    with pd.option_context('display.width', 200, 'display.max_columns', None):
        print(summary)


if __name__ == '__main__':
    main()
