"""
Walk-forward evaluation: re-optimize on each training window, test on the next.
"""

import argparse
import pandas as pd
from typing import Iterable, List, Optional, Tuple
from src.optimizer_runner import OPTIMIZERS, OptimizationRunner


SUMMARY_METRICS = ['test_sharpe_ratio', 'test_total_return', 'test_max_drawdown', 'test_profit_factor']


def walk_forward_splits(
    n: int,
    train_size: int,
    test_size: int,
    anchored: bool = False
) -> List[Tuple[int, int, int, int]]:
    """
    Generate walk-forward windows with consecutive, non-overlapping test periods.

    Parameters:
    -----------
    n : int
        Number of periods in the data
    train_size : int
        Periods in each training window (the first one if anchored)
    test_size : int
        Periods in each test window; windows advance by this much
    anchored : bool
        If True, every training window starts at 0 (expanding window);
        otherwise it is the train_size periods before the test window

    Returns:
    --------
    list
        List of (train_start, train_end, test_start, test_end) tuples
    """
    splits = []
    test_start = train_size
    while test_start + test_size <= n:
        train_start = 0 if anchored else test_start - train_size
        splits.append((train_start, test_start, test_start, test_start + test_size))
        test_start += test_size
    return splits


def walk_forward(
    log_prices: pd.DataFrame,
    optimizers: Iterable[str] = OPTIMIZERS,
    train_size: int = 504,
    test_size: int = 126,
    anchored: bool = False,
    verbose: bool = True,
    **runner_kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run each optimizer on every walk-forward window.

    Parameters:
    -----------
    log_prices : pd.DataFrame
        Log prices
    optimizers : Iterable[str]
        Optimizers to compare (see OPTIMIZERS)
    train_size, test_size, anchored :
        Window settings, see walk_forward_splits
    **runner_kwargs :
        Passed to OptimizationRunner (thresholds, transaction_cost, metric,
        n_trials, seed)

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        (per_window, summary): one row per window and optimizer, and the mean
        and standard deviation of the test metrics across windows
    """
    optimizers = list(optimizers)
    splits = walk_forward_splits(len(log_prices), train_size, test_size, anchored)
    if not splits:
        raise ValueError(f"Need at least {train_size + test_size} periods, got {len(log_prices)}")

    rows = []
    for window, (train_start, train_end, test_start, test_end) in enumerate(splits):
        train = log_prices.iloc[train_start:train_end]
        test = log_prices.iloc[test_start:test_end]
        if verbose:
            print(f"Window {window + 1}/{len(splits)}: "
                  f"test {test.index[0].date()} to {test.index[-1].date()}")

        result = OptimizationRunner(train, test, **runner_kwargs).run(optimizers, verbose=False)
        result = result.reset_index()
        result.insert(0, 'window', window)
        result.insert(1, 'test_start', test.index[0])
        result.insert(2, 'test_end', test.index[-1])
        rows.append(result)

    per_window = pd.concat(rows, ignore_index=True)

    summary = per_window.groupby('optimizer', sort=False)[SUMMARY_METRICS].agg(['mean', 'std'])
    summary[('windows', 'count')] = per_window.groupby('optimizer', sort=False).size()
    summary[('windows', 'positive_sharpe')] = (
        per_window.assign(positive=per_window['test_sharpe_ratio'] > 0)
        .groupby('optimizer', sort=False)['positive'].mean()
    )

    return per_window, summary


def main(argv: Optional[list] = None):
    from src.data_utils import fetch_price_data, prepare_data

    parser = argparse.ArgumentParser(description='Walk-forward comparison of weight optimizers')
    parser.add_argument('--assets', nargs='+', required=True, help='Asset symbols')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--optimizers', nargs='+', default=list(OPTIMIZERS), choices=OPTIMIZERS)
    parser.add_argument('--train-size', type=int, default=504, help='Training window in days (default: 504)')
    parser.add_argument('--test-size', type=int, default=126, help='Test window in days (default: 126)')
    parser.add_argument('--anchored', action='store_true', help='Expanding training window from the start')
    parser.add_argument('--n-trials', type=int, default=50, help='Evaluations per optimizer (default: 50)')
    parser.add_argument('--metric', default='sharpe', choices=['sharpe', 'return', 'profit_factor'])
    parser.add_argument('--entry-threshold', type=float, default=2.0)
    parser.add_argument('--exit-threshold', type=float, default=0.5)
    parser.add_argument('--transaction-cost', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', help='Optional CSV path for the per-window results')
    args = parser.parse_args(argv)

    prices = fetch_price_data(args.assets, args.start_date, args.end_date)
    log_prices, _ = prepare_data(prices)

    per_window, summary = walk_forward(
        log_prices,
        optimizers=args.optimizers,
        train_size=args.train_size,
        test_size=args.test_size,
        anchored=args.anchored,
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        transaction_cost=args.transaction_cost,
        metric=args.metric,
        n_trials=args.n_trials,
        seed=args.seed
    )

    with pd.option_context('display.width', 200, 'display.max_columns', None):
        print(summary.round(4))

    if args.output:
        per_window.to_csv(args.output, index=False)
        print(f"Per-window results saved to {args.output}")


if __name__ == '__main__':
    main()
