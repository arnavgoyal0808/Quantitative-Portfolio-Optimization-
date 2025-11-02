"""Utility functions for quantitative research project."""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

def calculate_returns(prices: pd.DataFrame, method: str = 'simple') -> pd.DataFrame:
    """Calculate returns from price data."""
    if method == 'simple':
        return prices.pct_change().dropna()
    elif method == 'log':
        return np.log(prices / prices.shift(1)).dropna()
    else:
        raise ValueError("Method must be 'simple' or 'log'")

def winsorize(data: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Winsorize data to remove outliers."""
    lower_bound = data.quantile(lower)
    upper_bound = data.quantile(upper)
    return data.clip(lower=lower_bound, upper=upper_bound)

def standardize(data: pd.Series) -> pd.Series:
    """Standardize data to have mean 0 and std 1."""
    return (data - data.mean()) / data.std()

def create_deciles(factor_scores: pd.Series) -> pd.Series:
    """Create decile rankings from factor scores."""
    return pd.qcut(factor_scores.rank(method='first'), 10, labels=False) + 1

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio."""
    excess_returns = returns - risk_free_rate / 252
    return np.sqrt(252) * excess_returns.mean() / returns.std()

def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def calculate_alpha_beta(portfolio_returns: pd.Series, market_returns: pd.Series) -> Tuple[float, float]:
    """Calculate alpha and beta relative to market."""
    from sklearn.linear_model import LinearRegression
    
    X = market_returns.values.reshape(-1, 1)
    y = portfolio_returns.values
    
    model = LinearRegression().fit(X, y)
    beta = model.coef_[0]
    alpha = model.intercept_ * 252  # Annualized
    
    return alpha, beta

def get_sp500_tickers() -> List[str]:
    """Get S&P 500 ticker symbols."""
    # Predefined list to avoid web scraping issues in CI
    tickers = [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'META', 'NVDA', 'BRK-B', 'UNH', 'JNJ',
        'V', 'PG', 'JPM', 'HD', 'CVX', 'MA', 'PFE', 'ABBV', 'BAC', 'KO', 'AVGO', 'PEP',
        'TMO', 'COST', 'WMT', 'DIS', 'ABT', 'DHR', 'VZ', 'ADBE', 'CMCSA', 'NKE', 'LIN',
        'NEE', 'CRM', 'ACN', 'TXN', 'RTX', 'QCOM', 'LOW', 'PM', 'SPGI', 'HON', 'UNP',
        'IBM', 'AMGN', 'ELV', 'SCHW', 'T', 'CAT', 'INTU', 'GS', 'BKNG', 'AXP', 'DE',
        'AMD', 'LMT', 'SYK', 'ADI', 'GILD', 'MDLZ', 'ADP', 'TJX', 'VRTX', 'CVS', 'MU',
        'CI', 'ISRG', 'ZTS', 'MMC', 'SO', 'DUK', 'BDX', 'SHW', 'CME', 'EOG', 'USB',
        'PNC', 'NSC', 'AON', 'REGN', 'CL', 'BSX', 'ITW', 'HCA', 'GD', 'APD', 'EMR',
        'CSX', 'WM', 'MMM', 'FCX', 'PLD', 'ICE', 'F', 'GM', 'FDX', 'NOC', 'TGT', 'COP',
        'KLAC', 'SLB', 'MPC', 'PSX', 'AIG', 'DG', 'BK', 'TFC', 'ORLY', 'ECL', 'EW'
    ]
    return tickers[:100]  # Return first 100 for faster processing

def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Safely divide two series, handling division by zero."""
    return numerator / denominator.replace(0, np.nan)

def rolling_correlation(x: pd.Series, y: pd.Series, window: int = 252) -> pd.Series:
    """Calculate rolling correlation between two series."""
    return x.rolling(window).corr(y)

def performance_summary(returns: pd.Series, benchmark_returns: pd.Series = None) -> Dict:
    """Generate comprehensive performance summary."""
    summary = {
        'Total Return': (1 + returns).prod() - 1,
        'Annualized Return': (1 + returns).prod() ** (252 / len(returns)) - 1,
        'Volatility': returns.std() * np.sqrt(252),
        'Sharpe Ratio': calculate_sharpe_ratio(returns),
        'Max Drawdown': calculate_max_drawdown(returns),
        'Skewness': returns.skew(),
        'Kurtosis': returns.kurtosis()
    }
    
    if benchmark_returns is not None:
        alpha, beta = calculate_alpha_beta(returns, benchmark_returns)
        summary['Alpha'] = alpha
        summary['Beta'] = beta
        summary['Information Ratio'] = (returns - benchmark_returns).mean() / (returns - benchmark_returns).std() * np.sqrt(252)
    
    return summary
