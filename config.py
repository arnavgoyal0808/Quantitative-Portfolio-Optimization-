"""Configuration file for the quantitative research project."""

from datetime import datetime, timedelta

# Data Configuration
START_DATE = '2015-01-01'
END_DATE = '2023-12-31'
BENCHMARK_TICKER = '^GSPC'  # S&P 500

# Factor Configuration
MOMENTUM_LOOKBACK_PERIODS = [21, 63, 252]  # 1 month, 3 months, 1 year
VOLATILITY_WINDOW = 252  # 1 year
REBALANCE_FREQUENCY = 'M'  # Monthly rebalancing

# Portfolio Configuration
TRANSACTION_COST = 0.001  # 10 basis points
MAX_POSITION_SIZE = 0.1   # 10% maximum position
MIN_POSITION_SIZE = 0.01  # 1% minimum position

# Optimization Configuration
RISK_FREE_RATE = 0.02  # 2% annual risk-free rate
TARGET_VOLATILITY = 0.15  # 15% target volatility
RISK_AVERSION = 1.0

# Backtesting Configuration
INITIAL_CAPITAL = 1000000  # $1M initial capital
LOOKBACK_WINDOW = 252  # 1 year for rolling metrics

# Output Configuration
RESULTS_DIR = 'results'
PLOTS_DIR = 'results/plots'
DATA_DIR = 'data'

# Performance Metrics
PERFORMANCE_METRICS = [
    'Total Return',
    'Annualized Return', 
    'Volatility',
    'Sharpe Ratio',
    'Max Drawdown',
    'Calmar Ratio',
    'Skewness',
    'Kurtosis'
]

# Factor Names
FACTOR_NAMES = [
    'momentum',
    'size', 
    'value',
    'volatility',
    'quality'
]

# Optimization Methods
OPTIMIZATION_METHODS = [
    'mean_variance',
    'risk_parity',
    'min_variance',
    'hrp'
]
