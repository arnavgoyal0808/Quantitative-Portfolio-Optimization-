"""Quantitative Research Project - Source Package."""

__version__ = "1.0.0"
__author__ = "Quantitative Researcher"
__email__ = "researcher@example.com"

from .data_loader import DataLoader
from .factor_analysis import FactorAnalyzer
from .portfolio_optimization import PortfolioOptimizer
from .backtesting import Backtester
from . import utils

__all__ = [
    "DataLoader",
    "FactorAnalyzer", 
    "PortfolioOptimizer",
    "Backtester",
    "utils"
]
