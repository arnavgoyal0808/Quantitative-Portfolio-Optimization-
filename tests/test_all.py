"""Comprehensive test suite for the quantitative research project."""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import (calculate_returns, winsorize, standardize, create_deciles,
                  calculate_sharpe_ratio, calculate_max_drawdown, get_sp500_tickers)
from data_loader import DataLoader
from factor_analysis import FactorAnalyzer
from portfolio_optimization import PortfolioOptimizer
from backtesting import Backtester

class TestUtils:
    """Test utility functions."""
    
    def test_calculate_returns(self):
        """Test return calculation."""
        prices = pd.Series([100, 105, 102, 108], index=pd.date_range('2020-01-01', periods=4))
        returns = calculate_returns(prices)
        
        expected = pd.Series([0.05, -0.0286, 0.0588], 
                           index=pd.date_range('2020-01-02', periods=3))
        
        assert len(returns) == 3
        assert abs(returns.iloc[0] - 0.05) < 0.001
    
    def test_winsorize(self):
        """Test winsorization."""
        data = pd.Series([1, 2, 3, 4, 5, 100])  # 100 is outlier
        winsorized = winsorize(data, lower=0.1, upper=0.9)
        
        assert winsorized.max() < 100
        assert len(winsorized) == len(data)
    
    def test_standardize(self):
        """Test standardization."""
        data = pd.Series([1, 2, 3, 4, 5])
        standardized = standardize(data)
        
        assert abs(standardized.mean()) < 1e-10
        assert abs(standardized.std() - 1.0) < 1e-10
    
    def test_create_deciles(self):
        """Test decile creation."""
        data = pd.Series(range(100))
        deciles = create_deciles(data)
        
        assert deciles.min() == 1
        assert deciles.max() == 10
        assert len(deciles.unique()) == 10
    
    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005])
        sharpe = calculate_sharpe_ratio(returns)
        
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
    
    def test_max_drawdown(self):
        """Test maximum drawdown calculation."""
        returns = pd.Series([0.1, -0.05, -0.1, 0.15, -0.02])
        max_dd = calculate_max_drawdown(returns)
        
        assert max_dd <= 0
        assert isinstance(max_dd, float)
    
    def test_get_sp500_tickers(self):
        """Test S&P 500 ticker retrieval."""
        tickers = get_sp500_tickers()
        
        assert isinstance(tickers, list)
        assert len(tickers) > 0
        assert 'AAPL' in tickers

class TestDataLoader:
    """Test data loading functionality."""
    
    def setup_method(self):
        """Setup test data loader."""
        self.loader = DataLoader(start_date='2020-01-01', end_date='2020-12-31')
    
    def test_initialization(self):
        """Test data loader initialization."""
        assert self.loader.start_date == '2020-01-01'
        assert self.loader.end_date == '2020-12-31'
        assert self.loader.data_dir == 'data'
    
    def test_create_price_matrix(self):
        """Test price matrix creation."""
        # Mock stock data
        dates = pd.date_range('2020-01-01', periods=10)
        stock_data = {
            'AAPL': pd.DataFrame({
                'Adj Close': np.random.randn(10) + 100,
                'Volume': np.random.randint(1000, 10000, 10)
            }, index=dates),
            'MSFT': pd.DataFrame({
                'Adj Close': np.random.randn(10) + 200,
                'Volume': np.random.randint(1000, 10000, 10)
            }, index=dates)
        }
        
        prices = self.loader.create_price_matrix(stock_data)
        
        assert isinstance(prices, pd.DataFrame)
        assert prices.shape[1] == 2
        assert 'AAPL' in prices.columns
        assert 'MSFT' in prices.columns

class TestFactorAnalysis:
    """Test factor analysis functionality."""
    
    def setup_method(self):
        """Setup test data for factor analysis."""
        dates = pd.date_range('2020-01-01', periods=500)  # Need enough data for factors
        n_stocks = 10
        
        # Create mock price data
        np.random.seed(42)
        prices_data = np.random.randn(500, n_stocks).cumsum(axis=0) + 100
        self.prices = pd.DataFrame(prices_data, 
                                 index=dates,
                                 columns=[f'STOCK_{i}' for i in range(n_stocks)])
        
        # Create mock volume data
        volumes_data = np.random.randint(1000, 10000, (500, n_stocks))
        self.volumes = pd.DataFrame(volumes_data,
                                  index=dates,
                                  columns=[f'STOCK_{i}' for i in range(n_stocks)])
        
        self.analyzer = FactorAnalyzer(self.prices, self.volumes)
    
    def test_initialization(self):
        """Test factor analyzer initialization."""
        assert self.analyzer.prices.shape == self.prices.shape
        assert self.analyzer.volumes.shape == self.volumes.shape
        assert len(self.analyzer.returns) == len(self.prices) - 1
    
    def test_momentum_factor(self):
        """Test momentum factor computation."""
        momentum = self.analyzer.compute_momentum_factor()
        
        assert isinstance(momentum, pd.DataFrame)
        assert momentum.shape[1] == self.prices.shape[1]
        assert not momentum.isna().all().all()
    
    def test_size_factor(self):
        """Test size factor computation."""
        size = self.analyzer.compute_size_factor()
        
        assert isinstance(size, pd.DataFrame)
        assert size.shape[1] == self.prices.shape[1]
    
    def test_volatility_factor(self):
        """Test volatility factor computation."""
        volatility = self.analyzer.compute_volatility_factor()
        
        assert isinstance(volatility, pd.DataFrame)
        assert volatility.shape[1] == self.prices.shape[1]
    
    def test_compute_all_factors(self):
        """Test computing all factors."""
        factors = self.analyzer.compute_all_factors()
        
        assert isinstance(factors, dict)
        assert len(factors) >= 4  # At least momentum, size, value, volatility
        
        for factor_name, factor_data in factors.items():
            assert isinstance(factor_data, pd.DataFrame)
            assert factor_data.shape[1] == self.prices.shape[1]
    
    def test_create_factor_portfolios(self):
        """Test factor portfolio creation."""
        self.analyzer.compute_momentum_factor()
        portfolios = self.analyzer.create_factor_portfolios('momentum')
        
        assert isinstance(portfolios, dict)
        assert len(portfolios) == 10  # 10 deciles
        
        # Check that all stocks are assigned
        all_stocks = []
        for stocks in portfolios.values():
            all_stocks.extend(stocks)
        
        assert len(set(all_stocks)) <= self.prices.shape[1]
    
    def test_long_short_portfolio(self):
        """Test long-short portfolio creation."""
        self.analyzer.compute_momentum_factor()
        long_stocks, short_stocks = self.analyzer.create_long_short_portfolio('momentum')
        
        assert isinstance(long_stocks, list)
        assert isinstance(short_stocks, list)
        assert len(long_stocks) > 0
        assert len(short_stocks) > 0
        assert set(long_stocks).isdisjoint(set(short_stocks))

class TestPortfolioOptimization:
    """Test portfolio optimization functionality."""
    
    def setup_method(self):
        """Setup test data for portfolio optimization."""
        dates = pd.date_range('2020-01-01', periods=252)  # 1 year of data
        n_stocks = 5
        
        # Create mock price data with realistic properties
        np.random.seed(42)
        returns_data = np.random.multivariate_normal(
            mean=[0.0005] * n_stocks,
            cov=np.eye(n_stocks) * 0.0001 + 0.00005,
            size=252
        )
        
        prices_data = (1 + returns_data).cumprod(axis=0) * 100
        self.prices = pd.DataFrame(prices_data,
                                 index=dates,
                                 columns=[f'STOCK_{i}' for i in range(n_stocks)])
        
        self.optimizer = PortfolioOptimizer(self.prices)
    
    def test_initialization(self):
        """Test portfolio optimizer initialization."""
        assert self.optimizer.prices.shape == self.prices.shape
        assert len(self.optimizer.returns) == len(self.prices) - 1
    
    def test_mean_variance_optimization(self):
        """Test mean-variance optimization."""
        weights = self.optimizer.mean_variance_optimization()
        
        assert isinstance(weights, dict)
        assert len(weights) == self.prices.shape[1]
        assert abs(sum(weights.values()) - 1.0) < 0.01  # Weights sum to 1
        assert all(w >= -0.01 for w in weights.values())  # No short selling (approximately)
    
    def test_risk_parity_optimization(self):
        """Test risk parity optimization."""
        weights = self.optimizer.risk_parity_optimization()
        
        assert isinstance(weights, dict)
        assert len(weights) == self.prices.shape[1]
        assert abs(sum(weights.values()) - 1.0) < 0.01
        assert all(w >= 0 for w in weights.values())  # Long-only
    
    def test_minimum_variance_optimization(self):
        """Test minimum variance optimization."""
        weights = self.optimizer.minimum_variance_optimization()
        
        assert isinstance(weights, dict)
        assert len(weights) == self.prices.shape[1]
        assert abs(sum(weights.values()) - 1.0) < 0.01
    
    def test_calculate_portfolio_performance(self):
        """Test portfolio performance calculation."""
        weights = self.optimizer.mean_variance_optimization()
        performance = self.optimizer.calculate_portfolio_performance(weights)
        
        assert isinstance(performance, dict)
        assert 'Annualized Return' in performance
        assert 'Volatility' in performance
        assert 'Sharpe Ratio' in performance
        assert 'Max Drawdown' in performance

class TestBacktesting:
    """Test backtesting functionality."""
    
    def setup_method(self):
        """Setup test data for backtesting."""
        dates = pd.date_range('2020-01-01', periods=252)
        n_stocks = 5
        
        # Create mock price data
        np.random.seed(42)
        returns_data = np.random.multivariate_normal(
            mean=[0.0005] * n_stocks,
            cov=np.eye(n_stocks) * 0.0001 + 0.00005,
            size=252
        )
        
        prices_data = (1 + returns_data).cumprod(axis=0) * 100
        self.prices = pd.DataFrame(prices_data,
                                 index=dates,
                                 columns=[f'STOCK_{i}' for i in range(n_stocks)])
        
        # Create benchmark
        benchmark_returns = np.random.normal(0.0005, 0.01, 252)
        self.benchmark_prices = pd.Series((1 + benchmark_returns).cumprod() * 100, index=dates)
        
        self.backtester = Backtester(self.prices, self.benchmark_prices)
    
    def test_initialization(self):
        """Test backtester initialization."""
        assert self.backtester.prices.shape == self.prices.shape
        assert len(self.backtester.returns) == len(self.prices) - 1
        assert len(self.backtester.benchmark_returns) == len(self.benchmark_prices) - 1
    
    def test_backtest_strategy(self):
        """Test strategy backtesting."""
        # Create equal weight strategy
        n_stocks = self.prices.shape[1]
        weights = pd.DataFrame(1/n_stocks, 
                             index=self.prices.index, 
                             columns=self.prices.columns)
        
        performance = self.backtester.backtest_strategy('equal_weight', weights)
        
        assert isinstance(performance, dict)
        assert 'Annualized Return' in performance
        assert 'Sharpe Ratio' in performance
        assert 'Max Drawdown' in performance
        assert 'equal_weight' in self.backtester.results
    
    def test_regime_detection(self):
        """Test market regime detection."""
        returns = self.backtester.returns.iloc[:, 0]  # Use first stock
        regimes = self.backtester.regime_detection(returns)
        
        assert isinstance(regimes, pd.Series)
        assert len(regimes) <= len(returns)
        assert regimes.min() >= 0
        assert regimes.max() < 2  # 2 regimes by default
    
    def test_performance_attribution(self):
        """Test performance attribution."""
        # Create mock factor returns
        factor_returns = {
            'market': np.random.normal(0.0005, 0.01, len(self.backtester.returns)),
            'size': np.random.normal(0.0002, 0.005, len(self.backtester.returns))
        }
        
        factor_returns = {k: pd.Series(v, index=self.backtester.returns.index) 
                         for k, v in factor_returns.items()}
        
        # First run a strategy
        n_stocks = self.prices.shape[1]
        weights = pd.DataFrame(1/n_stocks, 
                             index=self.prices.index, 
                             columns=self.prices.columns)
        
        self.backtester.backtest_strategy('test_strategy', weights)
        
        # Test attribution
        attribution = self.backtester.performance_attribution('test_strategy', factor_returns)
        
        assert isinstance(attribution, dict)
        assert 'Alpha' in attribution
        assert 'Factor Exposures' in attribution
        assert 'R-squared' in attribution

class TestIntegration:
    """Integration tests for the complete workflow."""
    
    def test_complete_workflow(self):
        """Test the complete analysis workflow."""
        # Create minimal test data
        dates = pd.date_range('2020-01-01', periods=100)
        n_stocks = 3
        
        np.random.seed(42)
        returns_data = np.random.multivariate_normal(
            mean=[0.001] * n_stocks,
            cov=np.eye(n_stocks) * 0.0004,
            size=100
        )
        
        prices_data = (1 + returns_data).cumprod(axis=0) * 100
        prices = pd.DataFrame(prices_data,
                            index=dates,
                            columns=['AAPL', 'MSFT', 'GOOGL'])
        
        # Test factor analysis
        analyzer = FactorAnalyzer(prices)
        factors = analyzer.compute_all_factors()
        
        assert len(factors) >= 4
        
        # Test portfolio optimization
        optimizer = PortfolioOptimizer(prices)
        weights = optimizer.mean_variance_optimization()
        
        assert isinstance(weights, dict)
        assert len(weights) == n_stocks
        
        # Test backtesting
        backtester = Backtester(prices)
        
        # Create simple strategy
        equal_weights = pd.DataFrame(1/n_stocks, index=prices.index, columns=prices.columns)
        performance = backtester.backtest_strategy('equal_weight', equal_weights)
        
        assert isinstance(performance, dict)
        assert 'Sharpe Ratio' in performance

def run_tests():
    """Run all tests."""
    pytest.main([__file__, '-v'])

if __name__ == "__main__":
    run_tests()
