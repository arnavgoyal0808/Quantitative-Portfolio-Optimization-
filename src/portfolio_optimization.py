"""Portfolio optimization module using PyPortfolioOpt."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.expected_returns import mean_historical_return
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import objective_functions
import matplotlib.pyplot as plt
import seaborn as sns
from utils import calculate_returns, calculate_sharpe_ratio, performance_summary

warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    """Class for portfolio optimization and construction."""
    
    def __init__(self, prices: pd.DataFrame, returns: pd.DataFrame = None):
        self.prices = prices
        self.returns = returns if returns is not None else calculate_returns(prices)
        self.optimized_weights = {}
        
    def mean_variance_optimization(self, target_return: Optional[float] = None,
                                 target_volatility: Optional[float] = None,
                                 risk_aversion: float = 1) -> Dict[str, float]:
        """Perform mean-variance optimization."""
        # Calculate expected returns and covariance matrix
        mu = expected_returns.mean_historical_return(self.prices, frequency=252)
        S = risk_models.sample_cov(self.prices, frequency=252)
        
        # Create efficient frontier
        ef = EfficientFrontier(mu, S)
        
        if target_return is not None:
            # Optimize for target return
            weights = ef.efficient_return(target_return)
        elif target_volatility is not None:
            # Optimize for target volatility
            weights = ef.efficient_risk(target_volatility)
        else:
            # Maximize Sharpe ratio
            weights = ef.max_sharpe()
        
        # Clean weights
        cleaned_weights = ef.clean_weights()
        
        self.optimized_weights['mean_variance'] = cleaned_weights
        return cleaned_weights
    
    def risk_parity_optimization(self) -> Dict[str, float]:
        """Perform risk parity optimization."""
        # Calculate covariance matrix
        S = risk_models.sample_cov(self.prices, frequency=252)
        
        # Initialize equal weights
        n_assets = len(self.prices.columns)
        weights = np.ones(n_assets) / n_assets
        
        # Risk parity optimization using iterative approach
        for _ in range(100):  # Max iterations
            # Calculate risk contributions
            portfolio_vol = np.sqrt(weights.T @ S @ weights)
            marginal_contrib = S @ weights / portfolio_vol
            contrib = weights * marginal_contrib
            
            # Target risk contribution (equal for all assets)
            target_contrib = portfolio_vol / n_assets
            
            # Update weights
            weights = weights * target_contrib / contrib
            weights = weights / weights.sum()  # Normalize
            
            # Check convergence
            if np.max(np.abs(contrib - target_contrib)) < 1e-6:
                break
        
        # Convert to dictionary
        risk_parity_weights = dict(zip(self.prices.columns, weights))
        
        self.optimized_weights['risk_parity'] = risk_parity_weights
        return risk_parity_weights
    
    def minimum_variance_optimization(self) -> Dict[str, float]:
        """Perform minimum variance optimization."""
        # Calculate covariance matrix
        S = risk_models.sample_cov(self.prices, frequency=252)
        
        # Create efficient frontier and minimize risk
        ef = EfficientFrontier(None, S)
        weights = ef.min_volatility()
        
        cleaned_weights = ef.clean_weights()
        
        self.optimized_weights['min_variance'] = cleaned_weights
        return cleaned_weights
    
    def factor_based_optimization(self, factor_exposures: pd.DataFrame,
                                factor_returns: pd.DataFrame) -> Dict[str, float]:
        """Optimize portfolio based on factor exposures."""
        # Calculate factor-based expected returns
        latest_exposures = factor_exposures.iloc[-1]
        factor_premia = factor_returns.mean() * 252  # Annualized
        
        expected_returns_factor = latest_exposures @ factor_premia
        
        # Use sample covariance for risk model
        S = risk_models.sample_cov(self.prices, frequency=252)
        
        # Optimize
        ef = EfficientFrontier(expected_returns_factor, S)
        weights = ef.max_sharpe()
        
        cleaned_weights = ef.clean_weights()
        
        self.optimized_weights['factor_based'] = cleaned_weights
        return cleaned_weights
    
    def black_litterman_optimization(self, views: Dict[str, float],
                                   confidences: Dict[str, float]) -> Dict[str, float]:
        """Perform Black-Litterman optimization with investor views."""
        from pypfopt import black_litterman
        
        # Market cap weights (equal weight as proxy)
        n_assets = len(self.prices.columns)
        market_caps = pd.Series(1/n_assets, index=self.prices.columns)
        
        # Calculate covariance matrix
        S = risk_models.sample_cov(self.prices, frequency=252)
        
        # Create views matrix
        viewdict = views
        
        # Black-Litterman expected returns
        bl = black_litterman.BlackLittermanModel(S, pi=market_caps, 
                                               absolute_views=viewdict)
        
        # Get posterior expected returns and covariance
        ret_bl = bl.bl_returns()
        S_bl = bl.bl_cov()
        
        # Optimize
        ef = EfficientFrontier(ret_bl, S_bl)
        weights = ef.max_sharpe()
        
        cleaned_weights = ef.clean_weights()
        
        self.optimized_weights['black_litterman'] = cleaned_weights
        return cleaned_weights
    
    def hierarchical_risk_parity(self) -> Dict[str, float]:
        """Perform Hierarchical Risk Parity optimization."""
        from pypfopt import HRPOpt
        
        # Calculate returns for HRP
        returns_data = self.returns.dropna()
        
        # Create HRP optimizer
        hrp = HRPOpt(returns_data)
        weights = hrp.optimize()
        
        self.optimized_weights['hrp'] = weights
        return weights
    
    def calculate_portfolio_performance(self, weights: Dict[str, float],
                                      benchmark_returns: pd.Series = None) -> Dict:
        """Calculate portfolio performance metrics."""
        # Convert weights to series
        weight_series = pd.Series(weights)
        
        # Align with returns
        aligned_weights = weight_series.reindex(self.returns.columns).fillna(0)
        
        # Calculate portfolio returns
        portfolio_returns = (self.returns * aligned_weights).sum(axis=1)
        
        # Calculate performance metrics
        performance = performance_summary(portfolio_returns, benchmark_returns)
        
        return performance
    
    def efficient_frontier_plot(self, n_portfolios: int = 100) -> plt.Figure:
        """Plot efficient frontier."""
        # Calculate expected returns and covariance
        mu = expected_returns.mean_historical_return(self.prices, frequency=252)
        S = risk_models.sample_cov(self.prices, frequency=252)
        
        # Generate efficient frontier
        returns_range = np.linspace(mu.min(), mu.max(), n_portfolios)
        volatilities = []
        
        for target_return in returns_range:
            try:
                ef = EfficientFrontier(mu, S)
                ef.efficient_return(target_return)
                vol = ef.portfolio_performance()[1]
                volatilities.append(vol)
            except:
                volatilities.append(np.nan)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(volatilities, returns_range, 'b-', linewidth=2, label='Efficient Frontier')
        
        # Plot individual assets
        individual_returns = mu.values
        individual_vols = np.sqrt(np.diag(S))
        ax.scatter(individual_vols, individual_returns, alpha=0.6, s=50, label='Individual Assets')
        
        # Plot optimized portfolios
        for name, weights in self.optimized_weights.items():
            weight_series = pd.Series(weights).reindex(mu.index).fillna(0)
            port_return = (weight_series * mu).sum()
            port_vol = np.sqrt(weight_series.T @ S @ weight_series)
            ax.scatter(port_vol, port_return, s=100, label=f'{name.replace("_", " ").title()}')
        
        ax.set_xlabel('Volatility')
        ax.set_ylabel('Expected Return')
        ax.set_title('Efficient Frontier')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def weights_comparison_plot(self) -> plt.Figure:
        """Plot comparison of different optimization methods."""
        if not self.optimized_weights:
            raise ValueError("No optimized weights to compare")
        
        # Create DataFrame of weights
        weights_df = pd.DataFrame(self.optimized_weights).fillna(0)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        weights_df.plot(kind='bar', ax=ax, width=0.8)
        
        ax.set_title('Portfolio Weights Comparison')
        ax.set_xlabel('Assets')
        ax.set_ylabel('Weight')
        ax.legend(title='Optimization Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def discrete_allocation(self, weights: Dict[str, float], 
                          total_value: float = 10000) -> Dict[str, int]:
        """Convert continuous weights to discrete share allocation."""
        latest_prices = get_latest_prices(self.prices)
        
        da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=total_value)
        allocation, leftover = da.lp_portfolio()
        
        return allocation, leftover

def main():
    """Main function to demonstrate portfolio optimization."""
    from data_loader import DataLoader
    from factor_analysis import FactorAnalyzer
    from utils import get_sp500_tickers
    
    # Load data
    loader = DataLoader()
    
    try:
        prices = loader.load_data('stock_prices.csv')
    except FileNotFoundError:
        print("Data files not found. Running data loader first...")
        tickers = get_sp500_tickers()
        stock_data = loader.download_stock_data(tickers)
        prices = loader.create_price_matrix(stock_data)
        loader.save_data(prices, 'stock_prices.csv')
    
    # Use subset for optimization (top 20 by market cap proxy)
    prices_subset = prices.iloc[:, :20]  # First 20 stocks
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer(prices_subset)
    
    print("Running portfolio optimizations...")
    
    # Mean-variance optimization
    mv_weights = optimizer.mean_variance_optimization()
    print(f"Mean-Variance weights: {len([w for w in mv_weights.values() if w > 0.01])} active positions")
    
    # Risk parity optimization
    rp_weights = optimizer.risk_parity_optimization()
    print(f"Risk Parity weights: {len([w for w in rp_weights.values() if w > 0.01])} active positions")
    
    # Minimum variance optimization
    mv_min_weights = optimizer.minimum_variance_optimization()
    print(f"Min Variance weights: {len([w for w in mv_min_weights.values() if w > 0.01])} active positions")
    
    # HRP optimization
    hrp_weights = optimizer.hierarchical_risk_parity()
    print(f"HRP weights: {len([w for w in hrp_weights.values() if w > 0.01])} active positions")
    
    # Calculate performance for each strategy
    print("\nPerformance Comparison:")
    for name, weights in optimizer.optimized_weights.items():
        performance = optimizer.calculate_portfolio_performance(weights)
        print(f"\n{name.replace('_', ' ').title()}:")
        print(f"  Annual Return: {performance['Annualized Return']:.2%}")
        print(f"  Volatility: {performance['Volatility']:.2%}")
        print(f"  Sharpe Ratio: {performance['Sharpe Ratio']:.3f}")
        print(f"  Max Drawdown: {performance['Max Drawdown']:.2%}")
    
    # Create plots
    print("\nGenerating plots...")
    
    # Efficient frontier plot
    ef_fig = optimizer.efficient_frontier_plot()
    ef_fig.savefig('results/efficient_frontier.png', dpi=300, bbox_inches='tight')
    
    # Weights comparison plot
    weights_fig = optimizer.weights_comparison_plot()
    weights_fig.savefig('results/weights_comparison.png', dpi=300, bbox_inches='tight')
    
    print("Portfolio optimization complete!")

if __name__ == "__main__":
    main()
