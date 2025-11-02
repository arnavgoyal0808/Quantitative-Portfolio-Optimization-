#!/usr/bin/env python3
"""
Main execution script for the quantitative research project.
Orchestrates the complete workflow from data loading to backtesting.
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import DataLoader
from factor_analysis import FactorAnalyzer
from portfolio_optimization import PortfolioOptimizer
from backtesting import Backtester
from utils import get_sp500_tickers, performance_summary

warnings.filterwarnings('ignore')

def setup_directories():
    """Create necessary directories."""
    directories = ['data', 'results', 'results/plots']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def main():
    """Main execution function."""
    print("=" * 60)
    print("QUANTITATIVE RESEARCH PROJECT")
    print("Multi-Factor Investment Strategy Analysis")
    print("=" * 60)
    
    # Setup
    setup_directories()
    
    # Configuration
    START_DATE = '2015-01-01'
    END_DATE = '2023-12-31'
    
    print(f"\nAnalysis Period: {START_DATE} to {END_DATE}")
    
    # Step 1: Data Loading
    print("\n" + "="*50)
    print("STEP 1: DATA LOADING")
    print("="*50)
    
    loader = DataLoader(start_date=START_DATE, end_date=END_DATE)
    tickers = get_sp500_tickers()
    
    print(f"Loading data for {len(tickers)} stocks...")
    
    # Check if data already exists
    try:
        prices = loader.load_data('stock_prices.csv')
        volumes = loader.load_data('stock_volumes.csv')
        market_data = loader.load_data('market_data.csv')
        print("Loaded existing data files.")
    except FileNotFoundError:
        print("Downloading fresh data...")
        
        # Download stock data
        stock_data = loader.download_stock_data(tickers)
        
        # Create matrices
        prices = loader.create_price_matrix(stock_data)
        volumes = loader.create_volume_matrix(stock_data)
        
        # Get market data
        market_data = loader.get_market_data()
        
        # Save data
        loader.save_data(prices, 'stock_prices.csv')
        loader.save_data(volumes, 'stock_volumes.csv')
        loader.save_data(market_data, 'market_data.csv')
    
    print(f"Final dataset: {prices.shape[0]} days, {prices.shape[1]} stocks")
    print(f"Date range: {prices.index.min().date()} to {prices.index.max().date()}")
    
    # Step 2: Factor Analysis
    print("\n" + "="*50)
    print("STEP 2: FACTOR ANALYSIS")
    print("="*50)
    
    # Get fundamental data for enhanced factors
    print("Downloading fundamental data...")
    fundamentals = loader.get_fundamental_data(prices.columns.tolist()[:50])  # Limit for speed
    
    # Initialize factor analyzer
    analyzer = FactorAnalyzer(prices, volumes, fundamentals)
    
    # Compute all factors
    print("Computing investment factors...")
    factors = analyzer.compute_all_factors()
    
    # Analyze factor performance
    print("\nFactor Performance Analysis:")
    factor_returns = {}
    
    for factor_name in factors.keys():
        try:
            factor_ret = analyzer.get_factor_returns(factor_name)
            factor_returns[factor_name] = factor_ret
            
            annual_return = factor_ret.mean() * 252
            volatility = factor_ret.std() * np.sqrt(252)
            sharpe = annual_return / volatility if volatility > 0 else 0
            
            print(f"\n{factor_name.upper()} Factor:")
            print(f"  Annual Return: {annual_return:.2%}")
            print(f"  Volatility: {volatility:.2%}")
            print(f"  Sharpe Ratio: {sharpe:.3f}")
            
        except Exception as e:
            print(f"Error analyzing {factor_name} factor: {e}")
    
    # Factor combination using PCA
    print("\nCombining factors using PCA...")
    pca_factors = analyzer.combine_factors_pca(n_components=3)
    
    print("PCA Factor Loadings:")
    print(analyzer.pca_loadings.round(3))
    
    # Step 3: Portfolio Optimization
    print("\n" + "="*50)
    print("STEP 3: PORTFOLIO OPTIMIZATION")
    print("="*50)
    
    # Use subset for optimization (computational efficiency)
    prices_subset = prices.iloc[:, :30]  # Top 30 stocks
    print(f"Optimizing portfolios with {prices_subset.shape[1]} stocks...")
    
    optimizer = PortfolioOptimizer(prices_subset)
    
    # Run different optimization methods
    optimization_methods = [
        ('mean_variance', 'Mean-Variance Optimization'),
        ('risk_parity', 'Risk Parity'),
        ('min_variance', 'Minimum Variance'),
        ('hrp', 'Hierarchical Risk Parity')
    ]
    
    for method_key, method_name in optimization_methods:
        try:
            print(f"\nRunning {method_name}...")
            
            if method_key == 'mean_variance':
                weights = optimizer.mean_variance_optimization()
            elif method_key == 'risk_parity':
                weights = optimizer.risk_parity_optimization()
            elif method_key == 'min_variance':
                weights = optimizer.minimum_variance_optimization()
            elif method_key == 'hrp':
                weights = optimizer.hierarchical_risk_parity()
            
            # Calculate performance
            performance = optimizer.calculate_portfolio_performance(weights)
            
            print(f"  Active positions: {len([w for w in weights.values() if abs(w) > 0.01])}")
            print(f"  Expected return: {performance['Annualized Return']:.2%}")
            print(f"  Volatility: {performance['Volatility']:.2%}")
            print(f"  Sharpe ratio: {performance['Sharpe Ratio']:.3f}")
            
        except Exception as e:
            print(f"Error in {method_name}: {e}")
    
    # Generate optimization plots
    print("\nGenerating optimization plots...")
    try:
        ef_fig = optimizer.efficient_frontier_plot()
        ef_fig.savefig('results/plots/efficient_frontier.png', dpi=300, bbox_inches='tight')
        plt.close(ef_fig)
        
        weights_fig = optimizer.weights_comparison_plot()
        weights_fig.savefig('results/plots/weights_comparison.png', dpi=300, bbox_inches='tight')
        plt.close(weights_fig)
        
    except Exception as e:
        print(f"Error generating optimization plots: {e}")
    
    # Step 4: Backtesting
    print("\n" + "="*50)
    print("STEP 4: BACKTESTING")
    print("="*50)
    
    # Initialize backtester
    benchmark_prices = market_data.iloc[:, 0] if not market_data.empty else None
    backtester = Backtester(prices, benchmark_prices)
    
    # Backtest factor strategies
    print("Backtesting factor strategies...")
    
    strategy_results = {}
    
    for factor_name in factors.keys():
        try:
            print(f"\nBacktesting {factor_name} factor strategy...")
            
            # Long-short strategy
            performance_ls = backtester.backtest_factor_strategy(
                factors[factor_name], f"{factor_name}_long_short", long_short=True
            )
            
            # Long-only strategy
            performance_lo = backtester.backtest_factor_strategy(
                factors[factor_name], f"{factor_name}_long_only", long_short=False
            )
            
            strategy_results[factor_name] = {
                'long_short': performance_ls,
                'long_only': performance_lo
            }
            
            print(f"  Long-Short - Return: {performance_ls['Annualized Return']:.2%}, "
                  f"Sharpe: {performance_ls['Sharpe Ratio']:.3f}")
            print(f"  Long-Only  - Return: {performance_lo['Annualized Return']:.2%}, "
                  f"Sharpe: {performance_lo['Sharpe Ratio']:.3f}")
            
        except Exception as e:
            print(f"Error backtesting {factor_name}: {e}")
    
    # Create equal-weight benchmark
    print("\nCreating equal-weight benchmark...")
    n_stocks = len(prices.columns)
    equal_weights = pd.DataFrame(1/n_stocks, index=prices.index, columns=prices.columns)
    benchmark_perf = backtester.backtest_strategy('equal_weight_benchmark', equal_weights)
    
    print(f"Equal-weight benchmark - Return: {benchmark_perf['Annualized Return']:.2%}, "
          f"Sharpe: {benchmark_perf['Sharpe Ratio']:.3f}")
    
    # Performance comparison
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)
    
    # Create summary table
    summary_data = []
    
    for strategy_name, results in backtester.results.items():
        perf = results['performance']
        summary_data.append({
            'Strategy': strategy_name.replace('_', ' ').title(),
            'Annual Return': f"{perf['Annualized Return']:.2%}",
            'Volatility': f"{perf['Volatility']:.2%}",
            'Sharpe Ratio': f"{perf['Sharpe Ratio']:.3f}",
            'Max Drawdown': f"{perf['Max Drawdown']:.2%}",
            'Calmar Ratio': f"{perf.get('Calmar Ratio', 0):.3f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n", summary_df.to_string(index=False))
    
    # Save summary
    summary_df.to_csv('results/performance_summary.csv', index=False)
    
    # Generate plots
    print("\nGenerating backtest plots...")
    
    try:
        # Cumulative returns plot
        cum_returns_fig = backtester.plot_cumulative_returns()
        cum_returns_fig.savefig('results/plots/cumulative_returns.png', dpi=300, bbox_inches='tight')
        plt.close(cum_returns_fig)
        
        # Find best strategy for detailed analysis
        best_strategy = max(backtester.results.keys(), 
                           key=lambda x: backtester.results[x]['performance']['Sharpe Ratio'])
        
        print(f"Best performing strategy: {best_strategy}")
        
        # Drawdown analysis
        drawdown_fig = backtester.plot_drawdowns(best_strategy)
        drawdown_fig.savefig('results/plots/drawdowns.png', dpi=300, bbox_inches='tight')
        plt.close(drawdown_fig)
        
        # Rolling performance analysis
        rolling_perf = backtester.rolling_performance_analysis(best_strategy)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        rolling_perf['Rolling Return'].plot(ax=axes[0,0], title='Rolling Annual Return')
        rolling_perf['Rolling Volatility'].plot(ax=axes[0,1], title='Rolling Volatility')
        rolling_perf['Rolling Sharpe'].plot(ax=axes[1,0], title='Rolling Sharpe Ratio')
        rolling_perf['Rolling Max DD'].plot(ax=axes[1,1], title='Rolling Max Drawdown')
        
        plt.tight_layout()
        plt.savefig('results/plots/rolling_performance.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
    except Exception as e:
        print(f"Error generating plots: {e}")
    
    # Generate comprehensive report
    print("\nGenerating comprehensive report...")
    try:
        backtester.generate_report('results/backtest_report.html')
    except Exception as e:
        print(f"Error generating report: {e}")
    
    # Final summary
    print("\n" + "="*60)
    print("PROJECT COMPLETION SUMMARY")
    print("="*60)
    
    print(f"✓ Data loaded for {prices.shape[1]} stocks over {prices.shape[0]} days")
    print(f"✓ Computed {len(factors)} investment factors")
    print(f"✓ Optimized {len(optimizer.optimized_weights)} portfolio strategies")
    print(f"✓ Backtested {len(backtester.results)} trading strategies")
    print(f"✓ Generated plots and comprehensive report")
    
    print(f"\nResults saved to 'results/' directory")
    print(f"Key files:")
    print(f"  - performance_summary.csv: Strategy performance metrics")
    print(f"  - backtest_report.html: Comprehensive HTML report")
    print(f"  - plots/: All visualization files")
    
    print(f"\nBest performing strategy: {best_strategy}")
    best_perf = backtester.results[best_strategy]['performance']
    print(f"  Annual Return: {best_perf['Annualized Return']:.2%}")
    print(f"  Sharpe Ratio: {best_perf['Sharpe Ratio']:.3f}")
    print(f"  Max Drawdown: {best_perf['Max Drawdown']:.2%}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
