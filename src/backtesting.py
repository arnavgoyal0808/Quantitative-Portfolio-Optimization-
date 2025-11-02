"""Backtesting module for portfolio performance evaluation."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from statsmodels.tsa.regime_switching import MarkovRegression
import warnings
from utils import (calculate_returns, calculate_sharpe_ratio, calculate_max_drawdown,
                  calculate_alpha_beta, performance_summary)

warnings.filterwarnings('ignore')

class Backtester:
    """Class for backtesting portfolio strategies."""
    
    def __init__(self, prices: pd.DataFrame, benchmark_prices: pd.Series = None):
        self.prices = prices
        self.returns = calculate_returns(prices)
        self.benchmark_prices = benchmark_prices
        self.benchmark_returns = calculate_returns(benchmark_prices) if benchmark_prices is not None else None
        self.results = {}
        
    def backtest_strategy(self, strategy_name: str, weights_series: pd.DataFrame,
                         rebalance_freq: str = 'M', transaction_cost: float = 0.001) -> Dict:
        """Backtest a strategy with given weights over time."""
        
        # Align weights with returns
        aligned_weights = weights_series.reindex(self.returns.index, method='ffill')
        aligned_weights = aligned_weights.fillna(0)
        
        # Calculate portfolio returns
        portfolio_returns = (self.returns * aligned_weights.shift(1)).sum(axis=1)
        
        # Apply transaction costs
        if transaction_cost > 0:
            # Calculate turnover
            weight_changes = aligned_weights.diff().abs().sum(axis=1)
            transaction_costs = weight_changes * transaction_cost
            portfolio_returns = portfolio_returns - transaction_costs
        
        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Performance metrics
        performance = performance_summary(portfolio_returns, self.benchmark_returns)
        
        # Additional metrics
        performance['Cumulative Return'] = cumulative_returns.iloc[-1] - 1
        performance['Calmar Ratio'] = performance['Annualized Return'] / abs(performance['Max Drawdown'])
        
        if self.benchmark_returns is not None:
            performance['Tracking Error'] = (portfolio_returns - self.benchmark_returns).std() * np.sqrt(252)
        
        # Store results
        self.results[strategy_name] = {
            'returns': portfolio_returns,
            'cumulative_returns': cumulative_returns,
            'performance': performance,
            'weights': aligned_weights
        }
        
        return performance
    
    def backtest_factor_strategy(self, factor_scores: pd.DataFrame, 
                               factor_name: str, long_short: bool = True) -> Dict:
        """Backtest a factor-based strategy."""
        
        # Create dynamic weights based on factor scores
        weights_series = pd.DataFrame(index=factor_scores.index, columns=factor_scores.columns)
        
        for date in factor_scores.index:
            scores = factor_scores.loc[date].dropna()
            
            if len(scores) > 0:
                if long_short:
                    # Long-short strategy
                    top_decile = scores >= scores.quantile(0.9)
                    bottom_decile = scores <= scores.quantile(0.1)
                    
                    weights = pd.Series(0.0, index=scores.index)
                    weights[top_decile] = 0.5 / top_decile.sum()  # 50% long
                    weights[bottom_decile] = -0.5 / bottom_decile.sum()  # 50% short
                    
                else:
                    # Long-only strategy
                    top_quintile = scores >= scores.quantile(0.8)
                    weights = pd.Series(0.0, index=scores.index)
                    weights[top_quintile] = 1.0 / top_quintile.sum()
                
                weights_series.loc[date] = weights
        
        # Backtest the strategy
        return self.backtest_strategy(f"{factor_name}_factor", weights_series)
    
    def rolling_performance_analysis(self, strategy_name: str, 
                                   window: int = 252) -> pd.DataFrame:
        """Analyze rolling performance metrics."""
        
        if strategy_name not in self.results:
            raise ValueError(f"Strategy {strategy_name} not found in results")
        
        returns = self.results[strategy_name]['returns']
        
        # Calculate rolling metrics
        rolling_metrics = pd.DataFrame(index=returns.index)
        rolling_metrics['Rolling Return'] = returns.rolling(window).mean() * 252
        rolling_metrics['Rolling Volatility'] = returns.rolling(window).std() * np.sqrt(252)
        rolling_metrics['Rolling Sharpe'] = rolling_metrics['Rolling Return'] / rolling_metrics['Rolling Volatility']
        
        # Rolling max drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.rolling(window).max()
        rolling_dd = (cumulative - rolling_max) / rolling_max
        rolling_metrics['Rolling Max DD'] = rolling_dd.rolling(window).min()
        
        return rolling_metrics
    
    def regime_detection(self, returns: pd.Series, n_regimes: int = 2) -> pd.Series:
        """Detect market regimes using Gaussian Mixture Model."""
        
        # Prepare data
        returns_clean = returns.dropna()
        X = returns_clean.values.reshape(-1, 1)
        
        # Fit Gaussian Mixture Model
        gmm = GaussianMixture(n_components=n_regimes, random_state=42)
        regimes = gmm.fit_predict(X)
        
        # Create regime series
        regime_series = pd.Series(regimes, index=returns_clean.index, name='Regime')
        
        return regime_series
    
    def regime_conditional_performance(self, strategy_name: str) -> Dict:
        """Analyze performance conditional on market regimes."""
        
        if strategy_name not in self.results:
            raise ValueError(f"Strategy {strategy_name} not found in results")
        
        returns = self.results[strategy_name]['returns']
        
        # Detect regimes using benchmark or strategy returns
        if self.benchmark_returns is not None:
            regimes = self.regime_detection(self.benchmark_returns)
        else:
            regimes = self.regime_detection(returns)
        
        # Align returns with regimes
        aligned_returns = returns.reindex(regimes.index)
        
        # Calculate performance by regime
        regime_performance = {}
        for regime in regimes.unique():
            regime_mask = regimes == regime
            regime_returns = aligned_returns[regime_mask]
            
            if len(regime_returns) > 0:
                regime_performance[f'Regime_{regime}'] = performance_summary(regime_returns)
        
        return regime_performance
    
    def drawdown_analysis(self, strategy_name: str) -> pd.DataFrame:
        """Detailed drawdown analysis."""
        
        if strategy_name not in self.results:
            raise ValueError(f"Strategy {strategy_name} not found in results")
        
        cumulative_returns = self.results[strategy_name]['cumulative_returns']
        
        # Calculate drawdowns
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        
        # Find drawdown periods
        drawdown_periods = []
        in_drawdown = False
        start_date = None
        
        for date, dd in drawdowns.items():
            if dd < -0.01 and not in_drawdown:  # Start of drawdown (>1%)
                in_drawdown = True
                start_date = date
                peak_value = running_max[date]
                
            elif dd >= -0.001 and in_drawdown:  # End of drawdown
                in_drawdown = False
                end_date = date
                trough_value = cumulative_returns[start_date:end_date].min()
                recovery_date = date
                
                drawdown_periods.append({
                    'Start': start_date,
                    'End': end_date,
                    'Recovery': recovery_date,
                    'Duration': (end_date - start_date).days,
                    'Max Drawdown': (trough_value - peak_value) / peak_value,
                    'Recovery Time': (recovery_date - start_date).days
                })
        
        return pd.DataFrame(drawdown_periods)
    
    def performance_attribution(self, strategy_name: str, 
                              factor_returns: Dict[str, pd.Series]) -> Dict:
        """Perform performance attribution analysis."""
        
        if strategy_name not in self.results:
            raise ValueError(f"Strategy {strategy_name} not found in results")
        
        returns = self.results[strategy_name]['returns']
        
        # Prepare factor return matrix
        factor_df = pd.DataFrame(factor_returns)
        factor_df = factor_df.reindex(returns.index).fillna(0)
        
        # Run regression
        from sklearn.linear_model import LinearRegression
        
        X = factor_df.values
        y = returns.values
        
        model = LinearRegression().fit(X, y)
        
        # Attribution results
        attribution = {
            'Alpha': model.intercept_ * 252,  # Annualized
            'Factor Exposures': dict(zip(factor_df.columns, model.coef_)),
            'R-squared': model.score(X, y),
            'Factor Contributions': {}
        }
        
        # Calculate factor contributions
        for i, factor_name in enumerate(factor_df.columns):
            factor_contribution = model.coef_[i] * factor_df[factor_name].mean() * 252
            attribution['Factor Contributions'][factor_name] = factor_contribution
        
        return attribution
    
    def plot_cumulative_returns(self, strategies: List[str] = None) -> plt.Figure:
        """Plot cumulative returns for strategies."""
        
        if strategies is None:
            strategies = list(self.results.keys())
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for strategy in strategies:
            if strategy in self.results:
                cumulative_returns = self.results[strategy]['cumulative_returns']
                ax.plot(cumulative_returns.index, cumulative_returns.values, 
                       label=strategy.replace('_', ' ').title(), linewidth=2)
        
        # Add benchmark if available
        if self.benchmark_returns is not None:
            benchmark_cumulative = (1 + self.benchmark_returns).cumprod()
            ax.plot(benchmark_cumulative.index, benchmark_cumulative.values,
                   label='Benchmark', linewidth=2, linestyle='--', color='black')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.set_title('Strategy Performance Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_drawdowns(self, strategy_name: str) -> plt.Figure:
        """Plot drawdown chart for a strategy."""
        
        if strategy_name not in self.results:
            raise ValueError(f"Strategy {strategy_name} not found in results")
        
        cumulative_returns = self.results[strategy_name]['cumulative_returns']
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Cumulative returns
        ax1.plot(cumulative_returns.index, cumulative_returns.values, linewidth=2)
        ax1.fill_between(cumulative_returns.index, cumulative_returns.values, 
                        running_max.values, alpha=0.3, color='red')
        ax1.set_ylabel('Cumulative Return')
        ax1.set_title(f'{strategy_name.replace("_", " ").title()} - Cumulative Returns')
        ax1.grid(True, alpha=0.3)
        
        # Drawdowns
        ax2.fill_between(drawdowns.index, drawdowns.values, 0, alpha=0.7, color='red')
        ax2.set_ylabel('Drawdown')
        ax2.set_xlabel('Date')
        ax2.set_title('Drawdowns')
        ax2.grid(True, alpha=0.3)
        
        return fig
    
    def generate_report(self, output_file: str = 'results/backtest_report.html'):
        """Generate comprehensive backtest report."""
        
        html_content = """
        <html>
        <head>
            <title>Quantitative Strategy Backtest Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: right; }
                th { background-color: #f2f2f2; }
                h1, h2 { color: #333; }
                .metric { margin: 10px 0; }
            </style>
        </head>
        <body>
            <h1>Quantitative Strategy Backtest Report</h1>
        """
        
        # Performance summary table
        html_content += "<h2>Performance Summary</h2><table><tr><th>Strategy</th>"
        
        # Get all metrics from first strategy
        if self.results:
            first_strategy = list(self.results.keys())[0]
            metrics = list(self.results[first_strategy]['performance'].keys())
            
            for metric in metrics:
                html_content += f"<th>{metric}</th>"
            html_content += "</tr>"
            
            # Add data for each strategy
            for strategy_name, results in self.results.items():
                html_content += f"<tr><td>{strategy_name.replace('_', ' ').title()}</td>"
                for metric in metrics:
                    value = results['performance'].get(metric, 'N/A')
                    if isinstance(value, float):
                        if 'Ratio' in metric or 'Return' in metric:
                            html_content += f"<td>{value:.3f}</td>"
                        else:
                            html_content += f"<td>{value:.4f}</td>"
                    else:
                        html_content += f"<td>{value}</td>"
                html_content += "</tr>"
        
        html_content += "</table></body></html>"
        
        # Save report
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"Report saved to {output_file}")

def main():
    """Main function to demonstrate backtesting."""
    import os
    from data_loader import DataLoader
    from factor_analysis import FactorAnalyzer
    from utils import get_sp500_tickers
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Load data
    loader = DataLoader()
    
    try:
        prices = loader.load_data('stock_prices.csv')
        market_data = loader.load_data('market_data.csv')
    except FileNotFoundError:
        print("Data files not found. Running data loader first...")
        tickers = get_sp500_tickers()
        stock_data = loader.download_stock_data(tickers)
        prices = loader.create_price_matrix(stock_data)
        market_data = loader.get_market_data()
        loader.save_data(prices, 'stock_prices.csv')
        loader.save_data(market_data, 'market_data.csv')
    
    # Initialize backtester
    benchmark_prices = market_data['SPY'] if 'SPY' in market_data.columns else market_data.iloc[:, 0]
    backtester = Backtester(prices, benchmark_prices)
    
    # Initialize factor analyzer
    analyzer = FactorAnalyzer(prices)
    factors = analyzer.compute_all_factors()
    
    print("Running factor strategy backtests...")
    
    # Backtest each factor strategy
    for factor_name in factors.keys():
        print(f"Backtesting {factor_name} factor...")
        performance = backtester.backtest_factor_strategy(factors[factor_name], factor_name)
        
        print(f"{factor_name.upper()} Factor Performance:")
        print(f"  Annual Return: {performance['Annualized Return']:.2%}")
        print(f"  Sharpe Ratio: {performance['Sharpe Ratio']:.3f}")
        print(f"  Max Drawdown: {performance['Max Drawdown']:.2%}")
        
        if 'Alpha' in performance:
            print(f"  Alpha: {performance['Alpha']:.2%}")
            print(f"  Beta: {performance['Beta']:.3f}")
    
    # Create equal-weight benchmark
    n_stocks = len(prices.columns)
    equal_weights = pd.DataFrame(1/n_stocks, index=prices.index, columns=prices.columns)
    backtester.backtest_strategy('equal_weight', equal_weights)
    
    print("\nGenerating plots...")
    
    # Plot cumulative returns
    cum_returns_fig = backtester.plot_cumulative_returns()
    cum_returns_fig.savefig('results/cumulative_returns.png', dpi=300, bbox_inches='tight')
    
    # Plot drawdowns for best performing strategy
    best_strategy = max(backtester.results.keys(), 
                       key=lambda x: backtester.results[x]['performance']['Sharpe Ratio'])
    
    drawdown_fig = backtester.plot_drawdowns(best_strategy)
    drawdown_fig.savefig('results/drawdowns.png', dpi=300, bbox_inches='tight')
    
    # Generate comprehensive report
    backtester.generate_report()
    
    print("Backtesting complete!")

if __name__ == "__main__":
    main()
