"""Factor analysis module for computing investment factors."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
from utils import calculate_returns, winsorize, standardize, create_deciles

warnings.filterwarnings('ignore')

class FactorAnalyzer:
    """Class for computing and analyzing investment factors."""
    
    def __init__(self, prices: pd.DataFrame, volumes: pd.DataFrame = None, 
                 fundamentals: Dict = None):
        self.prices = prices
        self.volumes = volumes
        self.fundamentals = fundamentals
        self.returns = calculate_returns(prices)
        self.factors = {}
        
    def compute_momentum_factor(self, lookback_periods: List[int] = [21, 63, 252]) -> pd.DataFrame:
        """Compute momentum factor (12-1 month momentum)."""
        momentum_scores = pd.DataFrame(index=self.prices.index, columns=self.prices.columns)
        
        for period in lookback_periods:
            if period <= len(self.prices):
                # Calculate momentum as cumulative return over period
                momentum = self.prices.pct_change(period).shift(21)  # Skip last month
                momentum_scores = momentum_scores.fillna(0) + momentum.fillna(0)
        
        # Average across periods
        momentum_scores = momentum_scores / len(lookback_periods)
        
        # Standardize cross-sectionally
        for date in momentum_scores.index:
            momentum_scores.loc[date] = standardize(momentum_scores.loc[date].dropna())
        
        self.factors['momentum'] = momentum_scores
        return momentum_scores
    
    def compute_size_factor(self) -> pd.DataFrame:
        """Compute size factor based on market capitalization."""
        if self.fundamentals is None:
            # Use price as proxy for size (inverse relationship)
            size_scores = -np.log(self.prices)
        else:
            # Use actual market cap data
            market_caps = pd.DataFrame(index=self.prices.index, columns=self.prices.columns)
            
            for ticker in self.prices.columns:
                if ticker in self.fundamentals:
                    market_cap = self.fundamentals[ticker].get('market_cap', np.nan)
                    if not pd.isna(market_cap):
                        market_caps[ticker] = np.log(market_cap)
            
            size_scores = -market_caps  # Negative because small cap premium
        
        # Standardize cross-sectionally
        for date in size_scores.index:
            if not size_scores.loc[date].isna().all():
                size_scores.loc[date] = standardize(size_scores.loc[date].dropna())
        
        self.factors['size'] = size_scores
        return size_scores
    
    def compute_value_factor(self) -> pd.DataFrame:
        """Compute value factor using P/E and P/B ratios."""
        value_scores = pd.DataFrame(index=self.prices.index, columns=self.prices.columns)
        
        if self.fundamentals is not None:
            for ticker in self.prices.columns:
                if ticker in self.fundamentals:
                    pe_ratio = self.fundamentals[ticker].get('pe_ratio', np.nan)
                    pb_ratio = self.fundamentals[ticker].get('pb_ratio', np.nan)
                    
                    # Value score (inverse of ratios)
                    value_score = 0
                    count = 0
                    
                    if not pd.isna(pe_ratio) and pe_ratio > 0:
                        value_score += 1 / pe_ratio
                        count += 1
                    
                    if not pd.isna(pb_ratio) and pb_ratio > 0:
                        value_score += 1 / pb_ratio
                        count += 1
                    
                    if count > 0:
                        value_scores[ticker] = value_score / count
        else:
            # Use price-based value proxy (inverse price momentum)
            value_scores = -self.prices.pct_change(252)
        
        # Standardize cross-sectionally
        for date in value_scores.index:
            if not value_scores.loc[date].isna().all():
                value_scores.loc[date] = standardize(value_scores.loc[date].dropna())
        
        self.factors['value'] = value_scores
        return value_scores
    
    def compute_volatility_factor(self, window: int = 252) -> pd.DataFrame:
        """Compute volatility factor (low volatility anomaly)."""
        # Calculate rolling volatility
        volatility = self.returns.rolling(window).std() * np.sqrt(252)
        
        # Low volatility factor (negative volatility)
        volatility_scores = -volatility
        
        # Standardize cross-sectionally
        for date in volatility_scores.index:
            if not volatility_scores.loc[date].isna().all():
                volatility_scores.loc[date] = standardize(volatility_scores.loc[date].dropna())
        
        self.factors['volatility'] = volatility_scores
        return volatility_scores
    
    def compute_quality_factor(self) -> pd.DataFrame:
        """Compute quality factor using ROE, profit margins, and debt ratios."""
        quality_scores = pd.DataFrame(index=self.prices.index, columns=self.prices.columns)
        
        if self.fundamentals is not None:
            for ticker in self.prices.columns:
                if ticker in self.fundamentals:
                    roe = self.fundamentals[ticker].get('roe', np.nan)
                    profit_margin = self.fundamentals[ticker].get('profit_margin', np.nan)
                    debt_to_equity = self.fundamentals[ticker].get('debt_to_equity', np.nan)
                    
                    quality_score = 0
                    count = 0
                    
                    if not pd.isna(roe):
                        quality_score += roe
                        count += 1
                    
                    if not pd.isna(profit_margin):
                        quality_score += profit_margin
                        count += 1
                    
                    if not pd.isna(debt_to_equity):
                        quality_score -= debt_to_equity / 100  # Penalize high debt
                        count += 1
                    
                    if count > 0:
                        quality_scores[ticker] = quality_score / count
        else:
            # Use return stability as quality proxy
            quality_scores = -self.returns.rolling(252).std()
        
        # Standardize cross-sectionally
        for date in quality_scores.index:
            if not quality_scores.loc[date].isna().all():
                quality_scores.loc[date] = standardize(quality_scores.loc[date].dropna())
        
        self.factors['quality'] = quality_scores
        return quality_scores
    
    def compute_all_factors(self) -> Dict[str, pd.DataFrame]:
        """Compute all factors."""
        print("Computing momentum factor...")
        self.compute_momentum_factor()
        
        print("Computing size factor...")
        self.compute_size_factor()
        
        print("Computing value factor...")
        self.compute_value_factor()
        
        print("Computing volatility factor...")
        self.compute_volatility_factor()
        
        print("Computing quality factor...")
        self.compute_quality_factor()
        
        return self.factors
    
    def create_factor_portfolios(self, factor_name: str, n_portfolios: int = 10) -> Dict[int, List[str]]:
        """Create portfolios based on factor scores."""
        if factor_name not in self.factors:
            raise ValueError(f"Factor {factor_name} not computed yet")
        
        factor_scores = self.factors[factor_name]
        portfolios = {}
        
        # Get latest factor scores
        latest_scores = factor_scores.iloc[-1].dropna()
        
        # Create decile portfolios
        deciles = create_deciles(latest_scores)
        
        for i in range(1, n_portfolios + 1):
            portfolios[i] = latest_scores[deciles == i].index.tolist()
        
        return portfolios
    
    def create_long_short_portfolio(self, factor_name: str) -> Tuple[List[str], List[str]]:
        """Create long-short portfolio based on factor."""
        portfolios = self.create_factor_portfolios(factor_name)
        
        # Long top decile, short bottom decile
        long_portfolio = portfolios[10]
        short_portfolio = portfolios[1]
        
        return long_portfolio, short_portfolio
    
    def combine_factors_pca(self, n_components: int = 3) -> pd.DataFrame:
        """Combine factors using PCA."""
        # Align all factors
        factor_data = pd.DataFrame()
        
        for name, factor in self.factors.items():
            factor_data[name] = factor.mean(axis=1)  # Average across stocks
        
        factor_data = factor_data.dropna()
        
        # Apply PCA
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(factor_data)
        
        pca = PCA(n_components=n_components)
        pca_factors = pca.fit_transform(scaled_data)
        
        pca_df = pd.DataFrame(pca_factors, index=factor_data.index,
                             columns=[f'PC{i+1}' for i in range(n_components)])
        
        # Store PCA results
        self.pca_factors = pca_df
        self.pca_loadings = pd.DataFrame(pca.components_.T, 
                                        index=factor_data.columns,
                                        columns=pca_df.columns)
        
        return pca_df
    
    def get_factor_returns(self, factor_name: str) -> pd.Series:
        """Calculate factor returns using long-short portfolio."""
        long_stocks, short_stocks = self.create_long_short_portfolio(factor_name)
        
        # Calculate portfolio returns
        long_returns = self.returns[long_stocks].mean(axis=1)
        short_returns = self.returns[short_stocks].mean(axis=1)
        
        factor_returns = long_returns - short_returns
        return factor_returns.dropna()

def main():
    """Main function to demonstrate factor analysis."""
    from data_loader import DataLoader
    from utils import get_sp500_tickers
    
    # Load data
    loader = DataLoader()
    
    try:
        prices = loader.load_data('stock_prices.csv')
        volumes = loader.load_data('stock_volumes.csv')
    except FileNotFoundError:
        print("Data files not found. Running data loader first...")
        tickers = get_sp500_tickers()
        stock_data = loader.download_stock_data(tickers)
        prices = loader.create_price_matrix(stock_data)
        volumes = loader.create_volume_matrix(stock_data)
        loader.save_data(prices, 'stock_prices.csv')
        loader.save_data(volumes, 'stock_volumes.csv')
    
    # Initialize factor analyzer
    analyzer = FactorAnalyzer(prices, volumes)
    
    # Compute all factors
    factors = analyzer.compute_all_factors()
    
    # Create factor portfolios
    for factor_name in factors.keys():
        long_stocks, short_stocks = analyzer.create_long_short_portfolio(factor_name)
        print(f"\n{factor_name.upper()} Factor:")
        print(f"Long portfolio: {len(long_stocks)} stocks")
        print(f"Short portfolio: {len(short_stocks)} stocks")
        
        # Calculate factor returns
        factor_returns = analyzer.get_factor_returns(factor_name)
        print(f"Factor return (annualized): {factor_returns.mean() * 252:.2%}")
        print(f"Factor volatility (annualized): {factor_returns.std() * np.sqrt(252):.2%}")
    
    # Combine factors using PCA
    pca_factors = analyzer.combine_factors_pca()
    print(f"\nPCA Factors shape: {pca_factors.shape}")
    print("PCA Loadings:")
    print(analyzer.pca_loadings)

if __name__ == "__main__":
    main()
