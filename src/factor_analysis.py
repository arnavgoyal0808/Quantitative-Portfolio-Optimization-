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
            size_scores = -np.log(self.prices.replace({0: np.nan}))
        else:
            # Use actual market cap data; may be sparse in demo mode
            market_caps = pd.DataFrame(index=self.prices.index, columns=self.prices.columns, dtype=float)
            available = 0
            for ticker in self.prices.columns:
                if ticker in self.fundamentals:
                    market_cap = self.fundamentals[ticker].get('market_cap', np.nan)
                    if not pd.isna(market_cap) and market_cap > 0:
                        market_caps[ticker] = np.log(market_cap)
                        available += 1
            coverage_ratio = available / max(1, len(self.prices.columns))
            size_scores = -market_caps  # Negative because small cap premium
            # Fallback to price-based proxy where fundamentals are missing or sparse
            fallback_size = -np.log(self.prices.replace({0: np.nan}))
            if coverage_ratio < 0.5:
                size_scores = size_scores.combine_first(fallback_size)

        # Standardize cross-sectionally
        for date in size_scores.index:
            row = size_scores.loc[date]
            if not row.isna().all():
                size_scores.loc[date] = standardize(row.dropna())

        self.factors['size'] = size_scores
        return size_scores
    
    def compute_value_factor(self) -> pd.DataFrame:
        """Compute value factor using P/E and P/B ratios."""
        value_scores = pd.DataFrame(index=self.prices.index, columns=self.prices.columns, dtype=float)

        if self.fundamentals is not None:
            for ticker in self.prices.columns:
                if ticker in self.fundamentals:
                    pe_ratio = self.fundamentals[ticker].get('pe_ratio', np.nan)
                    pb_ratio = self.fundamentals[ticker].get('pb_ratio', np.nan)

                    value_score = 0.0
                    count = 0

                    if not pd.isna(pe_ratio) and pe_ratio > 0:
                        value_score += 1.0 / pe_ratio
                        count += 1

                    if not pd.isna(pb_ratio) and pb_ratio > 0:
                        value_score += 1.0 / pb_ratio
                        count += 1

                    if count > 0:
                        # Use the same score across dates (fundamentals are not time-indexed here)
                        value_scores[ticker] = value_score / count
        else:
            # Use price-based value proxy: inverse of 1Y price momentum
            value_scores = -self.prices.pct_change(252)

        # Standardize cross-sectionally
        for date in value_scores.index:
            row = value_scores.loc[date]
            if not row.isna().all():
                value_scores.loc[date] = standardize(row.dropna())

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
        quality_scores = pd.DataFrame(index=self.prices.index, columns=self.prices.columns, dtype=float)

        if self.fundamentals is not None:
            for ticker in self.prices.columns:
                if ticker in self.fundamentals:
                    roe = self.fundamentals[ticker].get('roe', np.nan)  # percentage
                    profit_margin = self.fundamentals[ticker].get('profit_margin', np.nan)  # percentage
                    debt_to_equity = self.fundamentals[ticker].get('debt_to_equity', np.nan)  # percentage

                    quality_score = 0.0
                    count = 0

                    if not pd.isna(roe):
                        quality_score += roe
                        count += 1

                    if not pd.isna(profit_margin):
                        quality_score += profit_margin
                        count += 1

                    if not pd.isna(debt_to_equity):
                        quality_score -= debt_to_equity / 100.0  # Penalize high debt
                        count += 1

                    if count > 0:
                        quality_scores[ticker] = quality_score / count
        else:
            # Use return stability as quality proxy
            quality_scores = -self.returns.rolling(252).std()

        # Standardize cross-sectionally
        for date in quality_scores.index:
            row = quality_scores.loc[date]
            if not row.isna().all():
                quality_scores.loc[date] = standardize(row.dropna())

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
        portfolios: Dict[int, List[str]] = {}

        # Find the most recent date with at least 2 valid scores
        latest_scores = None
        for date in reversed(factor_scores.index.tolist()):
            row = factor_scores.loc[date].dropna()
            if len(row) >= 2:
                latest_scores = row
                break

        if latest_scores is None or latest_scores.empty:
            raise ValueError(f"No valid scores available to create portfolios for factor '{factor_name}'.")

        # If fewer stocks than requested portfolios, reduce number of portfolios
        k = min(n_portfolios, max(1, len(latest_scores)))

        # Try using utility deciles, fallback to qcut if needed
        try:
            deciles = create_deciles(latest_scores)
            # Ensure labels align to 1..k if utility returns fewer bins
            unique_bins = sorted(pd.unique(deciles.dropna()))
            if len(unique_bins) != k:
                # Recreate with qcut to enforce k bins or drop duplicates
                deciles = pd.qcut(latest_scores.rank(method='first'), q=k, labels=list(range(1, k + 1)), duplicates='drop')
        except Exception:
            deciles = pd.qcut(latest_scores.rank(method='first'), q=k, labels=list(range(1, k + 1)), duplicates='drop')

        # Build portfolios; empty lists allowed for bins with no members
        for i in range(1, n_portfolios + 1):
            if i <= k:
                portfolios[i] = latest_scores[deciles == i].index.tolist()
            else:
                portfolios[i] = []

        return portfolios
    
    def create_long_short_portfolio(self, factor_name: str) -> Tuple[List[str], List[str]]:
        """Create long-short portfolio based on factor."""
        portfolios = self.create_factor_portfolios(factor_name)

        # Prefer highest and lowest available deciles; fallback to top/bottom by score
        available_bins = [i for i, names in portfolios.items() if names]
        if available_bins:
            long_decile = max(available_bins)
            short_decile = min(available_bins)
            long_portfolio = portfolios[long_decile]
            short_portfolio = portfolios[short_decile]
        else:
            # Fallback: build directly from latest scores
            scores = self.factors[factor_name].iloc[-1].dropna()
            if len(scores) < 2:
                raise ValueError(f"Insufficient data to form long-short portfolio for factor '{factor_name}'.")
            sorted_idx = scores.sort_values()
            short_portfolio = sorted_idx.index[:max(1, len(sorted_idx) // 10)].tolist()
            long_portfolio = sorted_idx.index[-max(1, len(sorted_idx) // 10):].tolist()

        return long_portfolio, short_portfolio
    
    def combine_factors_pca(self, n_components: int = 3) -> pd.DataFrame:
        """Combine factors using PCA."""
        # Align all factors to a per-date summary across stocks
        factor_data = pd.DataFrame()

        for name, factor in self.factors.items():
            # Mean across stocks per date; ignore all-NaN rows with min_count
            factor_data[name] = factor.mean(axis=1, skipna=True)

        # Drop rows where all factors are NaN
        factor_data = factor_data.dropna(how='all')

        # If still empty, bail early to avoid StandardScaler error
        if factor_data.empty:
            print("Warning: PCA skipped due to no valid factor data.")
            self.pca_factors = pd.DataFrame()
            self.pca_loadings = pd.DataFrame()
            return self.pca_factors

        # Drop columns that are entirely NaN, then fill remaining NaNs with column means
        factor_data = factor_data.dropna(axis=1, how='all')
        factor_data = factor_data.apply(lambda col: col.fillna(col.mean()))

        # If n_components exceeds available samples, reduce it
        n_components = min(n_components, max(1, len(factor_data)))

        # Apply PCA
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(factor_data)

        pca = PCA(n_components=n_components)
        pca_factors = pca.fit_transform(scaled_data)

        pca_df = pd.DataFrame(
            pca_factors,
            index=factor_data.index,
            columns=[f'PC{i + 1}' for i in range(n_components)]
        )

        # Store PCA results
        self.pca_factors = pca_df
        self.pca_loadings = pd.DataFrame(
            pca.components_.T,
            index=factor_data.columns,
            columns=pca_df.columns
        )

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
