"""Data loading module for financial data from Yahoo Finance and FRED."""

import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
from typing import List, Dict, Optional, Tuple
import warnings
from tqdm import tqdm
import os
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

class DataLoader:
    """Class for loading and managing financial data."""
    
    def __init__(self, start_date: str = '2010-01-01', end_date: Optional[str] = None):
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.data_dir = 'data'
        os.makedirs(self.data_dir, exist_ok=True)
        
    def download_stock_data(self, tickers: List[str], batch_size: int = 50) -> Dict[str, pd.DataFrame]:
        """Download stock data for multiple tickers."""
        stock_data = {}
        
        # Process in batches to avoid rate limits
        for i in tqdm(range(0, len(tickers), batch_size), desc="Downloading stock data"):
            batch_tickers = tickers[i:i + batch_size]
            
            try:
                # Download batch data
                data = yf.download(batch_tickers, start=self.start_date, end=self.end_date, 
                                 group_by='ticker', progress=False, threads=True)
                
                # Process each ticker
                for ticker in batch_tickers:
                    try:
                        if len(batch_tickers) == 1:
                            ticker_data = data
                        else:
                            ticker_data = data[ticker]
                        
                        if not ticker_data.empty and len(ticker_data) > 252:  # At least 1 year of data
                            # Clean data
                            ticker_data = ticker_data.dropna()
                            stock_data[ticker] = ticker_data
                            
                    except Exception as e:
                        print(f"Error processing {ticker}: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error downloading batch {i//batch_size + 1}: {e}")
                continue
                
        print(f"Successfully downloaded data for {len(stock_data)} stocks")
        return stock_data
    
    def get_market_data(self) -> pd.DataFrame:
        """Download market benchmark data (S&P 500)."""
        try:
            spy = yf.download('^GSPC', start=self.start_date, end=self.end_date, progress=False)
            return spy['Adj Close'].to_frame('SPY')
        except Exception as e:
            print(f"Error downloading market data: {e}")
            return pd.DataFrame()
    
    def get_risk_free_rate(self) -> pd.Series:
        """Download risk-free rate from FRED (3-Month Treasury)."""
        try:
            rf_rate = web.DataReader('TB3MS', 'fred', self.start_date, self.end_date)
            rf_rate = rf_rate / 100  # Convert percentage to decimal
            return rf_rate.squeeze()
        except Exception as e:
            print(f"Error downloading risk-free rate: {e}")
            # Return constant 2% if FRED fails
            dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
            return pd.Series(0.02/365, index=dates, name='TB3MS')
    
    def get_fundamental_data(self, tickers: List[str]) -> Dict[str, Dict]:
        """Get fundamental data for tickers."""
        fundamental_data = {}
        
        for ticker in tqdm(tickers, desc="Downloading fundamental data"):
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # Extract key fundamental metrics
                fundamentals = {
                    'market_cap': info.get('marketCap', np.nan),
                    'pe_ratio': info.get('trailingPE', np.nan),
                    'pb_ratio': info.get('priceToBook', np.nan),
                    'debt_to_equity': info.get('debtToEquity', np.nan),
                    'roe': info.get('returnOnEquity', np.nan),
                    'profit_margin': info.get('profitMargins', np.nan),
                    'revenue_growth': info.get('revenueGrowth', np.nan),
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown')
                }
                
                fundamental_data[ticker] = fundamentals
                
            except Exception as e:
                print(f"Error getting fundamentals for {ticker}: {e}")
                continue
                
        return fundamental_data
    
    def create_price_matrix(self, stock_data: Dict[str, pd.DataFrame], 
                           price_type: str = 'Adj Close') -> pd.DataFrame:
        """Create price matrix from stock data dictionary."""
        prices = pd.DataFrame()
        
        for ticker, data in stock_data.items():
            if price_type in data.columns:
                prices[ticker] = data[price_type]
        
        # Forward fill missing values and drop stocks with too many NaNs
        prices = prices.fillna(method='ffill').dropna(axis=1, thresh=len(prices) * 0.8)
        
        return prices
    
    def create_volume_matrix(self, stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create volume matrix from stock data dictionary."""
        volumes = pd.DataFrame()
        
        for ticker, data in stock_data.items():
            if 'Volume' in data.columns:
                volumes[ticker] = data['Volume']
        
        volumes = volumes.fillna(method='ffill').dropna(axis=1, thresh=len(volumes) * 0.8)
        
        return volumes
    
    def save_data(self, data: pd.DataFrame, filename: str):
        """Save data to CSV file."""
        filepath = os.path.join(self.data_dir, filename)
        data.to_csv(filepath)
        print(f"Data saved to {filepath}")
    
    def load_data(self, filename: str) -> pd.DataFrame:
        """Load data from CSV file."""
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            return pd.read_csv(filepath, index_col=0, parse_dates=True)
        else:
            raise FileNotFoundError(f"File {filepath} not found")

def main():
    """Main function to demonstrate data loading."""
    from utils import get_sp500_tickers
    
    # Initialize data loader
    loader = DataLoader(start_date='2015-01-01')
    
    # Get tickers
    tickers = get_sp500_tickers()
    print(f"Loading data for {len(tickers)} stocks...")
    
    # Download stock data
    stock_data = loader.download_stock_data(tickers)
    
    # Create price and volume matrices
    prices = loader.create_price_matrix(stock_data)
    volumes = loader.create_volume_matrix(stock_data)
    
    # Get market data
    market_data = loader.get_market_data()
    
    # Save data
    loader.save_data(prices, 'stock_prices.csv')
    loader.save_data(volumes, 'stock_volumes.csv')
    loader.save_data(market_data, 'market_data.csv')
    
    print(f"Data loading complete. Price matrix shape: {prices.shape}")
    print(f"Date range: {prices.index.min()} to {prices.index.max()}")

if __name__ == "__main__":
    main()
