# Multi-Factor Investment Strategy Research

A comprehensive open-source quantitative research project for constructing and optimizing multi-factor investment strategies using free financial data.

## Features

- **Data Sources**: Yahoo Finance & FRED (Federal Reserve Economic Data)
- **Factor Analysis**: Momentum, Size, Value, Volatility, Quality
- **Portfolio Construction**: Decile-based long-short portfolios
- **Optimization**: Mean-variance, Risk parity via PyPortfolioOpt
- **Performance Metrics**: Sharpe ratio, Alpha, Maximum drawdown
- **Advanced Features**: Market regime detection, PCA factor combination

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run complete analysis
python main.py

# Run specific components
python src/data_loader.py
python src/factor_analysis.py
python src/portfolio_optimization.py
python src/backtesting.py
```

## Project Structure

```
├── src/
│   ├── data_loader.py          # Data acquisition from Yahoo Finance & FRED
│   ├── factor_analysis.py      # Factor computation and analysis
│   ├── portfolio_optimization.py # Portfolio construction and optimization
│   ├── backtesting.py          # Performance evaluation and backtesting
│   └── utils.py                # Utility functions
├── tests/                      # Unit tests
├── data/                       # Data storage
├── results/                    # Output results and plots
├── main.py                     # Main execution script
└── requirements.txt            # Dependencies
```

## Factors Implemented

1. **Momentum**: 12-1 month price momentum
2. **Size**: Market capitalization
3. **Value**: Price-to-book, Price-to-earnings ratios
4. **Volatility**: Historical volatility measures
5. **Quality**: ROE, Debt-to-equity, Profit margins

## License

MIT License
