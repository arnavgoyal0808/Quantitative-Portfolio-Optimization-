#!/bin/bash

# Script to push the quantitative research project to GitHub
# Repository: https://github.com/arnavgoyal0808/Quantitative-Portfolio-Optimization-.git

echo "Initializing Git repository and pushing to GitHub..."

# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Complete quantitative research project

- Multi-factor investment strategies (Momentum, Size, Value, Volatility, Quality)
- Portfolio optimization using PyPortfolioOpt (Mean-variance, Risk parity, HRP)
- Comprehensive backtesting with performance attribution
- Free data sources (Yahoo Finance + FRED)
- Complete CI/CD pipeline with GitHub Actions
- Extensive test suite with pytest
- Production-ready code with error handling
- Professional documentation and examples"

# Add remote origin
git remote add origin https://github.com/arnavgoyal0808/Quantitative-Portfolio-Optimization-.git

# Push to main branch
git branch -M main
git push -u origin main

echo "Code pushed successfully to GitHub!"
echo "Repository URL: https://github.com/arnavgoyal0808/Quantitative-Portfolio-Optimization-.git"
