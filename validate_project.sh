#!/bin/bash

echo "==================================="
echo "QUANTITATIVE RESEARCH PROJECT VALIDATION"
echo "==================================="

# Check project structure
echo "Checking project structure..."

required_files=(
    "README.md"
    "requirements.txt"
    "main.py"
    "setup.py"
    "Makefile"
    ".github/workflows/ci.yml"
    "src/__init__.py"
    "src/utils.py"
    "src/data_loader.py"
    "src/factor_analysis.py"
    "src/portfolio_optimization.py"
    "src/backtesting.py"
    "tests/__init__.py"
    "tests/test_all.py"
    "config.py"
)

required_dirs=(
    "src"
    "tests"
    "data"
    "results"
    "results/plots"
    ".github"
    ".github/workflows"
)

echo "Checking directories..."
for dir in "${required_dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "✓ Directory exists: $dir"
    else
        echo "✗ Missing directory: $dir"
    fi
done

echo -e "\nChecking files..."
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ File exists: $file"
    else
        echo "✗ Missing file: $file"
    fi
done

echo -e "\nChecking file sizes..."
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        size=$(wc -c < "$file")
        if [ $size -gt 100 ]; then
            echo "✓ $file ($size bytes)"
        else
            echo "⚠ $file is very small ($size bytes)"
        fi
    fi
done

echo -e "\nChecking Python imports in main files..."
main_files=("src/utils.py" "src/data_loader.py" "src/factor_analysis.py" "src/portfolio_optimization.py" "src/backtesting.py" "main.py")

for file in "${main_files[@]}"; do
    if [ -f "$file" ]; then
        imports=$(grep -c "^import\|^from.*import" "$file")
        echo "✓ $file has $imports import statements"
    fi
done

echo -e "\nChecking requirements.txt..."
if [ -f "requirements.txt" ]; then
    req_count=$(wc -l < requirements.txt)
    echo "✓ requirements.txt has $req_count dependencies"
    echo "Key dependencies:"
    grep -E "(pandas|numpy|yfinance|scikit-learn|PyPortfolioOpt)" requirements.txt || echo "⚠ Some key dependencies may be missing"
fi

echo -e "\nChecking CI configuration..."
if [ -f ".github/workflows/ci.yml" ]; then
    echo "✓ GitHub Actions CI configuration exists"
    if grep -q "pytest" ".github/workflows/ci.yml"; then
        echo "✓ CI includes pytest"
    fi
    if grep -q "python-version" ".github/workflows/ci.yml"; then
        echo "✓ CI includes Python version matrix"
    fi
fi

echo -e "\nProject validation complete!"
echo "==================================="

# Summary
total_files=${#required_files[@]}
existing_files=0
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        ((existing_files++))
    fi
done

total_dirs=${#required_dirs[@]}
existing_dirs=0
for dir in "${required_dirs[@]}"; do
    if [ -d "$dir" ]; then
        ((existing_dirs++))
    fi
done

echo "SUMMARY:"
echo "Files: $existing_files/$total_files"
echo "Directories: $existing_dirs/$total_dirs"

if [ $existing_files -eq $total_files ] && [ $existing_dirs -eq $total_dirs ]; then
    echo "✓ Project structure is complete!"
    exit 0
else
    echo "⚠ Project structure has missing components"
    exit 1
fi
