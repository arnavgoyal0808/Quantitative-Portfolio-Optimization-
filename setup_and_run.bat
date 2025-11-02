@echo off
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Creating necessary directories...
mkdir data 2>nul
mkdir results 2>nul
mkdir results\plots 2>nul

echo.
echo Running the project...
python main.py

echo.
echo Done!
pause
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Creating necessary directories...
mkdir data 2>nul
mkdir results 2>nul
mkdir results\plots 2>nul

echo.
echo Running the project...
python main.py

echo.
echo Done!
pause