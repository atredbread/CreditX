# Setup script for Credit Risk Analysis project
# Creates a virtual environment and installs dependencies

# Check if Python is installed
$pythonVersion = python --version
if (-not $?) {
    Write-Error "Python is not installed or not in PATH. Please install Python 3.8+ and try again."
    exit 1
}

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Cyan
python -m venv .venv

# Activate virtual environment
$activateScript = ".\.venv\Scripts\Activate.ps1"
if (-not (Test-Path $activateScript)) {
    Write-Error "Failed to create virtual environment. Please check your Python installation."
    exit 1
}

# Activate the environment
& $activateScript
if (-not $?) {
    Write-Error "Failed to activate virtual environment."
    exit 1
}

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Cyan
pip install -r requirements.txt

# Install development dependencies
Write-Host "Installing development dependencies..." -ForegroundColor Cyan
pip install pytest pytest-cov black flake8 mypy pre-commit

# Set up pre-commit hooks
Write-Host "Setting up pre-commit hooks..." -ForegroundColor Cyan
pre-commit install

Write-Host "`nSetup complete! Virtual environment is ready to use.`n" -ForegroundColor Green
Write-Host "To activate the virtual environment, run:"
Write-Host ".\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "`nTo deactivate, simply type 'deactivate'`n"
