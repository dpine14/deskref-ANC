$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".\host[dev]"

Write-Host "Python environment setup complete. To activate the virtual environment, run 'source .venv/bin/activate'."