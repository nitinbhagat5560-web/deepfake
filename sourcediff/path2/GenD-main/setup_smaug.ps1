$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$EnvDir = Join-Path $Root ".venv-smaug"
$Python = Join-Path $EnvDir "Scripts\python.exe"

if (-not (Test-Path $Python)) {
    Write-Host "Creating isolated Smaug environment..."
    $MainPython = Join-Path $Root ".venv\Scripts\python.exe"
    if (Test-Path $MainPython) {
        & $MainPython -m venv $EnvDir
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.12 -m venv $EnvDir
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv $EnvDir
    } else {
        throw "Python 3.12 or a base Python installation is required to create .venv-smaug."
    }
}

& $Python -m pip install --upgrade pip
& $Python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
& $Python -m pip install "transformers>=5.0,<6" "sentence-transformers==6.1.0" "huggingface-hub>=1.3,<2" pillow opencv-python-headless

Write-Host "Smaug environment ready at $EnvDir"
