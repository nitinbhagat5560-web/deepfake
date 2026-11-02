# SourceDiff Web Application Runner
Set-Location -Path $PSScriptRoot
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "  Starting SourceDiff Web Application" -ForegroundColor Green
Write-Host "  URL: http://127.0.0.1:8000" -ForegroundColor Yellow
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host ""
& "D:\deepfake\.venv311\Scripts\python.exe" -m uvicorn sourcediff.api:app --host 127.0.0.1 --port 8000
