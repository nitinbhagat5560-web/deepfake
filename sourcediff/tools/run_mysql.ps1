param(
    [switch]$MigrateOnly
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $projectRoot

$pythonCandidates = @(
    (Join-Path (Split-Path $projectRoot -Parent) ".venv311\Scripts\python.exe"),
    (Join-Path $projectRoot ".venv311\Scripts\python.exe"),
    "python.exe"
)

$python = $null
foreach ($candidate in $pythonCandidates) {
    if ($candidate -eq "python.exe") {
        $resolved = Get-Command python.exe -ErrorAction SilentlyContinue
        if ($resolved) {
            $python = $resolved.Source
            break
        }
    } elseif (Test-Path $candidate) {
        $python = (Resolve-Path $candidate).Path
        break
    }
}

if (-not $python) {
    throw "Python environment not found. Expected D:\deepfake\.venv311\Scripts\python.exe or the project's .venv311 environment."
}

$securePassword = Read-Host "MySQL application password" -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword)
try {
    $password = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
} finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
}

$env:SOURCEDIFF_DB_BACKEND = "mysql"
$env:SOURCEDIFF_MYSQL_HOST = "127.0.0.1"
$env:SOURCEDIFF_MYSQL_PORT = "3306"
$env:SOURCEDIFF_MYSQL_DATABASE = "sourcediff"
$env:SOURCEDIFF_MYSQL_USER = "sourcediff_app"
$env:SOURCEDIFF_MYSQL_PASSWORD = $password

& $python -c "import os, pymysql; pymysql.connect(host=os.environ['SOURCEDIFF_MYSQL_HOST'], port=int(os.environ['SOURCEDIFF_MYSQL_PORT']), database=os.environ['SOURCEDIFF_MYSQL_DATABASE'], user=os.environ['SOURCEDIFF_MYSQL_USER'], password=os.environ['SOURCEDIFF_MYSQL_PASSWORD'], connect_timeout=10).close()"
if ($LASTEXITCODE -ne 0) {
    throw "MySQL login failed for sourcediff_app. Enter the password assigned in MySQL, or reset the account using its root password."
}

if ($MigrateOnly) {
    & $python tools\migrate_sqlite_to_mysql.py --sqlite data\history.db
    exit $LASTEXITCODE
}

try {
    $uvicornProcesses = Get-CimInstance Win32_Process -ErrorAction Stop |
        Where-Object {
            $_.Name -match '^python(?:\.exe)?$' -and
            $_.CommandLine -match 'uvicorn\s+sourcediff\.api:app.*127\.0\.0\.1.*8000'
        }
    foreach ($process in $uvicornProcesses) {
        if ($process.ProcessId -ne $PID) {
            Stop-Process -Id $process.ProcessId -Force -ErrorAction Stop
        }
    }
    if ($uvicornProcesses) {
        Start-Sleep -Seconds 2
    }
} catch {
    throw "Could not stop an existing SourceDiff server on port 8000: $($_.Exception.Message)"
}

& $python -m uvicorn sourcediff.api:app --host 127.0.0.1 --port 8000
