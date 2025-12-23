# PowerShell helper: create venv, install deps, and run training with logfile
param(
    [string]$DataRoot = "$PWD\CAER",
    [int]$Epochs = 10,
    [int]$BatchSize = 32
)
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$Venv = Join-Path $RepoRoot ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"
$LogDir = Join-Path $RepoRoot "runs\caer_skeleton"
$LogFile = Join-Path $LogDir "train.log"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }
if (-not (Test-Path $Venv)) {
    Write-Host "Creating virtual environment..."
    python -m venv $Venv
    & $Python -m pip install --upgrade pip setuptools wheel
    Write-Host "Installing requirements..."
    & $Python -m pip install -r (Join-Path $RepoRoot "requirements.txt")
}
if (-not (Test-Path $DataRoot)) {
    Write-Error "Data root not found: $DataRoot. Set parameter -DataRoot or place dataset at $RepoRoot\CAER"
    exit 1
}
Start-Process -NoNewWindow -FilePath $Python -ArgumentList @("train.py","--data_root",$DataRoot,"--epochs",$Epochs.ToString(),"--batch_size",$BatchSize.ToString(),"--num_workers","0","--logfile",$LogFile) -RedirectStandardOutput (Join-Path $LogDir "train.out") -RedirectStandardError (Join-Path $LogDir "train.err") -WindowStyle Hidden
Write-Host "Training started. Tail logfile with:`nGet-Content $LogFile -Wait -Tail 50"