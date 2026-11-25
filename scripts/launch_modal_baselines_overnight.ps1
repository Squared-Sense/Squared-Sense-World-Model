<#
.SYNOPSIS
    Launches Stage 1A Baselines (B1-B4) + 20-epoch --no_vicreg Collapse Control in parallel on Modal GPUs.
#>

param(
    [switch]$SkipUpload,
    [string]$LogDir = "logs/modal_overnight"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

if (-not $SkipUpload -and (Test-Path "data/baked_regions")) {
    Write-Host "Uploading data/baked_regions to Modal volume squared-sense-vol:/data/baked_regions..." -ForegroundColor Cyan
    & uv run modal volume put squared-sense-vol data/baked_regions /data/baked_regions
}

$Jobs = @(
    @{ Name = "b4_uniform32"; Args = @("run", "modal", "run", "modal_app.py", "--action", "train-baselines", "--model", "b4_uniform32", "--epochs", "80", "--batch-size", "16") },
    @{ Name = "b1_lod0";      Args = @("run", "modal", "run", "modal_app.py", "--action", "train-baselines", "--model", "b1_lod0",      "--epochs", "80", "--batch-size", "32") },
    @{ Name = "b2_lod2";      Args = @("run", "modal", "run", "modal_app.py", "--action", "train-baselines", "--model", "b2_lod2",      "--epochs", "80", "--batch-size", "32") },
    @{ Name = "b3_mae";       Args = @("run", "modal", "run", "modal_app.py", "--action", "train-baselines", "--model", "b3_mae",       "--epochs", "80", "--batch-size", "32") },
    @{ Name = "no_vicreg";    Args = @("run", "modal", "run", "modal_app.py", "--action", "train-spatial",   "--epochs", "20", "--batch-size", "32", "--no-vicreg") }
)

$Procs = @()
foreach ($j in $Jobs) {
    $outLog = "$LogDir/$($j.Name).log"
    $errLog = "$LogDir/$($j.Name).err.log"
    Write-Host "Launching Modal job [$($j.Name)] -> $outLog" -ForegroundColor Green
    $p = Start-Process -FilePath "uv" -ArgumentList $j.Args -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru -NoNewWindow
    $Procs += @{ Name = $j.Name; Proc = $p }
}

Write-Host "Waiting for all $($Procs.Count) Modal GPU jobs to finish..." -ForegroundColor Cyan
foreach ($entry in $Procs) {
    $entry.Proc.WaitForExit()
    $code = $entry.Proc.ExitCode
    if ($code -eq 0) {
        Write-Host "  [OK] $($entry.Name)" -ForegroundColor Green
    } else {
        Write-Host "  [FAIL exit=$code] $($entry.Name) (check $LogDir/$($entry.Name).err.log)" -ForegroundColor Red
    }
}
