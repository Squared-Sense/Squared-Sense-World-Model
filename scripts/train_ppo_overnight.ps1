<#
.SYNOPSIS
    Runs PPO Navigation training with BC warm-start on baked Minecraft regions.
.DESCRIPTION
    You can edit the default variable values directly in this script,
    or override them by passing parameters, e.g.:
      .\scripts\train_ppo_overnight.ps1 -TotalSteps 20000 -BcEpochs 10
    Any additional arguments passed are forwarded directly to train_ppo_nav.py.
#>

param(
    [string]$RegionDir = "data/baked_regions",
    [string]$ModelType = "m1_full",
    [string]$Ckpt = "checkpoints/spatial_finetuned_best.pkl",
    [string]$TemporalCkpt = "checkpoints/temporal_best.pkl",
    [int]$BcEpochs = 20,
    [int]$BcSteps = 600,
    [int]$TotalSteps = 50000,
    [string]$SaveDir = "checkpoints/ppo_nav_m1_overnight",
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"

# Ensure we run from repository root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host " Squared Sense: PPO Navigation Training ($ModelType)" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "  Region Dir:    $RegionDir"
Write-Host "  Model Type:    $ModelType"
Write-Host "  Spatial Ckpt:  $Ckpt"
Write-Host "  Temporal Ckpt: $TemporalCkpt"
Write-Host "  BC Warm-Start: $BcEpochs epochs ($BcSteps demo steps)"
Write-Host "  Total Steps:   $TotalSteps"
Write-Host "  Save Dir:      $SaveDir"
Write-Host "  Device:        $Device"
if ($args.Count -gt 0) {
    Write-Host "  Extra Args:    $($args -join ' ')"
}
Write-Host "=======================================================" -ForegroundColor Cyan

$CmdArgs = @(
    "run", "python", "-m", "eval.train_ppo_nav",
    "--region_dir", $RegionDir,
    "--model_type", $ModelType,
    "--ckpt", $Ckpt,
    "--temporal_ckpt", $TemporalCkpt,
    "--bc_epochs", $BcEpochs,
    "--bc_steps", $BcSteps,
    "--total_steps", $TotalSteps,
    "--save_dir", $SaveDir,
    "--device", $Device
)

# Append any trailing unparsed arguments passed by the user
if ($args.Count -gt 0) {
    $CmdArgs += $args
}

& uv @CmdArgs
