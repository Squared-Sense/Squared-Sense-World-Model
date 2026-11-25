<#
.SYNOPSIS
    Runs real-time inference rollout of trained policy streaming to the 3D WebGL visualizer.
.DESCRIPTION
    You can edit the default variable values directly in this script,
    or override them by passing parameters, e.g.:
      .\scripts\run_inference_visualizer.ps1 -Episodes 20 -Fps 15
    Any additional arguments are forwarded to run_inference.py.
#>

param(
    [string]$RegionDir = "data/baked_regions",
    [string]$ModelType = "m1_full",
    [string]$Ckpt = "checkpoints/spatial_finetuned_best.pkl",
    [string]$TemporalCkpt = "checkpoints/temporal_best.pkl",
    [string]$PolicyCkpt = "checkpoints/ppo_nav_m1_overnight/best_ppo.pkl",
    [int]$Episodes = 10,
    [double]$Fps = 10.0,
    [int]$Seed = 42,
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

# Fallback to policy_best.pkl if best_ppo.pkl is not yet created
if (-not (Test-Path $PolicyCkpt)) {
    $AltCkpt = "checkpoints/ppo_nav/policy_best.pkl"
    if (Test-Path $AltCkpt) {
        Write-Host "Warning: $PolicyCkpt not found, falling back to $AltCkpt" -ForegroundColor Yellow
        $PolicyCkpt = $AltCkpt
    }
}

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host " Squared Sense: 3D Visualizer Real-Time Inference" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "  Region Dir:    $RegionDir"
Write-Host "  Model Type:    $ModelType"
Write-Host "  Policy Ckpt:   $PolicyCkpt"
Write-Host "  Spatial Ckpt:  $Ckpt"
Write-Host "  Temporal Ckpt: $TemporalCkpt"
Write-Host "  Episodes:      $Episodes"
Write-Host "  Target FPS:    $Fps"
Write-Host "  Device:        $Device"
if ($args.Count -gt 0) {
    Write-Host "  Extra Args:    $($args -join ' ')"
}
Write-Host "=======================================================" -ForegroundColor Cyan

$CmdArgs = @(
    "run", "python", "-m", "eval.run_inference",
    "--region_dir", $RegionDir,
    "--model_type", $ModelType,
    "--policy_ckpt", $PolicyCkpt,
    "--ckpt", $Ckpt,
    "--temporal_ckpt", $TemporalCkpt,
    "--episodes", $Episodes,
    "--fps", $Fps,
    "--seed", $Seed,
    "--device", $Device
)

if ($args.Count -gt 0) {
    $CmdArgs += $args
}

& uv @CmdArgs
