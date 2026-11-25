<#
.SYNOPSIS
    Runs RL2 (5-Target Chained Navigation with 50% Burst Compass Occlusion) PPO Fine-Tuning.
.DESCRIPTION
    Fine-tunes an RL1 ActorCriticHead checkpoint (or trains from scratch) on RL2.
    Any additional arguments passed are forwarded directly to eval.train_ppo_nav.
#>

param(
    [string]$RegionDir = "data/baked_regions",
    [string]$ModelType = "m1_full",
    [string]$Ckpt = "checkpoints/spatial_finetuned_best.pkl",
    [string]$TemporalCkpt = "checkpoints/temporal_best.pkl",
    [string]$InitPolicy = "checkpoints/ppo_nav_m1_overnight/best_ppo.pkl",
    [int]$NumTargets = 5,
    [double]$CompassOcclusion = 0.5,
    [int]$OcclusionBurst = 6,
    [double]$Lr = 1e-4,
    [int]$BcEpochs = 0,
    [int]$TotalSteps = 25000,
    [string]$SaveDir = "checkpoints/ppo_nav_m1_rl2",
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host " Squared Sense: RL2 PPO Navigation Training ($ModelType)" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "  Region Dir:        $RegionDir"
Write-Host "  Model Type:        $ModelType"
Write-Host "  Spatial Ckpt:      $Ckpt"
Write-Host "  Temporal Ckpt:     $TemporalCkpt"
Write-Host "  Init Policy:       $InitPolicy"
Write-Host "  Num Targets:       $NumTargets"
Write-Host "  Compass Occlusion: $CompassOcclusion (burst=$OcclusionBurst)"
Write-Host "  LR:                $Lr | BC Epochs: $BcEpochs"
Write-Host "  Total Steps:       $TotalSteps"
Write-Host "  Save Dir:          $SaveDir"
Write-Host "  Device:            $Device"
if ($args.Count -gt 0) {
    Write-Host "  Extra Args:        $($args -join ' ')"
}
Write-Host "=======================================================" -ForegroundColor Cyan

$CmdArgs = @(
    "run", "python", "-m", "eval.train_ppo_nav",
    "--region_dir", $RegionDir,
    "--model_type", $ModelType,
    "--ckpt", $Ckpt,
    "--task", "rl2",
    "--num_targets", $NumTargets,
    "--compass_occlusion", $CompassOcclusion,
    "--occlusion_burst", $OcclusionBurst,
    "--lr", $Lr,
    "--bc_epochs", $BcEpochs,
    "--total_steps", $TotalSteps,
    "--save_dir", $SaveDir,
    "--device", $Device
)

if ($ModelType -eq "m1_full" -and $TemporalCkpt) {
    $CmdArgs += @("--temporal_ckpt", $TemporalCkpt)
}
if ($InitPolicy -and (Test-Path $InitPolicy)) {
    $CmdArgs += @("--init_policy", $InitPolicy)
}
if ($args.Count -gt 0) {
    $CmdArgs += $args
}

& uv @CmdArgs
