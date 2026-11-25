<#
.SYNOPSIS
    Runs real-time 3D WebGL visualizer inference on an OtherWorlds/<WorldName> map.
.DESCRIPTION
    Streams live observations & telemetry from OtherWorlds/<WorldName>/baked_regions
    to the 3D WebGL visualizer via eval.run_inference.
    Automatically resolves checkpoints by ModelType (m1_full/m2_static) and Task (rl1/rl2).
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$WorldName,
    [ValidateSet("m1_full", "m2_static")]
    [string]$ModelType = "m1_full",
    [ValidateSet("rl1", "rl2")]
    [string]$Task = "rl1",
    [string]$Ckpt = "",
    [string]$TemporalCkpt = "",
    [string]$PolicyCkpt = "",
    [int]$Episodes = 10,
    [double]$Fps = 10.0,
    [int]$Seed = 42,
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

$RegionDir = Join-Path $RepoRoot "OtherWorlds/$WorldName/baked_regions"
if (-not (Test-Path $RegionDir)) {
    throw "Baked regions not found at $RegionDir. Run .\OtherWorlds\bake_world.ps1 -WorldName $WorldName first."
}

if (-not $Ckpt) {
    if ($ModelType -eq "m2_static" -and (Test-Path "checkpoints/m2_spatial_jepa/best.pkl")) {
        $Ckpt = "checkpoints/m2_spatial_jepa/best.pkl"
    } else {
        $Ckpt = "checkpoints/spatial_finetuned_best.pkl"
    }
}
if ($ModelType -eq "m1_full" -and -not $TemporalCkpt) {
    $TemporalCkpt = "checkpoints/temporal_best.pkl"
}
if (-not $PolicyCkpt) {
    $WorldCkpt = "checkpoints/other_worlds/${WorldName}_${ModelType}_${Task}/best_ppo.pkl"
    if (Test-Path $WorldCkpt) {
        $PolicyCkpt = $WorldCkpt
    } elseif ($ModelType -eq "m1_full") {
        $PolicyCkpt = if ($Task -eq "rl2") { "checkpoints/ppo_nav_m1_rl2/latest_ppo.pkl" } else { "checkpoints/ppo_nav_m1_overnight/best_ppo.pkl" }
    } else {
        $PolicyCkpt = if ($Task -eq "rl2") { "checkpoints/ppo_nav_m2_rl2/best_ppo.pkl" } else { "checkpoints/ppo_nav_m2_overnight/best_ppo.pkl" }
    }
}

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host " OtherWorld [$WorldName] 3D Visualizer Inference" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "  Region Dir:    $RegionDir"
Write-Host "  Model Type:    $ModelType ($Task)"
Write-Host "  Policy Ckpt:   $PolicyCkpt"
Write-Host "  Spatial Ckpt:  $Ckpt"
if ($TemporalCkpt) { Write-Host "  Temporal Ckpt: $TemporalCkpt" }
Write-Host "  Episodes:      $Episodes | FPS: $Fps | Device: $Device"
Write-Host "=======================================================" -ForegroundColor Cyan

$CmdArgs = @(
    "run", "python", "-m", "eval.run_inference",
    "--region_dir", $RegionDir,
    "--model_type", $ModelType,
    "--task", $Task,
    "--policy_ckpt", $PolicyCkpt,
    "--ckpt", $Ckpt,
    "--episodes", $Episodes,
    "--fps", $Fps,
    "--seed", $Seed,
    "--device", $Device
)
if ($ModelType -eq "m1_full" -and $TemporalCkpt) {
    $CmdArgs += @("--temporal_ckpt", $TemporalCkpt)
}
if ($args.Count -gt 0) {
    $CmdArgs += $args
}

& uv @CmdArgs
