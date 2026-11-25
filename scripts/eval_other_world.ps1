<#
.SYNOPSIS
    Zero-Shot Evaluation or Frozen-Backbone PPO Fine-Tuning on OtherWorlds/<WorldName>.
.DESCRIPTION
    Supports Mode = eval (default, zero-shot), train (PPO fine-tune only), or both (train + eval).
    Dynamically resolves backbone, temporal, and policy checkpoints by ModelType & Task.
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$WorldName,
    [ValidateSet("eval", "train", "both")]
    [string]$Mode = "eval",
    [ValidateSet("m1_full", "m2_static")]
    [string]$ModelType = "m1_full",
    [ValidateSet("rl1", "rl2")]
    [string]$Task = "rl1",
    [string]$BackboneCkpt = "",
    [string]$TemporalCkpt = "",
    [string]$PolicyCkpt = "",
    [int]$TotalSteps = 25000,
    [int]$NumEpisodes = 20,
    [int]$Seed = 1000,
    [string]$Device = "cpu"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

$BakedDir = Join-Path $RepoRoot "OtherWorlds/$WorldName/baked_regions"
if (-not (Test-Path $BakedDir)) {
    throw "Baked regions not found at $BakedDir. Run .\OtherWorlds\bake_world.ps1 -WorldName $WorldName first."
}

# Dynamic checkpoint resolution by ModelType & Task
if (-not $BackboneCkpt) {
    if ($ModelType -eq "m2_static" -and (Test-Path "checkpoints/m2_spatial_jepa/best.pkl")) {
        $BackboneCkpt = "checkpoints/m2_spatial_jepa/best.pkl"
    } else {
        $BackboneCkpt = "checkpoints/spatial_finetuned_best.pkl"
    }
}
if ($ModelType -eq "m1_full" -and -not $TemporalCkpt) {
    $TemporalCkpt = "checkpoints/temporal_best.pkl"
}
if (-not $PolicyCkpt) {
    if ($ModelType -eq "m1_full") {
        $PolicyCkpt = if ($Task -eq "rl2") { "checkpoints/ppo_nav_m1_rl2/latest_ppo.pkl" } else { "checkpoints/ppo_nav_m1_overnight/best_ppo.pkl" }
    } else {
        $PolicyCkpt = if ($Task -eq "rl2") { "checkpoints/ppo_nav_m2_rl2/best_ppo.pkl" } else { "checkpoints/ppo_nav_m2_overnight/best_ppo.pkl" }
    }
}

$TrainSaveDir = "checkpoints/other_worlds/${WorldName}_${ModelType}_${Task}"
$OutJson = "results/other_worlds/${WorldName}_${ModelType}_${Task}.json"
New-Item -ItemType Directory -Force -Path "results/other_worlds" | Out-Null

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host " OtherWorld [$WorldName] | Mode=$Mode | Model=$ModelType | Task=$Task" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "  Baked Dir:     $BakedDir"
Write-Host "  Backbone Ckpt: $BackboneCkpt"
if ($TemporalCkpt) { Write-Host "  Temporal Ckpt: $TemporalCkpt" }
Write-Host "  Policy Ckpt:   $PolicyCkpt"

if ($Mode -in @("train", "both")) {
    $TrainArgs = @(
        "run", "python", "-m", "eval.train_ppo_nav",
        "--region_dir", $BakedDir,
        "--model_type", $ModelType,
        "--ckpt", $BackboneCkpt,
        "--task", $Task,
        "--lr", "1e-4",
        "--bc_epochs", "0",
        "--total_steps", $TotalSteps,
        "--save_dir", $TrainSaveDir,
        "--device", $Device
    )
    if ($ModelType -eq "m1_full" -and $TemporalCkpt) {
        $TrainArgs += @("--temporal_ckpt", $TemporalCkpt)
    }
    if ($PolicyCkpt -and (Test-Path $PolicyCkpt)) {
        $TrainArgs += @("--init_policy", $PolicyCkpt)
    }
    & uv @TrainArgs
    if ($LASTEXITCODE -ne 0) { throw "train_ppo_nav failed with exit code $LASTEXITCODE" }
    $PolicyCkpt = "$TrainSaveDir/best_ppo.pkl"
}

if ($Mode -in @("eval", "both")) {
    $EvalArgs = @(
        "run", "python", "-m", "eval.eval_navigation",
        "--region_dir", $BakedDir,
        "--model_type", $ModelType,
        "--backbone_ckpt", $BackboneCkpt,
        "--policy_ckpt", $PolicyCkpt,
        "--task", $Task,
        "--num_episodes", $NumEpisodes,
        "--seed", $Seed,
        "--out_json", $OutJson
    )
    if ($ModelType -eq "m1_full" -and $TemporalCkpt) {
        $EvalArgs += @("--temporal_ckpt", $TemporalCkpt)
    }
    & uv @EvalArgs
    if ($LASTEXITCODE -ne 0) { throw "eval_navigation failed with exit code $LASTEXITCODE" }
}
