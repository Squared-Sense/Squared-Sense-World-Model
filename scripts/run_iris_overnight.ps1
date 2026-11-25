<#
.SYNOPSIS
    Overnight IRIS 27 Evaluation & RL2 Pipeline (Windows PowerShell).
.DESCRIPTION
    Runs all zero-retraining evaluations (temporal dynamics + ego-shift ablation,
    gating Pareto sweep, few-shot probes) and optionally RL2 fine-tuning + eval,
    saving JSONs to results/ and logs to logs/iris_overnight.log.
#>

param(
    [string]$RegionDir = "data/baked_regions",
    [string]$ResultsDir = "results",
    [string]$LogFile = "logs/iris_overnight.log",
    [switch]$RunRL2,
    [int]$EvalEpisodes = 20,
    [int]$Seed = 1000
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogFile) | Out-Null

$script:StepIdx = 0
$script:TotalStepsEst = 8

function Run-Step($Name, [string[]]$CmdArgs) {
    $script:StepIdx++
    $pct = [Math]::Min(99, [int](($script:StepIdx - 1) / $script:TotalStepsEst * 100))
    $barLen = 20
    $filled = [int][Math]::Round(($pct / 100.0) * $barLen)
    $bar = ("=" * $filled) + ("-" * ($barLen - $filled))
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $header = "[$ts] [$bar $pct%] ($($script:StepIdx)/$($script:TotalStepsEst)) === START: $Name ==="
    Write-Host $header -ForegroundColor Cyan
    Write-Progress -Activity "IRIS 27 Pipeline" -Status $Name -PercentComplete $pct
    Add-Content -Path $LogFile -Value $header
    & uv @CmdArgs 2>&1 | Tee-Object -FilePath $LogFile -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Step '$Name' failed with exit code $LASTEXITCODE"
    }
    $tsEnd = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $LogFile -Value "[$tsEnd] === DONE: $Name ==="
}

# 1. Temporal Dynamics & Pure Ego-Shift Ablation (M1)
if ((Test-Path "checkpoints/spatial_finetuned_best.pkl") -and (Test-Path "checkpoints/temporal_best.pkl")) {
    Run-Step "Temporal Dynamics & Ego-Shift Ablation (M1)" @(
        "run", "python", "-m", "eval.eval_temporal_dynamics",
        "--spatial_ckpt", "checkpoints/spatial_finetuned_best.pkl",
        "--temporal_ckpt", "checkpoints/temporal_best.pkl",
        "--region_dir", $RegionDir,
        "--out_json", "$ResultsDir/temporal_dynamics_ablation.json"
    )
} else {
    Write-Host "Skipping Temporal Dynamics (M1 checkpoints not found)." -ForegroundColor Yellow
}

# 2. Event-Driven Shell Gating Pareto Sweep (M1 RL1)
if (Test-Path "checkpoints/ppo_nav_m1_overnight/best_ppo.pkl") {
    Run-Step "Event-Driven Shell Gating Pareto Sweep (M1)" @(
        "run", "python", "-m", "eval.eval_gating_pareto",
        "--policy_ckpt", "checkpoints/ppo_nav_m1_overnight/best_ppo.pkl",
        "--backbone_ckpt", "checkpoints/spatial_finetuned_best.pkl",
        "--temporal_ckpt", "checkpoints/temporal_best.pkl",
        "--region_dir", $RegionDir,
        "--num_episodes", $EvalEpisodes,
        "--seed", $Seed,
        "--out_json", "$ResultsDir/gating_pareto.json"
    )
} else {
    Write-Host "Skipping Gating Pareto Sweep (checkpoints/ppo_nav_m1_overnight/best_ppo.pkl not found)." -ForegroundColor Yellow
}

# 3. Few-Shot Probe Sweep across available checkpoints
$ProbeTargets = @(
    @{ Name = "m1_full";      Ckpt = "checkpoints/spatial_finetuned_best.pkl";       Temp = "checkpoints/temporal_best.pkl" },
    @{ Name = "m2_static";    Ckpt = "checkpoints/m2_spatial_jepa/best.pkl";         Temp = $null },
    @{ Name = "b1_lod0";      Ckpt = "checkpoints/baselines/b1_lod0/best.pkl";       Temp = $null },
    @{ Name = "b2_lod2";      Ckpt = "checkpoints/baselines/b2_lod2/best.pkl";       Temp = $null },
    @{ Name = "b3_mae";       Ckpt = "checkpoints/baselines/b3_mae/best.pkl";        Temp = $null },
    @{ Name = "b4_uniform32"; Ckpt = "checkpoints/baselines/b4_uniform32/best.pkl";  Temp = $null }
)

foreach ($pt in $ProbeTargets) {
    if (Test-Path $pt.Ckpt) {
        $probeArgs = @(
            "run", "python", "-m", "eval.probe_suite",
            "--model_type", $pt.Name,
            "--ckpt", $pt.Ckpt,
            "--region_dir", $RegionDir,
            "--train_fractions", "0.1", "0.25", "1.0",
            "--out_json", "$ResultsDir/probe_$($pt.Name)_fewshot.json"
        )
        if ($pt.Temp -and (Test-Path $pt.Temp)) {
            $probeArgs += @("--temporal_ckpt", $pt.Temp)
        }
        Run-Step "Few-Shot Probes ($($pt.Name))" $probeArgs
    } else {
        Write-Host "Skipping Probe ($($pt.Name)): checkpoint $($pt.Ckpt) not found." -ForegroundColor Yellow
    }
}

# 4. Optional RL2 Training + Zero-Shot Evaluation
if ($RunRL2) {
    if (Test-Path "checkpoints/ppo_nav_m1_overnight/best_ppo.pkl") {
        & "$ScriptDir/train_ppo_rl2.ps1" -RegionDir $RegionDir -ModelType "m1_full" `
            -InitPolicy "checkpoints/ppo_nav_m1_overnight/best_ppo.pkl" `
            -SaveDir "checkpoints/ppo_nav_m1_rl2"
        Run-Step "RL2 Zero-Shot Eval (M1)" @(
            "run", "python", "-m", "eval.eval_navigation",
            "--model_type", "m1_full",
            "--backbone_ckpt", "checkpoints/spatial_finetuned_best.pkl",
            "--temporal_ckpt", "checkpoints/temporal_best.pkl",
            "--policy_ckpt", "checkpoints/ppo_nav_m1_rl2/best_ppo.pkl",
            "--region_dir", $RegionDir,
            "--task", "rl2",
            "--num_episodes", $EvalEpisodes,
            "--seed", $Seed,
            "--out_json", "$ResultsDir/nav_m1_rl2.json"
        )
    }
    if ((Test-Path "checkpoints/ppo_nav_m2_overnight/best_ppo.pkl") -and (Test-Path "checkpoints/m2_spatial_jepa/best.pkl")) {
        & "$ScriptDir/train_ppo_rl2.ps1" -RegionDir $RegionDir -ModelType "m2_static" `
            -Ckpt "checkpoints/m2_spatial_jepa/best.pkl" `
            -InitPolicy "checkpoints/ppo_nav_m2_overnight/best_ppo.pkl" `
            -SaveDir "checkpoints/ppo_nav_m2_rl2"
        Run-Step "RL2 Zero-Shot Eval (M2)" @(
            "run", "python", "-m", "eval.eval_navigation",
            "--model_type", "m2_static",
            "--backbone_ckpt", "checkpoints/m2_spatial_jepa/best.pkl",
            "--policy_ckpt", "checkpoints/ppo_nav_m2_rl2/best_ppo.pkl",
            "--region_dir", $RegionDir,
            "--task", "rl2",
            "--num_episodes", $EvalEpisodes,
            "--seed", $Seed,
            "--out_json", "$ResultsDir/nav_m2_rl2.json"
        )
    }
}

Write-Host "`nOvernight IRIS 27 pipeline complete! Outputs in $ResultsDir" -ForegroundColor Green
