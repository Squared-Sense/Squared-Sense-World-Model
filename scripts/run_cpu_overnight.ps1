<#
.SYNOPSIS
    CPU Parallel Overnight Runner for M1 RL2, M2 RL1->RL2, and All Zero-Retraining Evals.
.DESCRIPTION
    Runs two parallel CPU worker chains:
      - Chain A: M1 RL2 (25k steps)
      - Chain B: M2 RL1 (50k steps) -> M2 RL2 (25k steps)
    Once both chains complete, automatically runs:
      - Temporal Dynamics & Pure Ego-Shift Ablation (eval_temporal_dynamics.py)
      - Event-Driven Shell Gating Pareto Sweep (eval_gating_pareto.py)
      - Few-Shot Probe Sweep (probe_suite.py)
      - RL2 Zero-Shot Navigation Evaluations (M1 & M2)
    All logs are saved under logs/cpu_overnight/ and outputs under results/.
#>

param(
    [string]$RegionDir = "data/baked_regions",
    [string]$ResultsDir = "results",
    [string]$LogDir = "logs/cpu_overnight",
    [int]$EvalEpisodes = 20,
    [int]$Seed = 1000
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host " Squared Sense: CPU Parallel Overnight Training + Eval" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan

# Auto-resolve M2 spatial checkpoint
$M2Ckpt = if (Test-Path "checkpoints/m2_spatial_jepa/best.pkl") {
    "checkpoints/m2_spatial_jepa/best.pkl"
} else {
    "checkpoints/spatial_finetuned_best.pkl"
}
Write-Host "  M1 Spatial Ckpt: checkpoints/spatial_finetuned_best.pkl"
Write-Host "  M2 Spatial Ckpt: $M2Ckpt"

# Clear stale ppo_metrics.jsonl so progress starts cleanly at 0.0%
foreach ($d in @("checkpoints/ppo_nav_m1_rl2", "checkpoints/ppo_nav_m2_overnight", "checkpoints/ppo_nav_m2_rl2")) {
    $mf = Join-Path $d "ppo_metrics.jsonl"
    if (Test-Path $mf) { Remove-Item $mf -Force }
}

# Chain A: M1 RL2 (25k steps)
$m1Args = @(
    "run", "python", "-m", "eval.train_ppo_nav",
    "--region_dir", $RegionDir,
    "--model_type", "m1_full",
    "--ckpt", "checkpoints/spatial_finetuned_best.pkl",
    "--temporal_ckpt", "checkpoints/temporal_best.pkl",
    "--task", "rl2",
    "--num_targets", "5",
    "--compass_occlusion", "0.5",
    "--occlusion_burst", "6",
    "--lr", "1e-4",
    "--bc_epochs", "0",
    "--total_steps", "25000",
    "--save_dir", "checkpoints/ppo_nav_m1_rl2",
    "--device", "cpu"
)
if (Test-Path "checkpoints/ppo_nav_m1_overnight/best_ppo.pkl") {
    $m1Args += @("--init_policy", "checkpoints/ppo_nav_m1_overnight/best_ppo.pkl")
}

# Chain B Step 1: M2 RL1 (50k steps)
$m2Rl1Args = @(
    "run", "python", "-m", "eval.train_ppo_nav",
    "--region_dir", $RegionDir,
    "--model_type", "m2_static",
    "--ckpt", $M2Ckpt,
    "--task", "rl1",
    "--bc_epochs", "20",
    "--bc_steps", "600",
    "--total_steps", "50000",
    "--save_dir", "checkpoints/ppo_nav_m2_overnight",
    "--device", "cpu"
)

Write-Host "Launching Chain A (M1 RL2) -> $LogDir/m1_rl2.log" -ForegroundColor Green
$pM1 = Start-Process -FilePath "uv" -ArgumentList $m1Args `
    -RedirectStandardOutput "$LogDir/m1_rl2.log" `
    -RedirectStandardError "$LogDir/m1_rl2.err.log" `
    -PassThru -NoNewWindow

Write-Host "Launching Chain B Step 1 (M2 RL1) -> $LogDir/m2_rl1.log" -ForegroundColor Green
$pM2Rl1 = Start-Process -FilePath "uv" -ArgumentList $m2Rl1Args `
    -RedirectStandardOutput "$LogDir/m2_rl1.log" `
    -RedirectStandardError "$LogDir/m2_rl1.err.log" `
    -PassThru -NoNewWindow
$pM2Rl2 = $null
$m2Rl2Started = $false

Write-Host "Waiting for parallel CPU PPO chains to finish..." -ForegroundColor Cyan

function Get-PpoProgress($metricsPath, $targetUpdates) {
    if (-not (Test-Path $metricsPath)) { return 0.0 }
    try {
        $lines = (Get-Content $metricsPath -ErrorAction SilentlyContinue | Measure-Object -Line).Lines
        return [Math]::Min(100.0, ($lines / [Math]::Max(1, $targetUpdates)) * 100.0)
    } catch { return 0.0 }
}

function Format-Bar($pct, $width = 16) {
    $filled = [int][Math]::Round(($pct / 100.0) * $width)
    $filled = [Math]::Max(0, [Math]::Min($width, $filled))
    return ("=" * $filled) + ("-" * ($width - $filled))
}

$tStart = Get-Date
$lastWidth = 0
while ($true) {
    # Start M2 RL2 automatically when M2 RL1 finishes cleanly
    if (-not $m2Rl2Started -and $pM2Rl1.HasExited) {
        $m2Rl2Started = $true
        if ($pM2Rl1.ExitCode -eq 0) {
            $m2Rl2Args = @(
                "run", "python", "-m", "eval.train_ppo_nav",
                "--region_dir", $RegionDir,
                "--model_type", "m2_static",
                "--ckpt", $M2Ckpt,
                "--init_policy", "checkpoints/ppo_nav_m2_overnight/best_ppo.pkl",
                "--task", "rl2",
                "--num_targets", "5",
                "--compass_occlusion", "0.5",
                "--occlusion_burst", "6",
                "--lr", "1e-4",
                "--bc_epochs", "0",
                "--total_steps", "25000",
                "--save_dir", "checkpoints/ppo_nav_m2_rl2",
                "--device", "cpu"
            )
            $pM2Rl2 = Start-Process -FilePath "uv" -ArgumentList $m2Rl2Args `
                -RedirectStandardOutput "$LogDir/m2_rl2.log" `
                -RedirectStandardError "$LogDir/m2_rl2.err.log" `
                -PassThru -NoNewWindow
        }
    }

    $m1Done = $pM1.HasExited
    $m2Done = $pM2Rl1.HasExited -and (($pM2Rl1.ExitCode -ne 0) -or ($null -ne $pM2Rl2 -and $pM2Rl2.HasExited))

    $p1 = if ($m1Done) { if ($pM1.ExitCode -eq 0) { 100.0 } else { 0.0 } } else { Get-PpoProgress "checkpoints/ppo_nav_m1_rl2/ppo_metrics.jsonl" 195 }
    $p2_1 = if ($pM2Rl1.HasExited -and $pM2Rl1.ExitCode -eq 0) { 100.0 } else { Get-PpoProgress "checkpoints/ppo_nav_m2_overnight/ppo_metrics.jsonl" 390 }
    $p2_2 = if ($null -ne $pM2Rl2 -and $pM2Rl2.HasExited -and $pM2Rl2.ExitCode -eq 0) { 100.0 } else { Get-PpoProgress "checkpoints/ppo_nav_m2_rl2/ppo_metrics.jsonl" 195 }
    $p2 = ($p2_1 * 0.667) + ($p2_2 * 0.333)
    $overall = ($p1 + $p2) / 2.0

    $m1Tag = if ($m1Done -and $pM1.ExitCode -ne 0) { "ERR" } else { "{0,5:N1}%" -f $p1 }
    $m2Tag = if ($pM2Rl1.HasExited -and $pM2Rl1.ExitCode -ne 0) { "ERR" } elseif ($null -ne $pM2Rl2 -and $pM2Rl2.HasExited -and $pM2Rl2.ExitCode -ne 0) { "ERR" } else { "{0,5:N1}%" -f $p2 }

    $elap = ((Get-Date) - $tStart).ToString("hh\:mm\:ss")
    $bAll = Format-Bar $overall 16
    $line = "[$elap] [$bAll] {0,5:N1}% | M1-RL2:$m1Tag | M2:$m2Tag" -f $overall
    try { $cols = [Console]::WindowWidth } catch { $cols = 100 }
    if ($cols -lt 40) { $cols = 100 }
    $maxLen = $cols - 1
    if ($line.Length -gt $maxLen) { $line = $line.Substring(0, $maxLen) }
    $clearW = [Math]::Max($lastWidth, $line.Length)
    [Console]::Write("`r" + (" " * $clearW) + "`r" + $line)
    $lastWidth = $line.Length

    if ($m1Done -and $m2Done) { break }
    Start-Sleep -Seconds 2
}
[Console]::Write("`r" + (" " * $lastWidth) + "`r")
Write-Host ("[{0}] Finished | M1-RL2 exit={1} | M2-RL1 exit={2} | M2-RL2 exit={3}" -f ((Get-Date) - $tStart).ToString("hh\:mm\:ss"), $pM1.ExitCode, $pM2Rl1.ExitCode, $(if ($null -ne $pM2Rl2) { $pM2Rl2.ExitCode } else { "skipped" })) -ForegroundColor Cyan

# Phase 2: All Zero-Retraining Evaluations + RL2 Zero-Shot Nav Eval
$EvalSteps = @(
    @{ Name = "Zero-Retraining Suite"; Cmd = { & "$ScriptDir/run_iris_overnight.ps1" -RegionDir $RegionDir -ResultsDir $ResultsDir -LogFile "$LogDir/evals.log" -EvalEpisodes $EvalEpisodes -Seed $Seed } }
)
if (Test-Path "checkpoints/ppo_nav_m1_rl2/best_ppo.pkl") {
    $EvalSteps += @{ Name = "RL2 Eval (M1)"; Cmd = { & uv run python -m eval.eval_navigation --model_type m1_full --backbone_ckpt checkpoints/spatial_finetuned_best.pkl --temporal_ckpt checkpoints/temporal_best.pkl --policy_ckpt checkpoints/ppo_nav_m1_rl2/best_ppo.pkl --region_dir $RegionDir --task rl2 --num_episodes $EvalEpisodes --seed $Seed --out_json "$ResultsDir/nav_m1_rl2.json" 2>&1 | Tee-Object -FilePath "$LogDir/evals.log" -Append } }
}
if (Test-Path "checkpoints/ppo_nav_m2_rl2/best_ppo.pkl") {
    $EvalSteps += @{ Name = "RL2 Eval (M2)"; Cmd = { & uv run python -m eval.eval_navigation --model_type m2_static --backbone_ckpt checkpoints/m2_spatial_jepa/best.pkl --policy_ckpt checkpoints/ppo_nav_m2_rl2/best_ppo.pkl --region_dir $RegionDir --task rl2 --num_episodes $EvalEpisodes --seed $Seed --out_json "$ResultsDir/nav_m2_rl2.json" 2>&1 | Tee-Object -FilePath "$LogDir/evals.log" -Append } }
}

$totalEvals = $EvalSteps.Count
for ($i = 0; $i -lt $totalEvals; $i++) {
    $pct = (($i) / [Math]::Max(1, $totalEvals)) * 100.0
    Write-Progress -Activity "IRIS 27 Overnight Evaluations" -Status $EvalSteps[$i].Name -PercentComplete $pct
    Write-Host ("`n[Eval {0}/{1}] {2}" -f ($i + 1), $totalEvals, $EvalSteps[$i].Name) -ForegroundColor Cyan
    & $EvalSteps[$i].Cmd
}
Write-Progress -Activity "IRIS 27 Overnight Evaluations" -Completed

Write-Host "`nCPU Overnight Pipeline Complete! Results in $ResultsDir/, Logs in $LogDir/" -ForegroundColor Green
