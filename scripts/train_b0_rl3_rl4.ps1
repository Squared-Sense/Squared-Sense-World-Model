<#
.SYNOPSIS
    Trains B0 (b0_blind) and RL3/RL4 Dead-Reckoning PPO Policies (with optional post-train eval).
.DESCRIPTION
    Supports either a single (ModelType, Task) run or -RunSuite to train B0 (RL1)
    and fine-tune M1/M2 across rl3_beacon, rl3_flash, rl4_beacon, and rl4_flash.

    Examples:
      # 1. Train B0 (b0_blind) on RL1 (50K steps) + evaluate
      .\scripts\train_b0_rl3_rl4.ps1 -ModelType b0_blind -Task rl1

      # 2. Fine-tune M1 on RL4-Flash (5 targets, 3-tick compass flash per leg)
      .\scripts\train_b0_rl3_rl4.ps1 -ModelType m1_full -Task rl4_flash

      # 3. Run the full B0 + M1/M2 RL3/RL4 training & eval suite
      .\scripts\train_b0_rl3_rl4.ps1 -RunSuite
#>

param(
    [string]$RegionDir = "data/baked_regions",
    [ValidateSet("b0_blind", "m1_full", "m2_static", "b1_lod0", "b2_lod2", "b3_mae")]
    [string]$ModelType = "b0_blind",
    [ValidateSet("rl1", "rl2", "rl3_beacon", "rl3_flash", "rl4_beacon", "rl4_flash")]
    [string]$Task = "rl1",
    [string]$Ckpt = "",
    [string]$TemporalCkpt = "",
    [string]$InitPolicy = "",
    [int]$FlashCompassTicks = -1,
    [int]$BcEpochs = -1,
    [int]$BcSteps = 600,
    [double]$Lr = -1.0,
    [int]$TotalSteps = -1,
    [string]$SaveDir = "",
    [string]$Device = "auto",
    [int]$EvalEpisodes = 20,
    [int]$Seed = 1000,
    [switch]$RunSuite,
    [switch]$SkipEval
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

function Invoke-TrainAndEval {
    param(
        [string]$MType,
        [string]$TaskName,
        [string]$UserCkpt,
        [string]$UserTempCkpt,
        [string]$UserInitPol,
        [int]$UserFlash,
        [int]$UserBcEpochs,
        [int]$UserBcSteps,
        [double]$UserLr,
        [int]$UserSteps,
        [string]$UserSaveDir
    )

    $ModelTag = switch ($MType) {
        "b0_blind"  { "b0" }
        "m1_full"   { "m1" }
        "m2_static" { "m2" }
        default     { $MType }
    }

    # Resolve backbone & temporal checkpoints
    $ResolvedCkpt = $UserCkpt
    if ($MType -ne "b0_blind" -and -not $ResolvedCkpt) {
        if ($MType -eq "m1_full" -and (Test-Path "checkpoints/spatial_finetuned_best.pkl")) {
            $ResolvedCkpt = "checkpoints/spatial_finetuned_best.pkl"
        } elseif ($MType -eq "m2_static" -and (Test-Path "checkpoints/m2_spatial_jepa/best.pkl")) {
            $ResolvedCkpt = "checkpoints/m2_spatial_jepa/best.pkl"
        }
    }

    $ResolvedTemp = $UserTempCkpt
    if ($MType -eq "m1_full" -and -not $ResolvedTemp -and (Test-Path "checkpoints/temporal_best.pkl")) {
        $ResolvedTemp = "checkpoints/temporal_best.pkl"
    }

    # Resolve RL1 init policy when fine-tuning RL2/RL3/RL4
    $ResolvedInit = $UserInitPol
    if (-not $ResolvedInit -and $TaskName -ne "rl1") {
        $Candidates = switch ($MType) {
            "m1_full"   { @("checkpoints/ppo_nav_m1_overnight/best_ppo.pkl", "checkpoints/ppo_nav_m1/best_ppo.pkl") }
            "m2_static" { @("checkpoints/ppo_nav_m2_overnight/best_ppo.pkl", "checkpoints/ppo_nav_m2/best_ppo.pkl") }
            "b0_blind"  { @("checkpoints/ppo_nav_b0_rl1/best_ppo.pkl", "checkpoints/ppo_nav_b0/best_ppo.pkl") }
            default     { @() }
        }
        foreach ($c in $Candidates) {
            if (Test-Path $c) { $ResolvedInit = $c; break }
        }
    }

    $ResolvedSave = if ($UserSaveDir) { $UserSaveDir } else { "checkpoints/ppo_nav_${ModelTag}_${TaskName}" }
    $ResolvedSteps = if ($UserSteps -gt 0) { $UserSteps } elseif ($TaskName -eq "rl1" -or $MType -eq "b0_blind") { 50000 } else { 25000 }
    $ResolvedBc = if ($UserBcEpochs -ge 0) { $UserBcEpochs } elseif ($ResolvedInit) { 0 } else { 20 }
    $ResolvedLr = if ($UserLr -gt 0.0) { $UserLr } elseif ($ResolvedInit) { 1e-4 } else { 3e-4 }

    Write-Host "=======================================================" -ForegroundColor Cyan
    Write-Host " Squared Sense: Train $MType on $TaskName" -ForegroundColor Cyan
    Write-Host "=======================================================" -ForegroundColor Cyan
    Write-Host "  Region Dir:    $RegionDir"
    Write-Host "  Backbone Ckpt: $(if ($ResolvedCkpt) { $ResolvedCkpt } else { '<none>' })"
    Write-Host "  Temporal Ckpt: $(if ($ResolvedTemp) { $ResolvedTemp } else { '<none>' })"
    Write-Host "  Init Policy:   $(if ($ResolvedInit) { $ResolvedInit } else { '<scratch>' })"
    Write-Host "  BC Epochs:     $ResolvedBc (steps=$UserBcSteps) | LR: $ResolvedLr"
    Write-Host "  Total Steps:   $ResolvedSteps"
    Write-Host "  Save Dir:      $ResolvedSave"
    Write-Host "=======================================================" -ForegroundColor Cyan

    $TrainArgs = @(
        "run", "python", "-m", "eval.train_ppo_nav",
        "--region_dir", $RegionDir,
        "--model_type", $MType,
        "--task", $TaskName,
        "--bc_epochs", $ResolvedBc,
        "--bc_steps", $UserBcSteps,
        "--lr", $ResolvedLr,
        "--total_steps", $ResolvedSteps,
        "--save_dir", $ResolvedSave,
        "--device", $Device
    )
    if ($ResolvedCkpt) { $TrainArgs += @("--ckpt", $ResolvedCkpt) }
    if ($MType -eq "m1_full" -and $ResolvedTemp) { $TrainArgs += @("--temporal_ckpt", $ResolvedTemp) }
    if ($ResolvedInit -and (Test-Path $ResolvedInit)) { $TrainArgs += @("--init_policy", $ResolvedInit) }
    if ($UserFlash -ge 0) { $TrainArgs += @("--flash_compass_ticks", $UserFlash) }

    & uv @TrainArgs
    if ($LASTEXITCODE -ne 0) { throw "Training failed for $MType ($TaskName) with exit code $LASTEXITCODE" }

    if (-not $SkipEval) {
        $BestPol = Join-Path $ResolvedSave "best_ppo.pkl"
        if (-not (Test-Path $BestPol)) { $BestPol = Join-Path $ResolvedSave "latest_ppo.pkl" }
        $OutJson = "results/${ModelTag}_${TaskName}.json"

        $EvalArgs = @(
            "run", "python", "-m", "eval.eval_navigation",
            "--region_dir", $RegionDir,
            "--model_type", $MType,
            "--policy_ckpt", $BestPol,
            "--task", $TaskName,
            "--num_episodes", $EvalEpisodes,
            "--seed", $Seed,
            "--out_json", $OutJson
        )
        if ($ResolvedCkpt) { $EvalArgs += @("--backbone_ckpt", $ResolvedCkpt) }
        if ($MType -eq "m1_full" -and $ResolvedTemp) { $EvalArgs += @("--temporal_ckpt", $ResolvedTemp) }
        if ($UserFlash -ge 0) { $EvalArgs += @("--flash_compass_ticks", $UserFlash) }

        Write-Host "`nEvaluating $MType on $TaskName -> $OutJson" -ForegroundColor Yellow
        & uv @EvalArgs
        if ($LASTEXITCODE -ne 0) { throw "Evaluation failed for $MType ($TaskName) with exit code $LASTEXITCODE" }
    }
}

if ($RunSuite) {
    # 1. B0 Blind Compass-Only Baseline on RL1 (50K steps)
    Invoke-TrainAndEval -MType "b0_blind" -TaskName "rl1" -UserCkpt "" -UserTempCkpt "" -UserInitPol "" `
        -UserFlash $FlashCompassTicks -UserBcEpochs $BcEpochs -UserBcSteps $BcSteps -UserLr $Lr -UserSteps $TotalSteps -UserSaveDir ""

    # 2. M1 & M2 Fine-Tuning across RL3 & RL4 Dead-Reckoning Tasks
    foreach ($m in @("m1_full", "m2_static")) {
        foreach ($t in @("rl3_beacon", "rl3_flash", "rl4_beacon", "rl4_flash")) {
            Invoke-TrainAndEval -MType $m -TaskName $t -UserCkpt "" -UserTempCkpt "" -UserInitPol "" `
                -UserFlash $FlashCompassTicks -UserBcEpochs $BcEpochs -UserBcSteps $BcSteps -UserLr $Lr -UserSteps $TotalSteps -UserSaveDir ""
        }
    }
    Write-Host "`nFull B0 + RL3/RL4 Suite Complete!" -ForegroundColor Green
} else {
    Invoke-TrainAndEval -MType $ModelType -TaskName $Task -UserCkpt $Ckpt -UserTempCkpt $TemporalCkpt -UserInitPol $InitPolicy `
        -UserFlash $FlashCompassTicks -UserBcEpochs $BcEpochs -UserBcSteps $BcSteps -UserLr $Lr -UserSteps $TotalSteps -UserSaveDir $SaveDir
}
