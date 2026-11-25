<#
.SYNOPSIS
    Runs the full OtherWorlds cross-biome benchmarking pipeline and generates summary tables & plots.
#>

param(
    [string[]]$Worlds = @(),
    [string[]]$Models = @("m1_full", "m2_static"),
    [string[]]$Tasks = @("rl1", "rl2"),
    [int]$NumEpisodes = 20,
    [int]$Seed = 1000,
    [string]$LogFile = "logs/other_worlds_benchmark.log"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogFile) | Out-Null

if ($Worlds.Count -eq 0) {
    $Worlds = Get-ChildItem -Path "OtherWorlds" -Directory |
        Where-Object { Test-Path (Join-Path $_.FullName "baked_regions") } |
        Select-Object -ExpandProperty Name
}

if ($Worlds.Count -eq 0) {
    throw "No baked worlds found under OtherWorlds/*/baked_regions."
}

$Jobs = @()
foreach ($w in $Worlds) {
    foreach ($m in $Models) {
        foreach ($t in $Tasks) {
            $Jobs += @{ World = $w; Model = $m; Task = $t }
        }
    }
}

$total = $Jobs.Count
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host " OtherWorlds Benchmark Pipeline ($total runs across $($Worlds.Count) worlds)" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan

for ($i = 0; $i -lt $total; $i++) {
    $j = $Jobs[$i]
    $pct = [int](($i / [Math]::Max(1, $total)) * 100)
    Write-Progress -Activity "OtherWorlds Benchmark" -Status "$($j.World) | $($j.Model) | $($j.Task)" -PercentComplete $pct
    Write-Host ("`n[{0}/{1} | {2}%] {3} | {4} | {5}" -f ($i + 1), $total, $pct, $j.World, $j.Model, $j.Task) -ForegroundColor Cyan
    & "$ScriptDir/eval_other_world.ps1" -WorldName $j.World -Mode eval -ModelType $j.Model -Task $j.Task -NumEpisodes $NumEpisodes -Seed $Seed 2>&1 | Tee-Object -FilePath $LogFile -Append
    if ($LASTEXITCODE -ne 0) { throw "Benchmark run failed on $($j.World) $($j.Model) $($j.Task)" }
}
Write-Progress -Activity "OtherWorlds Benchmark" -Completed

Write-Host "`nAggregating CSV, JSON, and Springer LNEE benchmark figures..." -ForegroundColor Cyan
& uv run python -m eval.summarize_other_worlds 2>&1 | Tee-Object -FilePath $LogFile -Append
Write-Host "`nBenchmark complete! Outputs in results/other_worlds/ and paper_figs/other_worlds_benchmark.pdf" -ForegroundColor Green
