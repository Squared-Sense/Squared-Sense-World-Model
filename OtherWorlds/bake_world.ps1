<#
.SYNOPSIS
    Bakes Minecraft .mca region files for a world inside OtherWorlds/<WorldName>/.
.DESCRIPTION
    Converts OtherWorlds/<WorldName>/raw_regions/*.mca into baked memmaps and
    spawn indices under OtherWorlds/<WorldName>/baked_regions/.
    Pass -OverworldOnly to zero out subterranean cave spawns (0% caves).
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$WorldName,
    [int]$MaxRegions = 0,
    [switch]$Overwrite,
    [switch]$OverworldOnly
)

$ErrorActionPreference = "Stop"

$OtherWorldsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $OtherWorldsDir
Set-Location $RepoRoot

$RawDir = Join-Path $OtherWorldsDir "$WorldName/raw_regions"
$BakedDir = Join-Path $OtherWorldsDir "$WorldName/baked_regions"

if (-not (Test-Path $RawDir)) {
    throw "Raw regions folder not found: $RawDir. Place r.X.Z.mca files there first."
}

$McaFiles = Get-ChildItem -Path $RawDir -Filter "r.*.*.mca" -File
if ($McaFiles.Count -eq 0) {
    throw "No r.X.Z.mca files found in $RawDir."
}

New-Item -ItemType Directory -Force -Path $BakedDir | Out-Null

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host " Baking OtherWorld: $WorldName ($($McaFiles.Count) .mca files)" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan

$BakeArgs = @("run", "python", "-m", "env.mca_baker", "--input", $RawDir, "--output", $BakedDir)
if ($MaxRegions -gt 0) { $BakeArgs += @("--limit", $MaxRegions) }
if ($Overwrite) { $BakeArgs += "--overwrite" }

& uv @BakeArgs
if ($LASTEXITCODE -ne 0) { throw "mca_baker failed with exit code $LASTEXITCODE" }

Write-Host "`nIndexing traversable spawns..." -ForegroundColor Cyan
& uv run python -m env.spawn_indexer --region_dir $BakedDir
if ($LASTEXITCODE -ne 0) { throw "spawn_indexer failed with exit code $LASTEXITCODE" }

if ($OverworldOnly) {
    Write-Host "`nApplying -OverworldOnly filter (clearing subterranean cave spawns)..." -ForegroundColor Yellow
    & uv run python -c "
import glob, os, numpy as np
for p in glob.glob(os.path.join(r'$BakedDir', '*.spawns.npz')):
    with np.load(p) as d:
        surf, steep = d['surface'], d['steep']
    np.savez_compressed(p, surface=surf, subterranean=np.zeros((0, 3), dtype=np.int32), steep=steep)
    print('Cleared caves in', os.path.basename(p), '-> surface:', len(surf), 'steep:', len(steep))
"
    if ($LASTEXITCODE -ne 0) { throw "OverworldOnly filter failed with exit code $LASTEXITCODE" }
}

Write-Host "`nDone! Baked world ready at: $BakedDir" -ForegroundColor Green
