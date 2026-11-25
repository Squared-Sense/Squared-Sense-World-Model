<#
.SYNOPSIS
    Runs the fast 10-second coordinate, reward sign, and BFS oracle sanity diagnostic script.
.DESCRIPTION
    You can edit the default variable values directly in this script,
    or override them by passing parameters, e.g.:
      .\scripts\run_diagnostic.ps1 -RegionDir "data/baked_regions"
#>

param(
    [string]$RegionDir = "data/baked_regions"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

$CmdArgs = @(
    "run", "python", "scratch/diag_check.py"
)

if ($args.Count -gt 0) {
    $CmdArgs += $args
}

& uv @CmdArgs
