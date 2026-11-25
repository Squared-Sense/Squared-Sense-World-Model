#!/usr/bin/env bash
# ==============================================================================
# Bake .mca region files for a world under OtherWorlds/<world_name>/
# Usage:
#   ./OtherWorlds/bake_world.sh desert_biome
#   OVERWORLD_ONLY=1 ./OtherWorlds/bake_world.sh forest_biome
# ==============================================================================

set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 <world_name> [max_regions]"
  exit 1
fi

WORLD_NAME="$1"
MAX_REGIONS="${2:-0}"
OVERWRITE="${OVERWRITE:-0}"
OVERWORLD_ONLY="${OVERWORLD_ONLY:-0}"

OTHER_WORLDS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$OTHER_WORLDS_DIR")"
cd "$REPO_ROOT"

RAW_DIR="$OTHER_WORLDS_DIR/$WORLD_NAME/raw_regions"
BAKED_DIR="$OTHER_WORLDS_DIR/$WORLD_NAME/baked_regions"

if [[ ! -d "$RAW_DIR" ]]; then
  echo "Error: Raw regions folder not found: $RAW_DIR"
  exit 1
fi

mkdir -p "$BAKED_DIR"

BAKE_CMD=(uv run python -m env.mca_baker --input "$RAW_DIR" --output "$BAKED_DIR")
if [[ "$MAX_REGIONS" -gt 0 ]]; then
  BAKE_CMD+=(--limit "$MAX_REGIONS")
fi
if [[ "$OVERWRITE" == "1" ]]; then
  BAKE_CMD+=(--overwrite)
fi

"${BAKE_CMD[@]}"
uv run python -m env.spawn_indexer --region_dir "$BAKED_DIR"

if [[ "$OVERWORLD_ONLY" == "1" ]]; then
  uv run python -c "
import glob, os, numpy as np
for p in glob.glob(os.path.join('$BAKED_DIR', '*.spawns.npz')):
    with np.load(p) as d:
        surf, steep = d['surface'], d['steep']
    np.savez_compressed(p, surface=surf, subterranean=np.zeros((0, 3), dtype=np.int32), steep=steep)
    print('Cleared caves in', os.path.basename(p), '-> surface:', len(surf), 'steep:', len(steep))
"
fi

echo "Done! Baked world ready at: $BAKED_DIR"
