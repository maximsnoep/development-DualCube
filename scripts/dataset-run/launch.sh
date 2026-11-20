#!/bin/bash
# Run the skeleton pipeline on many meshes, each in its own process with a time limit.
# Usage: launch.sh RUN_DIR MESH_LIST
#   MESH_LIST: one absolute .obj path per line (under .../dataset/<folder>/).
# Env: JOBS meshes at a time (default 4), DISKSCISSORS_JOBS per mesh (default 4),
#      LIMIT seconds per mesh (default 1200). JOBS × DISKSCISSORS_JOBS should not exceed the cores.
set -e
run=$(realpath -m "$1"); list=$(realpath "$2")
here=$(dirname "$(realpath "$0")")
repo=$(git -C "$here" rev-parse --show-toplevel)
mkdir -p "$run"
cargo build --release -q -p cli --manifest-path "$repo/Cargo.toml"
cp "$repo/target/release/cli" "$run/cli"
git -C "$repo" log --oneline -1 > "$run/commit.txt"
git -C "$repo" status --short >> "$run/commit.txt"
cp "$list" "$run/meshes.txt"
export TETGEN_EXECUTABLE=${TETGEN_EXECUTABLE:-$HOME/.local/bin/tetgen}
export DISKSCISSORS_EXECUTABLE=${DISKSCISSORS_EXECUTABLE:-$HOME/.local/bin/diskscissors}
export DISKSCISSORS_JOBS=${DISKSCISSORS_JOBS:-4}
export RUN=$run
xargs -P "${JOBS:-4}" -n1 "$here/one.sh" < "$run/meshes.txt"
python3 "$here/summary.py" "$run" > "$run/summary.txt"
echo finished > "$run/finished.txt"
