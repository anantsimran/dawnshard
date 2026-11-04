#!/usr/bin/env zsh
# Tokenize many text files into uint32 shards in parallel, then write the manifest.
# One document per non-empty line. Run it inside screen so it survives a dropped SSH:
#
#   screen -S tokenize scripts/tokenize_shards.sh <out_dir> <merges.txt> <eos_id> <inputs...>
#
# Each input becomes <out_dir>/<input stem>_<n>.bin/.idx; progress goes to
# <out_dir>/tokenize.log. JOBS sets the parallelism (default: all cores).
set -euo pipefail

out_dir=$1 merges=$2 eos_id=$3
shift 3
mkdir -p "$out_dir"
log="$out_dir/tokenize.log"
jobs=${JOBS:-$(nproc)}

printf '%s\n' "$@" | xargs -P "$jobs" -I{} sh -c '
  stem=$(basename "{}"); stem=${stem%.*}
  PYTHONPATH=app/src uv run python -m training.dataload.shards \
    --input "{}" --out "'"$out_dir"'" --prefix "$stem" \
    --merges "'"$merges"'" --eos-id "'"$eos_id"'" \
  && echo "$(date +%T) done {}"
' 2>&1 | tee -a "$log"

PYTHONPATH=app/src uv run python -m training.dataload.shards --input /dev/null \
  --out "$out_dir" --prefix _ --merges "$merges" --eos-id "$eos_id" --manifest | tee -a "$log"
