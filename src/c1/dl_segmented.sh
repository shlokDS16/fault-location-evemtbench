#!/usr/bin/env bash
# Segmented, size-checked, resumable download from the FAU share (server honours byte ranges; 4 parallel
# ranges measured 1.4 MB/s vs 0.6 MB/s single on 2026-10-04). Usage: dl_segmented.sh <file> <bytes> [K]
# Each segment k is fetched into <file>.partK; a segment is resumed from its current size until it holds
# exactly its range. Then the parts are concatenated and the final byte count is verified.
cd "$(dirname "$0")/../../data/raw/evemt" || exit 1
F=$1; EXP=$2; K=${3:-8}
URL=https://data.fau.de/share/0e8d60feb7e65616c60aab78b93db77053275da53fd894bf5b75fc5e9ee7dfbf/$F
SEG=$(( (EXP + K - 1) / K ))
# never run two writers on the same .part files: wait for curls left over from an earlier (killed) run
while tasklist | grep -qi curl.exe; do echo "$(date +%T) waiting for running curl"; sleep 30; done

seg() {
  local k=$1 a=$(( $1 * SEG )) b=$(( ($1 + 1) * SEG - 1 ))
  [ $b -ge $EXP ] && b=$(( EXP - 1 ))
  local len=$(( b - a + 1 )) P="$F.part$1"
  for t in $(seq 1 500); do
    local s=$(stat -c %s "$P" 2>/dev/null || echo 0)
    [ "$s" -eq "$len" ] && { echo "seg $k done"; return 0; }
    [ "$s" -gt "$len" ] && { echo "seg $k OVERSIZE $s > $len"; return 2; }
    curl -sf --speed-limit 1000 --speed-time 60 -r $(( a + s ))-$b "$URL" >> "$P"
    sleep 3
  done
  echo "seg $k gave up"; return 3
}

for k in $(seq 0 $(( K - 1 ))); do seg $k & done
wait
for k in $(seq 0 $(( K - 1 ))); do
  ex=$SEG; [ $k -eq $(( K - 1 )) ] && ex=$(( EXP - (K - 1) * SEG ))
  [ "$(stat -c %s "$F.part$k")" -eq "$ex" ] || { echo "segment $k incomplete"; exit 3; }
done
cat $(for k in $(seq 0 $(( K - 1 ))); do echo "$F.part$k"; done) > "$F"
s=$(stat -c %s "$F")
if [ "$s" -eq "$EXP" ]; then rm -f "$F".part*; echo "DONE $F $s"; else echo "FINAL SIZE MISMATCH $s"; exit 4; fi
