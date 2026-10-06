#!/usr/bin/env bash
# Size-checked resumable download of adapt_grid-DoubleLine.tar.gz (FAU drops connections).
# Waits for any curl already writing the file, then resumes with -C - until the byte count matches.
cd "$(dirname "$0")/../../data/raw/evemt" || exit 1
F=adapt_grid-DoubleLine.tar.gz
EXP=17733535772
URL=https://data.fau.de/share/0e8d60feb7e65616c60aab78b93db77053275da53fd894bf5b75fc5e9ee7dfbf/adapt_grid-DoubleLine.tar.gz
while tasklist | grep -qi curl.exe; do sleep 30; done
for k in $(seq 1 200); do
  s=$(stat -c %s "$F" 2>/dev/null || echo 0)
  echo "$(date +%T) try $k size $s"
  if [ "$s" -eq "$EXP" ]; then echo "DONE size $s"; exit 0; fi
  if [ "$s" -gt "$EXP" ]; then echo "ERROR oversize $s"; exit 2; fi
  curl -sL -C - --retry 3 --retry-delay 10 -o "$F" "$URL"
  sleep 5
done
echo "GAVE UP size $(stat -c %s "$F")"; exit 3
