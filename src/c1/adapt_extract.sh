#!/usr/bin/env bash
# Pass 1 of the adapt_grid extraction: labels + graphs only, into data/raw/evemt/adapt/ (the benchmark family
# already lives in data/raw/evemt/DoubleLine, so the adapt family must never be extracted there).
# Pass 2 (after the window rule and splits are known): only the needed result files, via a member list.
set -e
cd "$(dirname "$0")/../../data/raw/evemt"
F=adapt_grid-DoubleLine.tar.gz
[ "$(stat -c %s "$F")" -eq 17733535772 ] || { echo "size mismatch"; exit 1; }
mkdir -p adapt
if [ "$1" = "pass2" ]; then
  tar -xzf "$F" -C adapt -T adapt/members.txt
else
  tar -xzf "$F" -C adapt --exclude='DoubleLine/data/*'
fi
find adapt -maxdepth 3 | head -20
