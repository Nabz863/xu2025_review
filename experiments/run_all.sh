#!/bin/bash
# Run every job in a jobs file, P at a time, skipping runs that already have a summary.
#   ./experiments/run_all.sh                       # experiments/jobs.txt, 3 in parallel
#   ./experiments/run_all.sh experiments/jobs.txt 1
#   STEPS=50 ./experiments/run_all.sh experiments/jobs_smoke.txt   # quick check that everything runs
# Each line of the jobs file is "name | flags". The shorthands THEIRS, THEIRS15, THEIRS5 and
# RECON expand to the flag sets defined below.
cd "$(dirname "$0")/.."
JOBS=${1:-experiments/jobs.txt}
P=${2:-3}
mkdir -p runs

THEIRS="--K 18 --lam 0.75 --v_init small --B_init small --lattice centres --lr_schedule warmup_decay --grad_clip 10 --loss_scale 40000"
THEIRS15="--K 18 --lam 0.375 --v_init small --B_init small --lattice centres --lr_schedule warmup_decay --grad_clip 10 --loss_scale 80000 --s 15"
THEIRS5="$THEIRS --s 5"
RECON="--K 144 --heading_interp --continuous_heading --lr_schedule gao_step"

LIST=$(mktemp)
grep -v '^#' "$JOBS" | grep '|' | while IFS='|' read -r name flags; do
  name=$(echo $name)
  flags=" $flags "
  flags=${flags// THEIRS15 / $THEIRS15 }
  flags=${flags// THEIRS5 / $THEIRS5 }
  flags=${flags// THEIRS / $THEIRS }
  flags=${flags// RECON / $RECON }
  echo $name $flags   # unquoted: collapses spaces, no trailing blank (xargs -L treats one as a continuation)
done > "$LIST"

export STEPS
xargs -P "$P" -L 1 bash -c '
  name=$0
  [ -f runs/$name/summary.json ] && exit 0
  echo "$(date +%T) start $name: $*"
  OMP_NUM_THREADS=1 python train.py --steps ${STEPS:-20000} "$@" --out runs/$name > runs/$name.log 2>&1
  python gridscore.py runs/$name > /dev/null 2>&1
  echo "$(date +%T) done  $name"' < "$LIST"
rm -f "$LIST"
touch runs/ALL_DONE
