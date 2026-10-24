#!/bin/bash
cd "$(dirname "$0")"
go() { name=$1; shift; [ -f runs/$name/summary.json ] || OMP_NUM_THREADS=1 python train.py --steps 20000 "$@" --out runs/$name > runs/$name.log 2>&1; python gridscore.py runs/$name > /dev/null 2>&1; }
THEIRS="--K 18 --lam 0.75 --v_init small --B_init small --lattice centres --lr_schedule warmup_decay --grad_clip 10 --loss_scale 40000"
(go base_seed3 --seed 3; go theirs_seed1 $THEIRS --seed 1; go vsmall_seed1 --v_init small --seed 1) &
(go base_seed4 --seed 4; go theirs_seed2 $THEIRS --seed 2; go vsmall_seed2 --v_init small --seed 2) &
(go base_seed5 --seed 5; go theirs_seed3 $THEIRS --seed 3) &
wait; touch runs/SEEDS_DONE
