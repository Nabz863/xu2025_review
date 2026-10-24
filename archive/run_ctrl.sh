#!/bin/bash
cd "$(dirname "$0")"
while [ ! -f runs/ALTS_DONE ]; do sleep 20; done
go() { name=$1; shift; [ -f runs/$name/summary.json ] || OMP_NUM_THREADS=1 python train.py --steps 20000 "$@" --out runs/$name > runs/$name.log 2>&1; }
THEIRS="--K 18 --lam 0.75 --v_init small --B_init small --lattice centres --lr_schedule warmup_decay --grad_clip 10 --loss_scale 40000"
(go ctrl_K18 --K 18; go ctrl_all_theirs $THEIRS) &
(go ctrl_lam075 --lam 0.75) &
(go ctrl_lrsched --lr_schedule warmup_decay; go ctrl_clip --grad_clip 10 --loss_scale 40000) &
wait; touch runs/CTRL_DONE
