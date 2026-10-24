#!/bin/bash
cd "$(dirname "$0")"
go() { name=$1; shift; [ -f runs/$name/summary.json ] || OMP_NUM_THREADS=1 python train.py --steps 20000 "$@" --out runs/$name > runs/$name.log 2>&1; python gridscore.py runs/$name > /dev/null 2>&1; }
(for sd in 1 5; do go rescue_lrsched_s$sd --lr_schedule warmup_decay --seed $sd; go rescue_clip_s$sd --grad_clip 10 --loss_scale 40000 --seed $sd; done; go vsmall_seed5 --v_init small --seed 5) &
(for sd in 1 5; do go rescue_lam075_s$sd --lam 0.75 --seed $sd; go rescue_K18_s$sd --K 18 --seed $sd; done) &
(for sd in 1 5; do go rescue_lattice_s$sd --lattice centres --seed $sd; go rescue_Bsmall_s$sd --B_init small --seed $sd; done) &
wait; touch runs/RESCUE_DONE
