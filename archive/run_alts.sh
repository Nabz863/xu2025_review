#!/bin/bash
cd "$(dirname "$0")"
go() { name=$1; shift; [ -f runs/$name/summary.json ] || OMP_NUM_THREADS=1 python train.py --steps 20000 "$@" --out runs/$name > runs/$name.log 2>&1; }
(go base_seed1 --seed 1; go radius_seed2 --radius_uniform --seed 2; go K72 --K 72) &
(go base_seed2 --seed 2; go K8 --K 8; go B_small --B_init small) &
(go radius_seed1 --radius_uniform --seed 1; go v_small --v_init small; go lattice_centres --lattice centres) &
wait; touch runs/ALTS_DONE
