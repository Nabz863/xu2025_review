## Part 1. Reconstruction from the paper and its references

| Condition | Hexagonal runs | Annulus gridness per seed | Mean valid % | Mean final L1 (×10⁻³) | s × scale |
|---|---|---|---|---|---|
| Reconstruction (seeds 0–5) | **4 / 6** | 1.61 / 0.05 / 1.61 / 1.59 / 0.04 / 1.64 | 67 | 7.83 | 3.57 |
| Paper text only, no reference conventions (seeds 0–5) | **4 / 6** | 1.50 / -0.87 / 1.45 / 1.62 / 1.63 / 0.03 | 67 | 3.42 | 3.37 |
| Authors' configuration (seeds 0–3) | **4 / 4** | 1.67 / 1.64 / 1.42 / 1.67 | 100 | 4.35 | 3.23 |

## Part 2. Authors' configuration, one silent choice changed at a time (seeds 0, 1)

| Condition | Hexagonal runs | Annulus gridness per seed | Mean valid % | Mean final L1 (×10⁻³) | s × scale |
|---|---|---|---|---|---|
| Authors' configuration, unchanged | **2 / 2** | 1.67 / 1.64 | 100 | 4.40 | 3.25 |
| Headings: 144 bins, interpolated, continuous (vs 18 bins, nearest) | **2 / 2** | 1.67 / 1.67 | 100 | 5.29 | 3.08 |
| v init: uniform, projected (vs N(0, 0.001)) | **2 / 2** | 1.66 / 1.66 | 100 | 4.53 | 3.15 |
| B init: He (vs N(0, 0.001)) | **2 / 2** | 0.66 / 1.67 | 100 | 4.40 | 3.22 |
| Lattice points on the walls (vs cell centres) | **1 / 2** | -0.19 / 1.61 | 50 | 3.39 | 3.30 |
| Δx uniform in length (vs uniform over the disc) | **2 / 2** | 1.62 / 1.65 | 100 | 3.49 | 3.22 |
| x + Δx may leave the square (vs kept inside) | **2 / 2** | 1.66 / 1.67 | 100 | 15.92 | 3.00 |
| λ = 1 (vs 0.75) | **2 / 2** | 1.68 / 1.68 | 100 | 5.24 | 3.09 |
| No gradient clipping (vs max-norm 10) | **2 / 2** | 1.36 / 1.67 | 100 | 4.67 | 3.16 |

## Part 3. Where the code departs from what the paper states (authors' configuration, seeds 0–2)

| Condition | Hexagonal runs | Annulus gridness per seed | Mean valid % | Mean final L1 (×10⁻³) | s × scale |
|---|---|---|---|---|---|
| Authors' configuration, as released | **3 / 3** | 1.67 / 1.64 / 1.42 | 100 | 4.28 | 3.26 |
| Learning rate constant 0.003, as the paper states (vs warmup + decay) | **2 / 3** | 1.58 / 1.49 / 0.07 | 67 | 3.98 | 3.29 |
| Losses as plain expectations, as eqs. 3–4 state (vs L1 summed, L2 ×30000) | **3 / 3** | 1.37 / 1.67 / 1.55 | 100 | 4.51 | 3.19 |
| s = 15 with the code's range s‖Δx‖ ≤ 1.125 | **1 / 2** | 1.18 / 0.03 | 50 | 4.90 | 3.05 |
| s = 15 with the paper's range s‖Δx‖ ≤ 1.25 | **2 / 2** | 1.19 / 1.18 | 100 | 5.38 | 3.34 |
| s = 5 (Table 2 check) | **1 / 1** | 1.28 | 100 | 3.47 | 3.35 |

## Part 4. The paper's ablations (Fig. 3), on the reconstruction and the authors' configuration (seeds 0, 1)

Paper's claims: removing L1 gives non-hexagonal patterns (Fig. 3h); removing L2 or normalisation disrupts the patterns (3f, 3g); removing non-negativity leaves hexagons (3a, 3e).

| Condition | Hexagonal runs | Annulus gridness per seed | Mean valid % | Mean final L1 (×10⁻³) | s × scale |
|---|---|---|---|---|---|
| Reconstruction, no conformal isometry loss (L1) | **0 / 2** | -0.11 / -0.16 | 0 | 0.00 | 0.80 |
| Reconstruction, no transformation loss (L2) | **2 / 2** | 0.46 / 0.46 | 65 | 2.54 | 3.67 |
| Reconstruction, no normalisation | **1 / 2** | 0.50 / -0.10 | 35 | 0.41 | 5.33 |
| Reconstruction, no non-negativity | **1 / 2** | 0.74 / -0.48 | 35 | 0.82 | 4.44 |
| Authors' configuration, no conformal isometry loss (L1) | **0 / 2** | -0.27 / -0.19 | 2 | 0.00 | 11.44 |
| Authors' configuration, no transformation loss (L2) | **2 / 2** | 0.52 / 0.46 | 65 | 3.26 | 3.54 |
| Authors' configuration, no normalisation | **0 / 2** | -0.02 / 0.23 | 6 | 1.66 | 4.33 |
| Authors' configuration, no non-negativity | **1 / 2** | 0.46 / 0.30 | 52 | 1.45 | 4.66 |
