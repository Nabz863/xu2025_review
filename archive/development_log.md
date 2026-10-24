# Implementation decisions: what the paper does not specify

Each row is a choice the paper (arXiv 2405.16865v4) leaves open, made while implementing from the paper alone, before comparing with the authors' code. These rows are the evidence for the Reproducibility section.

| # | Item | What the paper says | Our choice | Why |
|---|------|---------------------|------------|-----|
| 1 | λ in L = L1 + λL2 | "λ > 0 is a hyper-parameter" (§3.2); no value anywhere | 1.0 by default; swept | Neutral starting point; the sweep measures sensitivity |
| 2 | Heading bins for B(θ) | "discrete set of directions θ"; count not given | K = 16 (22.5°) | Balance of direction resolution and samples per matrix |
| 3 | Parameterisation of B(θ) | Not stated | Lookup table, one matrix per bin, nearest-bin rounding | Simplest reading of "discrete set" |
| 4 | Init of v | Not stated | Uniform [0,1), projected once before training | Starts inside Assumptions 3–4 |
| 5 | Init of B, A, b | Not stated | He init, std √(2/d); b = 0 | Standard for weight matrices |
| 6 | Lattice placement | "overlay a 40×40 lattice" (§3.2) | Corner points on the walls, spacing 1/39 m | Every position lies between lattice points |
| 7 | Distribution of Δx | "uniform" (§3.2) | Uniform over the disc: ‖Δx‖ = r√U, θ ~ U(0, 2π) | "Uniform" read as uniform in area. Matches reference code; chosen with prior knowledge of it. The predecessor Gao et al. (2021, suppl.) states Δr² uniform, i.e. the same disc reading, but the 2025 paper does not. Alternative tested: **matters** (see run log) |
| 8 | Heading for L2 samples | Not stated | Drawn from the K bin centres | B(θ) is only defined there. Matches reference code; chosen with prior knowledge of it. Alternative tested: does not matter |
| 9 | Positions leaving the square | Not stated | x drawn so that x + Δx stays inside | Avoids extrapolating outside the lattice. Matches reference code; chosen with prior knowledge of it. Alternative tested: does not matter for patterns |
| 10 | Learning rate schedule | "learning rate 0.003" (App. A) | Constant 0.003 | As stated |
| 11 | Gradient clipping | Not mentioned | None | As stated |
| 12 | Training length | 200,000 iterations (App. A) | 200,000 default; 20,000 for sweeps | CPU time |
| 13 | Gridness, valid rate, scale | Gridness cited to Langston (2010) and Sargolini (2006), no formula; valid rate never defined in the 2025 paper | Two scorers: NeuralPlayground GridScorer, and our annulus scorer (gridscore.py, Sargolini-style, max over annulus sizes as in Banino et al. 2018). Valid = gridness > 0.37 | The 0.37 threshold is stated in the predecessors Gao et al. (2021) and Xu et al. (2022), citing Banino et al. (2018), not in the 2025 paper |
| 14 | Ablation "remove L1" | Not stated how | Weight on L1 set to 0 | Removes the term entirely |

## Run log

| Date | Run | Settings | Result |
|------|-----|----------|--------|
| 2026-10-23 | runs/lam1_20k | linear, s = 10, λ = 1, 20k steps, constant lr, no clipping, all choices above | Hexagons in all 24 cells. Final L1 3.2e-3, L2 5.1e-3. NPG gridness median 1.15 (24/24 > 0); scale 0.35 m, s × scale 3.5 (paper 4.1) |
| 2026-10-23 | runs/alt_radius_uniform | as lam1_20k, but ‖Δx‖ uniform in length (`--radius_uniform`) | Hexagon-like but distorted. Gridness median 0.42 (vs 1.15); 24/24 > 0; s × scale 3.3. Uniform-in-length oversamples short Δx, so less of the loss comes from the finite range that §3.3–§4 say selects hexagons. The paper's "uniform" is therefore a consequential under-specification |
| 2026-10-23 | runs/alt_continuous_heading | as lam1_20k, but L2 headings continuous, rounded to nearest bin (`--continuous_heading`) | Hexagons. Gridness median 1.24; L2 higher (9.4e-3 vs 5.1e-3) from the rounding error, as expected |
| 2026-10-23 | runs/alt_leave_square | as lam1_20k, but x uniform over the square, x+Δx may leave it, border padding (`--leave_square`) | Hexagons. Gridness median 1.17; L1 higher (1.1e-2 vs 3.2e-3) because v is flat beyond the walls, which breaks isometry there |


## Controlled comparison (2026-10-24): our paper-only configuration vs the authors' code, one difference at a time

All runs: linear model, s = 10, 20k steps. "Ours" = every choice in the table above. Gridness is the median over 24 cells; valid = % of cells with NPG gridness > 0.37. Annulus = gridscore.py.

| Run | Change from ours | Seed | NPG gridness | Annulus gridness | Valid % | s × scale |
|---|---|---|---|---|---|---|
| lam1_20k / base_seed1 / base_seed2 | none | 0 / 1 / 2 | 1.15 / −1.19 / 1.15 | – / −0.87 / 1.45 | 100 / 0 / 100 | 3.5 / 3.3 / 3.4 |
| ctrl_K18 | K = 18 (theirs) | 0 | 0.81 | 1.00 | 100 | 3.1 |
| ctrl_lam075 | λ = 0.75 (theirs) | 0 | 1.21 | 1.64 | 100 | 3.3 |
| v_small | v init N(0, 0.001), not pre-projected (theirs) | 0 | 0.10 | 0.14 | 0 | 3.2 |
| B_small | B init N(0, 0.001) (theirs) | 0 | 1.10 | 1.48 | 100 | 3.5 |
| lattice_centres | lattice at cell centres, zero padding (theirs) | 0 | 1.09 | 1.36 | 100 | 3.4 |
| ctrl_lrsched | warmup 3k / decay from 6k (theirs) | 0 | 1.19 | 1.66 | 100 | 3.1 |
| ctrl_clip | grad clip 10 at loss scale 40000 (theirs) | 0 | 1.16 | 1.55 | 100 | 3.3 |
| ctrl_all_theirs | all of the above together | 0 | 1.23 | **1.67** | 100 | 3.3 |
| K8 / K72 | K = 8 / 72 (neither) | 0 | 1.22 / 1.23 | 1.64 / 1.66 | 100 / 100 | 3.4 / 3.2 |
| alt_radius_uniform / radius_seed1 / radius_seed2 | ‖Δx‖ uniform in length (neither) | 0 / 1 / 2 | 0.42 / −1.15 / 0.63 | – / −0.83 / 0.72 | 67 / 0 / 100 | 3.3 / 3.3 / 3.5 |
| alt_continuous_heading | L2 headings continuous (neither) | 0 | 1.24 | – | 100 | 3.4 |
| alt_leave_square | x+Δx may leave the square (neither) | 0 | 1.17 | – | 100 | 3.3 |

What this shows so far:
1. The authors' configuration, rebuilt in our code, scores 1.67 on the annulus scorer against the paper's 1.70. NeuralPlayground's scorer reads about 25% lower on every run. The gridness gap to Table 1 is the metric, not the model.
2. Seed matters. Our configuration gave hexagons on seeds 0 and 2 and a square-like lattice on seed 1 (negative gridness on both scorers, autocorrelogram with four-fold structure). Its L1 was higher (3.7e-3 vs 3.2–3.3e-3), consistent with §4's claim that the hexagon minimises the loss: seed 1 found a worse local minimum. Single-seed comparisons above are therefore indicative only.
3. s × scale is 3.1–3.5 in every condition, including all-theirs, against the paper's 4.1. Since the authors' own configuration gives the same value, the gap is the scale estimator, not the model.
4. v init alone (theirs, tiny and not pre-projected) gave no grid cells at seed 0, yet the full authors' configuration works. The lr warmup plausibly rescues the tiny init. Needs seeds before it is claimed.
5. Δx uniform in length: 1 of 3 seeds clean hexagons, 1 square-like, 1 weak. Worse than the disc reading, but not yet separable from seed variance without more seeds.

Still needed before the review states any of 2–5 firmly: 3 seeds each for ours, all-theirs, v_small, and radius-uniform.

## Seed study (2026-10-24)

A run counts as hexagonal if its median gridness exceeds 0.37 (the predecessors' valid-cell threshold; see references_audit.md). Failures were always whole-module: every cell square-like, never a mix.

| Configuration | Seeds | Hexagonal runs | Notes |
|---|---|---|---|
| Ours (paper-only) | 0–5 | **4 / 6** | Seeds 1 and 5 fail; both end with L1 3.7e-3 vs 3.2–3.3e-3 for successes, i.e. worse local minima |
| Authors' full configuration | 0–3 | **4 / 4** | Annulus gridness 1.42–1.67 (paper: 1.70) |
| Ours + their v init | 0–2, 5 | 2 / 4 | Fails on seeds 0 and 5 |
| Ours + Δx uniform in length | 0–2 | 2 / 3 | Seed 1 fails; successes are weaker (0.42, 0.63) |

### Paired rescue test on the two failing seeds

Same seed, one of the authors' choices added. The lr schedule, clipping and λ leave the initial values and the sampled batches identical, so those comparisons differ in that one choice only.

| Added choice | Seed 1 (NPG / annulus) | Seed 5 (NPG / annulus) | Rescued |
|---|---|---|---|
| none (ours) | −1.19 / −0.87 | −0.19 / 0.03 | 0 / 2 |
| lr warmup and decay | 0.99 / 1.27 | 1.20 / 1.67 | **2 / 2** |
| λ = 0.75 | 1.12 / 1.44 | 1.21 / 1.64 | **2 / 2** |
| lattice at cell centres | 0.71 / 0.89 | 0.69 / 0.81 | 2 / 2, weaker patterns |
| K = 18 | 0.44 / 0.56 | 1.21 / 1.64 | 2 / 2, seed 1 weak |
| grad clip at their loss scale | −1.16 / −0.77 | 0.84 / 1.01 | 1 / 2 |
| B init N(0, 0.001) | −0.15 / 0.02 | 1.21 / 1.64 | 1 / 2 |
| v init N(0, 0.001) | 1.15 / 1.50 | −0.11 / 0.00 | 1 / 2 |

Reading:
1. From the paper's description alone, hexagons emerge in about two thirds of runs. The authors' configuration produced them in every run tried. The paper's "100% valid rate" is reported without a seed count or variance, while the predecessor (Gao et al. 2021) reported 5 trials with standard deviations.
2. Which seeds fail depends on the optimisation path: almost any change can flip a given seed. The two choices that rescued both failing seeds cleanly were the learning-rate schedule (stated in the paper as a constant 0.003, but scheduled in the code and in both predecessors) and λ = 0.75 (never given in the paper).
3. This is n = 2 failing seeds. Before the review claims (2) firmly: run ours + lr schedule, and ours + λ = 0.75, on seeds 0–5, and count hexagonal runs.
