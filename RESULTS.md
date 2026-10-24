# Reproducing Xu et al. (ICLR 2025): results

Linear model, s = 10 unless stated, 20,000 steps, 24 cells. A run counts as **hexagonal** if its median gridness (expanding-annulus scorer, `gridscore.py`) exceeds 0.37, the valid-cell threshold the authors' predecessors use. Regenerate every table below with:

    python experiments/make_tables.py > experiments/tables.md

The tables marked "fill from tables.md" are produced by the runs in `experiments/jobs.txt`.

---

## Part 1. Reconstruction from the paper and its references

**Question:** can a careful reader build the model from the paper, following its citations where it is silent?

**Method:** every axis in `reconstruction_choices.md`. Where the paper is silent we adopt the convention of the work it cites or builds on (Gao et al. 2021; Xu et al. 2022); where those are silent too, we choose and say so.

**Results:** fill from tables.md, Part 1 (reconstruction, seeds 0–5; compared with the paper-text-only version and the authors' configuration).

**Already established (development runs, archive/):**
- The paper text alone, without reference conventions, gives hexagons in 4 of 6 seeds (seeds 1 and 5 fail).
- Failures are whole-module: every cell takes a square-like lattice (negative gridness, four-fold autocorrelogram). Failed runs end with a higher L1 (3.7×10⁻³ vs 3.2–3.3×10⁻³), so they are worse local minima. That is consistent with §4's argument that the hexagon minimises the loss.

## Part 2. The authors' configuration, varied one silent axis at a time

**Question:** which of the choices the paper leaves out actually matter?

**Method:** start from the authors' released configuration, rebuilt in our code. It reproduces Table 1: annulus gridness 1.67, 1.64, 1.42 and 1.67 on seeds 0–3 against the paper's 1.70. Change one silent axis to the alternative listed in `reconstruction_choices.md`, keep everything else, and run seeds 0 and 1.

**Results:** fill from tables.md, Part 2.

## Part 3. Seeds, and where the code departs from the paper

### 3a. Seed sensitivity

| Configuration | Hexagonal runs |
|---|---|
| Paper text only (archive) | 4 / 6 |
| Reconstruction | fill from tables.md |
| Authors' configuration | 4 / 4 |

The paper reports 100% valid rate with no number of runs or variance. Its predecessor (Gao et al. 2021) reported 5 trials with standard deviations.

Paired rescue test (archive/runs_dev/rescue_*): on the two failing paper-text seeds, adding the authors' learning-rate schedule rescued both, as did λ = 0.75. Other single changes rescued one or neither. With only two failing seeds this indicates, rather than establishes, which choices make the result reliable.

### 3b. What the paper states vs what the code does

Each row runs the authors' configuration with one element switched to what the paper says, on seeds 0–2.

| Paper states | Code does | Effect |
|---|---|---|
| Learning rate 0.003 (App. A) | Warmup, then linear decay to 0 | fill from tables.md, Part 3 |
| L1 and L2 are expectations (eqs. 3–4) | L1 summed, L2 averaged ×30000; with clipping at max-norm 10, the clip binds on every step | fill from tables.md, Part 3 |
| s‖Δx‖ ≤ 1.25 for all s | 1.125 at s = 15 | fill from tables.md, Part 3 (also gives s × scale at s = 5, 10, 15) |
| 200,000 iterations | 20,000 | Not tested (compute) |

### 3c. Metrics

- **Gridness:** the expanding-annulus scorer gives values on the paper's scale (1.67 vs 1.70). NeuralPlayground's GridScorer reads about 25% lower on every run. Barry & Burgess (2017) note that gridness variants cannot be compared across papers, and the 2025 paper does not say which variant it used.
- **s × scale:** 3.1–3.5 in every condition, including the authors' configuration, against the paper's 4.1. The authors' own setup gives the same value as ours, so the gap lies in the scale estimator, not the model.
- **Table 1 baselines** (Banino 0.18 / 25.2%, Sorscher 0.48 / 56.1%, Gao 0.90 / 73.1%) are identical to Gao et al. (2021) Table 1. They appear to be copied rather than re-run under the 2025 setup.
