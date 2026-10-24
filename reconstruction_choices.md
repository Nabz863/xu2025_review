# Reconstruction choices, axis by axis

Every choice the model needs, where the paper (Xu et al., ICLR 2025, arXiv 2405.16865v4) and its references stand on it, what our reconstruction uses, and what the authors' released code uses. "Silent" means the 2025 paper does not specify it. Rows marked **Part 2** are varied one at a time in the authors' configuration; rows marked **Part 3** are places where the code departs from what the paper states.

| Axis | 2025 paper | References (see references_audit.md) | Reconstruction (Part 1) | Authors' code | Tested |
|---|---|---|---|---|---|
| Model, losses, projections | §3.1–3.2: linear F, L1, L2, clamp then normalise | – | As stated | As stated | – |
| d, lattice size, optimiser, batch | 24, 40×40, Adam, 4000 (§3.4, App. A) | – | As stated | As stated | – |
| Learning rate | "0.003" (App. A) | Gao (2021): 0.003, halved every 500 iterations after 8,000. Xu (2022): scheduled | 0.003 with Gao's step decay | Warmup 3k, constant to 6k, linear decay to 0 | **Part 3** (constant, as stated) |
| Training length | 200,000 iterations (App. A) | – | 20,000 (compute) | 20,000 | Not tested |
| Loss form | Both terms written as E[·] (eqs. 3–4), L = L1 + λL2 | – | Both mean-reduced | L1 summed over batch; L2 averaged ×30000. Overall scale 40,000 | **Part 3** (plain expectations) |
| λ | Silent ("λ > 0") | Gao (2021): weights set so terms have "similar magnitudes" | 1 (L1 and L2 are of similar size in our runs) | Effectively 0.75 | **Part 2** |
| Heading directions for B(θ) | Silent ("discrete set") | Gao (2021), Xu (2022): 144 directions, linear interpolation | 144, interpolated; L2 headings continuous | 18, nearest bin; L2 headings on bin centres | **Part 2** |
| Δx distribution | "uniform" | Gao (2021): Δr² uniform, i.e. uniform over the disc | Uniform over the disc | Uniform over the disc | **Part 2** (uniform in length) |
| Samples leaving the square | Silent | Gao (2021): pairs kept only if both points lie in the domain | Draw x so x + Δx stays inside | Same | **Part 2** (allowed to leave) |
| Lattice placement | "overlay a 40×40 lattice" | Silent | Points on the walls (spacing 1/39 m) | Points at cell centres, zero padding outside | **Part 2** |
| v init | Silent | Silent | Uniform [0,1), projected before training | N(0, 0.001) | **Part 2** |
| B init | Silent | Silent | He, std √(2/d) | N(0, 0.001) | **Part 2** |
| Gradient clipping | Not mentioned | Not mentioned | None | Max-norm 10 (binds on every step at their loss scale) | **Part 2** |
| Isometry range at s = 15 | s‖Δx‖ ≤ 1.25 | – | 1.25 | 1.125 | **Part 3** |
| Gridness | Cited to Langston (2010), Sargolini (2006) | Two variants; expanding annulus (Langston/Banino) likely | gridscore.py (expanding annulus) and NeuralPlayground GridScorer | Not released as a stated method | – |
| Valid rate | Undefined | Gao (2021), Xu (2022): gridness > 0.37, from Banino (2018) | Cells with annulus gridness > 0.37 | – | – |
| Seeds / trials | Not reported | Gao (2021): 5 trials, mean ± sd | 6 seeds | – | **Part 3** (seed study) |
