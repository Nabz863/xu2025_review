# What the cited references specify

Rule: if Xu et al. (2025) leaves a detail out but cites, or builds directly on, a source that specifies it, we treat the source's choice as the likely intended one. Each row says where the trail leads and whether it ends in a usable specification.

| Gap in Xu et al. (2025) | Where the paper points | What that source says | Verdict |
|---|---|---|---|
| "Valid rate" (Table 1) | Nowhere in the 2025 paper | Gao et al. (2021, Table 1 caption) and Xu et al. (2022): valid = gridness > 0.37, "the same threshold ... as in [Banino et al. 2018]" | Recoverable from the predecessors |
| Where 0.37 comes from | Banino et al. (2018), via the predecessors | Banino Methods: thresholds were set per unit at the 95th percentile of a field-shuffle null distribution, and 0.37 is "the means, over units, of the thresholds obtained". Their Reporting Summary says 99th percentile instead | 0.37 is an average of data-specific thresholds from one study, reused as a fixed constant. Barry & Burgess (2017), the shuffling method Banino cites, recommend per-cell thresholds and show they depend on the gridness variant and field scale |
| Gridness formula | Langston et al. (2010); Sargolini et al. (2006) | Sargolini-style score: autocorrelogram; annulus from the central peak's edge to a circle enclosing the six nearest peaks; min(r60, r120) − max(r30, r90, r150). Langston-style variants take the maximum over expanding annuli (as described in Barry & Burgess 2017 and BNT/opexebo). Banino cites both | Recoverable in outline, but two variants exist and the paper does not say which. Barry & Burgess (2017): absolute values "cannot be compared between papers that use different versions of the Gridness measures" |
| Grid scale | Langston; Sargolini | Distance from the autocorrelogram centre to the six (or three) nearest peaks, averaged; exact peak detection varies by implementation | Recoverable in outline; estimator details vary |
| Table 1 baselines | Banino (2018), Sorscher, Gao (2021) | Gao et al. (2021) Table 1 reports exactly Banino 0.18 / 25.20%, Sorscher 0.48 / 56.10%, Gao 0.90 ± 0.044 / 73.10 ± 1.33% (5 trials). Gao does not say whether the baselines were retrained or how | The 2025 baselines are copied from Gao (2021), not re-run under the 2025 setup or scorer. Gao reported 5 trials with standard deviations; the 2025 paper reports single numbers. CHECK: which Sorscher year the 2025 Table 1 cites; Gao's number predates Sorscher (2023) |
| Δx "uniform" | Nowhere | Gao et al. (2021, suppl.): Δr² uniform on [0, 3] grids, θ uniform, i.e. uniform over the disc | The disc reading is the authors' prior convention |
| Heading bins | Nowhere | Gao (2021) and Xu (2022): 144 directions with linear interpolation between neighbours | Predecessors differ from the 2025 code (18 bins, nearest bin). Not carried over |
| λ | Nowhere | Xu (2022): λ = 0.05 in a different, multi-module model; Gao (2021): weights chosen so terms have "similar magnitudes" | Not transferable |
| lr schedule | Nowhere | Gao (2021): 0.003, halved every 500 iterations after 8,000. Xu (2022): linear decay 0.006 → 0.0003, flat, then to 0 | Both predecessors schedule the learning rate; the 2025 paper states only 0.003. A schedule was the authors' habit |
| Init, gradient clipping | Nowhere | Not stated in either predecessor | Unrecoverable |

## Consequences for our experiments

- Valid rate is now computed as % of cells with gridness > 0.37, on both scorers.
- Our annulus scorer (gridscore.py) is the expanding-annulus variant. On it, the authors' configuration scores 1.67 against Table 1's 1.70, so the expanding variant is the likely one the authors used.
- The lr-schedule result matters more given that both predecessors used one.

## Sources

- Gao, Xie, Wei, Zhu & Wu (2021), NeurIPS: paper and supplementary material, proceedings.neurips.cc
- Xu, Gao, Zhang, Wei & Wu (2022), PMLR v197 (NeurReps workshop)
- Banino et al. (2018), Nature, Methods and Reporting Summary
- Barry & Burgess (2017), "To be a grid cell: shuffling procedures for determining gridness", bioRxiv
- opexebo documentation (BNT gridness implementation)

Not read directly: Sargolini et al. (2006) and Langston et al. (2010) supplementary methods (paywalled). Their procedures are taken from Barry & Burgess (2017) and Banino et al. (2018), who describe and cite them.
