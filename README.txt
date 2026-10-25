Reproduction of Xu, Gao, Zhang, Wei & Wu (ICLR 2025),
"On Conformal Isometry of Grid Cells: Learning Distance-Preserving Position Embedding"
COMS4070A / COMS7081A review project, University of the Witwatersrand

WHAT THIS IS
  One codebase that can run (1) our reconstruction of the model from the paper and its
  references, (2) the authors' released configuration, and anything in between, one flag at
  a time. RESULTS.md explains the three parts of the study. reconstruction_choices.md lists
  every modelling choice. references_audit.md traces what the cited papers specify.

SETUP (Linux / WSL)
  python3 -m venv ~/.venvs/xu2025 && source ~/.venvs/xu2025/bin/activate
  pip install torch --index-url https://download.pytorch.org/whl/cpu    (or the CUDA wheel)
  pip install -r requirements.txt

CHECK IT WORKS (about 2 minutes)
  python model.py                                                     # self-tests
  STEPS=50 ./experiments/run_all.sh experiments/jobs_smoke.txt 3      # one tiny run of each kind
  rm -rf runs/A_recon_s0 runs/B_headings_s0 runs/C_lrconst_s0 runs/C_s15paper_s0 runs/C_s5_s0 runs/base_seed0 runs/ALL_DONE

RUN EVERYTHING
  ./experiments/run_all.sh experiments/jobs.txt 3
  34 runs of 20,000 steps. About 9 minutes per run on one core of a laptop, so roughly
  1.5-2 hours with 3 in parallel on a 4-core machine. Set the last number to (cores - 1).
  Finished runs are skipped, so it is safe to stop and restart.
  Progress: runs/run_all.out (if started with "> runs/run_all.out") or ls runs/

  To split the work between two machines, cut jobs.txt into two files and give each
  person one. Copy the finished runs/ folders back together before making tables.

MAKE THE TABLES AND FIGURES
  python experiments/make_tables.py > experiments/tables.md
  python experiments/make_figures.py --authors authors_code         # writes figures/figA*.pdf/png
  make_figures.py also reads the lambda sweep (runs/lam*_s0, lam*_s1) and the authors' scores
  (authors_code/authors_scores.json, see below). Missing runs are listed and skipped.

FILES
  model.py                     the model: embedding table, lookup, transformation, losses, sampling
  train.py                     one training run; every modelling choice is a flag (python train.py -h)
  gridscore.py                 expanding-annulus gridness (Sargolini/Langston style, as in Banino 2018)
  experiments/jobs.txt         every run in the study, by part (A_, B_, C_ prefixes)
  experiments/run_all.sh       runs a jobs file in parallel; defines THEIRS and RECON flag sets
  experiments/make_tables.py   builds the result tables in RESULTS.md
  experiments/make_figures.py  builds the appendix figures from the run folders
  experiments/score_authors.py scores checkpoints from the authors' released code with gridscore.py
  experiments/jobs_ablations.txt  the paper's Fig. 3 ablations (Part 4)
  experiments/jobs_nonlinear.txt  nonlinear model 1 (ReLU) on the reconstruction (Part 5)
  authors_code/changes_to_released_code.diff  every change needed to run the authors' repository
  authors_code/authors_scores.json            scores of our 15 runs of the authors' released code
  tex/review.tex, tex/ccn_style.bib, tex/ccn.sty   the review itself (figures are read from figures/)
  runs/                        seed-study runs already done (paper text only, authors' configuration)
  archive/                     development runs and log from building the codebase

UNITS
  Positions are in metres in the 1 m square. v[:, i, j] is lattice point (x index i, y index j).
  Rate maps are plotted with x horizontal.

ADDED (24 Oct): ABLATIONS AND THE AUTHORS' OWN CODE
  Part 4, the paper's Fig. 3 ablations on both the reconstruction and the authors' configuration:
    ./experiments/run_all.sh experiments/jobs_ablations.txt 3 > runs/run_ablations.out 2>&1 &
    16 runs, about 50 minutes with 3 in parallel. make_tables.py now prints a Part 4 table.
  The authors' released repository, run directly: see THEIR_REPO.md.

THE AUTHORS' RELEASED CODE (github.com/DehongXu/grid-cell-conformal-isometry)
  Run in its own environment, next to this folder. It needed these changes to run on current
  libraries (full diff in authors_code/changes_to_released_code.diff):
    - install tensorboard (missing from its dependencies)
    - matplotlib: cm.get_cmap -> colormaps[...].resampled, tostring_rgb -> buffer_rgba (utils.py)
    - x.get_device() -> x.device (returns -1 on CPU)
    - model.py: the w_trans = 0 path returned one value where three are expected (ablation crash)
    - seeds read from an environment variable (they are hard-coded as released)
  Ablations: the same code with each loss or projection switched off in the config.
  Score the checkpoints with the same gridness used for our runs:
    python experiments/score_authors.py ../authors_runs
    cp ../authors_runs/authors_scores.json authors_code/      # make_figures.py reads it from here
  Run time: about 2 minutes per run on an RTX 4060 laptop GPU, about 11 minutes on one CPU
  (we ran 2 seeds on the GPU and the other 13 runs on CPU).