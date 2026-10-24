# Running the authors' released code

Repository: github.com/DehongXu/grid-cell-conformal-isometry (released for PyTorch 1.10.1).
Purpose: check, with the authors' own code and their own evaluation, whether it reproduces what the paper reports. Our tables use their configuration rebuilt in our codebase; this is the direct check.

This guide was written from a reading of the repository on 6 October 2026, not by running it. Expect to fix small things; the likely ones are listed.

## 1. Get it, in its own folder and environment

    cd ..                                    # next to xu2025_mine, not inside it
    git clone https://github.com/DehongXu/grid-cell-conformal-isometry.git
    cd grid-cell-conformal-isometry
    python3 -m venv ~/.venvs/xu_authors && source ~/.venvs/xu_authors/bin/activate
    pip install torch                        # default Linux wheel includes CUDA; the authors used a GPU (2080 Ti)
    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
    pip install numpy scipy matplotlib ml_collections absl-py
    ls; ls configs                           # expect main.py, model.py, experiment.py, configs/linear_scale10.py ...

If there is a requirements file, install from it instead, keeping the torch line above.

## 2. Check before running (2 minutes)

    grep -n "cuda\|get_device\|device" *.py | head -30
    grep -n "seed" *.py configs/linear_scale10.py

- **GPU.** Run on the RTX 4060 so the code runs as released: `nvidia-smi` should work inside WSL (needs the NVIDIA Windows driver), and the check above should print True. If CUDA is unavailable, the fallback is to change `.cuda()` / `"cuda"` to `"cpu"` and `x.get_device()` to `x.device`. Write down every line you change: it is part of the reproducibility record.
- **Seed.** Note whether a seed is set and where. If the config has a seed field, runs differ by changing it; if the seed is fixed in code, edit it per run.

## 3. Run

    python main.py --config=configs/linear_scale10.py
    # if absl config flags are used, fields can be overridden on the command line, e.g.
    python main.py --config=configs/linear_scale10.py --config.seed=1

Run linear_scale10 on 3 seeds (0, 1, 2). If time allows: linear_scale5 and linear_scale15 (one seed each) for Table 2.

Each run is 20,000 steps; the paper reports under 15 minutes per run on a 2080 Ti, so record your run time as a check. Our code keeps running on the CPU in parallel.

## 4. Record

For each run, record:
- the gridness and valid rate their code reports (it evaluates with its own scorer), and the estimated scale;
- whether the saved response maps are hexagonal;
- every change you had to make to get it running.

Compare with Table 1 (linear: gridness 1.70, valid rate 100%) and Table 2 (scale 0.82 / 0.41 / 0.27 at s = 5 / 10 / 15).

If it will not run after the device fixes, record the error. "The released code does not run on current PyTorch without changes X, Y" is itself a reproducibility finding.
