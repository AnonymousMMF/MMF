# MMF Experiment Scripts (Reconstruction / Completion / Expressivity / Identifiability)

This repository contains **standalone Python scripts** used to run MMF experiments:

- **Reconstruction (fully observed synthetic matrix)**: `main_reconstruction.py`
- **Matrix completion (explicit ratings)**: `main_completion.py`
- **Expressivity via singular-value spectrum**: `vis_expressivity.py`
- **Identifiability / factor stability visualization**: `vis_identifiability.py`

> **Note:** Some scripts may still mention older filenames in their internal docstrings (e.g., `iden.py`, `vis_spectrum.py`).  
> In this repo, the correct entrypoints are the filenames listed above.

---

## 1) Setup

### Requirements

- Python **3.9+**
- Recommended: a CUDA-enabled GPU for the larger runs

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

> If you want GPU-enabled PyTorch, install the right CUDA wheel following the official PyTorch instructions, then install the remaining packages.

---

## 2) Quickstart (smoke tests)

These are **tiny** runs meant to verify everything works end-to-end.

- Bash (Linux/macOS/Git-Bash):
```bash
bash scripts/run_all_smoke.sh
```

- PowerShell (Windows):
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_all_smoke.ps1
```

Outputs are written under `outputs/`.

---

## 3) Reconstruction experiment (synthetic, fully observed)

The reconstruction script benchmarks MMF against common baselines (e.g., truncated SVD / MF variants) on a fully observed matrix and logs errors and timing.

### Example runs

- Budget sweep (vary rank budget, fixed `K_ratio`):
```bash
bash scripts/run_reconstruction_budget_sweep.sh
```

Or directly:

```bash
python main_reconstruction.py --device cuda --I 256 --J 256   --R_list 16,32,48,64,80 --K_ratio 0.75 --out_dir outputs/reconstruction_budget
```

---

## 4) Matrix completion (explicit ratings)

This script runs MF vs. MMF on explicit-rating matrix completion and saves per-seed + aggregate JSON results.

Supported datasets include MovieLens and (preprocessed) Flixster/Douban.

### MovieLens (small sanity run)

```bash
bash scripts/run_completion_movielens_small.sh
```

### Flixster / Douban

```bash
bash scripts/run_completion_flixster_douban.sh
```

### Windows note: MovieLens extraction `WinError 32`

On Windows, MovieLens download/extract may fail with a message like:

```
PermissionError: [WinError 32] ... './data_rec\_ml-1m_tmp\ml-1m' -> './data_rec\ml-1m'
```

This happens when **another process holds a file handle** (common with cloud-sync folders like Dropbox/OneDrive, Explorer preview, antivirus scans).

Workarounds (no code change required):
1. Use a local, non-synced directory for `--data_dir` (e.g., `C:\temp\mmf_data`).
2. Make sure the target folder is not open in Explorer and re-run.
3. If a partial folder exists, delete `data_rec\_ml-*_tmp` and `data_rec\ml-*` and re-run with `--download`.

---

## 5) Expressivity (singular value spectrum)

The expressivity script reconstructs synthetic matrices with SVD vs. MMF under a similar parameter budget and plots singular-value spectra.

Run:

```bash
bash scripts/run_expressivity.sh
```

Tip: use `--svdvals_method lowrank` for speed on large matrices.

---

## 6) Identifiability / factor stability visualization

This script trains two independent runs and compares base factors via an absolute cosine-similarity heatmap.

Run:

```bash
bash scripts/run_identifiability.sh
```

If you pass `--reorder`, it uses the Hungarian algorithm (requires SciPy) to correct permutation ambiguity in the visualization.

---

## 7) Output directories

All scripts write results to the specified output directory:

- Reconstruction: `--out_dir ...`
- Completion: `--save_dir ...`
- Expressivity: `--out_dir ...`
- Identifiability: `--out_dir ...`

---

## 8) Reproducibility

All scripts accept seeds (`--seed` or seed variants) and use deterministic seeding utilities where applicable.

---

## License / Citation

Add your preferred license and citation once you decide how you want this code released.
