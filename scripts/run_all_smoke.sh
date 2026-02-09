#!/usr/bin/env bash
set -euo pipefail

# Very small runs intended for quick "does it run?" checks.
mkdir -p outputs/smoke

python main_reconstruction.py --device cpu --I 128 --J 128 --R_list 16 --K_ratio 0.5 --epochs 2000 --out_dir outputs/smoke/reconstruction
python vis_expressivity.py --device cpu --out_dir outputs/smoke/expressivity --R_budget 20 --K_list 2,3,4,5 --mmf_epochs 2000 --svdvals_method lowrank
python vis_identifiability.py --device cpu --out_dir outputs/smoke/identifiability --I 128 --J 128 --R 16 --K 2 --epochs 2000
