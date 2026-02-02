$ErrorActionPreference = "Stop"

# Very small runs intended for quick "does it run?" checks.
New-Item -ItemType Directory -Force -Path "outputs/smoke" | Out-Null

python main_reconstruction.py --device cpu --I 128 --J 128 --R_list 16 --K_ratio 0.5 --epochs 50 --out_dir outputs/smoke/reconstruction
python vis_expressivity.py --device cpu --out_dir outputs/smoke/expressivity --R_budget 20 --K_list 2 --mmf_epochs 100 --svdvals_method lowrank
python vis_identifiability.py --device cpu --out_dir outputs/smoke/identifiability --I 128 --J 128 --R 16 --K 2 --epochs 200
