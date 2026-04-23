# scripts

Public-facing helper scripts live here.

Main entrypoints:

- `train_gameformer_waymo.sh`
- `eval_gameformer_waymo.sh`
- `train_imotion_waymo.sh`
- `eval_imotion_waymo.sh`
- `train_gameformer_nuplan.sh`
- `eval_gameformer_nuplan.sh`
- `train_imotion_nuplan.sh`
- `eval_imotion_nuplan.sh`
- `train_mtr_waymo_base.sh`
- `eval_mtr_waymo_base.sh`
- `train_mtr_waymo.sh`
- `eval_mtr_waymo.sh`
- `preprocess_waymo_gameformer.sh`
- `preprocess_nuplan_complex.sh`

Public usage policy:

- prefer these scripts over ad-hoc commands from legacy folders
- use `--act --act_dec` for conditional GameFormer variants
- use `IMOTION_MTR_EVAL_MODE` or `datasets.traj_align_valid.processor.valid.new_eval_mode=...` to switch paper evaluation modes
- the legacy experiment folders remain in the repo for traceability, but they are not the primary public interface
