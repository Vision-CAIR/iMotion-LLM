# configs

Reserved for experiment and runtime configuration files.

Planned examples:

- dataset preprocessing configs
- model/training configs
- evaluation configs
- ablation configs
Release-ready configs live in [`configs/release`](release).

Supported entrypoints:

- `imotion_waymo_train.yaml`
- `imotion_waymo_eval.yaml`
- `imotion_nuplan_train.yaml`
- `mtr_waymo_act.yaml`

The original legacy experiment YAMLs are preserved under `trajgpt/` for traceability and paper archaeology.
