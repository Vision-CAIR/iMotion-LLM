# configs

Release-ready configs live in [`configs/release`](release).

Supported entrypoints:

- `imotion_waymo_train.yaml`
- `imotion_waymo_eval.yaml`
- `imotion_nuplan_train.yaml`
- `imotion_nuplan_eval.yaml`
- `mtr_waymo.yaml`
- `mtr_waymo_act.yaml`

The original legacy experiment YAMLs are preserved under `trajgpt/` for traceability and paper archaeology.

Notes:

- `imotion_*` configs are the supported public YAML surface for iMotion-LLM
- MTR uses `mtr_waymo.yaml` for unconditional baseline runs and `mtr_waymo_act.yaml` for conditional baseline runs
- many paper ablations still live as preserved legacy YAMLs under `trajgpt/`
