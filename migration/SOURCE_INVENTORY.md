# Incoming Source Inventory

Use this file to track every incoming legacy source before and during migration into this repository.

## Protected Local Paths

These paths already exist on this machine and should not be edited in place as part of the cleanup effort:

| Path | Status | Notes |
| --- | --- | --- |
| `/ibex/project/c2278/felembaa/projects/iMotion-LLM-Jan/run_ibex/iMotion-LLM` | protected | Existing local copy discovered during bootstrap. Use only as reference if needed. |

## Incoming Sources To Audit

Add each source once it is provided.

| Source Path / Repo | Suspected Contents | Audit Status | Migration Target | Notes |
| --- | --- | --- | --- | --- |
| Pending user-provided source | Unknown | waiting | TBD | Awaiting legacy code locations/files. |

## Audit Rules

- Do not modify legacy source trees in place.
- Copy only the relevant iMotion-LLM components into this repository.
- Preserve attribution to upstream parent projects where code is inherited or adapted.
- Track anything missing for paper reproduction in `MIGRATION_CHECKLIST.md`.
