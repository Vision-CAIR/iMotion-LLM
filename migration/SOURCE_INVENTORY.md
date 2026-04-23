# Incoming Source Inventory

Use this file to track every incoming legacy source before and during migration into this repository.

## Protected Local Paths

These paths already exist on this machine and should not be edited in place as part of the cleanup effort:

| Path | Status | Notes |
| --- | --- | --- |
| `<legacy_repo_root>` | protected | Primary source repo identified during audit. Dirty working tree; read-only for migration. |
| `<legacy_snapshot_repo>` | protected | Secondary snapshot/reference copy with similar structure plus logs/artifacts. |
| `<legacy_old_repo>` | protected | Older precursor repo; reference only if needed. |
| `<legacy_snapshot_repo>/run_ibex/iMotion-LLM` | protected | Existing local copy discovered during bootstrap. Use only as reference if needed. |

## Incoming Sources To Audit

Primary candidate sources have now been identified.

| Source Path / Repo | Suspected Contents | Audit Status | Migration Target | Notes |
| --- | --- | --- | --- | --- |
| `<legacy_repo_root>` | Main iMotion-LLM working repo with `trajgpt`, `gameformer`, `instructions`, `mtr`, configs, and run scripts | audited | primary | Best candidate for migration. Refer to `migration/SOURCE_AUDIT_IMOTION_LLM_ICLR.md`. |
| `<legacy_snapshot_repo>` | Snapshot/mirror of the project with overlapping code plus logs, figures, examples, and packed experiment state | partially audited | secondary | Useful to recover missing files or compare variants, but not the first source to migrate from. |
| `<legacy_old_repo>` | Older precursor with similar layout | identified | tertiary | Use only when code is missing from `iMotion-LLM-ICLR`. |

## Audit Rules

- Do not modify legacy source trees in place.
- Copy only the relevant iMotion-LLM components into this repository.
- Preserve attribution to upstream parent projects where code is inherited or adapted.
- Track anything missing for paper reproduction in `MIGRATION_CHECKLIST.md`.
- Prefer `iMotion-LLM-ICLR` as the migration baseline unless a needed file exists only in a secondary source.
