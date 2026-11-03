# Worker Final Review — EXP107 CPU read-only inspection (server installer + migration/profile retention)

Scope: `setup.py data_folder(migrate=False)` / `--no-data-migration` propagation, `tools/setup_server.py`
private registry / preparation, `setup.build_engine` profile retention. CPU-only inspection. No GPU, no services,
no models/configs/registry/browser, no setup run, no legacy suite. Only file written: this one.

## Test results — `python -m unittest tools.test_setup_server`
```
Ran 36 tests in 0.894s
OK
Loaded accepted profile at native 262144; service momoe-ai-server.service is enabled.
```
All 36 pass (0 failures). Covered: normalized_config exactness/flags, canonical-hash & artifact validation,
constructor validation, unit render, install rollback (prepare-fail + success), `owns_port`, setup-CLI arg
validity, run-server argv, atomic replace over running inode, private-registry isolation, entry help, and the
two migration tests.

## Migration paths / persistence — findings
1. `data_folder(migrate=False)` is a true no-migration path. `setup.py:4565` sets `sources = [...] if migrate else []`
   and the merge block at `setup.py:4566` (`if migrate and ...`) is skipped, so no discovery, no `move_into`, no
   config repoint, no `elsewhere` population. Test `TestNoMigration.test_no_migration_never_discovers_or_moves_other_installations`
   asserts `other_installs`/`move_into` are never called and returns `(dest, [])`.
2. `--no-data-migration` propagates at BOTH `data_folder` call sites: main install `setup.py:6784`
   `data_folder(a.data_dir, migrate=not a.no_data_migration)` and model-prep `setup.py:6513`. Flag defined `setup.py:6694-6695`.
3. Dedicated installer disables migration explicitly AND privately. `tools/setup_server.py:185` passes
   `--no-data-migration`; `:191-199` reroutes `XDG_CONFIG_HOME` to `<root>/.momoe/setup-state` for the setup.sh run
   and restores the caller's value (or clears it) in `finally`. `TestPreparationIsolation` confirms the private
   registry value is seen during `prepare` and the env is restored on failure. `--no-data-migration` also disables
   the `elsewhere`-based nearby-checkout adoption in main(), so no checkout is discovered — intended for the
   dedicated transaction.
4. No stray migration entry points remain in setup-server.sh / run-server.sh / docker-entrypoint.sh / setup.sh
   (setup.sh only execs `setup.py`; grep for migrate/other_installs/previous_config = none).

## build_engine profile retention
- `want_profile` (`setup.py:4121-4123`): env `STRATA_AIXS_PAIRED_INT8` wins when set, else preserves installed stamp.
- `engine_ok` (`setup.py:4126-4127`) requires `bool(meta.get("profile", False)) == want_profile`.
- CMake always gets an explicit `-DSTRATA_AIXS_PAIRED_INT8=<ON|OFF>` (`setup.py:4147`); stamp records `"profile": want_profile` (`:4177`).
- Tests confirm: profile-off cache hit, unset preserves installed profile (True/False), switch-on forces build with
  `-DSTRATA_AIXS_PAIRED_INT8=ON`, explicit-off forces build with `-DSTRATA_AIXS_PAIRED_INT8=OFF` (old CMakeCache
  cannot keep ON).
- `setup_server.prepare()` double-checks retention end-to-end: `:209` raises unless `meta.get("profile") is True`,
  and `:204-207` rejects an engine outside `<root>/engine` or `engine-cuda12`.

## Verdict
No migration or persistence regression found in the inspected paths; profile retention is intact and explicit.
36/36 isolated tests green. (GPU build + end-to-end install owned by supervisor; not run here per instructions.)
