# R7P direct four-lane controller

Latest explicit user direction supersedes the old preflight-only/future-window/budget launch gate. Generator records verbatim messages and scientific parameters stay unchanged. No added preflight, archive-before-start or old timing gate. Actual code/parent identity and runtime nonfinite/OOM/count/disk safety remain mandatory.

Generate (does not launch):
PYTHON r7p_prepare.py --workroot R7PROOT --trainer-code R7PROOT/trainer --r6-root R6ROOT --worker-template R6_FORMAL_WORKER_CONFIG --child-env-config R6ROOT/formal-child-env.json [--backup-command-json PATH_TO_JSON_ARGV] [--python EXISTING_PYTHON]
Then immediately execute argv from R7PROOT/launch-command.json.

Generator verifies fixed R6/R1-1201/L48 parent cbcd57...bytes9167667; hardlinks its immutable model into R7PROOT/inputs/parent.pt without changing source bytes/mode so endpoint evaluator root-local contract works. parent-selection also binds actual R6 effective-settings path/SHA. parameters.json is reviewed plan bytes semantically unchanged; user-execution-override.json explicitly supersedes its stale launch-blocking text. Worker source config is the trainer's actual parameters schema.

Exactly P0/P1×1401/1402,128env each,pool65536,16prefill+48learning. Spawn2s stagger; worker flock serializes startup/prefill, controller releases all four verified iteration16 ready files together. Independent RNG/pool/path per arm. Worker saves L24/L48 model nodes, L24 fullresume, latest fullresume <=300s (macro-boundary latency applies). Driver writes node progress at start, every1800s and completion/failure, preserving all lane status and latest counters/checkpoint records.

Soft target12:45Z; do not use it as old launch blocking gate. Actual no-renew lease14:15Z; children receive14:14Z orderly-save deadline, controller grants at most60s close grace. No new jobs below12GiB, save-stop below10GiB. Source mutation, worker nonzero/failure or safety error stops all four owned process groups with save grace. No automatic restart, scientific parameter changes or unequal-stage ranking.

Optional backup_command is an explicit argv invoked asynchronously at30min and immediately at completion/failure, never before first launch. Parent owns independent R7P publisher/model-backup configuration; original R6 interfaces untouched. A separate existing30min backup loop is also supported. Models/state outputs retain R6-like field names; checkpoint internal metadata is r7p. Parent must verify publication receipts, node recording is not remote backup success.

After all4 L48 workers finish, separate r7p_evaluate_endpoint.py --config evaluation-runtime.json runs only frozen parent6+four arms24 cases with new141-143 D/S. Eval config {workroot,evaluation_code,evaluator,study_root,data_root,bootstrap_env}. Child environment reused, bootstrap binds new evaluate_sac_r7p.py/provider manifest. No heldout,turn,pulse or promotion. Does not delay training launch.

3 standard-library CPU tests pass (4-lane barrier/no early learning, node records, frozen input changes); py_compile passes. No actual server/GPU started here.

Python is explicit --python, else worker cfg python, else authorized /root/rivermind-data/tita-check/.venv/bin/python. Does not depend on any runtime-formal filename. Parent hardlink must be same filesystem; fail rather than silently copy. Both paths are used read-only and SHA checked; source permissions are not changed.

Kit exit0 safeguard: every already-exited worker is inspected both during staggered launch and monitoring. Even exit0 is a failure unless its response exists and status=trained. All other owned workers are then save-stopped immediately, original logs/exitcode/response status retained, never auto-restarted. Regression covers missing response, failed response and valid trained response.
