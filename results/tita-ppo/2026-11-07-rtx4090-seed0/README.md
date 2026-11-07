# Tita NP3O/PPO training artifacts

Date: 2026-11-07. Hardware: RTX 4090. Training seed: 0.
Target repository directory: results/tita-ppo/2026-11-07-rtx4090-seed0

Completed iterations: 3000. Selected checkpoints: 1000, 2000, 3000.
Selection is by reached milestone plus final checkpoint, NOT by best evaluation score.
Checkpoints retain original model, optimizer, normalization/BN state and iter metadata, with infos=None.
Raw checkpoint bytes are unchanged; SHA-256 values are independently verified against sidecars.

This is the original DDT NP3O (Barlow Twins PPO) run, bounded to at most 3000 iterations and its available training budget, not SAC or extra training.
A single training seed and short simulation evaluations do not establish reliable walking, robustness, real-robot safety, or statistically significant improvement.
Missing or skipped evaluations must not be interpreted as passed. There is no supplied numerical 64-environment baseline comparison. Hardware and parallel batch changes prevent attributing differences solely to the algorithm.

Configuration files are sanitized: <PROJECT_ROOT> replaces the private deployment root; <RECORDER_EXPORT_DIR> replaces the exact upstream default recorder directory (not an actual export from this run); deadline and credential fields are omitted.
Any command below is a SANITIZED, illustrative command, not the raw execution transcript. Private wrapper/deployment code is not redistributed.
Inspection command: sha256sum -c SHA256SUMS
Upstream fixed source versions are linked in sources.json. No upstream source, URDF, mesh, deployment files, stdout/stderr or access credentials are included.

SHA256SUMS covers every exported payload file except itself (a self-hash is mathematically circular).
This package being present locally does not establish that GitHub backup has completed.
