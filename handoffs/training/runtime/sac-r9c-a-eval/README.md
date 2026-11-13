# R9C A Frozen Evaluation Adapter

CPU syntax and two synthetic protocol tests pass. **Not GPU tested.** This is the Phase A standard development evaluator only, not a complete R9C campaign. No heldout, phase B or promotion is enabled.

Source: unchanged r4_protocol.py and adapted evaluate_sac_r6.py/evaluation_r6_helpers.py plus linux_common.py/r1_metrics.py extracted using HTTP Range from sac-r6-stability-20261109 / r6-core-e0cdecc8ef742c0e-e0cdecc8ef742c0e-core.zip.part000. ZIP CRC was verified for all extracted source files, including the standalone dependency slice. Verify runtime imports/assets before claiming acceptance.

Before invocation caller must have closed ALL training applications, retained full checkpoints, and verified the actual study checkout and DDT task are the frozen compatible version. This runner serializes four cases, never retries actual falls, and leaves all outputs on technical failure. Kit startup uses the specified lock. Existing output directories are rejected.

```
python run_a_probe.py --python /actual/venv/bin/python --study-root /actual/study --data-root /actual/data --checkpoint /actual/model.pt --output /actual/new-probe-dir --startup-lock /actual/kit.lock --node 16 --timeout-seconds 1800
```

Node16/768 = seeds261/262 × D/S four cases, one model. Run for each of six branch models, after all six reach the common node. Not the 20-case gate. For full A nodes invoke evaluate_sac_r9c_a.py with exactly seeds231–235,261–265 ×D/S, same flags, then independently aggregate using r4_protocol.analyze_rows.

`status=passed` means process completed, NOT control success. `protocol-result.json.task_pass` is the physical score. `development_gate_pass` now derives from that same scorer, no longer r1 d_gate. Early true termination is a failed physical case, not a retry reason.

Additional evidence captures full actual noisy policy history and observations, actual normalized policy history and actor features at before_action, actor mu/std/action and existing pre_reset physics. Environment critic observation is captured honestly as raw critic observation: no Critic/Q forward occurs during policy-only evaluation. R9C's complete offline Q diagnosis/256 states still requires separate implementation; this does not claim that scope completed. No extra observation generation, noise draw or normalizer update is introduced.

Production reward modifications are not applied merely to score policy trajectories; evaluator uses fixed task physics and commands from original R6 reference. Caller must verify physics/task config matches the intended evaluation baseline. Any saved reward decomposition therefore reflects evaluator task configuration, not automatically the training −8 parking reward.
