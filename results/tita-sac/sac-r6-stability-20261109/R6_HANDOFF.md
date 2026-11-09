# R6 experiment handoff

All eight arms completed 96 learning macro rounds. Development evaluation completed 80 cases at stage 48 and 80 at stage 96.

Original controller stopped after two zero-step Kit initialization failures. An approved evaluation-only infrastructure recovery completed after exact 600-step same-model/seed equivalence passed. Original failures and stopped state remain preserved.

Selected pair: R1 at learning48. Primary seed1202: heldout standard2/10. Secondary seed1201:7/10. Both fail the all-pass gate; turn and pulse were not run. The direction is not reproduced. Do not replace the existing parent model.

All32 stage48/96 model and full-resume assets are uploaded. Stage96 sixteen current local files were rehashed and matched remote SHA256 and size. A full remote redownload test did not complete; do not claim it passed. Local originals are retained.

See R6_HANDOFF.json for identities and source SHA256. Recovery artifacts live under recovery/infra-eval96-001. Download Release evidence pieces using the corresponding committed MANIFEST and local-verification sidecars. Final bulk archival was verified at 09:35 UTC: all 1763 in-workspace recovery evidence files were covered with matching hashes, no pending files. Final archive commit: 1e69eb8390c2b5cb4353e01acb0f0bf3aacadcf4. The two final incremental evidence archives are not substitutes for earlier batches.
