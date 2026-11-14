# R11 精度复制实验报告（precision replication）

- 实验：R11 precision replication，3 forks/arm（R10P 的 2 forks/arm 未能定论后的复制实验）
- 运行目录（服务器）：`/root/sj-tmp/r9c-source/R11-precision-replication-20261114/`
- 训练启动：2026-11-14 04:02:24 UTC（12:02 +08），控制器 PID 600208；训练完成约 04:42 UTC
- 评估：2026-11-14 05:30:57–09:11:38 UTC（13:31–17:12 +08），36 细胞全部完成
- 报告版本：2026-11-14（评估完成后撰写）；本地副本 `~/workspace/r9c-takeover/R11-REPORT.md`
- 结论先行：**36 细胞 0 个通过 gate；precision 项效应 INCONCLUSIVE（与 R10P 相同结论，现为 3 forks/arm 的复制结果）；不进 Phase B；不放宽阈值。**

## 1. 设计

- 暖启动：Z0-1603 L256（SHA256 `83e1d12954c7de7a9a4040635e4ea5a1e2679ae53efc14705412fa5570671063`，已验证）
- 两臂：precision 0（control-1/2/3，训练种子 1141/1142/1143）vs precision 0.5（precision-1/2/3，训练种子 1151/1152/1153）
- 每 fork：128 envs，max L128，parking 权重 −8，SAC 核心、更新密度、原始 gate 均与 R10P 一致；L32 检查点本次已保存（修复 R10P 的 L32 缺口）
- 评估种子（已锁定，seed-lock.json SHA `e520616cd8577802215b017823e958872c4a9e98dc709a8be38c3ac62729b6df`；修正案 `validation/seed-amendment-r11.json` SHA `7d552f8b1f9e55fa164e4374cedbc8aff51675767b8e964f6902fc03dfae47ea`）：
  - dev：1301–1305、1111–1115；held-out：1121–1125、1131–1135
- 评估：6 forks × L32/64/128 × dev/held-out = 36 细胞 × 20 例 = 720 例
- Gate（R9C 原公式，不变）：survival 20/20 且 passes 20/20 且 mean P ≤ 0.04 m/s 且 mean W ≤ 0.08 rad/s 且 mean Mavg ≤ 0.08 m/s

## 2. 训练结果（6 forks 全部 DONE，exit 0）

| fork | 训练种子 | iterations | transitions | critic/alpha/target/aux | actor | 失败 | wall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| control-1 | 1141 | 144 | 147,456 | 16,384 各 | 8,192 | 0 | 948–1134 s |
| control-2 | 1142 | 144 | 147,456 | 16,384 各 | 8,192 | 0 | 948–1134 s |
| control-3 | 1143 | 144 | 147,456 | 16,384 各 | 8,192 | 0 | 948–1134 s |
| precision-1 | 1151 | 144 | 147,456 | 16,384 各 | 8,192 | 0 | 948–1134 s |
| precision-2 | 1152 | 144 | 147,456 | 16,384 各 | 8,192 | 0 | 948–1134 s |
| precision-3 | 1153 | 144 | 147,456 | 16,384 各 | 8,192 | 0 | 948–1134 s |

- 计数以 segment-result.json 的真实 schema 为准（counters 嵌套在 `"counters"` 键下，非顶层键）。每 fork 另有 env_steps 1152、replay_samples 4,194,304、learning_skipped_iterations 0；formal_acceptance false。
- 模型（每 fork）：`full-L0128.pt` 216,117,465 B（SHA 已验证）+ `model-L0032.pt` / `model-L0064.pt` / `model-L0128.pt` 各 9,148,983 B + `segment-result.json` + `result.json` + `training.jsonl`。
- 捕获（captures）：每 fork 4 段，index 状态 closed，failure None，各段 transition 数之和为 147,456；`failures/` 为空；`gates/` 有 6 个 .gate 文件。
- 控制器：wave2_launched true；heartbeat 最后 2026-11-14T04:42:12 UTC；训练后 GPU 空闲（2 MiB，0%）。

## 3. 显存（实测，不注水）

- 训练期 vram-samples.txt（78 采样，正确解析 memory.used 列）：峰值 **12,657 MiB（51.5%）**，均值 9,131 MiB（37.2%），p50 6,817，p90 12,655；GPU 利用率峰值 99%，均值 54.3%。
- **≥80% 目标在锁定配置下未达成**（R10P 同为峰值 51.5%）。未做任何填充造假；达到 80% 需要规格决策，不是代码修复。评估期（4 路并行）监控读数约 5.7–8.9 GiB。

## 4. 评估结果：36 细胞，720 例，0 通过

评估运行器 `run_eval_r11.py`：36 细胞全部 exit 0，每细胞 20 例；单细胞耗时 1361.9–1623.0 s（中位 1459.1 s，约 24.3 分钟），细胞总耗时 14.68 h，墙钟 3 h 41 m（4 路并行）。评估结果文件 `eval/R11/<fork>/L<node>/<split>/{development-results.json|held-out-results.json}`；gate 汇总 `eval/R11/gate-summary.json`（SHA256 `769ebbee6685b641b2524a1af9114508a32356160f7e8941c01b2c4986cb6caa`）。

| fork | 节点 | split | surv | pass | mean P | mean W | mean M | qualified |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control-1 | L32 | dev | 20/20 | 7/20 | 0.0733 | 0.0609 | 0.0510 | 否 |
| control-1 | L32 | heldout | 20/20 | 1/20 | 0.0856 | 0.0786 | 0.0535 | 否 |
| control-1 | L64 | dev | 20/20 | 10/20 | 0.0540 | 0.0623 | 0.0457 | 否 |
| control-1 | L64 | heldout | 20/20 | 8/20 | 0.0513 | 0.0767 | 0.0480 | 否 |
| control-1 | L128 | dev | 20/20 | 9/20 | 0.0481 | 0.0889 | 0.0456 | 否 |
| control-1 | L128 | heldout | 20/20 | 5/20 | 0.0460 | 0.0929 | 0.0512 | 否 |
| control-2 | L32 | dev | 20/20 | 5/20 | 0.0673 | 0.0636 | 0.0518 | 否 |
| control-2 | L32 | heldout | 20/20 | 2/20 | 0.0727 | 0.0629 | 0.0568 | 否 |
| control-2 | L64 | dev | 20/20 | 7/20 | 0.0520 | 0.0638 | 0.0525 | 否 |
| control-2 | L64 | heldout | 20/20 | 12/20 | 0.0487 | 0.0692 | 0.0561 | 否 |
| control-2 | L128 | dev | 20/20 | 7/20 | 0.0606 | 0.0719 | 0.0496 | 否 |
| control-2 | L128 | heldout | 20/20 | 9/20 | 0.0562 | 0.0740 | 0.0480 | 否 |
| control-3 | L32 | dev | 20/20 | 9/20 | 0.0514 | 0.0555 | 0.0444 | 否 |
| control-3 | L32 | heldout | 20/20 | 14/20 | 0.0522 | 0.0700 | 0.0573 | 否 |
| control-3 | L64 | dev | 20/20 | 12/20 | 0.0486 | 0.0576 | 0.0604 | 否 |
| control-3 | L64 | heldout | 20/20 | 12/20 | 0.0439 | 0.0774 | 0.0591 | 否 |
| control-3 | L128 | dev | 20/20 | 11/20 | 0.0507 | 0.0594 | 0.0495 | 否 |
| control-3 | L128 | heldout | 20/20 | 9/20 | 0.0558 | 0.0586 | 0.0563 | 否 |
| precision-1 | L32 | dev | 20/20 | 10/20 | 0.0405 | 0.0828 | 0.0457 | 否 |
| precision-1 | L32 | heldout | 20/20 | 14/20 | 0.0404 | 0.0722 | 0.0489 | 否 |
| precision-1 | L64 | dev | 20/20 | 7/20 | 0.0488 | 0.1006 | 0.0447 | 否 |
| precision-1 | L64 | heldout | 20/20 | 8/20 | 0.0373 | 0.1342 | 0.0459 | 否 |
| precision-1 | L128 | dev | 20/20 | 13/20 | 0.0470 | 0.0612 | 0.0465 | 否 |
| precision-1 | L128 | heldout | 20/20 | 8/20 | 0.0526 | 0.0750 | 0.0486 | 否 |
| precision-2 | L32 | dev | 20/20 | 14/20 | 0.0456 | 0.0640 | 0.0481 | 否 |
| precision-2 | L32 | heldout | 20/20 | 15/20 | 0.0464 | 0.0660 | 0.0525 | 否 |
| precision-2 | L64 | dev | 20/20 | 9/20 | 0.0542 | 0.0676 | 0.0569 | 否 |
| precision-2 | L64 | heldout | 20/20 | 13/20 | 0.0431 | 0.0720 | 0.0508 | 否 |
| precision-2 | L128 | dev | 20/20 | 10/20 | 0.0484 | 0.0680 | 0.0440 | 否 |
| precision-2 | L128 | heldout | 20/20 | 5/20 | 0.0546 | 0.0813 | 0.0481 | 否 |
| precision-3 | L32 | dev | 20/20 | 8/20 | 0.0477 | 0.0837 | 0.0660 | 否 |
| precision-3 | L32 | heldout | 19/20 | 4/20 | null | null | null | 否 |
| precision-3 | L64 | dev | 20/20 | 12/20 | 0.0479 | 0.0726 | 0.0477 | 否 |
| precision-3 | L64 | heldout | 20/20 | 7/20 | 0.0489 | 0.0885 | 0.0470 | 否 |
| precision-3 | L128 | dev | 20/20 | 5/20 | 0.0619 | 0.0655 | 0.0664 | 否 |
| precision-3 | L128 | heldout | 20/20 | 8/20 | 0.0547 | 0.0825 | 0.0666 | 否 |

- 总计：survival 719/720；passes 319/720（44.3%）；**qualified 0/36**。
- 最好的单细胞：precision-2 L32 heldout 15/20；其次 control-3 L32 heldout 与 precision-1 L32 heldout 各 14/20。没有任何细胞达到 20/20 passes，因此无论均值如何都不可能通过。
- 约束分析（35 个有效细胞 + 1 个异常细胞）：
  - mean P ≤ 0.04：仅 1/35 通过（precision-1 L64 heldout 0.0373，该细胞 W=0.1342 仍远超标）。P 是最紧的约束。
  - mean W ≤ 0.08：26/36 通过；10 个细胞超标（最高 0.1342）。
  - mean M ≤ 0.08：全部通过（范围 0.0440–0.0666）。M 从不是瓶颈。
- 唯一异常细胞：precision-3 L32 heldout，survival 19/20，gate 错误 `missing_or_nonfinite_window`。原因已查明：种子 1132 执行 S 的 episode 仅 10 步即结束（loss 5.0，survived False，task_pass False），forward/reverse/stop1/stop2 全部窗口为 null。按严格 R9C gate，该细胞均值记为 null（有效 19 例的均值为 P 0.0491 / W 0.1030 / M 0.0610）。该细胞 passes 仅 4/20，即使没有缺失窗口也不可能通过，异常不改变任何结论。

## 5. 结论：precision 项效应 INCONCLUSIVE

- 两臂 passes 总计：control 149/360（均值 8.28/细胞），precision 170/360（均值 9.44/细胞）。臂间差 +1.17/细胞，小于臂内 fork 间波动：control 三 fork 为 40/42/67，precision 三 fork 为 60/66/44（极差 27 vs 22）。最好的单个 fork 是 control-3（67/120），与 precision-2（66/120）基本持平。**fork 级噪声主导了结果，臂间差在噪声之内，不能得出 precision 项有系统性效应的结论。**
- 节点趋势：L32 103/240、L64 117/240、L128 99/240——无单调提升，L128 不优于 L32；更长训练未带来更好评估。
- dev vs held-out：dev 165/360，held-out 154/360——无明显泛化落差。
- 与 R10P 的关系：R10P（2 forks/arm）结论为 inconclusive，最好的 precision-2 L128 dev 12/20；R11（3 forks/arm）复现了同样的模式——最好的 precision-2 L32 heldout 15/20，仍远未达 20/20，fork 排序在节点间翻转（R10P 亦然）。**R11 没有推翻 R10P，也没有证实 precision 项；它以三倍复制确认了“效应未定论”。**
- 不进 Phase B；阈值未放宽；不做系统性宣称。

## 6. 偏差与修正记录（诚实账）

1. **评估器缺口与修复（B18）**：`r10_eval.py` 原先只认 R10P 的 4 个 fork 且拒绝 `--node 32`；`r10_gate.py` 也同样钉住了 R10P 种子/fork/节点（B14 未发现，是恢复协调员查出的）。修复方式与 R10P adapter 相同：字节复制、只改 seed/fork 允许表 + L32 支持、diff 记录在 `provenance/changes.patch`、冻结原件未动。Gate 公式（P/W/Mavg、20/20、0.04/0.08/0.08）一字未改。验证：probe 全部 PASS（18/18 检查点头、允许表 20 接受 / 20 个 R10P 种子拒绝、解析、help、bash -n），报告 `validation/eval-driver-r11.json` 状态 PASS。修复中新驱动 SHA `35d1b0a4779f3483de5eb8bcb192ed95d8d643513d3c5b82b8c86559e81a5ced`，gate SHA `a1f6fb7cb6656115f1d704cdc62cc124f1189640460ada8dd15dec7e969507c1`，修正案 SHA `7d552f8b…`。
2. **B4 resume-guard 缺口由 B9 关闭**：`restore_checkpoint` 原先无 fork/config/seed-lock-SHA/parent 身份 guard，B9 六 fork 控制器补了 resume identity guard 与 run config 中的 `seed_lock_sha256`。
3. **上传 DEGRADED（非上传失败）**：6 个 `full-L0128.pt` 均已在 Release `r11-precision-replication-20261114`（prerelease）上架，每件精确 216,117,465 B，名称含已验证的本地 SHA12；控制器在上传时记录的完整 SHA256 与本地一致。DEGRADED 指上传后的下载 SHA 回读在 180 s 超时（服务器→GitHub 下载约 8 KB/s；完整回读 5 件约需 35 小时，不可行），presence + 精确大小已用 `gh api` 验证。precision-3 曾缺失（仅 QUEUED），已单独上传并确认上架；其下载回读同样因 `dial tcp … i/o timeout` 失败。**本地副本未删除；删除前不以回读为前提。**
4. **6-way 并行请求（2026-11-14 15:48 +08）未执行**：`run_eval_r11.py` 不支持断点续跑（预检断言 eval 目录无结果、main 建全新 manifest 全量跑、`--workers` 上限断言为 4）。按指令未停当前运行；从头重启将丢掉 20 个已完成细胞（约 2 小时 19 分），净亏，已向 parent 报告、等用户定夺。评估最终以 4-way 跑完。
5. **本地 /tmp 两次被清空与运行时重启**：评估监控脚本两次因本地 VM /tmp 清空而丢失（15:15、约 17:30 +08 前后），监控重建两次；服务器端训练与评估未受影响（评估 36/36 完成）。本地 `/tmp/r11_evalfix/` 副本丢失，服务器副本完好，关键 SHA 均已记入验证记录。
6. **Tailscale 两次中断**：04:48 UTC 与 08:27 UTC 附近 SSH 各断一次；训练与评估均在服务器端继续，未受影响；监控单次失败后自动重试。
7. **R9C 遗留偏差（与本实验无关，照录）**：R9C 无训练前 seed-lock 记录，不得声称为严格预注册；R11 的种子是在训练前锁定的（seed-lock e520616c…），不存在该问题。

## 7. 归档状态

- Release `r11-precision-replication-20261114`（prerelease）：6/6 `full-L0128.pt` 已上架（presence + 精确大小已验证；下载 SHA 回读不可行，见第 6.3 条）。
- Git 历史推送（报告 + 36 份评估结果 JSON + gate-summary.json + seed-lock + 修正案 + 评估 manifest，纯文本）：见验证记录的归档节；权重绝不进 Git。
- 评估逐例原始数据（`eval/R11` 下每例的 steps.jsonl 等，8.6 GB）与分段交互捕获（R11 5.5 GB + R10P 3.7 GB = 9.2 GB）为唯一副本，未进 Git/Release；删除前须 parent 另行决定（见验证记录）。
- 服务器租约到期日未知、无自动续费；本报告撰写时服务器仍在计费，停机由用户在算家云控制台操作。

## 8. 文件索引

- 本报告：`~/workspace/r9c-takeover/R11-REPORT.md`；服务器副本：`/root/sj-tmp/r9c-source/R11-precision-replication-20261114/R11-REPORT.md`
- 评估结果：`/root/sj-tmp/r9c-source/R11-precision-replication-20261114/eval/R11/`（36 细胞结果 JSON + `gate-summary.json`）
- 评估 manifest：`/root/sj-tmp/r9c-source/R11-precision-replication-20261114/logs/eval-manifest-r11.json`
- 种子锁定：`seed-lock.json`（`e520616c…`）；修正案：`validation/seed-amendment-r11.json`（`7d552f8b…`）
- 验证全记录：`~/workspace/r9c-takeover/verification-2026-11-14.md`（B1–B22 及归档节）
- R10P 报告：`~/workspace/r9c-takeover/R10P-REPORT.md`
