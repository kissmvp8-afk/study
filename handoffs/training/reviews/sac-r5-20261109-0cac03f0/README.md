# TITA SAC-R5独立审阅与R6建议

**训练完成，控制验收未通过。** 主候选标准留出4/6、备选2/6，12次均存活；转向、扰动未运行。

- [独立审阅与证据边界](review.txt)
- [R6建议：先复核早期候选，再检验更新稳定性](next-plan.txt)
- [R6拟议参数、候选权重清单及预算](parameters.json)
- [全部标准评价指标](all-case-metrics.csv)
- [每组、每fork、各阶段及估计速度误差](stage-fork-analysis.json)
- [开发集趋势图](development-trends.png)
- [正式更新计数及活体停走切换统计](training-audit.json)
- [并发、利用率与吞吐独立核验](concurrency-verification.json)
- [24份阶段模型实际下载/CPU核对](model-state-audit.json)
- [8份最新恢复点远端摘要与大小核对](latest-resume-asset-verification.json)
- [缺失证据清单](evidence-gaps.json)
- [输入版本与独立核验范围](input-evidence.json)

固定证据：[0cac03f0](https://github.com/kissmvp8-afk/study/tree/0cac03f0869039ece66dd410f66631f61d663195/results/tita-sac/sac-r5-park-20261109/20261109T041342Z)。

重要解释：C0符合当时冻结的选点规则，但该规则先按最后一轮选方向，再回溯该方向的历史模型，未对C3/144和C2/288做最终留出。不能据此认定C0优于全部历史候选。R6拟修订选点范围，先保存并复核早期完整配对检查点。

本目录仅为独立审阅及建议。没有启动新训练、应用R6参数、替换正式模型或验证实机；服务器本地归档修复说明不在固定快照中，本审阅未宣称已经核验那项修复。
