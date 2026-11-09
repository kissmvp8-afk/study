# TITA SAC-R4独立审阅及R5建议

**结论：训练及筛选执行完成，控制验收失败。** 固定提交内主、次候选均为存活6/6、标准任务通过0/6；用户摘要中的主候选1/6未得到该固定版本原始数据支持。转向及扰动均按门控未运行。

- [独立审阅、证据与不确定性](review.txt)
- [下一轮有对照方案：仅建议，未启动](next-plan.txt)
- [下一轮参数与预算](parameters.json)
- [12次留出逐阶段指标](heldout-stage-metrics.csv)
- [全部186次评价指标](all-case-metrics.csv)
- [停车失败轨迹图](heldout-command-response.png)
- [训练实际计数](training-audit.json)
- [各轮独立复算排名](recomputed-rankings.json)
- [28份模型CPU核对](model-state-audit.json)
- [12份最新恢复点资产核对](latest-resume-asset-verification.json)
- [输入版本及核验范围](input-evidence.json)

输入证据固定于 [a258b80a](https://github.com/kissmvp8-afk/study/tree/a258b80a2f47964b8cbb9993a7a326271439e98f/results/tita-sac/sac-r4-128-screen-20261108/20261108T234904Z)。该提交位于实验分支；审阅时默认分支仍在R4计划提交，不能只读默认分支就断言没有训练结果。

下一轮建议保留仓库SAC，以已核验的D2/1001、936学习宏轮模型作共同父权重，四组各两条暖启动轨迹，对照停车奖励强度和命令保持时长。这里仅发布结论与方案，不下发启动命令，不修改训练/评价程序，不替换正式配置。
