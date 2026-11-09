# SAC-R7：回放覆盖与停车精度

用户已请求安排下一轮。本目录是交给Dot的有条件执行计划，尚未由审阅窗口启动训练。

- [Dot执行入口](DOT_START_HERE.txt)
- [完整计划](execution-plan.txt)
- [参数与门控](parameters.json)
- [论文与证据边界](basis-and-papers.txt)
- [R6末次20条留出有限复算](R6-heldout-limited-audit.json)
- [实际下载并CPU读取的父模型](parent-model-check.json)
- [输入版本与未核验范围](input-evidence.json)

R6证据固定于 `1e69eb8390c2b5cb4353e01acb0f0bf3aacadcf4`。原主2/10、原备7/10，均存活10/10；未过标准验收。新父为R6原备R1/1201-L48（SHA `cbcd57a45c5addf7c29d30d40e5f753965bd62da9ba57a227ab0866fcc8acac3`），不是追认其为R6主候选。R7四组×两fork、每worker128环境，Actor5e-5，比较回放容量×零指令精度项；新独立留出231–235，预算目标7–9小时、硬上限9.5小时，有效果和资源门控。

R7训练适配尚未实现，GPU/真实恢复/新奖励测试由Dot按计划完成。本次未修改算法源码、正式配置、定时巡检或模型设置。独立审阅不等于真实机器人验收。
