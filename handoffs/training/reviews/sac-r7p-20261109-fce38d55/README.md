# SAC-R7P独立审阅与新seed冻结复核

固定输入 `fce38d5557595c9aaae8652a8d5593b8e2851b63`，raw归档 `986c9781afd0c0ed56ffb4e22e5e0c382e6d831e`。575文件、30评价18000步、256宏轮和262144条奖励贡献已复算；P0通过3/12、P1通过4/12，均存活12/12，父通过2/6。预定promising_short_budget成立，但停车效果两fork相反，未独立验收，不替换父。

- [完整独立审阅与证据限制](review.txt)
- [下一步：只冻结验证，不加训练](next-plan.txt)
- [固定五模型、新seed与65分钟预算](parameters.json)
- [逐案例评价表](case-table.csv)
- [独立复算判定](decision-recomputed.json)
- [四个最终模型和四个恢复点实际下载读取](model-and-resume-verification.json)
- [原始输入版本与指纹](input-evidence.json)

下一步计划未由本窗口启动。新231–235共50例，标准关卡与奖励效果复现分别判定；不调参、不转向/推扰、不续训。固定20:45北京时间收尾、独立阶段65分钟与真实租期边界取较早者，无法完整容纳就只备份。四路训练实测7.06分钟，30评价17.69分钟，另有归档/回读；不再沿用135/162分钟固定粗估。

独立验证脚本在本目录保存，需按input-evidence指向的原始归档布置evidence目录后使用。未在本审阅运行Isaac/GPU或实际训练恢复。
