# R8 Actor学习率对照训练交接

入口：[给Dot的执行入口](DOT_START_HERE.txt)、[配置与门控](parameters.json)、[详细执行计划](execution-plan.txt)、[完整R7P依据](basis-review.txt)、[逐例表](final-matched-cases.csv)。本目录授权在剩余租期可容纳时由Dot实施R8，当前只是计划，未在本窗口启动或验证新训练适配器。

audit_full.py是只读标准库审核工具，复用已冻结C1评分器并独立复算窗口/门槛；输入为C1、四批既有L24以及234/235新两批解压目录。默认/workspace/sac-r8-plan，可用--work/--c1/--old-l24替换；另依赖上轮明确路径的L24模型验证及model-inference-reference.json。实际运行已通过90例52858步，新增316文件SHA/大小/CRC通过。R7P原C1及早前L24文件沿用前两轮实际核验记录，不冒充这轮重新下载。

CSV的learning=0只表示R7P新增学习为0，父模型本身是R6 R1-L48；P/W为停车窗口最大均值，Mavg与Mmax分别正反均值和最大MAE。所有缺失完整阶段指标为空，不用存活者填充。新R8配置仅比较Actor LR，optimizer加载后的LR覆盖是关键，需记录除该值外继承完全一致。尚无新增训练、GPU、实机或抗扰效果证据。
