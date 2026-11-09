# R8独立审阅与H1冻结候选留出

入口：[审阅](review.txt)、[给Dot的下一步](next-plan.txt)、[参数](parameters.json)、[40例表](cases.csv)、[门槛图](case-gates.svg)。原始固定版本与核验范围见input-evidence.json；各归档、模型/恢复点、Adam增量和恢复回放边界分别有独立JSON回执。

audit_r8.py只读标准库工具，明确使用/workspace/sac-r8-review的L2/L48解压目录及已审计C1评分器，复算256宏轮、40例24000步，再独立复算窗口和门槛。check_models.py在CPU Torch2.7 weights_only加载已下载文件，检查模块/Adam/normalizer/计数/真实LR与两份回放；L0是预填充后，统计变化按原采集语义处理。复跑需按源码顶部输入位置放置固定归档、模型和原父模型，或显式修改这些路径，不改变审核公式。下载脚本不执行归档内脚本。

实际核验：L2 131文件与最终787文件CRC/SHA/大小通过，40例与报告/原评分一致，16预填充+48学习计数与实际step增量吻合，两fork恢复物理指纹配对一致；两份1502回放边界逐环境检查通过。核心checkpoint.py源码SHA在审阅中固定；没有本窗口Isaac恢复或训练测试。H1适配器尚未实施/验证，不能把计划当作已经执行。
