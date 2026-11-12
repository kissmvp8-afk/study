# SAC-R9C：4090质量优先长训交接

本目录把R9L的三fork固定预算迁移到4090服务器：每路128环境，六条轨迹按最多四并发的4+2两波推进。保留SAC核心，A启动停车合格后才进入B转向/抗扰，不恢复旧云端9小时截止。

服务器Codex/Dot先读 [4090_RUN.txt](4090_RUN.txt) 和 [parameters.json](parameters.json)。新Ubuntu实例用 `bash setup_4090.sh /实际数据盘/tita-r9c`；已有兼容环境先复用。脚本固定Python3.11、Torch2.7.0/cu128、IsaacSim5.1及IsaacLab/DDT/资产提交，隔离安装并进行真实物理采集检查。需要同目录的 [constraints.txt](constraints.txt) 和 [smoke_runtime.py](smoke_runtime.py)，依 [FILE_SHA256.json](FILE_SHA256.json) 核验。

安装脚本只验证环境/采集，不实现SAC监督器。正式控制器、完整学习/串并兼容/恢复、评价与归档仍须服务器代理实现并实测通过。本目录只在当前CPU工作区检查语法、预算/矩阵/校验等可验证项；GPU安装和Isaac运行未在此工作区执行。

接口：安装器参数是独立项目绝对路径，要求Ubuntu22.04/24.04 x86_64、可见4090、40GiB安装空间与root或既有免密sudo。所有大包/缓存留在项目盘，不改HOME、系统Python或主机驱动。失败退出保留setup.log和部分环境，重试不清空用户目录。`--print-plan`只打印安装组合，`--help`显示使用方式。

smoke_runtime.py使用scene/tita两个独立进程模式；tita输出采集步数、action_dim和GPU，但 `SAC_updates=0`、`control_success_evaluated=false`。完整学习和恢复的通过结论必须来自后续真实日志。核心依赖冲突停止留证，不以强制无依赖安装或升级SAC核心掩盖。

三fork来自同一暖启动，不是三个独立从头训练；新测试最多195例且按模型门控。固定预算上限不是运行时长保证，完整工期须扣除部署/全部开发/终评/备份，并受当前真实租期约束。
