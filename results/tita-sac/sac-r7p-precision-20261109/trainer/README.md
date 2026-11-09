# R7P实际训练适配

按用户最新直接开训/不另做预检归档/保存节点指令实施；原计划parameters.json保留原文，旧budget-only状态不再作为启动门控。没有放宽父身份、环境、公式、算法、计数或安全停止。此适配只训练，不做评价/归档，不替代controller核对真实租期。

入口 r7p_train.py --worker-config cfg.json --request req.json --response resp.json --headless --device cuda:0
worker配置：root(原tita-check)、study_root(已核验R6study)、r1_trainer、r2_tools、lock_file(四worker共用初始化锁)、use_cache:false。不需preflight_passed或任何预检报告。依赖/环境沿用既有R6，固定R1源码SHA不改。
request：plan_id=SAC-R7P-PRECISION-PILOT-20261109、workroot、parameters_path、parent_selection_path、direction=P0/P1、fork_seed=1401/1402、target_learning_macros=48、action=train_evaluate、execution_mode=four_lane、barrier_id唯一、barrier_timeout_s默认600、deadline_utc由controller实际截止指定。12:45软目标、14:15硬租期遵循controller最新授权安排；worker在request截止前60秒保存退出，不能越过硬租期。

parent-selection schema：
plan_id同上；selected={model_path绝对路径,model_sha256:cbcd57a45c5addf7c29d30d40e5f753965bd62da9ba57a227ab0866fcc8acac3,bytes:9167667,direction:R1,fork_seed:1201,learning_macros:48,source_config:{parking_body_velocity_weight:-8,command_resampling_time_range:[2,4],track_ang_vel_z_exp_weight:0.5}}；parent_effective_settings_path指向真实R6 R1-1201/effective-settings.json；parent_effective_settings_sha256为该文件真实SHA。参数文件使用本目录原R7P parameters.json，无需改成R6common schema。

窄warm-loader仅接受固定R6父、R1/1201/L48、上游C3/1101SHA及实际环境源证据；逐字段学习配置相等，只有新seed和描述字段不同。六modules/four optimizers/two normalizers原样复制并逐项fingerprint，Actor实际LR5e-5，alpha约7.54434e-5，std不重置。新池/task/episode计数零；新fork独立三generators及global RNG，在初始化后显式重设。fullresume保持严格同臂identity/config/池/RNG，不再次seed。

新项作为独立RewardTermCfg、weight=-w、函数返回mask*cost，无dt。旧-8项原样保留。每实际步单次RewardManager.compute，立即用其所有贡献分解验证原奖励+extra=总奖励、额外上界.0125、dt一次；P0 extra0严格保持原奖励分解。原旧项也独立断言。首次实际步写first-step-reward-check.json；不是预造GPU通过。日志包含每步mask/cost/weight/dt/extra/original/total以及原控制诊断。

四worker初始化+16prefill持共享flock，保存后释放锁；ready文件barriers/token/{P0-1401,P0-1402,P1-1401,P1-1402}.ready.json，含identity,counters,barrier_id。controller release.json必须{barrier_id,ready_sha256:{四lane:实际SHA}}。worker逐项验证lane/共同source identity/共同iterations后才学习。fresh同forkP0/P1在prefill前保存并比较模块/优化器/统计指纹，以及初始物理指纹。

每宏8x128采集；8x16Critic/alpha/target/aux attempts、64Actor。实际optimizer.step与soft_update独立累计并存checkpoint，aux成功/跳过分开。16prefill不学习。L24/L48分别保存model-learning-N.pt、fullresume-learning-N.pt和node-learning-N.json；latest-fullresume.pt周期300秒且节点更新。检查点fsync、weights_only独立读回、完整fingerprint验证后原子替换；最新记录只对应latest。full节点占用额外磁盘，controller须监控。完整恢复不恢复仿真现场；需要四臂共同节点恢复时用保留的L24/L48，不能冒称不同物理现场无缝接续。

SIGTERM/SIGINT在AppLauncher后和trainer构造后安装，宏边界保存退出2；deadline/低10GiB/屏障超时有序保存。非有限/OOM/配置/计数异常保留最近有效checkpoint，不将坏状态提升为有效；aux累计skip比例>1%停止。没有独立长GPU预检，没有自动归档。

验证：8项本地标准库测试通过，包含固定计划/父配置拒绝、source证据、四lane屏障、prefill/L24/L48计数、单次reward manager以及节点/信号源码契约；py_compile通过。真实Torch/GPU奖励由正式runtime首步和每宏断言给出，不能把本地测试写成GPU通过。本目录未操作服务器。
