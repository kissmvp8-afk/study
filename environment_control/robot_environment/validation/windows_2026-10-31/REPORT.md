# Windows 平地单环境最小验证（2026-10-31，UTC+8）

本目录保存精简证据，完整日志及检查点留在原机器 `C:\Users\86193\tita-check`。没有提交环境、缓存或模型大文件。

## 实际结果与边界

| 范围 | 结果 | 证据 |
|---|---|---|
| 历史 CPU | 文档、216 单元、18 CPU 集成通过，跳过 0；本轮未重跑全仓 | [既有汇总](t06-final.json) |
| Windows 路径修复 | 仅三处报告路径 as_posix；此前及本轮同文件 6 项均通过 | [此前记录](A-t06-regression.log)、[本轮记录](pr-t06-six-regression.log) |
| 官方空场景 | 原版 create_empty，经外部限步包装完成 120 物理步，退出 0 | [精简日志](B-official-empty-clean-close.log) |
| 平地 inspect | 8 动作、单帧 policy 33 / Q 54，周期 0.005×4=0.02 s | [metadata](flat-inspect-vulkan/metadata.json)、[summary](flat-inspect-vulkan/summary.json) |
| 平地交互 | 50 步：终止 1、超时 0；1005 步：终止 40、超时 0，均有限 | [50 步](flat-interact-50/summary.json)、[1005 步](flat-interact-1005/summary.json) |
| 短训练/恢复 | 3×8 步保存；恢复后再 8 步。Critic 3→4、Actor 2→2、alpha/目标/辅助各 3→4，无学习跳过 | [summary](flat-train/summary.json)、[配置](flat-train/run_config.json)、[检查点审计](flat-train/checkpoint-audit.json)、[模型摘要](checkpoint-manifest.json) |
| 本轮真实纯截断 | 首次尝试通过：计数器 998→999→1000，terminated=False / time_out=True | [诊断 metadata](timeout/metadata.json)、[summary](timeout/summary.json)、[真实观测和 D01 记录](timeout/captured_observations.json) |

**截断方法和限制**：正常 reset 后只提前设置 Isaac 的计数器；不改正式 20 秒/1000 步上限，不关闭 base_contact，不伪造标记。原终止管理器在随后 2 个真实策略步（8 个物理步，0.04 秒）触发纯时间截断，不能描述为机器人稳定运行完整 20 秒。自然 1005 步记录本来没有超时，两个结果分别保留。

终止步 next 历史/最后一帧/Q 观测与同一次 compute 捕获的重置前值逐张量相等；新 observation 与重置后值相等，历史有效长度 1、step_index=0、episode_id 0→1。随后 3 步真实交互未覆盖已保存 next。没有额外重新采样观测噪声用于比较。

真实记录通过原 D01 校验与 D02 读取，再经原 D03：valid=[True,True,False]，k=2，mask=1，discount=0.97²，reward_sum=-0.009830326773226261。尾部仍在旧回合 0；新回合奖励 -0.024698738008737564 被排除，不被改值或拼接。

短训练是 CPU 学习器 / GPU 仿真；检查点恢复学习状态和经验池，仿真现场重新开始。checkpoint 保存于 24 步，32 步续训结果只在 summary，没有再次保存。轮速显式 20 rad/s 是“首次仿真调试选择，尚未验证”，不是代码默认值。复杂地形、长期训练、性能与稳定行走均未验证。

## 命令与版本

[commands.json](commands.json) 保存实际命令、工作目录、时间和退出码；[versions](dependency-versions.json) 保存实际 Python/包版本；[evidence-manifest](evidence-manifest.json) 保存原始 JSON 文件摘要与来源。完整源码锁定：

- Python 3.11.18；Isaac Sim 5.1.0（wheel 5.1.0.0）；Isaac Lab v2.3.0 / 3c6e67bb5c7ada942a6d1884ab69338f57596f77。
- torch 2.7.0+cu128 / torchvision 0.22.0+cu128；DDT_Lab 701a49c6e78dfb93a8b30145f0e7723b22921c0b。
- ddt_ros2_control e7b32d9746df900680641ff2845ffb0b5ffe99ed；URDF SHA-256 0941b83136cd5c8c89bbb33165bff515a3a988d33319b67c8b06bfb9c71350ce。
- baseline/ppo 8fe40aa0ab3afdb8998cab1357f1f795d2cabc3c，静态读取使用固定树。
- Windows 11 22H2 / 22621，i7-12700H，16 GB RAM，RTX 3060 Laptop 6 GB，驱动 621.01。之前 C/D 总显存采样峰值约 3.3 GiB（含桌面，3 秒采样），不是精确瞬时峰值。

在 study 根目录复现本轮诊断，解释器指向已准备的环境，不重新安装：

```powershell
$env:OMNI_KIT_ACCEPT_EULA='YES'
$env:PYTHONDONTWRITEBYTECODE='1'
& C:\Users\86193\tita-check\.venv\Scripts\python.exe -B environment_control\robot_environment\diagnose_timeout_isaac.py --headless '--kit_args=--vulkan --/app/extensions/fsWatcherEnabled=false' --output_dir C:\Users\86193\tita-check\runs\flat-timeout-next
```

## 兼容问题与未消除的告警

最初普通 inspect 退出 3221225477。原 Kit 日志明确出现文件变化→扩展热重载→级联 shutdown→USD 访问违规，见 [首故障摘录](initial-hot-reload-failure.log)。引起文件变化的外部进程没有确定。后续关闭 `/app/extensions/fsWatcherEnabled`；本轮诊断从 carb 实际读取 False，且其完整日志没有热重载事件。此启动参数仅影响扩展文件监视，不修改机器人或算法。

仍有 `DriverShaderCacheManager::init() called with a different graphics interface`、`Aftermath Error 0xbad00009`、跳过 Intel GPU、point-instancer 等告警。保留了 --vulkan 试验参数，但图形警告仍在，不能把成功归因于消除 D3D12/Vulkan 冲突，也不宣称正式硬件支持或所有启动风险消失。参见 [本轮精简日志](timeout-real-isaac.log) 与 [Kit 设置说明](https://docs.omniverse.nvidia.com/kit/docs/kit-manual/107.2.0/guide/extensions_advanced.html#app-extensions-fswatcherenabled-default-true)。

最早空场景虽然完成步数，关闭时仍等待 STOP 回调；外部包装按 Env.close 顺序清理 callbacks/instance 后重跑成功。未修改官方脚本或 SDK。既有安装时的网络重置、flatdict 构建依赖和包约束问题已经处理，本轮没有安装依赖。

**T06 语义**：T06 不自动运行这些真实仿真，其 simulation/performance 仍为 not_run。本目录的独立 B/C/D 和真实截断证据不能被 CPU 测试替代，也不改写 T06 层级状态。
