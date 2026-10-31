# E01 · 机器人与任务环境｜人类阅读版

[同目录 AI 实施规范](README.md) · [Codex 初稿任务](../../CODEX.md) · [Gemini 润色任务](../../GEMINI.md) · [返回上层人类版](../README_HUMAN.md)

| 字段 | 当前内容 |
|---|---|
| 文档状态 | 待 Codex 编写 |
| 技术核对状态 | 待 Codex 阅读实际代码后核对 |
| 语言润色状态 | 待技术初稿完成后交 Gemini |
| 已核对代码版本 | 无 |
| 待同步代码版本 | 分支 `claude/environment-integration`，代码提交 `f3a1c743b8e9992aa77f218e1ef649d638db9796`：`assets.py`、`metadata.py`、`isaac_env.py`、`run_sac_isaac.py`、`test_environment.py`；配套 T01 修改 `bd99b7b87b09b5fd1124ebdd9d842547910a9831`（文档同步在其后的提交中） |
| 实际阅读材料 | 待 Codex 记录 |
| 验证依据与限制 | 尚未编写教学正文。代码侧：14 项 CPU 单元检查通过（Python 3.11 + torch 2.15.0 / 2.7.0，人工 URDF 与人工运行时元数据；真实 URDF 一项设置 `TITA_URDF` 后通过），记录见 [AI 规范](README.md#实现证据与文档交接)；`isaac_env.py`、`run_sac_isaac.py` 只做了语法与结构检查，**未在 Isaac 中运行**，等待 Windows + NVIDIA 验证；不代表机器人训练成功 |

此文件仅保留占位与导航，当前不撰写讲解正文。对应讲解范围内的代码完成后，由 Claude 提供带注释的实际代码、版本和验证记录，再由 Codex 根据这些材料编写。代码未完成、只有空壳或无法读取实际代码时，继续保留占位。

代码完成后的顺序：Codex 阅读实际代码并编写技术初稿 → Gemini 在本文件中润色表达。代码后续变化时先由 Codex 更新技术内容，再交 Gemini 润色；两者共用本文件，不新增第三套 README。
