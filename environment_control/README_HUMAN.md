# 环境与动作接口｜人类阅读版

[同目录 AI 实施规范](README.md) · [Codex 初稿任务](../CODEX.md) · [Gemini 润色任务](../GEMINI.md) · [返回上层人类版](../README_HUMAN.md)

| 字段 | 当前内容 |
|---|---|
| 文档状态 | 待 Codex 编写 |
| 技术核对状态 | 待 Codex 阅读实际代码后核对 |
| 语言润色状态 | 待技术初稿完成后交 Gemini |
| 已核对代码版本 | 无 |
| 待同步代码版本 | 分支 `claude/environment-integration`：E01 `f3a1c743b8e9992aa77f218e1ef649d638db9796`、E02 `8a0ee22937937e9d08ad059e025d416f380d4294`、E05 `3d4c995b2730b3183614f4cb5b7a62edfa83f927`；E03、E04 未改动 |
| 实际阅读材料 | 待 Codex 记录 |
| 验证依据与限制 | 尚未编写大模块教学正文。五个小模块均有 CPU 单元验证（见各自 AI 规范）；Isaac 接入尚未运行，等待 Windows + NVIDIA 验证 |

此文件仅保留占位与导航，当前不撰写讲解正文。对应讲解范围内的代码完成后，由 Claude 提供带注释的实际代码、版本和验证记录，再由 Codex 根据这些材料编写。代码未完成、只有空壳或无法读取实际代码时，继续保留占位。

## 下层人类版导航

- [E04 · 动作边界与初始中心](action_bounds/README_HUMAN.md)（待 Codex 同步）
- [E03 · 八维动作定义表](action_contract/README_HUMAN.md)（待 Codex 同步）
- [E05 · 终止、截断与重置前观测](episode_lifecycle/README_HUMAN.md)（待 Codex 编写）
- [E02 · 观测接口](observation_interface/README_HUMAN.md)（待 Codex 编写）
- [E01 · 机器人与任务环境](robot_environment/README_HUMAN.md)（待 Codex 编写）

代码完成后的顺序：Codex 阅读实际代码并编写技术初稿 → Gemini 在本文件中润色表达。代码后续变化时先由 Codex 更新技术内容，再交 Gemini 润色；两者共用本文件，不新增第三套 README。
