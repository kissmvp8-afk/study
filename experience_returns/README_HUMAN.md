# 经验与回报｜人类阅读版

[同目录 AI 实施规范](README.md) · [Codex 初稿任务](../CODEX.md) · [Gemini 润色任务](../GEMINI.md) · [返回上层人类版](../README_HUMAN.md)

| 字段 | 当前内容 |
|---|---|
| 文档状态 | 待 Codex 编写 |
| 技术核对状态 | 待 Codex 阅读实际代码后核对 |
| 语言润色状态 | 待技术初稿完成后交 Gemini |
| 已核对代码版本 | 无 |
| 待同步代码版本 | D01 `7fe7e93`（已合并）；D02～D05 在分支 `claude/experience-returns`：`3838fe4`、`c3b2fe0`、`e77bc66`、`807ee63` |
| 实际阅读材料 | 待 Codex 记录 |
| 验证依据与限制 | 尚未编写教学正文。五个小模块的代码与 CPU 单元检查都已完成（人工数据），记录见 [AI 规范](README.md#当前证据)；未接入仿真或网络。大模块讲解应在各小模块人类版完成后再写 |

此文件仅保留占位与导航，当前不撰写讲解正文。对应讲解范围内的代码完成后，由 Claude 提供带注释的实际代码、版本和验证记录，再由 Codex 根据这些材料编写。代码未完成、只有空壳或无法读取实际代码时，继续保留占位。

## 下层人类版导航

- [D05 · 历史窗口与有效性](history_windows/README_HUMAN.md)（等待代码完成）
- [D03 · 连续多步回报](n_step_returns/README_HUMAN.md)（等待代码完成）
- [D04 · 观测归一化与统计管理](normalization/README_HUMAN.md)（等待代码完成）
- [D02 · 经验池写入与抽样](replay_buffer/README_HUMAN.md)（等待代码完成）
- [D01 · 经历记录契约](transition_schema/README_HUMAN.md)（等待代码完成）

代码完成后的顺序：Codex 阅读实际代码并编写技术初稿 → Gemini 在本文件中润色表达。代码后续变化时先由 Codex 更新技术内容，再交 Gemini 润色；两者共用本文件，不新增第三套 README。
