# 网络与学习｜人类阅读版

[同目录 AI 实施规范](README.md) · [Codex 初稿任务](../CODEX.md) · [Gemini 润色任务](../GEMINI.md) · [返回上层人类版](../README_HUMAN.md)

| 字段 | 当前内容 |
|---|---|
| 文档状态 | 待 Codex 编写 |
| 技术核对状态 | 待 Codex 阅读实际代码后核对 |
| 语言润色状态 | 待技术初稿完成后交 Gemini |
| 已核对代码版本 | 无 |
| 待同步代码版本 | 分支 `claude/networks-learning`：N01～N08 代码提交 `7e7d34a`、`2e6c470`、`a7a469e`、`ef98631`、`e43b4ea`、`0b150dc`、`b186639`、`7f671f3` |
| 实际阅读材料 | 待 Codex 记录 |
| 验证依据与限制 | 尚未编写教学正文。八个小模块的代码与 CPU 单元检查都已完成（共 43 项，人工数据，含一次完整更新衔接检查），记录见 [AI 规范](README.md#当前证据)；未接入仿真或训练。大模块讲解应在各小模块人类版完成后再写 |

此文件仅保留占位与导航，当前不撰写讲解正文。对应讲解范围内的代码完成后，由 Claude 提供带注释的实际代码、版本和验证记录，再由 Codex 根据这些材料编写。代码未完成、只有空壳或无法读取实际代码时，继续保留占位。

## 下层人类版导航

- [N03 · SAC 动作分布](actor_distribution/README_HUMAN.md)（等待代码完成）
- [N06 · Actor 的参数更新](actor_learning/README_HUMAN.md)（等待代码完成）
- [N02 · 速度监督与 Barlow Twins](auxiliary_learning/README_HUMAN.md)（等待代码完成）
- [N05 · Q 的训练目标与更新](critic_learning/README_HUMAN.md)（等待代码完成）
- [N07 · 探索力度 alpha](entropy_temperature/README_HUMAN.md)（等待代码完成）
- [N01 · 历史状态编码](history_encoder/README_HUMAN.md)（等待代码完成）
- [N04 · 双 Q 评分网络](q_networks/README_HUMAN.md)（等待代码完成）
- [N08 · 目标 Q 的慢更新](target_updates/README_HUMAN.md)（等待代码完成）

代码完成后的顺序：Codex 阅读实际代码并编写技术初稿 → Gemini 在本文件中润色表达。代码后续变化时先由 Codex 更新技术内容，再交 Gemini 润色；两者共用本文件，不新增第三套 README。
