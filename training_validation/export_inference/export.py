"""T05 模型导出与运行一致性：把“归一化 → 历史切片 → 历史编码 → Actor 确定性动作”导出为 TorchScript，并核对一致性。

导出格式：TorchScript（torch.jit.trace），用 torch.jit.save 保存为 .pt，torch.jit.load 读取，只依赖 PyTorch。
（原 PPO 同样用 torch.jit.trace；ONNX 需要额外依赖，本版本不提供。）

推理管线（InferencePolicy.forward）与训练内推理是同一条代码路径：
    policy_history [B,10,33]（原始观测，含当前帧，从旧到新）
      → D04 normalize_policy（冻结统计）→ D05 split_windows（t-4..t）→ N01 build_actor_features（52 维）
      → N03 deterministic_action（u = mu，a = b + c·tanh(u)）→ action [B,8]
直接调用 N01 features_from_history 与 N03 deterministic_action，不另写公式，因此训练与导出读到同样的时间顺序。

历史对齐（原 PPO 的静态不一致）：原 _InferenceWrapper 另外接收 current，并把 history[:, 1:] 与 current 拼接后取最后 5 帧；
如果传入的 history 已经包含当前帧，当前帧会出现两次（窗口 t-3, t-2, t-1, t, t）。本管线只接收一份已含当前帧的历史，
不再追加当前帧（D05 测试与本模块测试都有回归检查）。

冻结：构造管线时深拷贝编码器、Actor 与 D04 统计，并设为 eval、不求梯度；之后的训练或统计更新不影响已导出的模型。
不进入动作执行链：Q、目标 Q、Barlow projector、随机采样与温度。

输出与物理换算的接口：导出模型输出**策略动作坐标**的 8 维动作（E03 顺序，位于 E04 区间 [L, U] 内），**不包含**物理换算。
物理目标 = default_target + scale × a，再按 clip 裁剪（E03 action_to_targets；在 Isaac Lab 中由环境动作项执行），
运行端只执行这一次换算；若导出模型里再换算一次，就会重复缩放。元数据记录关节顺序、scale、默认目标、clip 与控制周期。
控制周期 0.02 s（50 Hz）。CPU 上的导出一致性不代表真机部署通过。
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch
from torch import nn

_ROOT = Path(__file__).resolve().parents[2]
for _path in ("networks_learning/history_encoder", "training_validation/configuration"):
    sys.path.insert(0, str(_ROOT / _path))

from config import version_report  # noqa: E402
from encoder import features_from_history  # noqa: E402

EXPORT_FORMAT = "torchscript-trace"
METADATA_VERSION = 1
CONTROL_PERIOD_S = 0.02
HISTORY_SHAPE = (10, 33)


class InferencePolicy(nn.Module):
    """冻结的确定性推理管线：forward(policy_history [B,10,33]) -> action [B,8]（策略动作坐标）。"""

    def __init__(self, encoder: nn.Module, actor: nn.Module, normalizers) -> None:
        super().__init__()
        self.encoder = copy.deepcopy(encoder).eval()
        self.actor = copy.deepcopy(actor).eval()
        self.normalizers = copy.deepcopy(normalizers)  # D04 统计的冻结副本（不是 nn.Module，trace 时作为常量写入）
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, policy_history: torch.Tensor) -> torch.Tensor:
        return self.actor.deterministic_action(features_from_history(self.encoder, self.normalizers, policy_history))


def build_inference_pipeline(trainer) -> InferencePolicy:
    """由 T02 Trainer（或恢复后的 Trainer）构造冻结的推理管线；只读 Trainer。"""
    return InferencePolicy(trainer.encoder, trainer.actor, trainer.normalizers)


def export_metadata(trainer) -> dict:
    """导出模型的输入输出说明、动作接口、控制周期、统计计数与来源。"""
    entries = [{"index": e.index, "joint_name": e.joint_name, "control_type": e.control_type.name, "unit": e.unit,
                "default_target": e.default_target, "scale": e.scale, "clip": [e.clip_low, e.clip_high]}
               for e in trainer.contract.entries]
    return {
        "format": EXPORT_FORMAT,
        "metadata_version": METADATA_VERSION,
        "input": {"name": "policy_history", "shape": ["B", *HISTORY_SHAPE], "dtype": "float32",
                  "meaning": "原始（未归一化）策略观测历史，含当前帧，从旧到新；归一化在模型内部用冻结统计完成"},
        "output": {"name": "action", "shape": ["B", 8], "dtype": "float32", "coordinate": "policy_action",
                   "physical_conversion_included": False,
                   "meaning": "E03 顺序的策略动作；物理目标 = default_target + scale × a 再按 clip 裁剪，由运行端执行一次",
                   "lower": list(trainer.bounds.lower), "upper": list(trainer.bounds.upper)},
        "joints": entries,
        "control_period_s": CONTROL_PERIOD_S,
        "normalizer_counts": {"policy": trainer.normalizers.policy.count, "q": trainer.normalizers.q.count},
        "config_name": trainer.config.name,
        "versions": version_report(),
        "excluded": ["Q 与目标 Q", "Barlow projector", "随机采样", "温度 alpha"],
        "validation_scope": "CPU 上与训练内推理的一致性；未经真机或仿真部署验证",
    }


def export_policy(trainer, path: Path, example_batch: int = 4) -> dict:
    """trace 推理管线并保存到 path，同时写出 path.with_suffix('.json') 元数据，返回元数据。

    trace 使用一组非零的示例历史；trace 后立即与 eager 管线比较一次（torch.jit.trace 的 check_trace）。
    """
    pipeline = build_inference_pipeline(trainer)
    example = torch.linspace(-1.0, 1.0, example_batch * HISTORY_SHAPE[0] * HISTORY_SHAPE[1]).reshape(example_batch, *HISTORY_SHAPE)
    traced = torch.jit.trace(pipeline, example, check_trace=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, str(path))
    metadata = export_metadata(trainer)
    path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def load_exported_policy(path: Path) -> torch.jit.ScriptModule:
    """读取导出的 TorchScript 模型（eval 模式）。"""
    return torch.jit.load(str(path)).eval()


def compare_outputs(reference, exported, histories: torch.Tensor, atol: float = 1e-6) -> dict:
    """在同一批历史上比较两条路径的确定性动作，返回最大绝对差、是否在容限内与样本数。

    只比较确定性动作；随机采样结果不能逐元素比较。默认容限 1e-6：两条路径执行相同的 float32 运算，
    差异只来自算子实现，超出时应排查归一化统计、切片或动作变换是否不同。
    """
    with torch.no_grad():
        a, b = reference(histories), exported(histories)
    diff = (a - b).abs().max().item()
    return {"max_abs_diff": diff, "within_tolerance": diff <= atol, "atol": atol, "samples": int(histories.shape[0])}
