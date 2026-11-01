"""T06 检查组织与维护证据：一个简单的检查入口，汇总文档、单元、CPU 集成、仿真与性能五个层级的证据。

本模块不重新实现任何算法来“验证自己”：
    - check_documentation：核对 Markdown 相对链接、各小模块的状态描述与实现 / 测试文件是否一致、已实现模块是否残留计划接口表；
    - collect_module_results：用子进程运行各小模块目录中已有的 test_*.py，解析 unittest 输出；
    - check_integration：运行已有测试中覆盖跨模块风险的指定测试类（结束原因、历史错位、动作单位与物理换算、
      网络更新衔接、保存恢复流程、E05 环境适配器接 Trainer、E01 填写的 Isaac 配置驱动 Trainer），作为 CPU 集成检查；
      E05 / E01 两项用人工钩子代替 Isaac，只检查数据接口衔接，不是仿真检查；
    - build_summary：按层级汇总。仿真与性能层级目前没有可执行的检查，明确记为 not_run，不推测结果。
层级之间不能互相替代：文档链接通过不等于代码正确；单元测试通过、损失有限不等于机器人会走。

测试状态：passed（全部通过，跳过数单独记录）、failed（有失败或错误）、error（没有得到 unittest 结果，例如导入失败）、
not_run（没有测试文件或该层级尚无检查）。跳过（skipped）的测试计入 skipped，不算通过，也不算失败。

用法（仓库根目录，需要安装了 torch 的 Python；子进程使用同一个解释器）：
    uv run --no-project --python 3.11 --with torch==2.15.0 python -B training_validation/verification/verify.py
    ……/verify.py --docs            只做文档检查
    ……/verify.py --integration     只做 CPU 集成检查
    ……/verify.py --json report.json  另存 JSON
任一已执行的检查失败时退出码为 1。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BIG_MODULES = ("environment_control", "experience_returns", "networks_learning", "training_validation")
ROOT_DOCUMENTS = ("README.md", "README_HUMAN.md", "AGENTS.md")
LINK_PATTERN = re.compile(r"\]\(([^)\s]+)\)")
PLAN_SECTION = "## 计划文件与逐函数职责"

# 跨模块风险 → 已有的测试（文件, 测试名）。这些测试本身在各模块中实现，这里只组织运行。
INTEGRATION_CHECKS = (
    ("结束原因、末状态与 n 步回报（T02 假环境）", "training_validation/training_loop/test_runner.py", ("EpisodeBoundaryTest",)),
    ("结束原因与下一轮奖励隔离（D02 → D03）", "experience_returns/n_step_returns/test_returns.py", ("BoundaryTest",)),
    ("历史窗口切片与原 PPO 一致", "experience_returns/history_windows/test_history.py", ("SplitWindowsTest",)),
    ("导出历史不重复当前帧", "training_validation/export_inference/test_export.py",
     ("ExportConsistencyTest.test_frame_numbered_history_is_not_duplicated",)),
    ("动作单位与物理换算（E03）", "environment_control/action_contract/test_contract.py", ("AcceptanceCaseTest",)),
    ("导出输出到 E03 的单次换算", "training_validation/export_inference/test_export.py",
     ("InterfaceTest.test_metadata_and_single_physical_conversion",)),
    ("网络与学习一次完整更新", "networks_learning/target_updates/test_targets.py", ("UpdateCycleTest",)),
    ("训练、保存、恢复与继续训练", "training_validation/checkpoints/test_checkpoint.py",
     ("FullFlowTest", "RestoreTest.test_exact_replay_reproduces_next_learning")),
    # 以下两项的环境钩子是人工模拟（ScriptedSimulation），检查的是 E05 / E01 与 T02 / D03 / T03 的接口衔接。
    ("E05 重置前末状态进入经验池，接 Trainer 与保存恢复（人工钩子）", "environment_control/episode_lifecycle/test_lifecycle.py",
     ("TrainerIntegrationTest",)),
    ("E01 用运行时元数据填写的 Isaac 配置驱动 Trainer（人工钩子）", "environment_control/robot_environment/test_environment.py",
     ("RunConfigTest.test_trainer_runs_on_filled_config",)),
)
NOT_RUN_LEVELS = {
    "simulation": "本入口未执行 Isaac 仿真；独立 Windows 平地单环境与截断证据见 E01 验证记录，不能由 CPU 结果推定本层通过",
    "performance": "尚未进行 PPO / SAC 训练对照；T04 只提供评价与报告工具",
}


def small_modules(repo_root: Path = REPO_ROOT) -> list[Path]:
    """全部小模块目录（大模块下含 README.md 的子目录），按路径排序。"""
    return sorted(p.parent for big in BIG_MODULES for p in (repo_root / big).glob("*/README.md"))


def _markdown_files(repo_root: Path) -> list[Path]:
    files = [repo_root / name for name in ROOT_DOCUMENTS if (repo_root / name).exists()]
    for big in BIG_MODULES:
        files += sorted((repo_root / big).rglob("README*.md"))
    return [f for f in files if not f.name.startswith("._")]


def check_documentation(repo_root: Path = REPO_ROOT) -> list[dict]:
    """返回文档问题列表（空表示通过）：断开的相对链接、状态与文件不一致、已实现模块残留计划接口表。

    相对链接只检查目标文件或目录是否存在（去掉 #锚点）；外部链接（含 ://）与纯锚点不检查。
    状态一致性：小模块 README 中以“状态：”开头的一行；“待实现”却已有实现文件，或声称已实现却缺少实现或测试文件，都报告。
    """
    issues = []
    for path in _markdown_files(repo_root):
        for target in LINK_PATTERN.findall(path.read_text(encoding="utf-8")):
            if "://" in target or target.startswith(("#", "mailto:")):
                continue
            relative = target.split("#", 1)[0]
            if relative and not (path.parent / relative).exists():
                issues.append({"kind": "broken_link", "file": path.relative_to(repo_root).as_posix(), "target": target})
    for module in small_modules(repo_root):
        text = (module / "README.md").read_text(encoding="utf-8")
        status = next((line for line in text.splitlines() if line.startswith("状态：")), "")
        implementation = [p.name for p in module.glob("*.py") if not p.name.startswith(("test_", "._"))]
        tests = [p.name for p in module.glob("test_*.py")]
        where = module.relative_to(repo_root).as_posix()  # 报告路径统一用 /，避免 Windows 分隔符破坏结果索引。
        if status.startswith("状态：待实现"):
            if implementation:
                issues.append({"kind": "status_mismatch", "module": where, "detail": f"状态为待实现，但已有实现文件 {implementation}"})
        elif status:
            if not implementation or not tests:
                issues.append({"kind": "status_mismatch", "module": where,
                               "detail": f"状态声称已实现，但实现文件 {implementation}、测试文件 {tests} 不全"})
            if PLAN_SECTION in text:
                issues.append({"kind": "stale_plan", "module": where, "detail": "已实现，但仍保留“计划文件与逐函数职责”表"})
        else:
            issues.append({"kind": "status_mismatch", "module": where, "detail": "README 缺少“状态：”行"})
    return issues


def _parse_unittest(output: str, returncode: int) -> dict:
    """从 unittest 输出解析测试数、失败、错误与跳过。没有 “Ran N tests” 行时为 error。"""
    ran = re.search(r"^Ran (\d+) tests? in", output, flags=re.M)
    if not ran:
        tail = output.strip().splitlines()[-1] if output.strip() else ""
        return {"status": "error", "tests": 0, "failures": 0, "errors": 0, "skipped": 0, "detail": tail}
    counts = {key: int(m.group(1)) if (m := re.search(rf"{key}=(\d+)", output)) else 0 for key in ("failures", "errors", "skipped")}
    passed = returncode == 0 and counts["failures"] == 0 and counts["errors"] == 0
    return {"status": "passed" if passed else "failed", "tests": int(ran.group(1)), **counts}


def run_test_file(repo_root: Path, test_file: str, test_names: tuple[str, ...] = (), timeout: int = 600) -> dict:
    """用当前解释器在子进程中运行一个测试文件（可指定测试类或方法），返回解析结果。"""
    command = [sys.executable, "-B", str(repo_root / test_file), *test_names]
    result = subprocess.run(command, cwd=repo_root, capture_output=True, text=True, timeout=timeout)
    return {"file": test_file, "tests_selected": list(test_names), **_parse_unittest(result.stderr + result.stdout, result.returncode)}


def collect_module_results(repo_root: Path = REPO_ROOT, modules: list[Path] | None = None) -> list[dict]:
    """运行每个小模块目录中的 test_*.py；没有测试文件的小模块记为 not_run。"""
    results = []
    for module in modules if modules is not None else small_modules(repo_root):
        tests = sorted(p for p in module.glob("test_*.py") if not p.name.startswith("._"))
        where = module.relative_to(repo_root).as_posix()
        if not tests:
            results.append({"module": where, "status": "not_run", "detail": "没有测试文件"})
        for test in tests:
            results.append({"module": where, **run_test_file(repo_root, str(test.relative_to(repo_root)))})
    return results


def check_integration(repo_root: Path = REPO_ROOT, checks=INTEGRATION_CHECKS) -> list[dict]:
    """运行跨模块回归测试；测试文件不存在时记为 not_run。"""
    results = []
    for name, test_file, test_names in checks:
        if not (repo_root / test_file).exists():
            results.append({"check": name, "file": test_file, "status": "not_run", "detail": "测试文件不存在"})
            continue
        results.append({"check": name, **run_test_file(repo_root, test_file, test_names)})
    return results


def _level(results: list[dict] | None) -> dict:
    if results is None:
        return {"status": "not_run", "detail": "本次未选择该层级"}
    executed = [r for r in results if r["status"] != "not_run"]
    failed = [r for r in executed if r["status"] in ("failed", "error")]
    status = "not_run" if not executed else ("failed" if failed else "passed")
    return {"status": status, "tests": sum(r.get("tests", 0) for r in executed),
            "skipped": sum(r.get("skipped", 0) for r in executed), "not_run_items": len(results) - len(executed),
            "failed_items": failed, "results": results}


def build_summary(documentation: list[dict] | None, unit: list[dict] | None, integration: list[dict] | None) -> dict:
    """按五个层级汇总；documentation 为问题列表（None 表示未检查）。"""
    docs = ({"status": "not_run", "detail": "本次未选择该层级"} if documentation is None
            else {"status": "failed" if documentation else "passed", "issues": documentation})
    return {
        "documentation": docs,
        "unit": _level(unit),
        "cpu_integration": _level(integration),
        **{level: {"status": "not_run", "detail": reason} for level, reason in NOT_RUN_LEVELS.items()},
        "note": "文档链接通过不等于代码正确；CPU 单元与集成检查通过不等于 Isaac 仿真、训练或性能验证完成",
    }


def main(argv: list[str] | None = None) -> int:
    """命令行入口；返回退出码（任一已执行层级失败时为 1）。"""
    parser = argparse.ArgumentParser(description="study 仓库检查入口")
    parser.add_argument("--docs", action="store_true", help="只做文档检查（可与其他选项组合）")
    parser.add_argument("--unit", action="store_true", help="运行各小模块单元测试")
    parser.add_argument("--integration", action="store_true", help="运行 CPU 集成检查")
    parser.add_argument("--json", type=Path, help="把汇总另存为 JSON")
    args = parser.parse_args(argv)
    run_all = not (args.docs or args.unit or args.integration)
    summary = build_summary(
        check_documentation() if run_all or args.docs else None,
        collect_module_results() if run_all or args.unit else None,
        check_integration() if run_all or args.integration else None,
    )
    for level in ("documentation", "unit", "cpu_integration", "simulation", "performance"):
        item = summary[level]
        extra = f"，测试 {item['tests']} 项，跳过 {item['skipped']}" if "tests" in item else ""
        print(f"{level}: {item['status']}{extra}")
        for issue in item.get("issues", []):
            print(f"  - {issue}")
        for failed in item.get("failed_items", []):
            print(f"  - 失败：{failed}")
    print(summary["note"])
    if args.json:
        args.json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 1 if any(summary[level]["status"] == "failed" for level in ("documentation", "unit", "cpu_integration")) else 0


if __name__ == "__main__":
    sys.exit(main())
