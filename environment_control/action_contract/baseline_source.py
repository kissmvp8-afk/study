"""从固定原版 PPO 快照静态读取 TITA 动作配置，生成 E03 动作表的输入。

职责：
    - 通过 `git show <快照提交>:<路径>` 读取 baseline/ppo 快照中的原文件，并核对快照树哈希。
    - 用 ast 静态解析 rough_env_cfg.py 的 ActionsCfg 与 ddt_robot.py 的 DDT_TITA_CFG.init_state，
      得到 ActionTermSpec 序列与默认关节位置/速度。
    - 命令行运行时打印由原版配置生成的动作表。

不负责：不导入或运行 Isaac Lab，不加载 URDF。因此这里证明的是“原版配置写了什么”，
不是“仿真中实际加载出的关节顺序”；后者需在 E01 接入真实环境后用 ActionManager 再核对。

为什么静态解析：原配置文件依赖 isaaclab（需 Isaac Sim），本仓库编写环境没有这些依赖；
配置中的动作参数都是字面量，静态解析即可无损读取，而不必复制一份手写数值来“自证”。

来源与许可：被读取的文件属于 DDTRobot/DDT_Lab 701a49c（Isaac Lab 项目模板，文件头为
BSD-3-Clause）。本文件不复制其代码，只在运行时读取快照内容。
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    # 本目录不是 Python 包（小模块目录与 README 并列），直接运行时把本目录加入搜索路径。
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from contract import ActionContract, ActionContractError, ActionTermSpec, ControlType, build_action_contract

# 固定来源，与根目录 UPSTREAM.json 的 ppo.snapshot_commit / ppo.tree 一致。
BASELINE_SNAPSHOT_COMMIT = "8fe40aa0ab3afdb8998cab1357f1f795d2cabc3c"
BASELINE_TREE = "88bb121c956f8704302cc5fdfae10e614b77c634"
ENV_CFG_PATH = "source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/robots/tita/rough_env_cfg.py"
FLAT_ENV_CFG_PATH = "source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/robots/tita/flat_env_cfg.py"
ROBOT_CFG_PATH = "source/ddt_lab/ddt_lab/assets/ddt_robot.py"

REPO_ROOT = Path(__file__).resolve().parents[2]


def read_baseline_file(path: str, repo_dir: Path = REPO_ROOT, commit: str = BASELINE_SNAPSHOT_COMMIT) -> str:
    """从本地 Git 对象库读取快照中的一个文件，并确认快照树哈希未变。

    参数：
        path: 快照内相对路径。
        repo_dir: 仓库根目录。
        commit: 快照提交号，默认固定的 baseline/ppo 提交。
    返回：文件文本（UTF-8）。
    异常：
        FileNotFoundError：本机没有 git 可执行文件。
        RuntimeError：提交对象不在本地（需先 `git fetch origin baseline/ppo`）、路径不存在，
            或树哈希与 BASELINE_TREE 不一致（说明读到的不是登记过的快照）。
    副作用：调用 git 子进程，只读。
    """

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), *args], capture_output=True, text=True, encoding="utf-8"
        )
        if result.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} 失败：{result.stderr.strip()}")
        return result.stdout

    tree = git("rev-parse", f"{commit}^{{tree}}").strip()
    if commit == BASELINE_SNAPSHOT_COMMIT and tree != BASELINE_TREE:
        raise RuntimeError(f"快照 {commit} 的树哈希为 {tree}，与登记的 {BASELINE_TREE} 不符")
    return git("show", f"{commit}:{path}")


def _find_class(module: ast.Module, name: str) -> ast.ClassDef:
    """在模块顶层查找类定义；找不到时报错，避免静默得到空动作表。"""
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise ActionContractError(f"源文件中没有找到类 {name}")


def _call_name(call: ast.Call) -> str:
    """取调用表达式的末段名称，如 mdp.JointPositionActionCfg(...) → "JointPositionActionCfg"。"""
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    raise ActionContractError(f"无法识别的调用形式：{ast.dump(func)}")


def _literal_keywords(call: ast.Call) -> dict[str, Any]:
    """把调用的关键字参数按字面量求值；遇到非字面量（变量、函数调用）直接报错而不猜测。"""
    values: dict[str, Any] = {}
    for kw in call.keywords:
        try:
            values[kw.arg] = ast.literal_eval(kw.value)
        except ValueError as exc:
            raise ActionContractError(f"参数 {kw.arg} 不是字面量，静态解析无法确定其值") from exc
    return values


def assert_actions_not_modified(source: str, filename: str) -> None:
    """确认源文件中没有任何 `self.actions...` 访问。

    子类 __post_init__ 若改写 self.actions，静态读取 ActionsCfg 就不再代表最终配置。
    原版 rough/flat 配置都没有这样做；一旦上游改动，这里会报错提醒重新核对。
    """
    for node in ast.walk(ast.parse(source, filename=filename)):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "actions"
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            raise ActionContractError(f"{filename} 第 {node.lineno} 行访问了 self.actions，动作配置可能被运行时修改")


def parse_action_terms(env_cfg_source: str, class_name: str = "ActionsCfg") -> list[ActionTermSpec]:
    """解析环境配置中的动作项，按类体声明顺序返回。

    声明顺序就是动作向量顺序：Isaac Lab 的 configclass 保持字段声明顺序，
    ActionManager._prepare_terms 依次遍历 cfg.__dict__ 并拼接各项（核对版本 Isaac Lab v2.3.0）。
    值为 None 的字段在 Isaac Lab 中被跳过，这里同样跳过。

    参数：
        env_cfg_source: rough_env_cfg.py 的文本。
        class_name: 动作配置类名。
    返回：ActionTermSpec 列表。
    异常：ActionContractError：出现不支持的动作类、非字面量参数或缺少必需字段。
    """
    cls = _find_class(ast.parse(env_cfg_source), class_name)
    terms: list[ActionTermSpec] = []
    for stmt in cls.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            name, value = stmt.targets[0].id, stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
            name, value = stmt.target.id, stmt.value
        else:
            continue  # 文档字符串、注释掉的旧动作项等
        if isinstance(value, ast.Constant) and value.value is None:
            continue
        if not isinstance(value, ast.Call):
            raise ActionContractError(f"动作项 {name} 不是配置构造调用，无法静态解析")
        control_type = ControlType.from_cfg_class_name(_call_name(value))
        kw = _literal_keywords(value)
        if "joint_names" not in kw or "scale" not in kw:
            raise ActionContractError(f"动作项 {name} 缺少 joint_names 或 scale")
        if not isinstance(kw["scale"], (int, float)):
            raise ActionContractError(f"动作项 {name} 的 scale 不是标量，E03 暂不支持按关节字典缩放")
        terms.append(
            ActionTermSpec(
                term_name=name,
                control_type=control_type,
                joint_names=tuple(kw["joint_names"]),
                scale=float(kw["scale"]),
                # 以下缺省值与 Isaac Lab v2.3.0 JointActionCfg 的字段默认值一致。
                use_default_offset=bool(kw.get("use_default_offset", True)),
                offset=float(kw.get("offset", 0.0)),
                clip=kw.get("clip"),
                preserve_order=bool(kw.get("preserve_order", False)),
            )
        )
    return terms


def parse_init_state(robot_cfg_source: str, cfg_name: str = "DDT_TITA_CFG") -> tuple[dict[str, float], dict[str, float]]:
    """解析机器人配置的 init_state，返回 (joint_pos, joint_vel) 两个“关节名正则 → 数值”字典。

    joint_pos 单位 rad，joint_vel 单位 rad/s；原版 joint_vel 为 {".*": 0.0}，展开由
    contract.resolve_joint_values 完成。
    异常：ActionContractError：找不到配置或缺少 joint_pos / joint_vel。
    """
    for node in ast.parse(robot_cfg_source).body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == cfg_name for t in node.targets)
            and isinstance(node.value, ast.Call)
        ):
            for kw in node.value.keywords:
                if kw.arg == "init_state" and isinstance(kw.value, ast.Call):
                    init = _literal_keywords(kw.value)
                    if "joint_pos" not in init or "joint_vel" not in init:
                        raise ActionContractError(f"{cfg_name}.init_state 缺少 joint_pos 或 joint_vel")
                    return (
                        {k: float(v) for k, v in init["joint_pos"].items()},
                        {k: float(v) for k, v in init["joint_vel"].items()},
                    )
    raise ActionContractError(f"源文件中没有找到 {cfg_name} 的 init_state")


def load_baseline_action_contract(repo_dir: Path = REPO_ROOT) -> ActionContract:
    """读取固定快照并建立原版 TITA 动作表（未调用 validate，由调用方决定是否校验）。

    同时检查 rough 与 flat 配置都没有在运行时改写动作，保证两类任务共用同一张动作表。
    异常与副作用：同 read_baseline_file 与各解析函数。
    """
    env_src = read_baseline_file(ENV_CFG_PATH, repo_dir)
    flat_src = read_baseline_file(FLAT_ENV_CFG_PATH, repo_dir)
    robot_src = read_baseline_file(ROBOT_CFG_PATH, repo_dir)
    assert_actions_not_modified(env_src, ENV_CFG_PATH)
    assert_actions_not_modified(flat_src, FLAT_ENV_CFG_PATH)
    joint_pos, joint_vel = parse_init_state(robot_src)
    return build_action_contract(parse_action_terms(env_src), joint_pos, joint_vel)


if __name__ == "__main__":
    from contract import describe_action_contract, validate_action_contract

    contract = load_baseline_action_contract()
    validate_action_contract(contract)
    print(f"来源：baseline/ppo 快照 {BASELINE_SNAPSHOT_COMMIT}（树 {BASELINE_TREE}）")
    print(describe_action_contract(contract))
