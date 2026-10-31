"""T06 检查入口的验证程序（unittest + PyTorch CPU）。

运行：在仓库根目录执行
    uv run --no-project --python 3.11 --with torch==2.15.0 python -B training_validation/verification/test_verify.py -v

数据说明：文档与测试结果的检查使用临时目录中人工构造的 README 与测试文件；集成检查的变异案例把真实的
经验与回报模块复制到临时目录后故意改错 D05 的一处切片，确认集成检查能定位该错误。
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from verify import build_summary, check_documentation, check_integration, collect_module_results, small_modules  # noqa: E402


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text), encoding="utf-8")


class DocumentationTest(unittest.TestCase):
    """断开的链接、状态与文件不一致、残留计划表都能被发现。"""

    def test_detects_documentation_problems(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            write(repo / "README.md", "[好](environment_control/a/README.md) [坏](missing.md) [外部](https://example.com) [锚](#x)\n")
            write(repo / "environment_control/a/README.md", "# A\n\n状态：单元验证通过。\n\n[人类版](README_HUMAN.md#top)\n")
            write(repo / "environment_control/a/README_HUMAN.md", "# A\n")
            write(repo / "environment_control/a/impl.py", "")
            write(repo / "environment_control/a/test_impl.py", "")
            write(repo / "environment_control/b/README.md", "# B\n\n状态：单元验证通过。\n")  # 无实现、无测试
            write(repo / "environment_control/c/README.md", "# C\n\n状态：待实现。\n")
            write(repo / "environment_control/c/impl.py", "")  # 待实现却有实现文件
            write(repo / "environment_control/d/README.md", "# D\n\n状态：单元验证通过。\n\n## 计划文件与逐函数职责\n")
            write(repo / "environment_control/d/impl.py", "")
            write(repo / "environment_control/d/test_impl.py", "")
            issues = check_documentation(repo)
        kinds = sorted((i["kind"], i.get("target") or i.get("module")) for i in issues)
        self.assertEqual(kinds, [
            ("broken_link", "missing.md"),
            ("stale_plan", "environment_control/d"),
            ("status_mismatch", "environment_control/b"),
            ("status_mismatch", "environment_control/c"),
        ])

    def test_lists_all_small_modules(self) -> None:
        self.assertEqual(len(small_modules(ROOT)), 24)  # 5 + 5 + 8 + 6 个小模块


class ResultCollectionTest(unittest.TestCase):
    """解析已有测试的结果：通过、失败、跳过、导入错误与没有测试文件。"""

    def test_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            head = "import unittest\n"
            write(repo / "training_validation/ok/README.md", "状态：单元验证通过。\n")
            write(repo / "training_validation/ok/test_ok.py", head + textwrap.dedent("""
                class T(unittest.TestCase):
                    def test_a(self): pass
                    @unittest.skip("示例跳过")
                    def test_b(self): pass
                unittest.main()
            """))
            write(repo / "training_validation/bad/README.md", "状态：单元验证通过。\n")
            write(repo / "training_validation/bad/test_bad.py", head + textwrap.dedent("""
                class T(unittest.TestCase):
                    def test_a(self): self.assertEqual(1, 2)
                    def test_b(self): raise RuntimeError("x")
                unittest.main()
            """))
            write(repo / "training_validation/broken/README.md", "状态：单元验证通过。\n")
            write(repo / "training_validation/broken/test_broken.py", "import module_that_does_not_exist\n")
            write(repo / "training_validation/none/README.md", "状态：待实现。\n")
            results = {r["module"].split("/")[-1]: r for r in collect_module_results(repo)}
        self.assertEqual((results["ok"]["status"], results["ok"]["tests"], results["ok"]["skipped"]), ("passed", 2, 1))
        self.assertEqual((results["bad"]["status"], results["bad"]["failures"], results["bad"]["errors"]), ("failed", 1, 1))
        self.assertEqual(results["broken"]["status"], "error")
        self.assertEqual(results["none"]["status"], "not_run")


class IntegrationTest(unittest.TestCase):
    """集成检查运行已有回归测试，并能定位注入的跨模块错误。"""

    def test_detects_injected_history_misalignment(self) -> None:
        check = (("历史窗口切片与原 PPO 一致", "experience_returns/history_windows/test_history.py", ("SplitWindowsTest",)),)
        self.assertEqual(check_integration(ROOT, check)[0]["status"], "passed")
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            # test_history.py 会导入 D01～D04，因此复制整个经验与回报大模块（这些模块不依赖 git 快照）。
            shutil.copytree(ROOT / "experience_returns", repo / "experience_returns", ignore=shutil.ignore_patterns("._*", "__pycache__"))
            history = repo / "experience_returns/history_windows/history.py"
            source = history.read_text(encoding="utf-8")
            broken = source.replace("recent=history[:, end - WINDOW_FRAMES : end],", "recent=history[:, end - WINDOW_FRAMES - 1 : end - 1],")
            self.assertNotEqual(broken, source)  # 变异确实生效：决策窗口错位一帧
            history.write_text(broken, encoding="utf-8")
            result = check_integration(repo, check)[0]
        self.assertEqual(result["status"], "failed")
        self.assertGreater(result["failures"], 0)

    def test_missing_test_file_is_not_run(self) -> None:
        result = check_integration(ROOT, (("不存在的检查", "training_validation/none/test_none.py", ()),))[0]
        self.assertEqual(result["status"], "not_run")


class SummaryTest(unittest.TestCase):
    """五个层级分开记录；仿真与性能未执行；任何失败都会体现在对应层级。"""

    def test_levels(self) -> None:
        summary = build_summary([], [{"module": "m", "status": "passed", "tests": 3, "skipped": 1}],
                                [{"check": "c", "status": "failed", "tests": 2}])
        self.assertEqual(summary["documentation"]["status"], "passed")
        self.assertEqual((summary["unit"]["status"], summary["unit"]["tests"], summary["unit"]["skipped"]), ("passed", 3, 1))
        self.assertEqual(summary["cpu_integration"]["status"], "failed")
        self.assertEqual((summary["simulation"]["status"], summary["performance"]["status"]), ("not_run", "not_run"))
        self.assertEqual(build_summary(None, None, None)["unit"]["status"], "not_run")


if __name__ == "__main__":
    unittest.main()
