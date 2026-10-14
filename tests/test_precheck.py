"""
vOS-VirtualVoice - Phase 3 訓練前自檢測試

守著一個核心價值：**把「訓練三小時後才發現」變成「開跑前 10 秒就知道」。**

訓練腳本通常在讀完資料、建好模型、進入第一個 backward pass 之後才會
碰到真正的限制。這組測試確保自檢能在那之前把問題找出來，而且找得準
（不誤報、不漏報）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from hwinfo import DeviceInfo, EnvReport  # noqa: E402
from precheck import (  # noqa: E402
    CheckResult,
    PrecheckReport,
    render_precheck,
    run_precheck,
    total_system_memory_gb,
)
from trainfit import fit_training_params  # noqa: E402


def _cpu_env(cpu_count: int = 8) -> EnvReport:
    rep = EnvReport(cpu_count=cpu_count, torch_version="2.13.0+cpu")
    rep.torch_is_cpu_only_build = True
    rep.devices.append(DeviceInfo("cpu", "Test CPU"))
    return rep


class TestReportModel:
    def test_passed_ignores_warnings(self) -> None:
        """warn 不該阻擋訓練 —— 只有 fail 才算。"""
        rep = PrecheckReport()
        rep.add("a", "pass")
        rep.add("b", "warn")
        assert rep.passed
        assert len(rep.warnings) == 1
        assert not rep.failures

    def test_fail_blocks(self) -> None:
        rep = PrecheckReport()
        rep.add("a", "pass")
        rep.add("b", "fail", "壞了")
        assert not rep.passed
        assert len(rep.failures) == 1

    def test_skip_does_not_block(self) -> None:
        """無法查詢的東西不該被當成失敗（例如非 POSIX 的記憶體查詢）。"""
        rep = PrecheckReport()
        rep.add("a", "skip", "無法查詢")
        assert rep.passed

    @pytest.mark.parametrize(
        "status,expect_ok",
        [
            ("pass", True),
            ("skip", True),
            # warn 刻意**不算** ok：它代表「需要注意」，只是不阻擋訓練。
            # 若把 warn 當成 ok，渲染時就不會標記它，使用者會漏看。
            ("warn", False),
            ("fail", False),
        ],
    )
    def test_ok_property(self, status: str, expect_ok: bool) -> None:
        assert CheckResult("x", status).ok is expect_ok


class TestSystemMemory:
    def test_returns_positive_values_on_this_machine(self) -> None:
        """本模組必須能在 Windows 上工作 —— 那是主要的訓練平台。"""
        total, free = total_system_memory_gb()
        assert total > 0, "應能查到總記憶體（Windows 走 ctypes，POSIX 走 sysconf）"
        assert free >= 0
        assert free <= total, f"可用 {free} 不該大於總量 {total}"

    def test_does_not_require_wmi(self) -> None:
        """
        不得依賴 WMI/CIM —— 它可能被權限或企業政策擋住。
        本機實測 Win32_OperatingSystem 查詢就是「拒絕存取」。
        """
        import inspect

        src = inspect.getsource(total_system_memory_gb)
        # 注意：docstring 裡「說明為什麼不用 WMI」是良好註解，不該被判為違規。
        # 所以要檢查的是「有沒有真的呼叫它」，而不是字串是否出現。
        code_lines = [
            ln for ln in src.splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        body = "\n".join(code_lines)
        assert "import wmi" not in body and "Win32_OperatingSystem" not in body, (
            "不可實際呼叫 WMI/CIM（本機實測會拒絕存取）"
        )
        assert "ctypes" in body, "Windows 應走 ctypes"


class TestPrecheck:
    def test_cpu_only_environment_fails_with_guidance(self) -> None:
        """純 CPU 環境必須 fail，並指向說明文件。"""
        env = _cpu_env()
        report = run_precheck(env, fit_training_params(env))
        assert not report.passed
        gpu_checks = [c for c in report.checks if "GPU" in c.name]
        assert gpu_checks and gpu_checks[0].status == "fail"
        assert "PHASE3" in gpu_checks[0].hint or "CPU" in gpu_checks[0].hint

    def test_runs_backward_pass_on_cpu(self) -> None:
        """CPU 上仍要驗證 autograd 能走完一步 —— 這是訓練的最低門檻。"""
        env = _cpu_env()
        report = run_precheck(env, fit_training_params(env))
        names = {c.name: c for c in report.checks}
        assert names["反向傳播"].status == "pass"
        assert names["optimizer"].status == "pass"

    def test_reports_pytorch_version(self) -> None:
        env = _cpu_env()
        report = run_precheck(env, fit_training_params(env))
        assert any("PyTorch" in c.name for c in report.checks)

    def test_disk_space_checked(self) -> None:
        env = _cpu_env()
        report = run_precheck(env, fit_training_params(env), checkpoint_dir=REPO)
        assert any("空間" in c.name for c in report.checks)

    def test_missing_pytorch_is_reported_not_raised(self, monkeypatch) -> None:
        """
        PyTorch 缺席時必須回報，而不是拋例外 ——
        自檢的價值就在於「給出可行動的訊息」。
        """
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("simulated missing torch")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        env = _cpu_env()
        report = run_precheck(env, fit_training_params(env))
        assert not report.passed
        assert any("PyTorch" in c.name and c.status == "fail" for c in report.checks)

    def test_never_raises_on_any_environment(self) -> None:
        """各種奇怪的環境都不該讓自檢崩潰。"""
        for dev in (
            DeviceInfo("cpu", "C"),
            DeviceInfo("cuda", "G", total_vram_gb=0.0),
            DeviceInfo("mps", "M"),
        ):
            rep = EnvReport(cpu_count=4, torch_version="2.0.0")
            rep.devices.append(dev)
            report = run_precheck(rep, fit_training_params(rep))
            assert isinstance(report, PrecheckReport)


class TestRendering:
    def test_shows_every_check_with_mark(self) -> None:
        env = _cpu_env()
        text = render_precheck(run_precheck(env, fit_training_params(env)))
        assert "[OK]" in text
        assert "[X]" in text

    def test_failure_message_explains_why_early_detection_matters(self) -> None:
        """
        失敗訊息必須說明「為什麼要在這裡擋下來」——
        否則使用者會覺得只是又多一個紅字。
        """
        env = _cpu_env()
        text = render_precheck(run_precheck(env, fit_training_params(env)))
        assert "backward" in text or "訓練" in text

    def test_all_pass_message_is_clear(self) -> None:
        rep = PrecheckReport()
        rep.add("a", "pass", "ok")
        text = render_precheck(rep)
        assert "可以開始訓練" in text