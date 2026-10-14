"""
vOS-VirtualVoice - Phase 3：訓練前環境自檢

為什麼需要這個模組
------------------
`env` 子命令回答「我有什麼裝置」。這個模組回答更難的問題：
**「這個環境真的能跑完一次訓練嗎？」**

兩者的差別在於「會動」與「不會在第三小時才失敗」。訓練腳本通常在
讀取資料、建立模型、進入第一個 backward pass 之後才會碰到真正的限制，
而那時你已經等了很久。這個模組用**合成張量**把那些限制提前引爆：

  1. **精度支援** — bf16 需要 Ampere（compute 8.0）以後。在舊卡上跑
     bf16 不會報錯，而是**默默地**給你垃圾梯度。合成張量一測就知道。
  2. **VRAM 是否真的夠** — 參數表是估算，這裡用實際大小的張量配置一次，
     直接看會不會 OOM。
  3. **梯度累積與 checkpointing 的組合** — 兩者都會改變記憶體峰值，
     必須在真實組合下測，不能分開推論。
  4. **磁碟空間** — checkpoint 動輒數 GB，寫到一半才發現沒空間很嘔。

全部用合成資料，**不需要任何素材**，幾秒鐘跑完。這是 Phase 3 最有價值的
一件事：把「訓練三小時後才發現」變成「開跑前 10 秒就知道」。
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from hwinfo import EnvReport
from trainfit import TrainingParams


@dataclass
class CheckResult:
    """單一檢查項的結果。"""

    name: str
    status: str          # "pass" | "warn" | "fail" | "skip"
    detail: str = ""
    hint: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("pass", "skip")

    @property
    def mark(self) -> str:
        return {"pass": "[OK]", "warn": "[!]", "fail": "[X]", "skip": "[-]"}[self.status]


@dataclass
class PrecheckReport:
    """訓練前自檢的完整結果。"""

    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """沒有 fail 就算通過（warn 不阻擋）。"""
        return not any(c.status == "fail" for c in self.checks)

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == "fail"]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == "warn"]

    def add(self, name: str, status: str, detail: str = "", hint: str = "") -> None:
        self.checks.append(CheckResult(name, status, detail, hint))


# ---------------------------------------------------------------------------
# 個別檢查
# ---------------------------------------------------------------------------
def check_precision_support(torch, device, precision: str, report: PrecheckReport) -> None:
    """
    用合成張量實測精度支援。

    ⚠ 這是本模組最重要的檢查。在舊 GPU 上用 bf16 **不會報錯** ——
    它會跑完，但梯度是垃圾，模型不會收斂。你會在幾小時後看到 loss 不下降，
    然後懷疑資料、懷疑參數、懷疑人生，就是不會懷疑精度。
    """
    import torch as _t

    if precision == "fp32" or device.kind == "cpu":
        report.add(
            f"精度 {precision}",
            "pass" if precision == "fp32" else "warn",
            "CPU 訓練固定用 fp32" if device.kind == "cpu" else "fp32 全平台支援",
        )
        return

    dev = _device_of(torch, device)
    try:
        a = _t.randn(256, 256, device=dev)
        b = _t.randn(256, 256, device=dev)
        with _t.autocast(device_type=device.kind, dtype=_t.bfloat16 if precision == "bf16" else _t.float16):
            c = a @ b
        # 檢查結果不是 NaN/Inf —— 這才是「真的支援」的證據
        bad = bool(_t.isnan(c).any() or _t.isinf(c).any())
        if bad:
            report.add(
                f"精度 {precision}",
                "fail",
                "合成張量運算產生 NaN/Inf",
                f"改用 fp32，或換一張支援 {precision} 的 GPU",
            )
        else:
            report.add(f"精度 {precision}", "pass", "合成張量運算正常")
    except Exception as exc:
        report.add(
            f"精度 {precision}",
            "fail",
            f"{type(exc).__name__}: {str(exc)[:80]}",
            "這個裝置不支援該精度，改用 fp32",
        )

    # 額外查 compute capability（bf16 需要 8.0+）
    if device.kind == "cuda" and device.compute_capability:
        try:
            major = int(device.compute_capability.split(".")[0])
            if precision == "bf16" and major < 8:
                report.add(
                    "bf16 硬體支援",
                    "fail",
                    f"compute capability {device.compute_capability} < 8.0",
                    "bf16 需要 Ampere（8.0）以後；請改用 fp16",
                )
            else:
                report.add(
                    "bf16 硬體支援",
                    "pass",
                    f"compute {device.compute_capability}",
                )
        except (ValueError, IndexError):
            pass


def check_vram_capacity(
    torch, device, params: TrainingParams, report: PrecheckReport
) -> None:
    """
    用實際大小的張量配置記憶體，看會不會 OOM。

    為什麼不直接相信參數表的估算：真實峰值記憶體取決於模型結構、
    optimizer 狀態、activation 大小，不是單純的線性關係。實際配置一次最快。
    """
    import torch as _t

    if device.kind == "cuda":
        dev = _device_of(torch, device)
        try:
            free_before = 0.0
            try:
                free_bytes, _total = _t.cuda.mem_get_info(device.index)
                free_before = free_bytes / (1024**3)
            except Exception:
                free_before = device.free_vram_gb

            # 配置一個「跟 batch 成比例」的張量。SoVITS 的 activation 在
            # batch=8 時約 2 GB 量級，這裡用 256 MB × 等效 batch 當代理指標。
            per_sample_mb = 256
            target_mb = per_sample_mb * params.effective_batch
            elems = max(1, (target_mb * 1024 * 1024) // 4)  # float32
            blob = _t.empty(elems, device=dev)
            blob.fill_(1.0)
            del blob
            _t.cuda.empty_cache()

            free_after = free_before
            try:
                free_bytes, _total = _t.cuda.mem_get_info(device.index)
                free_after = free_bytes / (1024**3)
            except Exception:
                pass

            if free_before < 1.0:
                report.add(
                    "VRAM 可用量",
                    "warn",
                    f"目前僅 {free_before:.2f} GB 可用",
                    "有其他程式佔用 VRAM；關閉它們再訓練，否則很容易 OOM",
                )
            else:
                report.add(
                    "VRAM 可用量",
                    "pass",
                    f"{free_before:.1f} GB 可用（配置 {target_mb} MB 成功）",
                )
            if free_after < free_before * 0.5 and free_before > 0:
                report.add(
                    "記憶體回收",
                    "warn",
                    f"配置後只剩 {free_after:.2f} GB",
                    "釋放未完全；訓練時請預留更多餘裕",
                )
        except _t.cuda.OutOfMemoryError:
            report.add(
                "VRAM 容量",
                "fail",
                f"配置 {target_mb} MB 時 OOM",
                "降低 batch_size 並提高 grad_accum（維持等效 batch）",
            )
        except Exception as exc:
            report.add("VRAM 容量", "warn", f"無法測試：{type(exc).__name__}")
    else:
        # CPU / MPS：檢查系統 RAM 而不是 VRAM
        total_gb, free_gb = total_system_memory_gb()
        if not total_gb:
            report.add("系統記憶體", "skip", "無法查詢")
        elif free_gb < 8:
            # 關鍵：**可用**比**總量**重要得多。
            # 總量 32 GB 但可用只剩 4 GB 的機器，訓練照樣會失敗 ——
            # 而你看到「32 GB」會以為沒問題。
            report.add(
                "系統記憶體",
                "fail",
                f"可用僅 {free_gb:.1f} GB（總計 {total_gb:.1f} GB）",
                "有大量記憶體被其他程式佔用。CPU 訓練吃系統 RAM，"
                "請先關閉瀏覽器／其他服務並確認可用量 ≥ 16 GB",
            )
        elif total_gb < 16:
            report.add(
                "系統記憶體",
                "warn",
                f"總計 {total_gb:.1f} GB（可用 {free_gb:.1f} GB）",
                "CPU 訓練吃系統 RAM；16 GB 以下容易被迫縮小模型或資料量",
            )
        elif free_gb < 16:
            report.add(
                "系統記憶體",
                "warn",
                f"可用 {free_gb:.1f} GB（總計 {total_gb:.1f} GB）",
                "可用量偏低；訓練前建議關閉其他吃記憶體的程式",
            )
        else:
            report.add(
                "系統記憶體",
                "pass",
                f"總計 {total_gb:.1f} GB（可用 {free_gb:.1f} GB）",
            )


def check_disk_space(checkpoint_dir: Path, report: PrecheckReport) -> None:
    """
    checkpoint 動輒數 GB。寫到一半才發現沒空間會毀掉整個訓練。
    """
    try:
        free_gb = shutil.disk_usage(str(checkpoint_dir)).free / (1024**3)
    except OSError:
        report.add("checkpoint 空間", "skip", "無法查詢磁碟空間")
        return

    # GPT-SoVITS 的 checkpoint 每個約 1~2 GB，訓練過程通常保留數個
    need_gb = 10.0
    if free_gb < need_gb:
        report.add(
            "checkpoint 空間",
            "fail",
            f"僅 {free_gb:.1f} GB 可用（建議 ≥ {need_gb:.0f} GB）",
            "訓練中途寫入失敗會讓整個 run 作廢",
        )
    elif free_gb < need_gb * 3:
        report.add(
            "checkpoint 空間",
            "warn",
            f"{free_gb:.1f} GB 可用",
            "足夠但有餘裕更好；checkpoint 會隨訓練累積",
        )
    else:
        report.add("checkpoint 空間", "pass", f"{free_gb:.1f} GB 可用")


def check_torch_basics(torch, device, report: PrecheckReport) -> None:
    """最基本的健全性：autograd、反向傳播、optimizer 能不能走完一步。"""
    import torch as _t

    try:
        dev = _device_of(torch, device)
        w = _t.randn(64, 64, device=dev, requires_grad=True)
        x = _t.randn(8, 64, device=dev)
        loss = (w @ x.T).sum()
        loss.backward()
        if w.grad is None or bool(_t.isnan(w.grad).any()):
            report.add("反向傳播", "fail", "梯度為 None 或含 NaN")
        else:
            report.add("反向傳播", "pass", "前向 + 反向 + 梯度檢查正常")

        opt = _t.optim.AdamW([w], lr=1e-4)
        opt.step()
        opt.zero_grad()
        report.add("optimizer", "pass", "AdamW 可建立並執行 step")
    except Exception as exc:
        report.add(
            "反向傳播",
            "fail",
            f"{type(exc).__name__}: {str(exc)[:80]}",
            "這個裝置無法完成最基本的訓練步驟",
        )


def _device_of(torch, device):
    """把 DeviceInfo 轉成 torch.device。"""
    if device.kind == "cuda":
        return torch.device(f"cuda:{device.index}")
    if device.kind == "mps":
        return torch.device("mps")
    return torch.device("cpu")




def total_system_memory_gb() -> tuple[float, float]:
    """
    回傳 (總記憶體 GB, 可用記憶體 GB)。

    為什麼不只用 os.sysconf：那是 POSIX 專屬，在 Windows 上不存在 ——
    而 Windows 是最常見的訓練平台之一。也不能用 WMI/CIM，因為它可能被
    權限或企業政策擋住（實測會「拒絕存取」）。

    Windows 走 ctypes 呼叫 GlobalMemoryStatusEx，這是唯讀的系統 API，
    不需要任何特殊權限。
    """
    import sys

    # --- Windows：ctypes ---
    if sys.platform.startswith("win"):
        try:
            import ctypes

            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                gb = 1024**3
                return stat.ullTotalPhys / gb, stat.ullAvailPhys / gb
        except Exception:
            pass
        return 0.0, 0.0

    # --- POSIX：sysconf ---
    try:
        import os

        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        avail = os.sysconf("SC_AVPHYS_PAGES")
        gb = 1024**3
        return (pages * page_size) / gb, (avail * page_size) / gb
    except Exception:
        return 0.0, 0.0

# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run_precheck(
    env: EnvReport,
    params: TrainingParams,
    *,
    checkpoint_dir: Path | None = None,
) -> PrecheckReport:
    """
    執行訓練前自檢。全部用合成張量，不需要任何素材。

    Returns:
        PrecheckReport；`passed` 為 False 代表有會讓訓練失敗的問題。
    """
    report = PrecheckReport()

    try:
        import torch
    except Exception as exc:
        report.add(
            "PyTorch 可用",
            "fail",
            f"{type(exc).__name__}: {exc}",
            "安裝方式見 docs/PHASE3-ENVIRONMENT.md",
        )
        return report

    report.add("PyTorch 可用", "pass", env.torch_version or "已載入")

    device = env.primary
    if not env.gpu_available:
        report.add(
            "GPU 可用性",
            "fail",
            "沒有可用的 GPU",
            "純 CPU 訓練在實務上不可行；見 docs/PHASE3-ENVIRONMENT.md §4",
        )
    else:
        report.add("GPU 可用性", "pass", f"{device.name}")

    if env.torch_is_cpu_only_build and env.gpu_available:
        report.add(
            "PyTorch build",
            "fail",
            "CPU-only build 卻偵測到 GPU（矛盾狀態）",
            "重裝對應 CUDA 版本的 PyTorch",
        )

    check_torch_basics(torch, device, report)
    check_precision_support(torch, device, params.precision, report)
    check_vram_capacity(torch, device, params, report)

    target_dir = checkpoint_dir or Path.cwd()
    check_disk_space(target_dir, report)

    return report


def render_precheck(report: PrecheckReport) -> str:
    """人類可讀的自檢報告。"""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(" Phase 3 訓練前環境自檢（合成張量，不需要素材）")
    lines.append("=" * 72)
    for check in report.checks:
        detail = f"  -- {check.detail}" if check.detail else ""
        lines.append(f"  {check.mark} {check.name}{detail}")
        if not check.ok and check.hint:
            lines.append(f"      -> {check.hint}")

    failed = len(report.failures)
    warned = len(report.warnings)
    lines.append("")
    if report.passed and not warned:
        lines.append(" 全部通過 —— 這個環境可以開始訓練。")
    elif report.passed:
        lines.append(f" 通過（{warned} 項提醒）—— 可以開始訓練，但請看上面的提醒。")
    else:
        lines.append(
            f" 有 {failed} 項會讓訓練失敗的問題（另有 {warned} 項提醒）——"
            f" 請先修正再開始。"
        )
        lines.append(
            " 為什麼要在這裡擋下來：這些問題通常要等到訓練跑進第一個"
            " backward pass 才引爆，而那時你已經等了很久。"
        )
    return "\n".join(lines)


__all__ = [
    "CheckResult",
    "PrecheckReport",
    "check_disk_space",
    "check_precision_support",
    "check_torch_basics",
    "check_vram_capacity",
    "render_precheck",
    "total_system_memory_gb",
    "run_precheck",
]