"""
vOS-VirtualVoice - Phase 3：跨平台環境偵測 (CUDA / MPS / CPU)

為什麼需要這個模組
------------------
Phase 3 的目標是「確保模型在不同硬體架構下順利跑通」。這件事的第一個
難題不是設定，而是**準確知道自己在什麼硬體上**：

  1. **`torch.cuda.is_available()` 回 False 有兩種完全不同的原因**
     - 機器沒有 NVIDIA GPU
     - 機器有 GPU，但裝到的是 CPU-only 的 PyTorch build（`x.y.z+cpu`）
     兩者的解法天差地遠，但對使用者來說症狀一模一樣。

  2. **Windows 上的 WMI/CIM 查詢可能被權限或政策擋住**
     依賴 `Win32_VideoController` 會在不該失敗的地方失敗。PyTorch 本身
     就是最可靠的探測來源 —— 它已經知道自己的 build 與可見裝置。

  3. **VRAM 決定了一切訓練參數**
     batch size、梯度累積、是否啟用 gradient checkpointing、精度格式，
     全都取決於可用 VRAM。猜錯就是 OOM 或浪費。

這個模組只做唯讀探測，不安裝、不改設定。
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass, field

#: PyTorch build 標記：版本字串帶 `+cpu` 表示這是 CPU-only 建置。
CPU_ONLY_BUILD_MARKER = "+cpu"

#: 常見消費級 GPU 的 VRAM 對照（GB）。
#: 用途：PyTorch 在某些平台上無法直接回報總 VRAM 時，用型號名稱反推。
KNOWN_GPU_VRAM_GB: dict[str, float] = {
    "4090": 24, "4080": 16, "4070 ti": 12, "4070": 12, "4060 ti": 16, "4060": 8,
    "3090": 24, "3080 ti": 12, "3080": 10, "3070 ti": 8, "3070": 8, "3060 ti": 8,
    "3060": 12, "3050": 8,
    "2080 ti": 11, "2080": 8, "2070": 8, "2060": 6,
    "1660 ti": 6, "1660": 6, "1650": 4,
    "1080 ti": 11, "1080": 8, "1070": 8, "1060": 6,
    # 專業／運算卡
    "a100": 40, "a6000": 48, "a5000": 24, "a4000": 16,
    "v100": 16, "t4": 16, "l4": 24, "l40": 48,
    "rtx 6000": 48, "rtx 5000": 32, "rtx 4000": 16,
}


@dataclass
class DeviceInfo:
    """單一運算裝置的資訊。"""

    kind: str                  # "cuda" | "mps" | "cpu"
    name: str
    total_vram_gb: float = 0.0
    free_vram_gb: float = 0.0
    index: int = 0
    compute_capability: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def has_gpu(self) -> bool:
        return self.kind in ("cuda", "mps")


@dataclass
class EnvReport:
    """整台機器的運算環境。"""

    devices: list[DeviceInfo] = field(default_factory=list)
    torch_version: str = ""
    torch_cuda_build: str | None = None
    torch_is_cpu_only_build: bool = False
    python_version: str = ""
    os_name: str = ""
    cpu_count: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def primary(self) -> DeviceInfo:
        """最適合訓練的裝置（cuda > mps > cpu）。"""
        for kind in ("cuda", "mps"):
            for dev in self.devices:
                if dev.kind == kind:
                    return dev
        return self.devices[0] if self.devices else DeviceInfo("cpu", "unknown")

    @property
    def gpu_available(self) -> bool:
        return any(d.has_gpu for d in self.devices)


def _guess_vram_from_name(name: str) -> float:
    """從 GPU 型號名稱推測 VRAM（GB）。找不到就回 0。"""
    lowered = name.lower()
    for needle, gb in KNOWN_GPU_VRAM_GB.items():
        if needle in lowered:
            return float(gb)
    return 0.0


def detect_torch_environment() -> EnvReport:
    """
    用 PyTorch 探測運算環境（跨平台，不依賴 WMI/CIM）。

    刻意把「沒有 GPU」與「裝到 CPU-only build」區分開來 —— 這是本模組
    存在的核心理由，因為兩者的解法完全不同。
    """
    report = EnvReport(
        python_version=platform.python_version(),
        os_name=f"{platform.system()} {platform.release()}",
        cpu_count=os.cpu_count() or 0,
    )

    try:
        import torch
    except Exception as exc:
        report.warnings.append(
            f"未安裝 PyTorch（{type(exc).__name__}）—— Phase 3 之後的階段都需要它。"
            f"安裝方式見 docs/PHASE3-ENVIRONMENT.md"
        )
        report.devices.append(DeviceInfo("cpu", platform.processor() or "CPU"))
        return report

    report.torch_version = getattr(torch, "__version__", "")
    report.torch_cuda_build = getattr(torch.version, "cuda", None)
    report.torch_is_cpu_only_build = CPU_ONLY_BUILD_MARKER in report.torch_version

    # ---- CUDA ----
    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False

    if cuda_available:
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            total_gb = props.total_memory / (1024**3)
            free_gb = total_gb
            try:
                free_bytes, _total = torch.cuda.mem_get_info(idx)
                free_gb = free_bytes / (1024**3)
            except Exception:
                pass
            cap = f"{props.major}.{props.minor}"
            dev = DeviceInfo(
                kind="cuda",
                name=props.name,
                total_vram_gb=round(total_gb, 2),
                free_vram_gb=round(free_gb, 2),
                index=idx,
                compute_capability=cap,
            )
            if props.major < 7:
                dev.notes.append(
                    f"compute capability {cap} 偏舊，混合精度訓練可能不支援或不穩定"
                )
            report.devices.append(dev)
    else:
        # 關鍵診斷：分清楚「沒有 GPU」與「裝錯 build」
        if report.torch_is_cpu_only_build:
            report.warnings.append(
                f"PyTorch 是 CPU-only 建置（{report.torch_version}）—— "
                f"即使這台機器有 NVIDIA GPU，這個 build 也**永遠**看不到它。"
                f"要啟用 GPU 必須改裝對應 CUDA 版本的 PyTorch，"
                f"詳見 docs/PHASE3-ENVIRONMENT.md"
            )
        elif report.torch_cuda_build is None:
            report.warnings.append(
                "PyTorch 未以 CUDA 支援編譯（torch.version.cuda 為 None）"
            )
        else:
            report.warnings.append(
                f"PyTorch 以 CUDA {report.torch_cuda_build} 編譯，但找不到可見的 GPU。"
                f"常見原因：驅動版本過舊、GPU 被其他程序獨占、"
                f"或在容器／遠端桌面工作階段中看不到裝置。"
            )

        if "nvidia" not in (platform.processor() or "").lower():
            report.warnings.append(
                "若確定這台機器有 NVIDIA 顯示卡，上面的訊息代表是**軟體層**"
                "（PyTorch build 或驅動）而非硬體層的問題。"
            )

    # ---- Apple MPS ----
    try:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            report.devices.append(
                DeviceInfo(kind="mps", name="Apple Silicon (MPS)", index=0)
            )
        elif mps is not None and platform.system() == "Darwin":
            report.warnings.append(
                "在 macOS 上但 MPS 不可用 —— 需要 macOS 12.3+ 與 Apple Silicon"
                "（或 AMD GPU 的特定機型）"
            )
    except Exception:
        pass

    # ---- CPU（永遠可用，當作後備） ----
    report.devices.append(
        DeviceInfo(
            kind="cpu",
            name=platform.processor() or platform.machine() or "CPU",
        )
    )
    return report


def render_env_report(report: EnvReport) -> str:
    """人類可讀的環境報告。"""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(" Phase 3 運算環境偵測")
    lines.append("=" * 72)
    lines.append(f" 作業系統   {report.os_name}")
    lines.append(f" Python     {report.python_version}")
    lines.append(f" PyTorch    {report.torch_version or '（未安裝）'}")
    if report.torch_cuda_build:
        lines.append(f" CUDA 編譯   {report.torch_cuda_build}")
    elif report.torch_version:
        lines.append(" CUDA 編譯   （無 —— 這是 CPU-only 建置）")
    lines.append(f" CPU 執行緒  {report.cpu_count}")

    lines.append("\n 可用運算裝置：")
    for dev in report.devices:
        if dev.kind == "cuda":
            lines.append(
                f"   [GPU] {dev.name}  VRAM {dev.total_vram_gb:.1f} GB"
                f"（可用 {dev.free_vram_gb:.1f} GB）  compute {dev.compute_capability}"
            )
        elif dev.kind == "mps":
            lines.append(f"   [GPU] {dev.name}")
        else:
            lines.append(f"   [CPU] {dev.name}（{report.cpu_count} 執行緒）")
        for note in dev.notes:
            lines.append(f"         ! {note}")

    if report.warnings:
        lines.append("\n 注意：")
        for w in report.warnings:
            lines.append(f"   * {w}")

    lines.append("")
    if report.gpu_available:
        dev = report.primary
        vram = dev.total_vram_gb or 8.0
        lines.append(f" 結論：可用 GPU 訓練（{dev.name}，約 {vram:.0f} GB VRAM）")
    else:
        lines.append(" 結論：**只能用 CPU 訓練**")
        lines.append(
            "   GPT-SoVITS 微調在純 CPU 上需要數小時到數天，實務上不可行。"
        )
        lines.append(
            "   建議先解決 GPU 問題（見 docs/PHASE3-ENVIRONMENT.md），"
            "或改用 Google Colab / 雲端 GPU 進行訓練。"
        )
    return "\n".join(lines)


__all__ = [
    "CPU_ONLY_BUILD_MARKER",
    "KNOWN_GPU_VRAM_GB",
    "DeviceInfo",
    "EnvReport",
    "detect_torch_environment",
    "render_env_report",
]