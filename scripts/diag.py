"""
vOS-VirtualVoice - 工具鏈自我診斷 (Phase 1)

為什麼需要這個模組
------------------
Phase 1 只有兩個外部依賴（vgmstream-cli、ffmpeg），但它們的**安裝失敗方式
極具誤導性**。這個模組把「看起來像別的問題」的症狀變成一則明確的診斷：

  1. **vgmstream 找到 exe 但少了 DLL**
     Windows 上 vgmstream-cli 需要 10 個以上的 .dll 在**同一個資料夾**。
     少了 libvorbis.dll，Wwise 的 Vorbis 語音會解不出來 —— 而錯誤訊息
     通常只是「不支援此格式」，非常容易誤判成「這些素材壞了」。

  2. **設定檔指到一個路徑，但 doctor 說找不到**
     設定檔的 unpack.vgmstream_bin 沒有傳進檢查，於是使用者明明設好了
     卻看到紅字，接著開始懷疑設定檔沒生效。

  3. **exe 存在但根本跑不起來**（架構不符、缺 VC runtime、被防毒隔離）
     只檢查 is_file() 會漏掉這種情況。

本模組只做「觀察與回報」，不修改任何東西。
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from config import ENV_TOOL_OVERRIDES, PipelineConfig, find_external_bin

#: Windows 上 vgmstream-cli 需要的執行期 DLL（取自官方 doc/USAGE.md）。
#: 這些檔案必須與 vgmstream-cli.exe 放在同一個資料夾。
WINDOWS_VGMSTREAM_DLLS: tuple[str, ...] = (
    "libvorbis.dll",
    "libmpg123-0.dll",
    "libg719_decode.dll",
    "avcodec-vgmstream-59.dll",
    "avformat-vgmstream-59.dll",
    "avutil-vgmstream-57.dll",
    "swresample-vgmstream-4.dll",
    "libatrac9.dll",
    "libcelt-0061.dll",
    "libcelt-0110.dll",
    "libspeex-1.dll",
)

#: 這幾個 DLL 缺席時的症狀最容易被誤判，所以特別點名。
CRITICAL_DLLS: dict[str, str] = {
    "libvorbis.dll": "Wwise 的 Vorbis 語音會解不出來（最常見的遊戲語音編碼）",
    "libmpg123-0.dll": "MPEG 音訊（部分 BGM 與過場）會失敗",
    "avcodec-vgmstream-59.dll": "FFmpeg 支援的編碼（AAC/ATRAC9 等）會失敗",
}


@dataclass
class ToolStatus:
    """單一外部工具的診斷結果。"""

    name: str
    path: str | None = None
    available: bool = False
    source: str = ""            # 在哪裡找到的（設定檔／環境變數／tools/bin／PATH）
    runs: bool = False          # 實際跑起來是否成功
    version: str = ""
    missing_dlls: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        """可用、跑得起來、且沒有會導致誤判的缺件。"""
        return self.available and self.runs and not self.missing_dlls


def _which_found(name: str) -> str | None:
    """在 PATH 上找工作（與 config.find_external_bin 的 PATH 邏輯一致）。"""
    exe_names = [name]
    if os.name == "nt":
        exe_names = [f"{name}{ext}" for ext in (".exe", ".cmd", ".bat", "")]
    for exe in exe_names:
        found = shutil.which(exe)
        if found:
            return found
    return None


def locate_tool(name: str, cfg: PipelineConfig | None = None) -> tuple[str | None, str]:
    """
    找出工具實際是從哪裡來的，回傳 (路徑, 來源說明)。

    來源說明很重要：使用者常常「明明放進 tools/bin 了卻還是找不到」，
    或「設定檔改了卻沒生效」。把來源講出來，這兩種情況自己就解開了。
    """
    from config import repo_root

    hint = None
    if cfg is not None and name == "vgmstream-cli":
        hint = cfg.unpack_vgmstream_bin

    # 環境變數（由 CLI 依設定檔匯出）
    env_var = ENV_TOOL_OVERRIDES.get(name)
    if env_var and os.environ.get(env_var):
        found = find_external_bin(name)
        if found:
            return found, f"環境變數 {env_var}"

    # 設定檔明確指定
    if hint and hint.lower() != "auto":
        candidate = Path(hint)
        if not candidate.is_absolute():
            candidate = repo_root() / candidate
        if candidate.is_file():
            return str(candidate), f"設定檔 unpack.vgmstream_bin（{hint}）"

    # repo 內建 tools/bin/
    local_bin = repo_root() / "tools" / "bin"
    exe_names = [name]
    if os.name == "nt":
        exe_names = [f"{name}{ext}" for ext in (".exe", ".cmd", ".bat", "")]
    for exe in exe_names:
        candidate = local_bin / exe
        if candidate.is_file():
            return str(candidate), "repo 內建 tools/bin/"

    # 系統 PATH
    found = _which_found(name)
    if found:
        return found, "系統 PATH"

    return None, "未找到"


def check_dlls(exe_path: str | None) -> list[str]:
    """
    檢查 vgmstream 同目錄缺少哪些 DLL。

    非 Windows 一律回空清單 —— 其他平台的相依由套件管理器處理，
    沒有「放在隔壁」這種規則。
    """
    if os.name != "nt" or not exe_path:
        return []
    exe = Path(exe_path)
    if not exe.is_file():
        return []
    try:
        present = {p.name.lower() for p in exe.parent.iterdir() if p.is_file()}
    except OSError:
        return []
    return [dll for dll in WINDOWS_VGMSTREAM_DLLS if dll.lower() not in present]


def diagnose_vgmstream(cfg: PipelineConfig | None = None, *, probe: bool = True) -> ToolStatus:
    """
    診斷 vgmstream-cli。

    Args:
        probe: 是否實際執行一次（會稍微慢，但能抓到「檔在卻跑不起來」）。
    """
    status = ToolStatus(name="vgmstream-cli")
    path, source = locate_tool("vgmstream-cli", cfg)
    status.path = path
    status.source = source
    status.available = path is not None

    if not path:
        status.notes.append(
            "處理 .wem/.awb/.acb/.fsb/.bnk 等遊戲容器時必需；"
            "素材已是 .wav/.ogg/.flac 則可略過"
        )
        return status

    status.missing_dlls = check_dlls(path)
    if status.missing_dlls:
        critical = [d for d in status.missing_dlls if d in CRITICAL_DLLS]
        for dll in critical:
            status.notes.append(f"缺少 {dll} -> {CRITICAL_DLLS[dll]}")
        if not critical:
            status.notes.append(
                "缺少部分 DLL。若遇到特定編碼解不開，先補齊同目錄的 .dll 再試"
            )

    if probe:
        try:
            from _audioio import run_tool

            proc = run_tool([path], check=False, timeout=30)
            combined = f"{proc.stdout or ''}{proc.stderr or ''}"
            status.runs = proc.returncode == 0 or bool(combined.strip())
            if status.runs:
                first = next(
                    (ln.strip() for ln in combined.splitlines() if ln.strip()), ""
                )
                status.version = first[:60]
            else:
                status.notes.append(
                    f"執行檔存在但跑不起來（exit {proc.returncode}）—— "
                    "可能是架構不符、缺少 VC++ 執行期，或被防毒隔離"
                )
        except Exception as exc:
            status.notes.append(f"無法執行：{exc}")
    else:
        status.runs = True  # 沒實際探測就不宣稱它壞了

    return status


def diagnose_ffmpeg(cfg: PipelineConfig | None = None, *, probe: bool = True) -> ToolStatus:
    """診斷 ffmpeg（通用解碼後備）。"""
    status = ToolStatus(name="ffmpeg")
    path, source = locate_tool("ffmpeg", cfg)
    status.path = path
    status.source = source
    status.available = path is not None

    if not path:
        status.notes.append(
            "處理 .m4a/.aac/.wma 等 libsndfile 吃不下來的格式時需要；"
            "純 .wav/.ogg/.flac 可略過"
        )
        return status

    if probe:
        try:
            from _audioio import run_tool

            proc = run_tool([path, "-version"], check=False, timeout=30)
            status.runs = proc.returncode == 0
            first = (proc.stdout or "").splitlines()
            status.version = first[0][:60] if first else ""
            if not status.runs:
                status.notes.append("執行檔存在但 `-version` 失敗")
        except Exception as exc:
            status.notes.append(f"無法執行：{exc}")
    else:
        status.runs = True

    return status


def diagnose_all(cfg: PipelineConfig | None = None, *, probe: bool = True) -> list[ToolStatus]:
    """診斷所有外部工具。"""
    return [
        diagnose_vgmstream(cfg, probe=probe),
        diagnose_ffmpeg(cfg, probe=probe),
    ]


__all__ = [
    "CRITICAL_DLLS",
    "WINDOWS_VGMSTREAM_DLLS",
    "ToolStatus",
    "check_dlls",
    "diagnose_all",
    "diagnose_ffmpeg",
    "diagnose_vgmstream",
    "locate_tool",
]