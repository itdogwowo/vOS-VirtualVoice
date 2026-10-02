"""
vOS-VirtualVoice — 設定檔載入與驗證 (Phase 1)

設計原則
--------
1. 單一真實來源 (single source of truth)：所有門檻值都放在 configs/pipeline.*.json，
   程式碼裡不寫死魔術數字，方便 Phase 1→5 逐步調參。
2. 相對路徑：設定檔內的路徑一律相對於 repo 根目錄，跨平台（Windows/Mac/Linux）
   都不會綁死某台機器的絕對路徑。
3. 明確報錯：設定寫錯時直接拋 ConfigError 並講清楚哪個欄位錯，不要靜默用預設值。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """設定檔內容不合法。"""


# ---------------------------------------------------------------------------
# 路徑工具
# ---------------------------------------------------------------------------
#: 環境變數：可強制指定 repo 根目錄。用途：
#:   1. 自動化測試在獨立沙箱內跑完整 CLI（不污染真正的 data/）
#:   2. 素材放在外接硬碟、想把產物寫到別處時
#: 未設定時一律以「本檔案的所在位置」推導，因此正常使用完全不受影響。
ENV_REPO_ROOT = "VOS_REPO_ROOT"


def repo_root() -> Path:
    """回傳 repo 根目錄（本檔案位於 <root>/scripts/config.py）。"""
    override = os.environ.get(ENV_REPO_ROOT)
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parent.parent


def default_config_path() -> Path:
    """優先使用本機覆寫檔 configs/pipeline.local.json，其次才用預設檔。"""
    root = repo_root()
    local = root / "configs" / "pipeline.local.json"
    if local.is_file():
        return local
    return root / "configs" / "pipeline.default.json"


def find_external_bin(name: str, hint: str | None = None) -> str | None:
    """
    跨平台尋找外部執行檔 (ffmpeg / ffprobe / vgmstream-cli ...)。

    搜尋順序：
      1. settings 明確指定的路徑（相對 repo 根目錄）
      2. repo 內建 tools/bin/（各平台自行放二進位檔的地方）
      3. 系統 PATH（含 Windows 的 .exe/.cmd/.bat 與 Unix 無副檔名）

    回傳：可執行檔路徑字串，找不到則回 None。
    """
    exe_names = [name]
    if os.name == "nt":  # Windows：PATH 查找需要帶副檔名
        exe_names = [f"{name}{ext}" for ext in (".exe", ".cmd", ".bat", "")]

    if hint and hint.lower() != "auto":
        p = Path(hint)
        if not p.is_absolute():
            p = repo_root() / p
        if p.is_file():
            return str(p)

    local_bin = repo_root() / "tools" / "bin"
    if local_bin.is_dir():
        for exe in exe_names:
            candidate = local_bin / exe
            if candidate.is_file():
                return str(candidate)

    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if not directory:
            continue
        for exe in exe_names:
            candidate = Path(directory) / exe
            try:
                if candidate.is_file():
                    return str(candidate)
            except OSError:
                continue
    return None


# ---------------------------------------------------------------------------
# 設定資料結構
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TargetFormat:
    """Phase 1 產出格式：24bit / 48kHz / 單聲道 PCM WAV。"""

    sample_rate: int = 48_000
    bit_depth: int = 24
    channels: int = 1

    @property
    def subtype(self) -> str:
        """soundfile 的編碼子類型字串。"""
        return {16: "PCM_16", 24: "PCM_24", 32: "PCM_32"}.get(
            self.bit_depth, "PCM_24"
        )


@dataclass(frozen=True)
class DurationFilter:
    """任務 2：時長篩選門檻（秒）。"""

    min_seconds: float = 1.5
    max_seconds: float = 15.0


@dataclass(frozen=True)
class QualityGate:
    """任務 2 附加：壞資料早期偵測。"""

    enabled: bool = True
    peak_dbfs_max: float = -0.5
    min_rms_dbfs: float = -50.0


@dataclass(frozen=True)
class Paths:
    raw_dir: Path
    unpacked_dir: Path
    standardized_dir: Path
    rejected_dir: Path
    reports_dir: Path

    def ensure(self) -> None:
        """建立所有 Phase 1 會用到的目錄（含舊資料夾會自動略過）。"""
        for p in (
            self.raw_dir,
            self.unpacked_dir,
            self.standardized_dir,
            self.rejected_dir,
            self.reports_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class PipelineConfig:
    target: TargetFormat
    duration_filter: DurationFilter
    quality_gate: QualityGate
    paths: Paths
    manifest_filename: str = "manifest_phase1.csv"
    manifest_include_rejected: bool = True
    #: "none" = 所有產物攤平在同一層（單一角色）
    #: "folder" = 依 data/00_raw/<群組>/ 的第一層子資料夾分組（多角色／多來源）
    group_by: str = "none"
    unpack_vgmstream_bin: str = "auto"
    unpack_keep_subdirs: bool = True
    unpack_extra_args: tuple[str, ...] = ()
    #: 單一容器允許解出的 subsong 上限。防呆用：一個 .awb 可能內含數千條語音，
    #: 意外全解會佔滿磁碟並讓後續階段變得極慢。
    unpack_max_subsongs: int = 3000
    scan_extensions: tuple[str, ...] = (".wem", ".ogg", ".wav", ".flac")
    decode_backend_priority: tuple[str, ...] = ("soundfile", "ffmpeg", "vgmstream")
    source_file: Path = field(default_factory=default_config_path)

    # -- 便利屬性 ---------------------------------------------------------
    @property
    def manifest_path(self) -> Path:
        return self.paths.reports_dir / self.manifest_filename

    def group_of(self, src: Path) -> str:
        """
        回傳檔案所屬群組名稱。

        多角色專案建議把素材放成 data/00_raw/<角色名>/xxx.wem，
        這樣每個角色的產物會各自進到 02_wav24_48k/<角色名>/，
        Phase 4 打標籤與 Phase 5 選參考音訊時都不會混到別的角色。

        判定順序：00_raw 的第一層子資料夾 → 01_unpacked 的第一層 → "default"。
        """
        if self.group_by != "folder":
            return "default"
        for base in (self.paths.raw_dir, self.paths.unpacked_dir):
            try:
                rel_parts = src.resolve().relative_to(base.resolve()).parts
            except ValueError:
                continue
            if len(rel_parts) > 1:  # 有子資料夾才算一個群組
                return rel_parts[0]
        return "default"

    def resolve_output_for(self, src: Path, base: Path) -> Path:
        """
        決定某個來源檔在輸出資料夾中的位置。

        * group_by="folder"：先加一層群組子資料夾（多角色專案）
        * keep_subdirs=True：再保留來源相對結構，避免同名檔案互撞
        * 全部攤平時：base/<stem>.wav
        """
        group = self.group_of(src)
        target_dir = base / group if self.group_by == "folder" else base

        if self.unpack_keep_subdirs:
            try:
                rel = src.relative_to(self.paths.raw_dir)
            except ValueError:
                rel = None
            if rel is not None:
                # 重點：不要讓群組名重複兩層。
                # data/00_raw/A/sub/x.wem → 02_wav24_48k/A/sub/x.wav（而非 A/A/sub/x.wav）
                if group != "default" and rel.parts and rel.parts[0] == group:
                    rel = Path(*rel.parts[1:]) if len(rel.parts) > 1 else Path(src.name)
                return (target_dir / rel).with_suffix(".wav")
        return (target_dir / src.name).with_suffix(".wav")


# ---------------------------------------------------------------------------
# 載入 + 驗證
# ---------------------------------------------------------------------------
def _require(section: dict[str, Any], key: str, where: str) -> Any:
    if key not in section:
        raise ConfigError(f"設定檔缺少必要欄位：{where}.{key}")
    return section[key]


def load_config(path: str | os.PathLike[str] | None = None) -> PipelineConfig:
    """讀取並驗證設定檔，回傳 PipelineConfig。"""
    cfg_path = Path(path) if path else default_config_path()
    cfg_path = cfg_path if cfg_path.is_absolute() else (repo_root() / cfg_path)
    if not cfg_path.is_file():
        raise ConfigError(f"找不到設定檔：{cfg_path}")

    try:
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"設定檔 JSON 格式錯誤（{cfg_path}）：{exc}") from exc

    root = repo_root()

    tgt_raw = raw.get("target", {})
    target = TargetFormat(
        sample_rate=int(tgt_raw.get("sample_rate", 48_000)),
        bit_depth=int(tgt_raw.get("bit_depth", 24)),
        channels=int(tgt_raw.get("channels", 1)),
    )
    if target.sample_rate <= 0:
        raise ConfigError("target.sample_rate 必須為正整數")
    if target.bit_depth not in (16, 24, 32):
        raise ConfigError("target.bit_depth 僅支援 16 / 24 / 32")
    if target.channels not in (1, 2):
        raise ConfigError("target.channels 僅支援 1（單聲道，建議）或 2")

    dur_raw = raw.get("duration_filter", {})
    duration = DurationFilter(
        min_seconds=float(_require(dur_raw, "min_seconds", "duration_filter")),
        max_seconds=float(_require(dur_raw, "max_seconds", "duration_filter")),
    )
    if duration.min_seconds <= 0 or duration.max_seconds <= duration.min_seconds:
        raise ConfigError(
            "duration_filter 門檻不合理：需滿足 0 < min_seconds < max_seconds"
        )

    q_raw = raw.get("quality_gate", {})
    quality = QualityGate(
        enabled=bool(q_raw.get("enabled", True)),
        peak_dbfs_max=float(q_raw.get("peak_dbfs_max", -0.5)),
        min_rms_dbfs=float(q_raw.get("min_rms_dbfs", -50.0)),
    )
    if quality.peak_dbfs_max > 0.0:
        raise ConfigError("quality_gate.peak_dbfs_max 不可大於 0.0 dBFS")

    p_raw = raw.get("paths", {})

    def _p(key: str, fallback: str) -> Path:
        value = str(p_raw.get(key, fallback))
        path = Path(value)
        return path if path.is_absolute() else (root / path)

    paths = Paths(
        raw_dir=_p("raw_dir", "data/00_raw"),
        unpacked_dir=_p("unpacked_dir", "data/01_unpacked"),
        standardized_dir=_p("standardized_dir", "data/02_wav24_48k"),
        rejected_dir=_p("rejected_dir", "data/99_rejected"),
        reports_dir=_p("reports_dir", "data/_reports"),
    )

    man_raw = raw.get("manifest", {})
    unpack_raw = raw.get("unpack", {})
    scan_raw = raw.get("scan", {})
    decode_raw = raw.get("decode", {})
    group_raw = raw.get("grouping", {})

    group_by = str(group_raw.get("by", "none")).lower()
    if group_by not in ("none", "folder"):
        raise ConfigError(
            f"grouping.by 僅支援 'none'（單一角色）或 'folder'（多角色分組），得到 {group_by!r}"
        )

    exts = tuple(
        e.lower() if e.startswith(".") else f".{e.lower()}"
        for e in scan_raw.get(
            "extensions", [".wem", ".ogg", ".wav", ".flac", ".acb", ".fsb"]
        )
    )
    if not exts:
        raise ConfigError("scan.extensions 不可為空")

    return PipelineConfig(
        target=target,
        duration_filter=duration,
        quality_gate=quality,
        paths=paths,
        manifest_filename=str(man_raw.get("filename", "manifest_phase1.csv")),
        manifest_include_rejected=bool(man_raw.get("include_rejected", True)),
        group_by=group_by,
        unpack_vgmstream_bin=str(unpack_raw.get("vgmstream_bin", "auto")),
        unpack_keep_subdirs=bool(unpack_raw.get("keep_subdirs", True)),
        unpack_extra_args=tuple(unpack_raw.get("extra_args", []) or []),
        unpack_max_subsongs=int(unpack_raw.get("max_subsongs", 3000)),
        scan_extensions=exts,
        decode_backend_priority=tuple(
            decode_raw.get("backend_priority", ["soundfile", "ffmpeg", "vgmstream"])
        ),
        source_file=cfg_path,
    )


__all__ = [
    "ConfigError",
    "DurationFilter",
    "ENV_REPO_ROOT",
    "Paths",
    "PipelineConfig",
    "QualityGate",
    "TargetFormat",
    "default_config_path",
    "find_external_bin",
    "load_config",
    "repo_root",
]
