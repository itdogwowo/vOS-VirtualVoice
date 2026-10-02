"""
vOS-VirtualVoice — 遊戲容器解包 (Phase 1, 任務 1)

為什麼獨立成一個模組
--------------------
解包是 Phase 1 唯一「依賴外部二進位檔」的環節，也是錯誤成本最高的環節
（一個 .awb 可能內含 3000 條語音）。把它從 CLI 抽出來，才能用假的
vgmstream-cli 做完整單元測試 —— 不需要真的下載二進位檔、也不需要真實素材。

vgmstream-cli 的關鍵行為（取自官方 doc/USAGE.md）
------------------------------------------------
* 典型用法：`vgmstream-cli -o happy.wav happy.adx`
* `-i`        不要循環（遊戲語音常有 loop 標記，這對我們是必要的）
* `-o` 支援萬用字元：
    - `?f`  輸入檔名
    - `?s`  目前 subsong 編號
    - `?0Ns` subsong 編號左補零到 N 位
* `-S 0`      從第 1 個 subsong 一路寫到最大（**可能產生數千個檔案**）
* `-m`        只印資訊、不解碼
* 預設輸出檔名是 `?f.wav`；有設 `-s/-S` 時是 `?f#?s.wav`

兩個真實踩到的坑（本模組就是為了修它們而存在）
----------------------------------------------
1. **多 subsong 會互相覆蓋**：若 `-o` 不含 subsong 萬用字元，
   官方文件明講 `-s 1 -S 5 -o bgm.wav` 會「5 個 subsong 全部覆蓋同一個檔案 = 錯誤」。
   這會讓你以為解包成功，實際上只拿到最後一條語音。
2. **把檔案搬離原目錄會壞掉**：CriWare 的加密 `.hca` 需要同目錄的 `.hcakey`，
   `.awb` 需要配對的 `.acb`（cue/名稱表），`.adx` 需要 `.adxkey`。
   vgmstream 是「在輸入檔所在目錄」尋找這些 companion file 的。
   所以我們**必須在來源目錄就地解包**，再把產物搬走，而不是叫 vgmstream 直接寫到別處。
"""

from __future__ import annotations

import re
import shutil
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from _audioio import run_tool

#: vgmstream-cli 可執行檔名稱（各平台一致，Windows 加 .exe）
TOOL_NAME = "vgmstream-cli"

#: 「需要 companion 檔」的來源副檔名 → 提醒文字。
#: 這些格式若少了配對檔，vgmstream 會解出無聲或直接失敗，而且錯誤訊息常常含糊。
COMPANION_HINTS: dict[str, str] = {
    ".hca": "加密 HCA 需要同目錄的 .hcakey（或 .awb 內附金鑰）",
    ".adx": "加密 ADX 需要同目錄的 .adxkey",
    ".ahx": "加密 AHX 需要同目錄的 .ahxkey",
    ".usm": "加密 USM 需要同目錄的 .usmkey",
    ".fsb": "加密 FSB 需要同目錄的 .fsbkey",
    ".awb": "AWB 需要配對的 .acb 才能取得 cue 名稱；加密者另需 .hcakey/.adxkey",
    ".acb": "ACB 只是索引表，音訊在配對的 .awb（請把兩者放在同一資料夾）",
    ".xwb": "XWB 需要配對的 .xsb 才有 cue 名稱",
    ".sli": "部分 .ogg/.opus 的循環點資訊存在 .sli/.sfl 內",
}

#: 從 vgmstream 的 `-m` 輸出解析 subsong 數量。
#: 實測輸出形如：
#:   encoding: Vorbis
#:   channels: 2
#:   sample rate: 48000
#:   subsongs: 1250
_SUBSONG_PATTERNS = (
    re.compile(r"subsongs?\s*[:=]\s*(\d+)", re.IGNORECASE),
    re.compile(r"(\d+)\s+subsongs?\b", re.IGNORECASE),
    re.compile(r"streams?\s*[:=]\s*(\d+)", re.IGNORECASE),
)


@dataclass
class UnpackOptions:
    """一次解包的所有可調參數。"""

    #: 單一容器允許解出的 subsong 上限。超過就跳過，避免一個 .awb 產生數千個檔案。
    max_subsongs: int = 3000
    #: 是否先跑 `-m` 探測 subsong 數量（建議開啟，代價是一次極快的探測）
    probe_subsongs: bool = True
    #: 額外傳給 vgmstream-cli 的參數（例如 `--loop-count 1`）
    extra_args: tuple[str, ...] = ()
    #: 逾時（秒）。大型 .awb 解包可能很久。
    timeout: int = 3600


@dataclass
class UnpackResult:
    """單一容器的解包結果。"""

    src: Path
    out_paths: list[Path] = field(default_factory=list)
    subsongs: int = 0            # 探測到的 subsong 數；-1 = 無法判定
    skipped: bool = False
    error: str = ""
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.out_paths) and not self.error


# ---------------------------------------------------------------------------
# 命令列建構（純函式，方便測試斷言）
# ---------------------------------------------------------------------------
def build_probe_cmd(exe: str, src: Path, extra_args: tuple[str, ...] = ()) -> list[str]:
    """
    `-m`：只印資訊不解碼。用來在真正解包前得知 subsong 數量。
    """
    return [exe, "-m", *extra_args, str(src)]


def build_unpack_cmd(
    exe: str,
    src: Path,
    out_pattern: Path,
    *,
    all_subsongs: bool,
    extra_args: tuple[str, ...] = (),
) -> list[str]:
    """
    組出解包命令。

    Args:
        out_pattern:  輸出樣式。**必須含 subsong 萬用字元**（例如 `song#?s.wav`），
                      否則多 subsong 會互相覆蓋 —— 這是官方文件明講的錯誤用法。
        all_subsongs: True 時加 `-S 0`（第 1 個寫到最大）。
                      單一 subsong 的容器不加，避免無意義的展開。
    """
    cmd = [exe, "-i", "-o", str(out_pattern)]  # -i：不要循環
    if all_subsongs:
        cmd += ["-S", "0"]
    cmd += list(extra_args)
    cmd.append(str(src))
    return cmd


def parse_subsong_count(text: str) -> int:
    """
    從 `-m` 的輸出解析 subsong 數量。

    找不到時回 -1 而不是 0 或 1 —— 「不確定」必須和「確定只有 1 個」區分開，
    否則安全機制會誤判為安全。
    """
    for pattern in _SUBSONG_PATTERNS:
        match = pattern.search(text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue
    return -1


def companion_hint(src: Path, siblings: set[str] | None = None) -> str:
    """
    回傳該格式的 companion file 提醒（沒有則空字串）。

    siblings 有給時，會檢查同目錄是否真的存在配對檔；缺少時講得更明確。
    """
    ext = src.suffix.lower()
    hint = COMPANION_HINTS.get(ext, "")
    if not hint:
        return ""
    if siblings is None:
        return hint

    base = src.stem
    present = {s.lower() for s in siblings}
    if ext == ".acb":
        if f"{base.lower()}.awb" not in present:
            return f"{hint}（⚠ 同目錄找不到 {base}.awb）"
    elif ext == ".awb":
        if f"{base.lower()}.acb" not in present:
            return f"{hint}（ℹ 同目錄找不到 {base}.acb，cue 名稱會缺失但音訊可解）"
    elif ext in (".hca", ".adx", ".ahx", ".usm", ".fsb"):
        if f".{ext.lstrip('.')}key" not in present:
            return f"{hint}（ℹ 同目錄無金鑰檔；若該檔未加密則可正常解出）"
    return hint


# ---------------------------------------------------------------------------
# 解包執行
# ---------------------------------------------------------------------------
def _collect_outputs(work_dir: Path, before: set[str]) -> list[Path]:
    """列出 work_dir 內「新增」的 .wav（用集合差集，不依賴檔名樣式猜測）。"""
    after = {p.name for p in work_dir.glob("*.wav")}
    new_names = after - before
    return sorted((work_dir / n) for n in new_names)


def _tool_exists(exe: str) -> bool:
    """檢查可執行檔是否存在（支援絕對路徑與 PATH 上的名稱，跨平台）。"""
    return Path(exe).is_file() or shutil.which(exe) is not None


def unpack_one(
    src: Path,
    dest_dir: Path,
    exe: str,
    options: UnpackOptions | None = None,
    *,
    force: bool = False,
    log=None,
) -> UnpackResult:
    """
    解包單一容器，產物放到 dest_dir。

    流程（順序很重要）：
      1. `-m` 探測 subsong 數量 → 超過 max_subsongs 就跳過（附可行動的建議）
      2. 若目的資料夾已有對應產物 → 直接略過（冪等性；`force=True` 可強制重做）
      3. 在**來源目錄**建立暫存子資料夾 —— 這樣 vgmstream 才找得到
         .hcakey / .acb / .adxkey 等 companion file
      4. 用含 `?s` 萬用字元的輸出樣式解包（多 subsong 時）
      5. 把產物搬到 dest_dir，刪除暫存資料夾

    Args:
        force: True 時忽略既有產物，強制重新解包。
        log:   可選的 callable(str)，用來即時回報（通常是 CLI 的 print）。

    Returns:
        UnpackResult（永不拋例外；失敗原因放在 error）。
    """
    opts = options or UnpackOptions()
    src = Path(src)
    dest_dir = Path(dest_dir)

    def _log(msg: str) -> None:
        if log:
            log(msg)

    result = UnpackResult(src=src)
    if not src.is_file():
        result.error = f"來源檔不存在：{src}"
        return result
    if not _tool_exists(exe):
        result.error = f"找不到 vgmstream-cli：{exe}"
        return result

    # ---- 1. 探測 subsong 數量 ----
    if opts.probe_subsongs:
        try:
            probe = run_tool(build_probe_cmd(exe, src, opts.extra_args), check=False, timeout=120)
            combined = f"{probe.stdout}\n{probe.stderr}"
            result.subsongs = parse_subsong_count(combined)
        except RuntimeError as exc:
            result.error = f"探測失敗：{exc}"
            return result

        if result.subsongs > opts.max_subsongs:
            result.skipped = True
            result.error = (
                f"內含 {result.subsongs} 個 subsong，超過上限 {opts.max_subsongs}"
                f"（單一容器解出數千個檔案會佔滿磁碟且拖慢後續階段）"
            )
            _log(
                f"    {result.error}\n"
                f"    → 若確定要全部解出，把 configs 的 unpack.max_subsongs 調高，"
                f"或先用 --limit 小量測試"
            )
            return result

    # ---- 2 & 3. 在來源目錄就地解包 ----
    # 暫存資料夾名稱只用「序號」構成，絕不從來源檔名改寫而來。
    #
    # ⚠ 這裡有個真實踩過的坑：先前版本把檔名經過字元清洗後放進暫存資料夾名稱，
    # 例如 bank=01 被清成 bank_01。結果同一支程式在「探測階段」用原始檔名、
    # 在「解包階段」用清洗後的檔名，兩邊對同一個來源給出不同路徑 ——
    # 生產環境會直接找不到檔案。用序號當唯一識別就沒有這個問題。
    work_dir: Path | None = None
    for seq in range(1, 100):
        candidate = src.parent / f".vos_unpack_{seq:03d}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        except OSError as exc:
            result.error = f"無法在來源目錄建立暫存資料夾（就地解包是必要的）：{exc}"
            return result
        work_dir = candidate
        break
    if work_dir is None:
        result.error = "來源目錄內已有過多 .vos_unpack_* 暫存資料夾，請手動清理後重試"
        return result

    try:
        before = {p.name for p in work_dir.glob("*.wav")}

        # ------------------------------------------------------------------
        # 輸出樣式的選擇（三個真實踩過的坑都在這裡）
        #
        # 1. 多 subsong 時**必須**含 `?s` 萬用字元，否則所有 subsong 會覆蓋
        #    同一個檔案（vgmstream 官方文件明講這是錯誤用法）→ 你只拿到最後一條。
        #
        # 2. `?s` 只有在「展開多 subsong」時才有機會被替換掉。若容器只有 1 個
        #    subsong 而我們仍傳 `#?s.wav`，那個 `?` 會原封不動留在檔名裡 ——
        #    而 `?` 在 Windows 是非法檔名字元，直接 OSError [Errno 22]，
        #    **所有單 subsong 容器全部解包失敗**（Linux/macOS 不會，所以是隱形殺手）。
        #
        # 3. 真實 vgmstream 的 `-m` 對「只有 1 個 subsong」的檔案**不會**印
        #    `subsongs:` 那行（只有 N>1 才印）。所以探測結果 -1 的代表意義是
        #    「沒有多個 subsong」，不是「不確定」。
        #
        # 綜合以上：探測不到（-1）或確定是 1 → 走單檔路徑；
        # 只有明確 > 1 才走多 subsong 展開。
        # 額外保險：若走了展開卻只產出 1 個檔案，事後把 `#1` 拿掉，
        # 避免幾千個語音檔全部變成 `voice#1.wav` 這種醜名字。
        # ------------------------------------------------------------------
        explicit_multi = result.subsongs > 1
        #: --no-probe 時 subsongs 恆為 0；此時沒有資訊可判斷，保守走展開路徑
        unknown_without_probe = (not opts.probe_subsongs) and result.subsongs == 0
        all_subsongs = explicit_multi or unknown_without_probe

        if all_subsongs:
            out_pattern = work_dir / f"{src.stem}#?s.wav"
        else:
            out_pattern = work_dir / f"{src.stem}.wav"

        # ------------------------------------------------------------------
        # 冪等性：上一次已經解包過就不要再解一次。
        #
        # 沒有這道檢查的症狀：重跑 `unpack`（甚至跑 `all`）時，產物會變成
        # `bank#1__1.wav`、`bank#1__2.wav` 一路累積，幾千條語音重跑三次就爆量。
        # convert 已經有同樣的「已存在就略過」行為，unpack 也該一致。
        # 需要強制重解時用 --overwrite。
        # ------------------------------------------------------------------
        expected_names = (
            [f"{src.stem}#{n}.wav" for n in range(1, result.subsongs + 1)]
            if (all_subsongs and result.subsongs > 1)
            else [f"{src.stem}.wav"]
        )
        if not force and all((dest_dir / name).is_file() for name in expected_names):
            result.out_paths = [dest_dir / name for name in expected_names]
            result.skipped = True
            result.error = f"已解包過（{len(expected_names)} 個檔案已存在，未重做）"
            return result

        try:
            proc = run_tool(
                build_unpack_cmd(
                    exe, src, out_pattern, all_subsongs=all_subsongs, extra_args=opts.extra_args
                ),
                check=False,
                timeout=opts.timeout,
            )
        except RuntimeError as exc:
            result.error = str(exc)
            return result

        result.stdout = proc.stdout or ""
        result.stderr = proc.stderr or ""

        made = _collect_outputs(work_dir, before)
        if not made:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
            hint = companion_hint(src, {p.name for p in src.parent.iterdir()})
            result.error = f"vgmstream 未產生任何輸出（exit {proc.returncode}）"
            if tail:
                result.error += f"：{' | '.join(tail)}"
            if hint:
                result.error += f"\n    ℹ {hint}"
            return result

        # ---- 4. 搬到目的資料夾 ----
        dest_dir.mkdir(parents=True, exist_ok=True)
        moved: list[Path] = []
        for f in made:
            name = f.name
            # 走了多 subsong 展開但只產出 1 個檔案 → 這是單 subsong 容器，
            # 把 `#1` 拿掉還原乾淨檔名（幾千個語音檔才不會都變成 voice#1.wav）
            if all_subsongs and len(made) == 1 and "#1" in name:
                name = name.replace("#1", "")

            target = dest_dir / name
            # 同名時加序號，不覆蓋既有產物（重跑時很常見）
            counter = 1
            while target.exists():
                target = dest_dir / f"{Path(name).stem}__{counter}{Path(name).suffix}"
                counter += 1
            shutil.move(str(f), str(target))
            moved.append(target)

        result.out_paths = moved
        if result.subsongs < 0:
            result.subsongs = len(moved)
        return result
    finally:
        if work_dir is not None and work_dir.is_dir():
            shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 預估產物數量（給 CLI 事先警告用）
# ---------------------------------------------------------------------------
def estimate_total_subsongs(
    sources: list[Path], exe: str, options: UnpackOptions | None = None, *, sample: int = 5
) -> tuple[int, int]:
    """
    抽樣探測前 `sample` 個容器，回傳 (抽樣總 subsong 數, 實際抽樣數)。

    用途：在開始解包前給使用者一個「這次可能會產生幾萬個檔案」的警告。
    全量探測太慢，所以用抽樣 + 平均推算。
    """
    opts = options or UnpackOptions()
    total = 0
    probed = 0
    for src in sources[:sample]:
        try:
            proc = run_tool(build_probe_cmd(exe, src, opts.extra_args), check=False, timeout=120)
            count = parse_subsong_count(f"{proc.stdout}\n{proc.stderr}")
        except RuntimeError:
            continue
        if count > 0:
            total += count
            probed += 1
    return total, probed


@dataclass
class UnpackSummary:
    """整批解包的彙總（給 CLI 印表用）。"""

    containers: int = 0
    ok: int = 0
    failed: int = 0
    skipped: int = 0
    files: int = 0
    results: deque[UnpackResult] = field(default_factory=deque)


__all__ = [
    "COMPANION_HINTS",
    "TOOL_NAME",
    "UnpackOptions",
    "UnpackResult",
    "UnpackSummary",
    "build_probe_cmd",
    "build_unpack_cmd",
    "companion_hint",
    "estimate_total_subsongs",
    "parse_subsong_count",
    "unpack_one",
]
