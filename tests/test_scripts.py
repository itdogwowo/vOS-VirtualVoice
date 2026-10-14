"""
vOS-VirtualVoice — 啟動腳本的跨平台契約測試

守著三個「本機測試全綠、換台機器／換個 shell 就壞掉」的坑：

1. **`run_phase1.ps1` 不得含非 ASCII 字元。**
   Windows PowerShell 5.1（Windows 內建的預設版本）讀 `.ps1` 時，
   若檔案沒有 UTF-8 BOM，會用**系統 ANSI 編碼**解讀（台灣是 cp950、
   中國 cp936、日本 cp932）。一個中文字就能讓多位元組序列錯位、
   引號配對錯亂，整個腳本以 ParserError 收場 —— 即使那個字在**註解裡**。

   這個問題真實發生過：開發時一路用 PowerShell 7（pwsh，預設 UTF-8），
   完全正常；使用者用內建的 5.1 卻連「顯示用法」都跑不到。
   加上 BOM 可以解決，但 BOM 很容易被編輯器／工具默默吃掉，
   所以更穩健的做法是讓檔案本身不含非 ASCII。

2. **Shell script 必須是 LF。**
   CRLF 的 `.sh` 在 Linux/macOS 上會失敗：
       bash: ./scripts/run_phase1.sh: /bin/bash^M: bad interpreter

3. **`.gitattributes` 必須存在並聲明這些規則。**
   沒有它的話，行尾取決於每個人的 `core.autocrlf`，行為不可預測。
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PS1 = REPO / "scripts" / "run_phase1.ps1"
SH = REPO / "scripts" / "run_phase1.sh"
GITATTRIBUTES = REPO / ".gitattributes"


# ---------------------------------------------------------------------------
# 1. PowerShell 腳本：ASCII-only + UTF-8 友善
# ---------------------------------------------------------------------------
class TestPowerShellScript:
    def test_script_exists(self) -> None:
        assert PS1.is_file(), f"找不到 {PS1}"

    def test_is_pure_ascii(self) -> None:
        """
        最重要的一條：整個檔案不得含任何非 ASCII 位元組。

        這比「加上 BOM」更穩健 —— BOM 會被編輯器與各種工具默默移除，
        而 ASCII-only 的檔案在任何編碼假設下都能正確解析。

        若這條測試失敗，請把新增的中文註解／訊息改成英文，
        而不是「補上 BOM 就好」。
        """
        raw = PS1.read_bytes()
        offenders: list[str] = []
        for lineno, line in enumerate(raw.split(b"\n"), start=1):
            for column, byte in enumerate(line, start=1):
                if byte > 127:
                    offenders.append(
                        f"line {lineno} col {column}: byte 0x{byte:02X} "
                        f"({line.decode('utf-8', 'replace').strip()[:60]})"
                    )
        assert not offenders, (
            "run_phase1.ps1 含非 ASCII 字元，Windows PowerShell 5.1（cp950/cp936/cp932）"
            "會解析失敗。請改用英文：\n  " + "\n  ".join(offenders[:10])
        )

    def test_declares_cmdletbinding_and_remaining_args(self) -> None:
        """ValueFromRemainingArguments 是「把參數原封不動轉給 Python」的關鍵。"""
        text = PS1.read_text(encoding="ascii")
        assert "ValueFromRemainingArguments" in text
        assert "phase1_prep.py" in text, "必須轉呼叫真正的 CLI"

    def test_sets_utf8_environment_for_python(self) -> None:
        """Python 的 stdout 要走 UTF-8，否則 cp950 主控台遇到符號就炸。"""
        text = PS1.read_text(encoding="ascii")
        assert "PYTHONIOENCODING" in text
        assert "PYTHONUTF8" in text

    def test_restores_working_directory(self) -> None:
        """Push-Location 必須有對應的 finally/Pop-Location。"""
        text = PS1.read_text(encoding="ascii")
        assert "Push-Location" in text
        assert "Pop-Location" in text
        assert "finally" in text, "Pop-Location 應放在 finally，例外時也要還原"

    def test_documents_every_subcommand(self) -> None:
        """用法提示不可與實際 CLI 脫節（會誤導使用者）。"""
        import sys

        sys.path.insert(0, str(REPO / "scripts"))
        from phase1_prep import build_parser

        parser = build_parser()
        text = PS1.read_text(encoding="ascii")
        for action in parser._subparsers._group_actions:  # type: ignore[attr-defined]
            for name in action.choices:
                assert name in text, f"用法提示缺少子命令：{name}"


# ---------------------------------------------------------------------------
# 2. Shell script：LF + 正確 shebang
# ---------------------------------------------------------------------------
class TestShellScript:
    def test_script_exists(self) -> None:
        assert SH.is_file(), f"找不到 {SH}"

    def test_uses_lf_line_endings(self) -> None:
        """
        CRLF 的 shell script 會讓 Linux/macOS 的 bash 直接失敗：
            /bin/bash^M: bad interpreter: No such file or directory
        這個錯誤訊息完全不會讓人想到「行尾」，所以必須自動擋下。
        """
        raw = SH.read_bytes()
        assert b"\r\n" not in raw, (
            "run_phase1.sh 含 CRLF 行尾，Linux/macOS 執行時會出現 "
            "'bad interpreter' 錯誤。請確認 .gitattributes 的 *.sh text eol=lf 生效。"
        )

    def test_starts_with_shebang_and_no_bom(self) -> None:
        raw = SH.read_bytes()
        assert raw.startswith(b"#!"), "缺少 shebang"
        assert not raw.startswith(b"\xef\xbb\xbf"), "shell script 不可有 UTF-8 BOM"

    def test_quotes_variables(self) -> None:
        """路徑含空白時未加引號會壞掉（跨平台常見地雷）。"""
        text = SH.read_text(encoding="utf-8")
        assert '"$@"' in text or '"$@"' in text, "轉發參數時必須用 \"$@\""
        assert 'exec "${PYTHON_EXE}"' in text

    def test_documents_every_subcommand(self) -> None:
        import sys

        sys.path.insert(0, str(REPO / "scripts"))
        from phase1_prep import build_parser

        parser = build_parser()
        text = SH.read_text(encoding="utf-8")
        for action in parser._subparsers._group_actions:  # type: ignore[attr-defined]
            for name in action.choices:
                assert name in text, f"用法提示缺少子命令：{name}"


# ---------------------------------------------------------------------------
# 3. .gitattributes：行尾規則必須被明確聲明
# ---------------------------------------------------------------------------
class TestGitAttributes:
    def test_exists(self) -> None:
        assert GITATTRIBUTES.is_file(), (
            "缺少 .gitattributes。沒有它的話，行尾取決於每個人的 core.autocrlf，"
            "同一份 repo 在不同機器上會產生假 diff，shell script 也可能壞掉。"
        )

    @pytest.mark.parametrize(
        "pattern",
        [
            "text=auto",       # 預設正規化
            "*.sh",            # shell script 必須 LF
            "*.ps1",           # PowerShell
            "*.wav",           # 音訊不可被換行轉換破壞
        ],
    )
    def test_declares_pattern(self, pattern: str) -> None:
        text = GITATTRIBUTES.read_text(encoding="utf-8")
        assert pattern in text, f".gitattributes 缺少規則：{pattern}"

    def test_binary_audio_is_marked_binary(self) -> None:
        """
        音訊檔必須標為 binary。git 的換行轉換會**直接損壞**二進位檔，
        而這種損壞在聽之前完全看不出來。
        """
        text = GITATTRIBUTES.read_text(encoding="utf-8")
        for ext in (".wav", ".wem", ".awb", ".acb", ".fsb"):
            line = next(
                (ln for ln in text.splitlines() if ln.strip().startswith(f"*{ext}")),
                None,
            )
            assert line is not None, f"缺少 {ext} 的規則"
            assert "binary" in line, f"{ext} 應標為 binary，實際：{line}"


# ---------------------------------------------------------------------------
# 4. 現況檢查：repo 內不該有 CRLF 的 shell script
# ---------------------------------------------------------------------------
class TestRepoLineEndings:
    def test_no_crlf_in_shell_scripts(self) -> None:
        for script in REPO.rglob("*.sh"):
            if ".git" in script.parts or ".pytest_tmp" in script.parts:
                continue
            raw = script.read_bytes()
            assert b"\r\n" not in raw, f"{script.relative_to(REPO)} 含 CRLF 行尾"


# ---------------------------------------------------------------------------
# 5. 文件與程式碼不可脫節
# ---------------------------------------------------------------------------
class TestDocsCoverEverySubcommand:
    """
    文件必須提到每一個子命令。

    為什麼要自動化：這已經**脫節兩次**了。
      1. 啟動腳本的用法提示漏了 audition / tune / report
      2. selftest 補回後，README 與主手冊都沒提到它 —— 而它正好是
         「還沒有素材」時唯一能跑的指令，也就是使用者最需要看到的那一個

    這種脫節不會報錯、不會有症狀，只會讓照著文件操作的人卡住。
    只能靠自動化守住。
    """

    @staticmethod
    def _subcommands() -> list[str]:
        import sys as _sys

        _sys.path.insert(0, str(REPO / "scripts"))
        from phase1_prep import build_parser

        parser = build_parser()
        names: list[str] = []
        for action in parser._subparsers._group_actions:  # type: ignore[attr-defined]
            names.extend(action.choices)
        return sorted(names)

    def test_readme_covers_every_subcommand(self) -> None:
        text = (REPO / "README.md").read_text(encoding="utf-8")
        missing = [n for n in self._subcommands() if n not in text]
        assert not missing, f"README.md 未提到這些子命令：{missing}"

    def test_manual_covers_every_subcommand(self) -> None:
        text = (REPO / "docs" / "PHASE1-INGEST.md").read_text(encoding="utf-8")
        missing = [n for n in self._subcommands() if n not in text]
        assert not missing, f"PHASE1-INGEST.md 未提到這些子命令：{missing}"

    def test_readme_documents_selftest_as_first_step(self) -> None:
        """
        selftest 必須出現在「還沒有素材時該做什麼」的位置 ——
        它是唯一不需要素材就能驗證環境的指令。
        """
        text = (REPO / "README.md").read_text(encoding="utf-8")
        assert "selftest" in text
        # 它應該排在 scan 之前（先驗證環境，再處理素材）
        assert text.index("selftest") < text.index("phase1_prep.py scan"), (
            "selftest 應排在 scan 之前：先確認工具能動，再處理素材"
        )

    def test_documented_paths_exist(self) -> None:
        """文件引用的關鍵路徑必須真的存在。"""
        for rel in (
            "docs/PHASE1-INGEST.md",
            "docs/FORMAT-PLAYBOOK.md",
            "configs/pipeline.default.json",
            "requirements.txt",
            "scripts/phase1_prep.py",
            "scripts/run_phase1.ps1",
            "scripts/run_phase1.sh",
        ):
            assert (REPO / rel).exists(), f"文件引用了不存在的路徑：{rel}"