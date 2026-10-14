#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# vOS-VirtualVoice — Phase 1 一鍵執行腳本 (macOS / Linux)
#
# 用法：
#   bash scripts/run_phase1.sh doctor
#   bash scripts/run_phase1.sh scan
#   bash scripts/run_phase1.sh unpack
#   bash scripts/run_phase1.sh convert --limit 100 -v
#   bash scripts/run_phase1.sh all
#
# Windows 請用 scripts/run_phase1.ps1
# ---------------------------------------------------------------------------
set -euo pipefail

# 專案根目錄（本檔案位於 <root>/scripts/）
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

resolve_python() {
    # 1) 專案虛擬環境（若存在）
    if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
        echo "${REPO_ROOT}/.venv/bin/python"
        return
    fi
    # 2) python3 優先（多數 Linux/macOS 沒有 python）
    if command -v python3 >/dev/null 2>&1; then
        command -v python3
        return
    fi
    if command -v python >/dev/null 2>&1; then
        command -v python
        return
    fi
    echo "找不到 Python。請先安裝 Python 3.10+，或建立 .venv。" >&2
    exit 1
}

PYTHON_EXE="$(resolve_python)"
PREP_SCRIPT="${REPO_ROOT}/scripts/phase1_prep.py"

if [[ $# -eq 0 ]]; then
    cat >&2 <<'USAGE'
用法：bash scripts/run_phase1.sh <指令> [選項]

指令：
  doctor     檢查 Python 依賴 / 外部工具 / 資料夾 / 目標格式
  scan       嗅探素材真實格式分布（不寫檔，最重要的一步）
  unpack     用 vgmstream-cli 解包遊戲容器
  convert    標準化為 24bit/48kHz WAV + 時長篩選
  audition   產生試聽帶與對照清單（人工驗收用）
  tune       分析 manifest，建議該不該調整門檻
  report     產生綜合報告（驗收清單 + 下一步）
  selftest   用合成素材驗證工具鏈能不能動（不需要你的素材）
  env        偵測 CUDA/MPS/CPU 並依 VRAM 擬合訓練參數
  precheck   訓練前環境自檢（合成張量，提前引爆會失敗的問題）
  all        依序執行 scan -> unpack -> convert

例：  bash scripts/run_phase1.sh scan
      bash scripts/run_phase1.sh convert --limit 50 -v
USAGE
    exit 64
fi

# 確保輸出一律 UTF-8（macOS 預設語系有時為其他編碼）
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

echo "Python : ${PYTHON_EXE}"
echo "腳本   : ${PREP_SCRIPT}"
echo

cd "${REPO_ROOT}"
exec "${PYTHON_EXE}" "${PREP_SCRIPT}" "$@"
