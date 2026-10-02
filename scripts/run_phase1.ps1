<#
.SYNOPSIS
    vOS-VirtualVoice — Phase 1 一鍵執行腳本 (Windows / PowerShell)

.DESCRIPTION
    自動尋找 Python 解譯器（優先 .venv，其次 py launcher / python），
    然後把參數原封不動轉給 scripts/phase1_prep.py。

    macOS / Linux 請用 scripts/run_phase1.sh。

.EXAMPLE
    .\scripts\run_phase1.ps1 doctor
    .\scripts\run_phase1.ps1 scan
    .\scripts\run_phase1.ps1 unpack
    .\scripts\run_phase1.ps1 convert --limit 100 -v
    .\scripts\run_phase1.ps1 all
#>
[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Args
)

$ErrorActionPreference = 'Stop'

# 專案根目錄（本檔案位於 <root>/scripts/）
$RepoRoot = Split-Path -Parent $PSScriptRoot

function Resolve-Python {
    # 1) 專案虛擬環境（若存在）
    $venv = Join-Path $RepoRoot '.venv\Scripts\python.exe'
    if (Test-Path $venv) { return $venv }

    # 2) Windows py launcher（能挑到已安裝的版本）
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) { return $py.Source }

    # 3) PATH 上的 python
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) { return $python.Source }

    throw "找不到 Python。請先安裝 Python 3.10+ 並加入 PATH，或建立 .venv。"
}

$PythonExe = Resolve-Python
$PrepScript = Join-Path $RepoRoot 'scripts\phase1_prep.py'

if (-not $Args -or $Args.Count -eq 0) {
    Write-Host "用法：.\scripts\run_phase1.ps1 <doctor|scan|unpack|convert|all> [選項]" -ForegroundColor Yellow
    Write-Host "例：  .\scripts\run_phase1.ps1 convert --limit 50 -v"
    exit 64
}

Write-Host "Python : $PythonExe" -ForegroundColor DarkGray
Write-Host "腳本   : $PrepScript" -ForegroundColor DarkGray
Write-Host ""

# 讓 Python 的 stdout 走 UTF-8，避免 cp950/cp936 主控台遇到符號就炸
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'

Push-Location $RepoRoot
try {
    & $PythonExe $PrepScript @Args
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
