<#
.SYNOPSIS
    vOS-VirtualVoice - Phase 1 one-command runner (Windows / PowerShell).

.DESCRIPTION
    Finds a Python interpreter (prefers .venv, then the py launcher, then
    python on PATH) and forwards all arguments to scripts/phase1_prep.py.

    macOS / Linux: use scripts/run_phase1.sh instead.

    !! ASCII-ONLY FILE - DO NOT ADD NON-ASCII CHARACTERS !!
    Windows PowerShell 5.1 reads .ps1 files using the system ANSI codepage
    (e.g. cp950/cp936/cp932) when the file has no UTF-8 BOM. A single
    non-ASCII character can then corrupt quote pairing and the whole script
    fails with a ParserError - even if the character is inside a comment.
    Keeping this file pure ASCII makes it work on every locale.
    tests/test_scripts.py guards this rule.

.EXAMPLE
    .\scripts\run_phase1.ps1 scan
    .\scripts\run_phase1.ps1 convert --limit 50 -v
    .\scripts\run_phase1.ps1 report
#>
[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Args
)

$ErrorActionPreference = 'Stop'

# Repo root (this file lives in <root>/scripts/)
$RepoRoot = Split-Path -Parent $PSScriptRoot

function Resolve-Python {
    # 1) Project virtual environment, if present
    $venv = Join-Path $RepoRoot '.venv\Scripts\python.exe'
    if (Test-Path $venv) { return $venv }

    # 2) Windows py launcher (knows about installed versions)
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) { return $py.Source }

    # 3) python on PATH
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) { return $python.Source }

    throw "Python not found. Install Python 3.10+ and add it to PATH, or create a .venv."
}

$PythonExe = Resolve-Python
$PrepScript = Join-Path $RepoRoot 'scripts\phase1_prep.py'

if (-not $Args -or $Args.Count -eq 0) {
    Write-Host "Usage: .\scripts\run_phase1.ps1 <command> [options]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  doctor     Check Python deps / external tools / folders / target format"
    Write-Host "  scan       Sniff the real format of your material (writes nothing)"
    Write-Host "  unpack     Unpack game containers with vgmstream-cli"
    Write-Host "  convert    Standardize to 24bit/48kHz WAV + duration filtering"
    Write-Host "  audition   Build an audition reel + reference list (manual QA)"
    Write-Host "  tune       Analyze the manifest and suggest threshold changes"
    Write-Host "  report     Generate the consolidated Phase 1 report"
    Write-Host "  all        Run scan -> unpack -> convert in order"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\scripts\run_phase1.ps1 scan"
    Write-Host "  .\scripts\run_phase1.ps1 convert --limit 50 -v"
    exit 64
}

# Make Python's stdout UTF-8 so symbols/Chinese survive the cp950 console
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
