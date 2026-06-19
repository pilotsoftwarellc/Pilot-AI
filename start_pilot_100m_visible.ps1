$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Runner = Join-Path $Root "scripts\run_pilot_100m_first_run.ps1"
$QuotedRunner = '"' + $Runner + '"'

Start-Process powershell.exe -WorkingDirectory $Root -ArgumentList "-NoExit -ExecutionPolicy Bypass -File $QuotedRunner"
