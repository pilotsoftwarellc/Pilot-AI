$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Runner = Join-Path $Root "scripts\download_120gb_bilingual.ps1"
$QuotedRunner = '"' + $Runner + '"'

Start-Process powershell.exe -WorkingDirectory $Root -ArgumentList "-NoExit -ExecutionPolicy Bypass -File $QuotedRunner"
