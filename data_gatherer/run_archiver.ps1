param(
  [string]$Config = "config.json",
  [int]$IntervalS = 60,
  [int]$DiscoverEveryS = 3600,
  [string]$LogFile = "archiver.log",
  [switch]$Foreground
)

$ErrorActionPreference = "Stop"

$pidFile = "archiver.pid"
if (Test-Path $pidFile) {
  $existingPid = (Get-Content $pidFile -Raw).Trim()
  if ($existingPid) {
    $existing = Get-Process -Id ([int]$existingPid) -ErrorAction SilentlyContinue
    if ($existing) {
      "Archiver already running (PID $existingPid). Stop it with .\\stop_archiver.ps1"
      exit 0
    }
  }
  Remove-Item -Force $pidFile -ErrorAction SilentlyContinue
}

$python = (Get-Command python).Source
$args = @(
  "-u",
  "polymarket_archive.py",
  "--config",
  $Config,
  "run",
  "--interval-s",
  "$IntervalS",
  "--discover-every-s",
  "$DiscoverEveryS",
  "--log-file",
  $LogFile
)

if ($Foreground) {
  & $python @args
  exit $LASTEXITCODE
}

$proc = Start-Process `
  -FilePath $python `
  -ArgumentList $args `
  -WindowStyle Hidden `
  -PassThru

$proc.Id | Out-File -Encoding ascii -Force $pidFile
"Started archiver PID $($proc.Id). Log: $LogFile"
