param(
  [string]$PidFile = "archiver.pid"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $PidFile)) {
  "No PID file found at $PidFile"
  exit 1
}

$processId = (Get-Content $PidFile -Raw).Trim()
if (-not $processId) {
  "PID file is empty: $PidFile"
  exit 1
}

Stop-Process -Id ([int]$processId) -Force
Remove-Item -Force $PidFile
"Stopped archiver PID $processId"
