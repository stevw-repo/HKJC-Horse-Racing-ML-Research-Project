# Register a Windows Task Scheduler job that runs the HKJC race-day pipeline at a fixed cutoff.
# It only writes a recommendation card + logs odds -- it NEVER places a bet.
#
# Usage (from a normal PowerShell, in the repo root):
#   .\scripts\register_raceday_task.ps1 -Date 2026-06-21 -Venue ST -Time 13:00
#
# Then, to also log live odds through the meeting, schedule a second job or run manually:
#   .\.venv\Scripts\hkjc.exe log-odds --date 2026-06-21 --venue ST --rounds 120 --interval 30
#
# Remove a task later with:  Unregister-ScheduledTask -TaskName "HKJC-RaceDay-2026-06-21-ST"

param(
    [Parameter(Mandatory = $true)][string]$Date,
    [string]$Venue = "ST",
    [string]$Time = "13:00",
    [string]$Model = "logit"
)

$repo = Split-Path -Parent $PSScriptRoot
$hkjc = Join-Path $repo ".venv\Scripts\hkjc.exe"
if (-not (Test-Path $hkjc)) { throw "hkjc.exe not found at $hkjc -- run 'uv sync' first." }

$action = New-ScheduledTaskAction -Execute $hkjc `
    -Argument "race-day --date $Date --venue $Venue --model $Model" -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -Once -At ([datetime]::Parse("$Date $Time"))
$task = "HKJC-RaceDay-$Date-$Venue"

Register-ScheduledTask -TaskName $task -Action $action -Trigger $trigger -Force `
    -Description "HKJC race-day recommendations (recommends only; never places a bet)." | Out-Null
Write-Host "Registered '$task' to run 'hkjc race-day --date $Date --venue $Venue' at $Date $Time."
