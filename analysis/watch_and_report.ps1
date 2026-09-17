# Wait for the 33 full-experiment training runs to finish, then evaluate them
# and rebuild the point-5 document with the ten-seed numbers in place.
#
#   powershell -ExecutionPolicy Bypass -File analysis\watch_and_report.ps1
#
# Safe to kill at any point: it only reads until training is done, and the two
# commands it then runs are the same ones you would type by hand.

$ErrorActionPreference = "Stop"
Set-Location (Split-Path (Split-Path $MyInvocation.MyCommand.Path))

$expected = 33
$log = "results\models\full\watcher.log"

function Say($msg) {
    $line = "{0}  {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Write-Output $line
    Add-Content -Path $log -Value $line -Encoding utf8
}

Say "watching for $expected training runs to finish"

while ($true) {
    $procs = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
               Where-Object { $_.CommandLine -like "*full_experiment*" })
    $done = @(Get-ChildItem "results\models\full\*.zip" -ErrorAction SilentlyContinue).Count

    if ($procs.Count -eq 0) {
        Say "no training processes left; $done/$expected models on disk"
        break
    }
    Say "$($procs.Count) still training, $done/$expected models saved"
    Start-Sleep -Seconds 120
}

$done = @(Get-ChildItem "results\models\full\*.zip" -ErrorAction SilentlyContinue).Count
if ($done -eq 0) {
    Say "ABORT: training produced no models -- check results\models\full\*.txt"
    exit 1
}
if ($done -lt $expected) {
    Say "WARNING: only $done of $expected models were saved; evaluating what exists"
}

Say "evaluating"
python analysis\full_experiment.py --mode eval
if (-not $?) { Say "evaluation failed"; exit 1 }

Say "rebuilding the point-5 document"
python analysis\point5_report.py --docx
if (-not $?) { Say "report build failed"; exit 1 }

Say "done: results\point5_algorithm_comparison.docx now carries the ten-seed numbers"
