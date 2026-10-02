# 来源：本项目原创。查看已有演示；可选生成新的统一任务记录，不覆盖旧归档。
param(
    [switch]$RunMission,
    [switch]$Open,
    [ValidateRange(0, 65535)][int]$Port = 8798,
    [string]$Python
)
$ErrorActionPreference = 'Stop'
if (-not $Python) {
    $bundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    if (Test-Path -LiteralPath $bundledPython) { $Python = $bundledPython }
    else { $Python = (Get-Command python -ErrorAction Stop).Source }
}
Push-Location $PSScriptRoot
try {
    $portalArguments = @('-m', 'drone_nav.demo_portal', '--port', "$Port")
    if ($RunMission) {
        $runOutput = Join-Path 'work' ('mission-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fffffff'))
        & $Python -u tools/supervised_mission_experiment.py --output $runOutput
        if ($LASTEXITCODE -ne 0) { throw 'Mission failed; saved output is preserved.' }
        & $Python -u tools/supervised_mission_experiment.py --output $runOutput --verify
        if ($LASTEXITCODE -ne 0) { throw 'Replay verification failed; saved output is preserved.' }
        $portalArguments += @('--mission-run', $runOutput)
    }
    if ($Open) { $portalArguments += '--open' }
    & $Python @portalArguments
    if ($LASTEXITCODE -ne 0) { throw 'Demo service failed. See output above.' }
}
finally { Pop-Location }
