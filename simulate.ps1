# 来源：本项目原创。统一仿真入口；每次新建归档并自动核验。
param(
    [ValidateSet('list', 'run', 'verify', 'view')][string]$Action = 'list',
    [ValidateSet('open', 'building', 'sensor-fault', 'control-fault')][string]$Scenario = 'building',
    [int[]]$Goal,
    [ValidateRange(1, 64)][int]$MaxTicks = 24,
    [string]$Output,
    [string]$InputDirectory,
    [switch]$Open,
    [ValidateRange(0, 65535)][int]$Port = 0,
    [string]$Python
)
$ErrorActionPreference = 'Stop'
if (-not $Python) {
    $simulationPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    if (Test-Path -LiteralPath $simulationPython) { $Python = $simulationPython }
    else { $Python = (Get-Command python -ErrorAction Stop).Source }
}
if ($Action -in @('verify', 'view') -and -not $InputDirectory) {
    throw 'verify / view 必须提供 -InputDirectory。'
}
if ($Goal -and $Goal.Count -ne 2) { throw '-Goal 必须恰好包含两个整数。' }
Push-Location $PSScriptRoot
try {
    $simulationArguments = @('-u', '-m', 'drone_nav.simulation', $Action)
    if ($Action -eq 'run') {
        $simulationArguments += @('--scenario', $Scenario, '--max-ticks', "$MaxTicks")
        if ($Goal) { $simulationArguments += @('--goal', "$($Goal[0])", "$($Goal[1])") }
        if ($Output) { $simulationArguments += @('--output', $Output) }
    }
    if ($Action -in @('verify', 'view')) { $simulationArguments += @('--input', $InputDirectory) }
    if ($Action -eq 'view') { $simulationArguments += @('--port', "$Port") }
    if ($Open -and $Action -in @('run', 'view')) { $simulationArguments += '--open' }
    & $Python @simulationArguments
    if ($LASTEXITCODE -ne 0) { throw '仿真入口未完成，已有输出保留；请查看上方原因。' }
}
finally { Pop-Location }
