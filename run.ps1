# 来源：本项目原创。优先使用已存在的稳定 Python；不安装软件、不修改系统配置。
param(
    [ValidateSet('vision', 'baseline', 'camera', 'explore', 'motion', 'localization', 'planning', 'delivery', 'physics', 'coupled', 'descent', 'realvision', 'scale')][string]$Mode = 'vision',
    [switch]$Test,
    [switch]$Open,
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
    if ($Test) {
        & $Python -m unittest discover -s tests -v
        if ($LASTEXITCODE -ne 0) { throw 'Tests failed. See output above.' }
    }
    $runOutput = Join-Path 'work' ('run-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fffffff'))
    & $Python -m drone_nav --mode $Mode --output $runOutput
    if ($LASTEXITCODE -ne 0) { throw 'Experiment failed. See output above.' }
    if ($Open) { Invoke-Item -LiteralPath (Join-Path $runOutput 'demo.html') }
}
finally { Pop-Location }
