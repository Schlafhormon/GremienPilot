# Run with: powershell -NoProfile -File tests/setup.Tests.ps1
# No Docker daemon, downloads or changes to running containers are needed.
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$tokens = $null
$parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    (Join-Path $repoRoot "setup.ps1"), [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count) { throw ($parseErrors | Out-String) }

# Load only the functions under test, without executing the setup entrypoint.
foreach ($name in @("Test-Truthy", "Invoke-BuildLocalImages", "Remove-ExistingContainersForRebuild",
                    "Get-ProjectVolumeName", "Wait-ForServices", "Show-StartupProgress", "Show-FailureDiagnostics")) {
    $definition = $ast.Find({ param($node)
        $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name
    }, $true)
    . ([scriptblock]::Create($definition.Extent.Text))
}
function Write-Info { Write-Host $args }
function Write-Success { Write-Host $args }
function Write-Err { Write-Host $args }
function Write-Warn { Write-Host $args }

# Deliberately emit stdout and stderr, as Docker/BuildKit does on failure.
function docker {
    $script:Calls += ,@($args)
    "Docker output"
    Write-Error "Docker diagnostic" -ErrorAction Continue
    $global:LASTEXITCODE = if ($script:Calls.Count -eq $script:FailAt) { 1 } else { 0 }
}

$ScriptDir = $repoRoot
$BACKEND_GPU_IMAGE = "test-backend:gpu"
$BACKEND_CPU_IMAGE = "test-backend:cpu"
$FRONTEND_IMAGE = "test-frontend:local"
$PROTOKOLL_PRECACHE_MODELS = "0"
$PROTOKOLL_BUILD_NO_CACHE = "true"

foreach ($gpu in @($false, $true)) {
    $script:USE_GPU = $gpu
    foreach ($failure in @(1, 2, 0)) {
        $script:Calls = @()
        $script:FailAt = $failure
        $result = @(Invoke-BuildLocalImages)
        if ($result.Count -ne 1 -or $result[0] -isnot [bool]) {
            throw "Docker output leaked into Boolean result (GPU=$gpu, failure=$failure)"
        }
        if ($result[0] -ne ($failure -eq 0)) { throw "Incorrect build status" }
        $expectedCalls = if ($failure -eq 1) { 1 } else { 2 }
        if ($script:Calls.Count -ne $expectedCalls) { throw "Build continued after backend failure" }
        if ($script:Calls[0] -notcontains "--progress=plain") { throw "Missing full build diagnostics" }
        if ($script:Calls[0] -notcontains "--no-cache") { throw "No-cache option lost" }
        $expectedFile = if ($gpu) { ".\app\backend\Dockerfile.gpu" } else { ".\app\backend\Dockerfile" }
        if ($script:Calls[0] -notcontains $expectedFile) { throw "Wrong backend Dockerfile" }
    }
}

$script:Calls = @()
$script:FailAt = 3 # ps -q, ps, down
$result = @(Remove-ExistingContainersForRebuild)
if ($result.Count -ne 1 -or $result[0] -isnot [bool] -or $result[0]) {
    throw "Container removal failure was hidden by Docker output"
}

# Timeout must preserve a single false return and actually show Docker diagnostics.
$script:Calls = @()
$script:FailAt = 0
$script:HostOutput = @()
function Out-Host { process { $script:HostOutput += [string]$_ } }
$captured = @(Wait-ForServices -MaxWaitSeconds 0 6>&1)
$values = @($captured | Where-Object { $_ -is [bool] })
$messages = ($captured | Out-String)
if ($values.Count -ne 1 -or $values[0]) { throw "Incorrect timeout result" }
if (($script:HostOutput -join "`n") -notmatch 'Docker output' -or $messages -notmatch 'beendet weder Container noch Downloads') {
    throw "Missing visible timeout diagnostics"
}
if ($messages -match 'Dienste konnten nicht gestartet werden|setup.ps1 cleanup') {
    throw "Timeout incorrectly claims startup failure or recommends deleting model data"
}

$PORT_BACKEND = 8010
$PORT_FRONTEND = 3000
$script:BrowserOpened = $false
function Invoke-WebRequest { [pscustomobject]@{ StatusCode = 200 } }
function Show-SuccessMessage { Write-Host "Ready" }
function Start-Process { $script:BrowserOpened = $true }
$result = @(Wait-ForServices -MaxWaitSeconds 1)
if ($result.Count -ne 1 -or $result[0] -isnot [bool] -or -not $result[0] -or -not $script:BrowserOpened) {
    throw "Ready service did not return success"
}

function docker {
    $global:LASTEXITCODE = 0
    if ($args -contains 'config') {
        '{"volumes":{"ollama_data":{"name":"custom-project_ollama_data"}}}'
    } else {
        "ollama | pulling model: 38%$([char]27)[K"
        "ollama | [GIN] HEAD /"
    }
}
if ((Get-ProjectVolumeName "ollama_data") -ne 'custom-project_ollama_data') {
    throw "Resolved Compose volume name was ignored"
}
$captured = @(Show-StartupProgress 6>&1)
$messages = $captured | Out-String
if ($messages -notmatch '38%' -or $messages.Contains([string][char]27)) {
    throw "Download progress is missing or still contains terminal escape sequences"
}
Write-Host "PASS: 11 setup regression cases and PowerShell syntax"
