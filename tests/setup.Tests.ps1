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
                    "Get-ProjectVolumeName", "Wait-ForServices", "Show-StartupProgress", "Show-FailureDiagnostics",
                    "Initialize-Configuration", "Invoke-Start", "Get-ConfiguredPort",
                    "Confirm-ModelCacheHandling", "Test-VolumeExists")) {
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
$PORT_LLM = 8080
$script:BrowserOpened = $false
function Invoke-WebRequest { [pscustomobject]@{ StatusCode = 200 } }
function Show-SuccessMessage { Write-Host "Ready" }
function Start-Process { $script:BrowserOpened = $true }
$result = @(Wait-ForServices -MaxWaitSeconds 1)
if ($result.Count -ne 1 -or $result[0] -isnot [bool] -or -not $result[0] -or -not $script:BrowserOpened) {
    throw "Ready service did not return success"
}

$script:ModelProbes = 0
function Invoke-WebRequest {
    param($Uri)
    $code = 200
    if ($Uri -match ':8080/') {
        if ($Uri -ne 'http://127.0.0.1:8080/health') { throw 'Model probe must match the IPv4-only Compose binding' }
        $script:ModelProbes++
        if ($script:ModelProbes -eq 1) { $code = 503 }
    }
    [pscustomobject]@{StatusCode=$code}
}
$result = @(Wait-ForServices -MaxWaitSeconds 3)
if (-not $result[-1] -or $script:ModelProbes -lt 2) { throw 'Backend health bypassed model readiness' }

function docker {
    $global:LASTEXITCODE = 0
    if ($args -contains 'config') {
        '{"volumes":{"backend_hf_cache":{"name":"custom-project_backend_hf_cache"}}}'
    } else {
        "ollama | pulling model: 38%$([char]27)[K"
        "ollama | [GIN] HEAD /"
    }
}
if ((Get-ProjectVolumeName "backend_hf_cache") -ne 'custom-project_backend_hf_cache') {
    throw "Resolved Compose volume name was ignored"
}
$captured = @(Show-StartupProgress 6>&1)
$messages = $captured | Out-String
if ($messages -notmatch '38%' -or $messages.Contains([string][char]27)) {
    throw "Download progress is missing or still contains terminal escape sequences"
}

# Gemma has no ollama_data volume. Cache operations must use the resolved
# Compose names and exclude both the session database and other instances.
foreach ($scenario in @('keep', 'refresh', 'fresh', 'missing', 'invalid', 'compose_error')) {
    $script:Calls = @()
    $script:Prompted = $false
    function Read-Host {
        $script:Prompted = $true
        if ($scenario -eq 'refresh') { return 'n' }
        return '' # Default: keep models.
    }
    function docker {
        $script:Calls += ,@($args)
        $global:LASTEXITCODE = 0
        if ($args -contains 'config') {
            if ($scenario -eq 'invalid') { return 'not JSON' }
            if ($scenario -eq 'compose_error') { $global:LASTEXITCODE = 1; return }
            if ($scenario -eq 'missing') {
                return '{"volumes":{"backend_hf_cache":{"name":"custom-project_hf"}}}'
            }
            return '{"volumes":{"backend_hf_cache":{"name":"custom-project_hf"},"backend_torch_cache":{"name":"custom-project_torch"},"backend_state":{"name":"custom-project_state"}}}'
        }
        if ($args -contains 'inspect' -and $scenario -eq 'fresh') { $global:LASTEXITCODE = 1 }
        if ($args[-1] -notin @('custom-project_hf', 'custom-project_torch')) {
            throw "Operation targeted a session, legacy or foreign volume: $args"
        }
    }
    $result = @(Confirm-ModelCacheHandling)
    $expected = $scenario -in @('keep', 'refresh', 'fresh')
    if ($result.Count -ne 1 -or $result[0] -isnot [bool] -or $result[0] -ne $expected) {
        throw "Incorrect Gemma cache handling result: $scenario"
    }
    $removals = @($script:Calls | Where-Object { $_ -contains 'rm' })
    if ($removals.Count -ne $(if ($scenario -eq 'refresh') { 2 } else { 0 })) {
        throw "Unexpected model cache deletion: $scenario"
    }
    if ($script:Prompted -ne ($scenario -in @('keep', 'refresh'))) {
        throw "Unexpected cache prompt: $scenario"
    }
}
function docker {
    $global:LASTEXITCODE = 0
    if ($args[-1] -eq 'llama') {
        'llama | [Gemma] Pruefe Modelldateien'
        'llama | weights/gemma-4-31B-it-Q4_K_M.gguf: OK'
    } else {
        'backend | INFO: "GET /health HTTP/1.1" 200 OK'
    }
}
$messages = @(Show-StartupProgress 6>&1) | Out-String
if ($messages -notmatch '\.gguf: OK' -or $messages -match 'GET /health') {
    throw 'Model verification progress was hidden by health requests'
}
$testDirectory = Join-Path ([IO.Path]::GetTempPath()) ('gp-setup-' + [guid]::NewGuid())
New-Item -ItemType Directory -Path $testDirectory | Out-Null
$ScriptDir = $testDirectory
$testTemplate = Join-Path $testDirectory '.env.example'
$testEnv = Join-Path $testDirectory '.env'
try {
    [IO.File]::WriteAllText($testTemplate, 'LLM_MODEL=qwen3.5:9b')
    if (-not (Initialize-Configuration)) { throw 'Fresh configuration failed' }
    if ([IO.File]::ReadAllText($testEnv) -ne 'LLM_MODEL=qwen3.5:9b') { throw 'Wrong defaults copied' }
    [IO.File]::WriteAllText($testEnv, 'LLM_MODEL=custom-model')
    if (-not (Initialize-Configuration)) { throw 'Existing configuration failed' }
    if ([IO.File]::ReadAllText($testEnv) -ne 'LLM_MODEL=custom-model') { throw 'Existing settings overwritten' }
    [IO.File]::WriteAllText($testEnv, "TEST_GEMMA_PORT='3001' # local port")
    if ((Get-ConfiguredPort 'TEST_GEMMA_PORT' 3000) -ne 3001) { throw 'Local port ignored' }
    $env:TEST_GEMMA_PORT = '3002'
    if ((Get-ConfiguredPort 'TEST_GEMMA_PORT' 3000) -ne 3002) { throw 'Shell port ignored' }
} finally {
    Remove-Item Env:TEST_GEMMA_PORT -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $testEnv,$testTemplate -Force -ErrorAction SilentlyContinue
    [IO.Directory]::Delete($testDirectory)
    $ScriptDir = $repoRoot
}

function Test-Docker { return $true }
function Invoke-Build { $script:Built = $true }
function Wait-ForServices { return $true }
foreach ($scenario in @('fresh','existing','invalid')) {
    $script:Built = $false
    $script:Started = $false
    function docker {
        $global:LASTEXITCODE = 0
        if ($args -contains 'ps') {
            if ($scenario -eq 'existing') { 'container-id' }
            if ($scenario -eq 'invalid') { $global:LASTEXITCODE = 1 }
        } elseif ($args -contains 'start') { $script:Started = $true }
    }
    Invoke-Start | Out-Null
    if ($script:Built -ne ($scenario -eq 'fresh')) { throw "Wrong initial installation: $scenario" }
    if ($script:Started -ne ($scenario -eq 'existing')) { throw "Wrong existing start: $scenario" }
}
Write-Host "PASS: setup regressions, fresh installation, settings preservation and PowerShell syntax"
