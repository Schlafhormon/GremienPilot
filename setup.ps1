#
# GremienPilot - Setup Script for Windows
# Intelligentes Setup-Skript fuer nicht-technische Benutzer
#
# Verwendung:
#   .\setup.ps1 build     # Lokale Images bauen und Container neu erstellen
#   .\setup.ps1 start     # Vorhandene Container starten
#   .\setup.ps1 stop      # Anwendung stoppen, Container behalten
#   .\setup.ps1 remove    # Container stoppen und entfernen
#   .\setup.ps1 status    # Status anzeigen
#   .\setup.ps1 restart   # Anwendung neu starten
#   .\setup.ps1 logs      # Live-Logs anzeigen
#   .\setup.ps1 cleanup   # Alle Daten loeschen und neu starten
#   .\setup.ps1 help      # Hilfe anzeigen
#

param(
    [Parameter(Position=0)]
    [string]$Command = "start"
)

$ErrorActionPreference = "Continue"

# Docker images. Defaults keep the established deployment path; set
# PROTOKOLL_IMAGE_TAG=v1.2.3 to pin application images to a release tag.
$IMAGE_BASE = "ghcr.io/aihpi/pilotproject-protokollierungsassistenz"
$PROTOKOLL_IMAGE_TAG = $env:PROTOKOLL_IMAGE_TAG
$USER_FRONTEND_IMAGE = $env:FRONTEND_IMAGE
$USER_BACKEND_IMAGE = $env:BACKEND_IMAGE
$USER_BACKEND_GPU_IMAGE = $env:BACKEND_GPU_IMAGE
$PROTOKOLL_BUILD_LOCAL = if ($env:PROTOKOLL_BUILD_LOCAL) { $env:PROTOKOLL_BUILD_LOCAL } else { "auto" }
$PROTOKOLL_PRECACHE_MODELS = if ($env:PROTOKOLL_PRECACHE_MODELS) { $env:PROTOKOLL_PRECACHE_MODELS } else { "0" }
$PROTOKOLL_BUILD_NO_CACHE = if ($env:PROTOKOLL_BUILD_NO_CACHE) { $env:PROTOKOLL_BUILD_NO_CACHE } else { "false" }
$script:PROTOKOLL_PULL_POLICY = if ($env:PROTOKOLL_PULL_POLICY) { $env:PROTOKOLL_PULL_POLICY } else { "missing" }

if ($PROTOKOLL_IMAGE_TAG) {
    $FRONTEND_IMAGE = if ($env:FRONTEND_IMAGE) { $env:FRONTEND_IMAGE } else { "${IMAGE_BASE}/frontend:${PROTOKOLL_IMAGE_TAG}" }
    $BACKEND_CPU_IMAGE = if ($env:BACKEND_IMAGE) { $env:BACKEND_IMAGE } else { "${IMAGE_BASE}/backend:${PROTOKOLL_IMAGE_TAG}" }
    $BACKEND_GPU_IMAGE = if ($env:BACKEND_GPU_IMAGE) { $env:BACKEND_GPU_IMAGE } else { "${IMAGE_BASE}/backend:${PROTOKOLL_IMAGE_TAG}-gpu" }
} else {
    $FRONTEND_IMAGE = if ($env:FRONTEND_IMAGE) { $env:FRONTEND_IMAGE } else { "${IMAGE_BASE}/frontend:latest" }
    $BACKEND_CPU_IMAGE = if ($env:BACKEND_IMAGE) { $env:BACKEND_IMAGE } else { "${IMAGE_BASE}/backend:cpu-latest" }
    $BACKEND_GPU_IMAGE = if ($env:BACKEND_GPU_IMAGE) { $env:BACKEND_GPU_IMAGE } else { "${IMAGE_BASE}/backend:gpu-latest" }
}

$OLLAMA_IMAGE_TAG = if ($env:OLLAMA_IMAGE_TAG) { $env:OLLAMA_IMAGE_TAG } else { "latest" }
$OLLAMA_IMAGE = if ($env:OLLAMA_IMAGE) { $env:OLLAMA_IMAGE } else { "ollama/ollama:${OLLAMA_IMAGE_TAG}" }
$OLLAMA_MODEL = if ($env:LLM_MODEL) { $env:LLM_MODEL } else { "gemma4:31b-it-q4_K_M" }

$env:FRONTEND_IMAGE = $FRONTEND_IMAGE
$env:BACKEND_IMAGE = $BACKEND_CPU_IMAGE
$env:BACKEND_GPU_IMAGE = $BACKEND_GPU_IMAGE
$env:OLLAMA_IMAGE = $OLLAMA_IMAGE

# Ports used by the application
$PORT_FRONTEND = 3000
$PORT_BACKEND = 8010
$PORT_OLLAMA = 11434

# Global state
$script:USE_GPU = $false
$script:MissingItems = @()
$script:BUILD_LOCAL_IMAGES = $false

# Print colored messages (German)
function Write-Info { Write-Host "[INFO] " -ForegroundColor Blue -NoNewline; Write-Host $args }
function Write-Success { Write-Host "[OK] " -ForegroundColor Green -NoNewline; Write-Host $args }
function Write-Warn { Write-Host "[WARNUNG] " -ForegroundColor Yellow -NoNewline; Write-Host $args }
function Write-Err { Write-Host "[FEHLER] " -ForegroundColor Red -NoNewline; Write-Host $args }

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

function Test-Truthy {
    param([string]$Value)
    return $Value -match "^(1|true|yes|ja|on)$"
}

function Test-Falsy {
    param([string]$Value)
    return $Value -match "^(0|false|no|nein|off)$"
}

$hasLocalDockerfiles = (Test-Path "app\backend\Dockerfile") -and (Test-Path "app\frontend\Dockerfile")
$explicitApplicationImage =
    $USER_FRONTEND_IMAGE -or $USER_BACKEND_IMAGE -or $USER_BACKEND_GPU_IMAGE -or $PROTOKOLL_IMAGE_TAG

if (Test-Truthy $PROTOKOLL_BUILD_LOCAL) {
    $script:BUILD_LOCAL_IMAGES = $true
} elseif ((-not (Test-Falsy $PROTOKOLL_BUILD_LOCAL)) -and $PROTOKOLL_BUILD_LOCAL.ToLower() -eq "auto") {
    $script:BUILD_LOCAL_IMAGES = $hasLocalDockerfiles -and (-not $explicitApplicationImage)
}

if ($script:BUILD_LOCAL_IMAGES) {
    $FRONTEND_IMAGE = "ki-protokollierung-frontend:local"
    $BACKEND_CPU_IMAGE = "ki-protokollierung-backend:local"
    $BACKEND_GPU_IMAGE = "ki-protokollierung-backend:gpu-local"
    $env:FRONTEND_IMAGE = $FRONTEND_IMAGE
    $env:BACKEND_IMAGE = $BACKEND_CPU_IMAGE
    $env:BACKEND_GPU_IMAGE = $BACKEND_GPU_IMAGE
}

#######################################
# Show help message
#######################################
function Show-Help {
    Write-Host ""
    Write-Host "==============================================" -ForegroundColor Cyan
    Write-Host "  GremienPilot - Hilfe" -ForegroundColor Cyan
    Write-Host "==============================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Verwendung: .\setup.ps1 [BEFEHL]"
    Write-Host ""
    Write-Host "Befehle:"
    Write-Host "  (ohne)     Vorhandene Container starten"
    Write-Host "  start      Vorhandene Container starten, ohne neu zu bauen"
    Write-Host "  build      Lokale Images neu bauen und Container neu erstellen"
    Write-Host "  stop       Anwendung stoppen, Container bleiben erhalten"
    Write-Host "  remove     Anwendung stoppen und Container entfernen"
    Write-Host "  status     Status der Dienste anzeigen"
    Write-Host "  restart    Anwendung neu starten"
    Write-Host "  logs       Live-Logs anzeigen (Strg+C zum Beenden)"
    Write-Host "  cleanup    Alle Daten loeschen und neu starten"
    Write-Host "  help       Diese Hilfe anzeigen"
    Write-Host ""
    Write-Host "Optionale Umgebungsvariablen:"
    Write-Host "  PROTOKOLL_IMAGE_TAG=v1.2.3       Stabile App-Images verwenden"
    Write-Host "  PROTOKOLL_BUILD_LOCAL=auto       Lokale Repo-Images bauen: auto, true, false"
    Write-Host "  PROTOKOLL_BUILD_NO_CACHE=true    Lokalen Docker-Build ohne Cache ausfuehren"
    Write-Host "  PROTOKOLL_PRECACHE_MODELS=0      Modelle beim lokalen Build vorladen: 0 oder 1"
    Write-Host "  PROTOKOLL_PULL_POLICY=missing    Pull-Verhalten: missing, always, never"
    Write-Host "  FRONTEND_IMAGE/BACKEND_IMAGE/... Vollstaendige Image-Referenzen setzen"
    Write-Host ""
    Write-Host "Beispiele:"
    Write-Host "  .\setup.ps1 build     # Nach Code-Aenderungen neu bauen"
    Write-Host "  .\setup.ps1 start     # Vorhandene Container nur starten"
    Write-Host "  `$env:PROTOKOLL_BUILD_NO_CACHE='true'; .\setup.ps1 build"
    Write-Host "  .\setup.ps1 status    # Pruefen ob alles laeuft"
    Write-Host "  .\setup.ps1 logs      # Fehlersuche mit Logs"
    Write-Host ""
}

#######################################
# Use the local image names produced by this repository
#######################################
function Set-LocalApplicationImages {
    $script:BUILD_LOCAL_IMAGES = $true
    $script:PROTOKOLL_PULL_POLICY = "missing"
    $script:FRONTEND_IMAGE = "ki-protokollierung-frontend:local"
    $script:BACKEND_CPU_IMAGE = "ki-protokollierung-backend:local"
    $script:BACKEND_GPU_IMAGE = "ki-protokollierung-backend:gpu-local"
    $env:FRONTEND_IMAGE = $script:FRONTEND_IMAGE
    $env:BACKEND_IMAGE = $script:BACKEND_CPU_IMAGE
    $env:BACKEND_GPU_IMAGE = $script:BACKEND_GPU_IMAGE
}

function Get-ProjectVolumeName {
    param([string]$Suffix)
    return "$(Split-Path -Leaf $ScriptDir)_$Suffix"
}

function Test-VolumeExists {
    param([string]$VolumeName)
    $null = docker volume inspect $VolumeName 2>&1
    return $LASTEXITCODE -eq 0
}

function Remove-ExistingContainersForRebuild {
    $containers = docker compose ps -a -q 2>&1
    if (-not $containers -or $containers.Count -eq 0) {
        return $true
    }

    Write-Host ""
    Write-Warn "Es sind bereits Container fuer diese Anwendung vorhanden."
    Write-Host ""
    docker compose ps -a 2>&1
    Write-Host ""
    Write-Host "Beim Build werden die vorhandenen Container automatisch entfernt."
    Write-Host "Modell-Volumes bleiben erhalten, solange Sie spaeter nicht ausdruecklich"
    Write-Host "das Neuladen der Modelle waehlen."

    Write-Info "Stoppe und entferne vorhandene Container (Volumes bleiben erhalten)..."
    docker compose down --remove-orphans 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Vorhandene Container konnten nicht entfernt werden."
        return $false
    }
    return $true
}

function Confirm-ModelCacheHandling {
    $volumeNames = @(
        (Get-ProjectVolumeName "ollama_data"),
        (Get-ProjectVolumeName "backend_hf_cache"),
        (Get-ProjectVolumeName "backend_torch_cache")
    )
    $existingVolumes = @($volumeNames | Where-Object { Test-VolumeExists $_ })

    if ($existingVolumes.Count -eq 0) {
        Write-Info "Keine Modell-Volumes gefunden; fehlende Modelle werden beim Start geladen."
        return $true
    }

    Write-Host ""
    Write-Info "Vorhandene Modell-Volumes gefunden:"
    foreach ($volume in $existingVolumes) {
        Write-Host "  - $volume"
    }
    Write-Host ""
    Write-Host "Standard: Modelle behalten. Fehlende oder durch Modellwechsel neue Modelle"
    Write-Host "werden automatisch nachgeladen. Nur bei defektem Cache oder bewusstem"
    Write-Host "Komplett-Refresh sollten die Modell-Volumes geloescht werden."
    $response = Read-Host "Modell-Volumes behalten? (J/n)"

    if ($response -match "^[Nn]$") {
        Write-Warn "Loesche Modell-Volumes. Modelle werden danach erneut heruntergeladen."
        foreach ($volume in $existingVolumes) {
            docker volume rm $volume 2>&1 | Out-Null
            if ($LASTEXITCODE -ne 0) {
                Write-Err "Volume konnte nicht geloescht werden: $volume"
                return $false
            }
        }
    } else {
        Write-Success "Modell-Volumes bleiben erhalten"
    }

    return $true
}

#######################################
# Build local application images from this repository
#######################################
function Invoke-BuildLocalImages {
    Write-Info "Baue lokale Docker-Images aus dem geklonten Repository..."

    $buildArgs = @()
    if (Test-Truthy $PROTOKOLL_BUILD_NO_CACHE) {
        $buildArgs += "--no-cache"
    }

    $backendTag = if ($script:USE_GPU) { $BACKEND_GPU_IMAGE } else { $BACKEND_CPU_IMAGE }
    $backendDockerfile = if ($script:USE_GPU) { ".\app\backend\Dockerfile.gpu" } else { ".\app\backend\Dockerfile" }

    Write-Info "Baue Backend-Image: $backendTag"
    docker build @buildArgs --build-arg "PRECACHE_MODELS=$PROTOKOLL_PRECACHE_MODELS" -f $backendDockerfile -t $backendTag ".\app\backend"
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Backend-Image konnte nicht gebaut werden."
        return $false
    }

    Write-Info "Baue Frontend-Image: $FRONTEND_IMAGE"
    docker build @buildArgs -t $FRONTEND_IMAGE ".\app\frontend"
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Frontend-Image konnte nicht gebaut werden."
        return $false
    }

    Write-Success "Lokale Docker-Images wurden gebaut"
    return $true
}

#######################################
# Pull images according to policy
#######################################
function Invoke-PullImages {
    switch ($script:PROTOKOLL_PULL_POLICY.ToLower()) {
        "always" {
            Write-Info "Pruefe auf Aktualisierungen (Pull-Policy: always)..."
        }
        "missing" {
            if ($script:MissingItems.Count -eq 0) {
                Write-Info "Lokale Images vorhanden; ueberspringe Pull (PROTOKOLL_PULL_POLICY=missing)."
                return
            }
            Write-Info "Lade fehlende Images (Pull-Policy: missing)..."
        }
        "never" {
            Write-Info "Ueberspringe Image-Pull (PROTOKOLL_PULL_POLICY=never)."
            return
        }
        default {
            Write-Warn "Unbekannte PROTOKOLL_PULL_POLICY '$script:PROTOKOLL_PULL_POLICY'; verwende 'missing'."
            if ($script:MissingItems.Count -eq 0) {
                return
            }
        }
    }

    if ($script:USE_GPU) {
        docker compose -f docker-compose.yml -f docker-compose.gpu.yml pull 2>&1 | Out-Null
    } else {
        docker compose pull 2>&1 | Out-Null
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Konnte Images nicht aktualisieren - verwende lokale Images"
    }
}

#######################################
# Check if Docker is installed and running
#######################################
function Test-Docker {
    Write-Info "Ueberpruefe Docker-Installation..."

    $dockerInstalled = Get-Command docker -ErrorAction SilentlyContinue
    if (-not $dockerInstalled) {
        Write-Err "Docker ist nicht installiert!"
        Write-Host ""
        Write-Host "Bitte installieren Sie Docker Desktop:"
        Write-Host "  https://docs.docker.com/desktop/install/windows-install/"
        Write-Host ""
        return $false
    }

    try {
        $null = docker info 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "Docker not running"
        }
    } catch {
        Write-Err "Docker ist installiert, aber nicht gestartet!"
        Write-Host ""
        Write-Host "Bitte starten Sie Docker Desktop und versuchen Sie es erneut."
        return $false
    }

    Write-Success "Docker ist installiert und laeuft"
    return $true
}

#######################################
# Check if a Docker image exists locally
#######################################
function Test-ImageExists {
    param([string]$ImageName)

    try {
        $null = docker image inspect $ImageName 2>&1
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

#######################################
# Check if Ollama model is downloaded
#######################################
function Test-OllamaModelExists {
    $volumeName = (Split-Path -Leaf $ScriptDir) + "_ollama_data"

    try {
        $null = docker volume inspect $volumeName 2>&1
        if ($LASTEXITCODE -eq 0) {
            # Volume exists, assume model might be downloaded
            $volumeInfo = docker system df -v 2>&1 | Select-String $volumeName
            if ($volumeInfo -and $volumeInfo -notmatch "0B") {
                return $true
            }
        }
    } catch {
        # Volume doesn't exist
    }
    return $false
}

#######################################
# Smart disk space check
#######################################
function Test-DiskSpace {
    Write-Info "Ueberpruefe verfuegbaren Speicherplatz..."

    # Calculate required space based on what's already downloaded
    $requiredGB = 3  # Base runtime buffer
    $script:MissingItems = @()

    if (-not (Test-ImageExists $BACKEND_CPU_IMAGE) -and -not (Test-ImageExists $BACKEND_GPU_IMAGE)) {
        $requiredGB += 9
        $script:MissingItems += "Backend-Image (~9GB)"
    }

    if (-not (Test-ImageExists $FRONTEND_IMAGE)) {
        $requiredGB += 1
        $script:MissingItems += "Frontend-Image (~1GB)"
    }

    if (-not (Test-ImageExists $OLLAMA_IMAGE)) {
        $requiredGB += 2
        $script:MissingItems += "Ollama-Image (~2GB)"
    }

    if (-not (Test-OllamaModelExists)) {
        $requiredGB += 5
        $script:MissingItems += "Sprachmodell (~5GB)"
    }

    # Get available disk space
    $drive = (Get-Location).Drive.Name
    $disk = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='${drive}:'" -ErrorAction SilentlyContinue
    if (-not $disk) {
        $disk = Get-WmiObject Win32_LogicalDisk -Filter "DeviceID='${drive}:'"
    }
    $availableGB = [math]::Floor($disk.FreeSpace / 1GB)

    if ($script:MissingItems.Count -eq 0) {
        Write-Success "Alle Images bereits heruntergeladen"
        Write-Success "Verfuegbarer Speicherplatz: ${availableGB}GB (nur ~3GB benoetigt)"
        return $true
    }

    Write-Host "  Noch herunterzuladen:"
    foreach ($item in $script:MissingItems) {
        Write-Host "    - $item"
    }
    Write-Host ""

    if ($availableGB -lt $requiredGB) {
        Write-Err "Nicht genuegend Speicherplatz!"
        Write-Host ""
        Write-Host "  Verfuegbar: ${availableGB}GB"
        Write-Host "  Benoetigt:  ${requiredGB}GB"
        Write-Host ""
        Write-Host "Bitte geben Sie Speicherplatz frei und versuchen Sie es erneut."
        return $false
    }

    Write-Success "Speicherplatz OK (${availableGB}GB verfuegbar, ~${requiredGB}GB benoetigt)"
    return $true
}

#######################################
# Check RAM
#######################################
function Test-RAM {
    Write-Info "Ueberpruefe verfuegbaren Arbeitsspeicher..."

    $computerSystem = Get-CimInstance Win32_ComputerSystem -ErrorAction SilentlyContinue
    if (-not $computerSystem) {
        $computerSystem = Get-WmiObject Win32_ComputerSystem
    }
    $totalRAMGB = [math]::Floor($computerSystem.TotalPhysicalMemory / 1GB)

    if ($totalRAMGB -lt 8) {
        Write-Warn "Wenig Arbeitsspeicher erkannt (${totalRAMGB}GB). Empfohlen: 8GB+"
        Write-Host "  Die Anwendung koennte langsam laufen."
    } else {
        Write-Success "Arbeitsspeicher OK (${totalRAMGB}GB verfuegbar)"
    }
    return $true
}

#######################################
# Check if a port is in use
#######################################
function Test-PortInUse {
    param([int]$Port)

    try {
        $connection = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        return $null -ne $connection
    } catch {
        return $false
    }
}

#######################################
# Get process using a port
#######################################
function Get-PortProcess {
    param([int]$Port)

    try {
        $connection = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($connection) {
            $process = Get-Process -Id $connection.OwningProcess -ErrorAction SilentlyContinue
            if ($process) {
                return "$($process.ProcessName) (PID: $($process.Id))"
            }
        }
    } catch {
        # Ignore errors
    }
    return "Unbekannter Prozess"
}

#######################################
# Kill process using a port
#######################################
function Stop-PortProcess {
    param([int]$Port)

    try {
        $connection = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($connection) {
            Stop-Process -Id $connection.OwningProcess -Force -ErrorAction SilentlyContinue
            return $true
        }
    } catch {
        # Ignore errors
    }
    return $false
}

#######################################
# Check for port conflicts
#######################################
function Test-Ports {
    Write-Info "Ueberpruefe Port-Verfuegbarkeit..."

    $conflicts = @()
    $conflictPorts = @()

    # Check each port (but ignore if our own containers are using them)
    foreach ($port in @($PORT_FRONTEND, $PORT_BACKEND, $PORT_OLLAMA)) {
        if (Test-PortInUse $port) {
            # Check if it's our own Docker container
            $isOurContainer = $false
            $composePs = docker compose ps 2>&1
            if ($composePs -match "0.0.0.0:${port}->") {
                $isOurContainer = $true
            }

            if (-not $isOurContainer) {
                $processInfo = Get-PortProcess $port
                $conflicts += "Port $port`: $processInfo"
                $conflictPorts += $port
            }
        }
    }

    if ($conflicts.Count -eq 0) {
        Write-Success "Alle Ports verfuegbar ($PORT_FRONTEND, $PORT_BACKEND, $PORT_OLLAMA)"
        return $true
    }

    Write-Warn "Port-Konflikte erkannt!"
    Write-Host ""
    foreach ($conflict in $conflicts) {
        Write-Host "  - $conflict"
    }
    Write-Host ""

    $response = Read-Host "Sollen die Ports automatisch freigegeben werden? (j/N)"

    if ($response -match "^[Jj]$") {
        foreach ($port in $conflictPorts) {
            Write-Info "Gebe Port $port frei..."
            if (Stop-PortProcess $port) {
                Write-Success "Port $port freigegeben"
            } else {
                Write-Err "Konnte Port $port nicht freigeben"
                Write-Host "  Bitte beenden Sie den Prozess manuell und versuchen Sie es erneut."
                return $false
            }
        }
        Start-Sleep -Seconds 1  # Give processes time to release ports
        return $true
    } else {
        Write-Host ""
        Write-Host "Bitte beenden Sie die konfliktierenden Prozesse manuell im Task-Manager."
        return $false
    }
}

#######################################
# Check for existing installation
#######################################
function Test-ExistingInstallation {
    # Check if docker compose containers exist
    $containers = docker compose ps -q 2>&1

    if (-not $containers -or $containers.Count -eq 0) {
        return $true  # No existing installation, continue
    }

    # Check container states
    $runningContainers = docker compose ps --status running -q 2>&1
    $runningCount = if ($runningContainers) { @($runningContainers).Count } else { 0 }

    $composePs = docker compose ps 2>&1
    $unhealthyCount = ($composePs | Select-String "unhealthy").Count

    $exitedContainers = docker compose ps --status exited -q 2>&1
    $exitedCount = if ($exitedContainers) { @($exitedContainers).Count } else { 0 }

    # All healthy and running
    if ($runningCount -ge 3 -and $unhealthyCount -eq 0) {
        Write-Host ""
        Write-Host "Die Anwendung laeuft bereits!" -ForegroundColor Green
        Write-Host ""
        Write-Host "  Frontend: http://localhost:$PORT_FRONTEND"
        Write-Host ""
        Write-Host "Optionen:"
        Write-Host "  1. Browser oeffnen"
        Write-Host "  2. Anwendung neu starten"
        Write-Host "  3. Anwendung stoppen"
        Write-Host "  4. Status anzeigen"
        Write-Host "  5. Nichts tun (beenden)"
        Write-Host ""
        $choice = Read-Host "Ihre Wahl (1/2/3/4/5)"

        switch ($choice) {
            "1" {
                Start-Process "http://localhost:$PORT_FRONTEND"
                exit 0
            }
            "2" {
                Invoke-Restart
                exit 0
            }
            "3" {
                Invoke-Stop
                exit 0
            }
            "4" {
                Show-Status
                exit 0
            }
            default {
                exit 0
            }
        }
    }

    # Some containers exist but not all healthy - partial/failed installation
    if ($exitedCount -gt 0 -or $unhealthyCount -gt 0 -or $runningCount -lt 3) {
        Write-Host ""
        Write-Warn "Es wurde eine unvollstaendige Installation gefunden!"
        Write-Host ""
        Write-Host "Container-Status:"
        docker compose ps 2>&1 | Select-Object -First 10
        Write-Host ""
        Write-Host "Optionen:"
        Write-Host "  1. Aufraeumen und neu starten (empfohlen)"
        Write-Host "  2. Versuchen, bestehende Container zu reparieren"
        Write-Host "  3. Abbrechen"
        Write-Host ""
        $choice = Read-Host "Ihre Wahl (1/2/3)"

        switch ($choice) {
            "1" {
                Write-Info "Raeume bestehende Container auf..."
                docker compose down --remove-orphans 2>&1 | Out-Null
                Write-Success "Aufgeraeumt. Starte neu..."
                return $true  # Continue with fresh setup
            }
            "2" {
                Write-Info "Versuche Container zu reparieren..."
                docker compose up -d 2>&1 | Out-Null
                Wait-ForServices
                exit 0
            }
            default {
                exit 0
            }
        }
    }

    return $true
}

#######################################
# Check for NVIDIA GPU
#######################################
function Test-GPU {
    $script:USE_GPU = $false

    Write-Info "Ueberpruefe NVIDIA GPU..."

    # Step 1: Detect NVIDIA GPU via Windows device manager (reliable, no nvidia-smi needed)
    $nvidiaGPU = $null
    $nvidiaGPU = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Where-Object { $_.Name -match "NVIDIA" } | Select-Object -First 1
    if (-not $nvidiaGPU) {
        $nvidiaGPU = Get-WmiObject Win32_VideoController -ErrorAction SilentlyContinue | Where-Object { $_.Name -match "NVIDIA" } | Select-Object -First 1
    }

    if (-not $nvidiaGPU) {
        Write-Info "Keine NVIDIA GPU im System gefunden, verwende CPU-Modus"
        return
    }

    Write-Success "NVIDIA GPU erkannt: $($nvidiaGPU.Name)"

    # Step 2: Verify NVIDIA driver works via nvidia-smi
    $nvidiaSmiCmd = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvidiaSmiCmd) {
        Write-Host "  Pruefe NVIDIA-Treiber ($($nvidiaSmiCmd.Source))..."
        $tempOut = Join-Path $env:TEMP "nvsmi_out.txt"
        $tempErr = Join-Path $env:TEMP "nvsmi_err.txt"
        try {
            $proc = Start-Process -FilePath $nvidiaSmiCmd.Source -NoNewWindow -Wait -PassThru -RedirectStandardOutput $tempOut -RedirectStandardError $tempErr
            if ($proc.ExitCode -ne 0) {
                $errMsg = ""
                if (Test-Path $tempErr) { $errMsg = (Get-Content $tempErr -Raw -ErrorAction SilentlyContinue).Trim() }
                Write-Warn "nvidia-smi fehlgeschlagen (Exit-Code: $($proc.ExitCode))"
                if ($errMsg) { Write-Host "  $errMsg" }
                Write-Host "  Treiber-Update empfohlen: https://www.nvidia.com/Download/index.aspx"
            } else {
                Write-Success "NVIDIA-Treiber OK"
            }
        } catch {
            Write-Warn "nvidia-smi konnte nicht ausgefuehrt werden: $($_.Exception.Message)"
        } finally {
            Remove-Item $tempOut -ErrorAction SilentlyContinue
            Remove-Item $tempErr -ErrorAction SilentlyContinue
        }
    } else {
        Write-Warn "nvidia-smi nicht gefunden - Treiberstatus konnte nicht geprueft werden"
    }

    # Step 3: Ask user about GPU mode
    Write-Host ""
    Write-Host "GPU-Modus bietet deutlich schnellere Transkription."
    $response = Read-Host "Moechten Sie den GPU-Modus verwenden? (j/N)"

    if ($response -match "^[Jj]$") {
        $dockerInfo = docker info 2>&1
        if ($dockerInfo -match "nvidia") {
            $script:USE_GPU = $true
            Write-Success "GPU-Modus aktiviert"
        } else {
            Write-Warn "NVIDIA Container Toolkit moeglicherweise nicht konfiguriert"
            Write-Host "  Fuer GPU-Modus stellen Sie sicher, dass Docker Desktop GPU-Unterstuetzung hat."
            Write-Host "  Siehe: https://docs.docker.com/desktop/gpu/"
            Write-Host ""
            $continue = Read-Host "GPU-Modus trotzdem versuchen? (j/N)"
            if ($continue -match "^[Jj]$") {
                $script:USE_GPU = $true
            } else {
                Write-Host "  Fahre mit CPU-Modus fort..."
            }
        }
    }
}

#######################################
# Wait for services to be ready
#######################################
function Wait-ForServices {
    Write-Host ""
    Write-Info "Warte auf Dienste..."
    Write-Host "Das System laedt KI-Modelle. Dies kann einige Minuten dauern."
    Write-Host ""

    $maxWait = 600  # 10 minutes
    $waitCount = 0
    $ready = $false

    while ($waitCount -lt $maxWait) {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:$PORT_BACKEND/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction SilentlyContinue
            if ($response.StatusCode -eq 200) {
                $ready = $true
                break
            }
        } catch {
            # Service not ready yet
        }

        # Show progress every 15 seconds
        if ($waitCount % 15 -eq 0) {
            Write-Host "  Laedt noch... (${waitCount}s vergangen)"
        }

        Start-Sleep -Seconds 1
        $waitCount++
    }

    if (-not $ready) {
        Write-Host ""
        Write-Err "Dienste konnten nicht gestartet werden!"
        Write-Host ""
        Show-FailureDiagnostics
        return $false
    }

    Write-Host ""
    Write-Success "Anwendung ist bereit!"
    Show-SuccessMessage
    Start-Process "http://localhost:$PORT_FRONTEND"
    return $true
}

#######################################
# Show failure diagnostics
#######################################
function Show-FailureDiagnostics {
    Write-Host "========== Fehlerdiagnose ==========" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Container-Status:"
    docker compose ps 2>&1
    Write-Host ""
    Write-Host "Letzte Log-Eintraege:"
    docker compose logs --tail=20 2>&1
    Write-Host ""
    Write-Host "========== Moegliche Ursachen ==========" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "1. Nicht genuegend Arbeitsspeicher"
    Write-Host "   -> Schliessen Sie andere Programme"
    Write-Host "   -> Erhoehen Sie Docker-Speicher in Docker Desktop Einstellungen"
    Write-Host ""
    Write-Host "2. Netzwerkprobleme"
    Write-Host "   -> Ueberpruefen Sie Ihre Internetverbindung"
    Write-Host "   -> Versuchen Sie: docker compose pull"
    Write-Host ""
    Write-Host "3. Docker-Ressourcen"
    Write-Host "   -> Docker Desktop -> Einstellungen -> Resources"
    Write-Host "   -> Empfohlen: Mindestens 8GB RAM, 4 CPUs"
    Write-Host ""
    Write-Host "Naechste Schritte:"
    Write-Host "  1. .\setup.ps1 logs      # Detaillierte Logs anzeigen"
    Write-Host "  2. .\setup.ps1 cleanup   # Alles loeschen und neu starten"
    Write-Host ""
}

#######################################
# Show success message
#######################################
function Show-SuccessMessage {
    Write-Host ""
    Write-Host "==============================================" -ForegroundColor Cyan
    Write-Host "  Installation erfolgreich!" -ForegroundColor Cyan
    Write-Host "==============================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Oeffnen Sie Ihren Browser:"
    Write-Host "  http://localhost:$PORT_FRONTEND" -ForegroundColor Green
    Write-Host ""
    Write-Host "Nuetzliche Befehle:"
    Write-Host "  .\setup.ps1 stop      Anwendung stoppen"
    Write-Host "  .\setup.ps1 remove    Container entfernen"
    Write-Host "  .\setup.ps1 status    Status anzeigen"
    Write-Host "  .\setup.ps1 logs      Logs anzeigen"
    Write-Host ""
}

#######################################
# Build/recreate the application from this repository
#######################################
function Invoke-Build {
    Write-Host ""
    Write-Host "==============================================" -ForegroundColor Cyan
    Write-Host "  GremienPilot - Build" -ForegroundColor Cyan
    Write-Host "==============================================" -ForegroundColor Cyan
    Write-Host ""

    Set-LocalApplicationImages
    if (-not $env:PROTOKOLL_PRECACHE_MODELS) {
        $script:PROTOKOLL_PRECACHE_MODELS = "0"
    }

    # Pre-flight checks
    if (-not (Test-Docker)) {
        Read-Host "Druecken Sie Enter zum Beenden"
        exit 1
    }

    if (-not (Remove-ExistingContainersForRebuild)) {
        exit 1
    }

    if (-not (Test-DiskSpace)) {
        Read-Host "Druecken Sie Enter zum Beenden"
        exit 1
    }

    Test-RAM | Out-Null

    if (-not (Test-Ports)) {
        Read-Host "Druecken Sie Enter zum Beenden"
        exit 1
    }

    Test-GPU

    if (-not (Confirm-ModelCacheHandling)) {
        Read-Host "Druecken Sie Enter zum Beenden"
        exit 1
    }

    # Create uploads directory
    New-Item -ItemType Directory -Force -Path "uploads" | Out-Null

    # Start the application
    Write-Host ""
    Write-Info "Baue und starte die Anwendung..."

    if ($script:MissingItems.Count -gt 0) {
        Write-Host "Downloads koennen einige Minuten dauern."
    }
    Write-Host ""

    if ($script:BUILD_LOCAL_IMAGES) {
        if (-not (Invoke-BuildLocalImages)) {
            Read-Host "Druecken Sie Enter zum Beenden"
            exit 1
        }
        Write-Info "Pruefe Runtime-Image fuer Ollama..."
        docker compose pull ollama 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "Konnte Ollama-Image nicht aktualisieren. Docker versucht es beim Start erneut."
        }
    } else {
        Invoke-PullImages
    }
    Write-Host ""

    if ($script:USE_GPU) {
        Write-Info "Starte im GPU-Modus..."
        docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --force-recreate
    } else {
        Write-Info "Starte im CPU-Modus..."
        docker compose up -d --force-recreate
    }

    $success = Wait-ForServices

    Write-Host ""
    Read-Host "Druecken Sie Enter zum Schliessen"
}

#######################################
# Start existing containers without rebuilding
#######################################
function Invoke-Start {
    Write-Host ""
    Write-Host "==============================================" -ForegroundColor Cyan
    Write-Host "  GremienPilot - Start" -ForegroundColor Cyan
    Write-Host "==============================================" -ForegroundColor Cyan
    Write-Host ""

    if (-not (Test-Docker)) {
        Read-Host "Druecken Sie Enter zum Beenden"
        exit 1
    }

    $containers = docker compose ps -a -q 2>&1
    if (-not $containers -or $containers.Count -eq 0) {
        Write-Err "Keine vorhandenen Container gefunden."
        Write-Host ""
        Write-Host "Fuehren Sie zuerst aus:"
        Write-Host "  .\setup.ps1 build"
        Write-Host ""
        exit 1
    }

    Write-Info "Starte vorhandene Container ohne Neubau..."
    docker compose start
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Container konnten nicht gestartet werden."
        exit 1
    }

    Wait-ForServices | Out-Null
}

#######################################
# Stop the application
#######################################
function Invoke-Stop {
    Write-Host ""
    Write-Info "Stoppe die Anwendung..."
    docker compose stop
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Anwendung konnte nicht gestoppt werden."
        exit 1
    }
    Write-Success "Anwendung gestoppt"
    Write-Host ""
}

#######################################
# Stop and remove containers
#######################################
function Invoke-Remove {
    Write-Host ""
    Write-Info "Stoppe und entferne die Container..."
    docker compose down --remove-orphans
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Container konnten nicht entfernt werden."
        exit 1
    }
    Write-Success "Container entfernt"
    Write-Host ""
}

#######################################
# Show status
#######################################
function Show-Status {
    Write-Host ""
    Write-Host "==============================================" -ForegroundColor Cyan
    Write-Host "  GremienPilot - Status" -ForegroundColor Cyan
    Write-Host "==============================================" -ForegroundColor Cyan
    Write-Host ""

    # Check if containers exist
    $containers = docker compose ps -q 2>&1
    if (-not $containers -or $containers.Count -eq 0) {
        Write-Host "Die Anwendung ist nicht gestartet."
        Write-Host ""
        Write-Host "Erst bauen mit: .\setup.ps1 build"
        Write-Host "Danach starten mit: .\setup.ps1 start"
        return
    }

    Write-Host "Container-Status:"
    docker compose ps
    Write-Host ""

    # Check health
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:$PORT_BACKEND/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            Write-Host "Backend: Erreichbar" -ForegroundColor Green
        } else {
            Write-Host "Backend: Nicht erreichbar" -ForegroundColor Red
        }
    } catch {
        Write-Host "Backend: Nicht erreichbar" -ForegroundColor Red
    }

    try {
        $response = Invoke-WebRequest -Uri "http://localhost:$PORT_FRONTEND" -UseBasicParsing -TimeoutSec 2 -ErrorAction SilentlyContinue
        Write-Host "Frontend: Erreichbar" -ForegroundColor Green
    } catch {
        Write-Host "Frontend: Nicht erreichbar" -ForegroundColor Red
    }

    try {
        $response = Invoke-WebRequest -Uri "http://localhost:$PORT_OLLAMA/api/tags" -UseBasicParsing -TimeoutSec 2 -ErrorAction SilentlyContinue
        Write-Host "Ollama: Erreichbar" -ForegroundColor Green
    } catch {
        Write-Host "Ollama: Nicht erreichbar" -ForegroundColor Red
    }

    Write-Host ""
    Write-Host "URL: http://localhost:$PORT_FRONTEND"
    Write-Host ""
}

#######################################
# Restart the application
#######################################
function Invoke-Restart {
    Write-Host ""
    Write-Info "Starte die Anwendung neu..."
    docker compose restart
    Write-Success "Anwendung neu gestartet"
    Write-Host ""

    # Brief wait and check
    Start-Sleep -Seconds 3
    Show-Status
}

#######################################
# Show logs
#######################################
function Show-Logs {
    Write-Host ""
    Write-Info "Zeige Live-Logs (Strg+C zum Beenden)..."
    Write-Host ""
    docker compose logs -f
}

#######################################
# Cleanup everything
#######################################
function Invoke-Cleanup {
    Write-Host ""
    Write-Warn "ACHTUNG: Dies loescht alle Anwendungsdaten!"
    Write-Host ""
    Write-Host "Folgendes wird entfernt:"
    Write-Host "  - Alle Docker-Container"
    Write-Host "  - Alle Volumes (inkl. heruntergeladener Modelle)"
    Write-Host "  - Hochgeladene Dateien bleiben erhalten (uploads/)"
    Write-Host ""
    $response = Read-Host "Sind Sie sicher? (j/N)"

    if ($response -notmatch "^[Jj]$") {
        Write-Host "Abgebrochen."
        return
    }

    Write-Info "Raeume auf..."
    docker compose down -v --remove-orphans 2>&1 | Out-Null
    Write-Success "Aufraeumen abgeschlossen"
    Write-Host ""

    $response = Read-Host "Moechten Sie die Anwendung jetzt neu installieren? (J/n)"

    if ($response -notmatch "^[Nn]$") {
        Invoke-Build
    }
}

#######################################
# Main entry point
#######################################
switch ($Command.ToLower()) {
    "build" {
        Invoke-Build
    }
    "start" {
        Invoke-Start
    }
    "" {
        Invoke-Start
    }
    "stop" {
        Invoke-Stop
    }
    "remove" {
        Invoke-Remove
    }
    "status" {
        Show-Status
    }
    "restart" {
        Invoke-Restart
    }
    "logs" {
        Show-Logs
    }
    "cleanup" {
        Invoke-Cleanup
    }
    "help" {
        Show-Help
    }
    "--help" {
        Show-Help
    }
    "-h" {
        Show-Help
    }
    default {
        Write-Err "Unbekannter Befehl: $Command"
        Show-Help
        exit 1
    }
}
