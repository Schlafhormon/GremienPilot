# GremienPilot

<p align="center">
  <img src="app/frontend/public/gremienpilot-icon.svg" alt="GremienPilot Icon" width="144">
</p>

GremienPilot ist eine Webanwendung zur automatischen Transkription,
Sprechererkennung, TOP-Zuordnung, Zusammenfassung und Protokollerstellung aus
Audioaufnahmen kommunaler Sitzungen.

Dieses Repository ist ein weiterentwickelter Fork der ursprünglichen
Protokollierungsassistenz. Die Weiterentwicklung wird durch die **Stadt Doberlug-Kirchhain** vorangetrieben.

<p align="center">
  <img src="app/frontend/public/logos/Keule-Logo.png" alt="Keule-Services Logo" height="64">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="app/frontend/public/logos/Logo-jpeg_4c.jpg" alt="Stadt Doberlug-Kirchhain Logo" height="72">
</p>

## Was bietet GremienPilot?

- End-to-End-Pipeline vom Audio-Upload bis zur prüfbaren Protokollvorlage
- automatische Sprecherdiarisierung mit lokalen Sprecherprofilen
- automatische Sprecher-Wiedererkennung mit Opt-in, Profilvorschlägen und manueller Bestätigung
- Verwaltung gespeicherter Sprecherprofile inklusive Umbenennen, Archivieren und Löschen gespeicherter Embeddings
- automatische TOP-Erkennung und Segmentierung aus Transkript und optionaler PDF-Einladung
- PDF-Upload zur Extraktion von Tagesordnungspunkten
- Review-Oberfläche für unsichere TOP-Grenzen, Sprecherzuordnungen und Zusammenfassungen
- editierbares Transkript mit Zusammenführen/Splitten von Zeilen und Segmenten
- sitzungsübergreifende Persistenz in SQLite inklusive Sitzungswiederherstellung
- Export als DOCX, PDF oder TXT mit Metadaten, Sprecherliste, Transkript-Auszug und Generierungshinweis
- lokales Docker-Setup mit CPU- oder optionalem NVIDIA-GPU-Betrieb

## Einblicke in die Anwendung

Vom Upload der Sitzungsunterlagen bis zum geprüften und exportierbaren
Protokoll führt die Anwendung durch einen übersichtlichen, weitgehend
automatisierten Ablauf.

<table>
  <tr>
    <td align="center" width="50%">
      <a href="docs/screenshots/gremienpilot%20start.png">
        <img src="docs/screenshots/gremienpilot%20start.png" alt="Start einer neuen Sitzung mit Audio- und PDF-Upload" width="100%">
      </a>
      <br>
      <strong>1. Sitzung starten</strong><br>
      Audio und optional eine Einladung hochladen, Automatisierung auswählen
      und die Verarbeitung starten.
    </td>
    <td align="center" width="50%">
      <a href="docs/screenshots/gremienpilot%20ende.png">
        <img src="docs/screenshots/gremienpilot%20ende.png" alt="Geprüftes Protokoll mit Tagesordnung, Transkript und Exportoptionen" width="100%">
      </a>
      <br>
      <strong>2. Ergebnis prüfen und exportieren</strong><br>
      Erkannte TOPs, Sprecher und Zusammenfassungen kontrollieren und das
      fertige Protokoll exportieren.
    </td>
  </tr>
</table>

> Die Screenshots zeigen nur einen kleinen Ausschnitt der Möglichkeiten. Unter
> anderem bietet die Anwendung weitere Funktionen zur Sprecherverwaltung,
> Sitzungswiederherstellung, gezielten Nachbearbeitung und zum Export.

## Typischer Ablauf

1. Audioaufnahme hochladen.
2. Optional PDF-Einladung hochladen oder TOPs manuell eingeben.
3. Optional "Sprecher dauerhaft merken" aktivieren, wenn Sprecherprofile für künftige Sitzungen genutzt werden sollen.
4. "Automatisch verarbeiten" starten.
5. Erkannte TOP-Segmente, Sprecher und unsichere Stellen prüfen und bei Bedarf korrigieren.
6. Zusammenfassungen je TOP prüfen. Änderungen an Sprechernamen und TOP-Titeln
   werden ohne LLM übernommen. Inhaltlich betroffene TOPs können bestätigt,
   manuell bearbeitet oder selektiv neu generiert werden.
7. Protokoll mit Sitzungsmetadaten als DOCX, PDF oder TXT exportieren.

## Gemeinsamer Sitzungsverlauf

Alle Sitzungen werden serverseitig in SQLite gespeichert und sind innerhalb der
Anwendung nicht an einen bestimmten Browser gebunden. Der gemeinsame Verlauf ist
über den Button **Verlauf** im Kopfbereich oder direkt unter
`http://localhost:3000/sessions` erreichbar. Eine konkrete Sitzung kann außerdem
über `/sessions/<session_id>` direkt geöffnet und als Link weitergegeben werden.

Der Verlauf bietet Suche, Statusfilter und eine kompakte Übersicht über TOPs,
Transkript, Zusammenfassungen und verfügbares Audio. Geöffnete Sitzungen können
von allen Besuchern der Anwendung bearbeitet und exportiert werden. Wenn zwei
Browser dieselbe Sitzung gleichzeitig bearbeiten, verhindert eine
Revisionsprüfung, dass ein neuerer Stand unbemerkt überschrieben wird.

Die Anwendung selbst enthält keine Benutzer- oder Rechteverwaltung. Der Zugriff
auf Verlauf, Transkripte, Audio und Bearbeitungsfunktionen muss daher auf
Infrastrukturebene auf den vorgesehenen internen Personenkreis beschränkt werden.

Wenn keine TOPs angegeben werden, kann die Anwendung ein gesamtes Gespräch als
einen Gesamt-TOP zusammenfassen.

## Systemanforderungen

| Anforderung | Minimum | Empfohlen |
| --- | --- | --- |
| Betriebssystem | Windows 10/11, macOS 11+, Linux | Windows/Linux für GPU |
| Docker | Docker Desktop oder Docker Engine mit Compose | aktuelle Docker-Version |
| RAM | 8 GB | 16 GB oder mehr |
| Speicherplatz | 25 GB | 40 GB oder mehr |
| Internet | für Installation, Images und Modell-Downloads | stabile Verbindung |
| GPU | optional | NVIDIA-GPU mit Container Toolkit |

Für die lokale Transkription nutzt das Backend WhisperX und PyAnnote. Wenn die
Modelle nicht im Docker-Image vorinstalliert sind, wird für die PyAnnote-
Diarisierung ein HuggingFace-Token über `HF_TOKEN` benötigt.

## Installation

### 1. Repository beziehen

```bash
git clone https://github.com/Schlafhormon/ki-protokollierung.git
cd ki-protokollierung
```

Alternativ kann das Repository als ZIP von GitHub heruntergeladen und entpackt
werden.

### 2. Docker installieren und starten

Installieren Sie Docker Desktop bzw. Docker Engine mit Docker Compose:

- Windows: https://docs.docker.com/desktop/install/windows-install/
- macOS: https://docs.docker.com/desktop/install/mac-install/
- Linux: https://docs.docker.com/desktop/install/linux-install/

Starten Sie Docker, bevor Sie das Setup ausführen.

### 3. Optional `.env` anlegen

Für viele lokale Tests reichen die Standardwerte. Wenn Sie lokale Images ohne
vorinstallierte Modelle verwenden, setzen Sie für die Sprecherdiarisierung einen
HuggingFace-Token:

```bash
cp .env.example .env
# In .env setzen:
# HF_TOKEN=hf_...
```

Unter PowerShell:

```powershell
Copy-Item .env.example .env
# Danach .env bearbeiten und HF_TOKEN setzen, falls erforderlich.
```

### 4. Setup bauen und starten

Windows:

```powershell
.\setup.ps1 build
```

macOS/Linux:

```bash
chmod +x ./setup.sh
./setup.sh build
```

Der Build-Befehl prüft Docker, erkennt optional eine NVIDIA-GPU, baut lokale
Docker-Images aus diesem Repository und startet Frontend, Backend und Ollama per
Docker Compose. Wenn bereits Container vorhanden sind, entfernt das Skript diese
automatisch vor dem Neuerstellen.

Nach erfolgreichem Start ist die Anwendung erreichbar unter:

```text
http://localhost:3000
```

Für spätere Starts ohne Neubau verwenden Sie:

```powershell
.\setup.ps1 start
```

oder unter macOS/Linux:

```bash
./setup.sh start
```

## Setup-Befehle

| Befehl | Windows | macOS/Linux |
| --- | --- | --- |
| Lokale Images bauen und Container neu erstellen | `.\setup.ps1 build` | `./setup.sh build` |
| Vorhandene Container ohne Neubau starten | `.\setup.ps1 start` oder `.\setup.ps1` | `./setup.sh start` oder `./setup.sh` |
| Stoppen, Container behalten | `.\setup.ps1 stop` | `./setup.sh stop` |
| Stoppen und Container entfernen | `.\setup.ps1 remove` | `./setup.sh remove` |
| Status prüfen | `.\setup.ps1 status` | `./setup.sh status` |
| Neustart | `.\setup.ps1 restart` | `./setup.sh restart` |
| Logs anzeigen | `.\setup.ps1 logs` | `./setup.sh logs` |
| Daten löschen und neu starten | `.\setup.ps1 cleanup` | `./setup.sh cleanup` |

Nach Änderungen im Repository führen Sie `build` aus. Das baut Frontend und
Backend neu, entfernt vorhandene Container automatisch und erstellt die Container
neu. Die Modell-Volumes bleiben
standardmäßig erhalten; fehlende Modelle oder Modelle, die sich durch geänderte
Konfiguration ergeben, werden beim Start nachgeladen. Nur wenn Sie die Frage zum
Behalten der Modell-Volumes ausdrücklich mit `n` beantworten, werden die Modelle
gelöscht und anschließend erneut heruntergeladen.

Wenn Sie nur eine vorhandene Installation starten möchten, verwenden Sie
`start`. Dieser Befehl baut keine Images und meldet einen Fehler, wenn noch keine
Container existieren.

Wichtige Setup-Variablen:

| Variable | Bedeutung | Standard |
| --- | --- | --- |
| `PROTOKOLL_BUILD_LOCAL` | lokale Images aus diesem Repository bauen (`auto`, `true`, `false`) | `auto` |
| `PROTOKOLL_PRECACHE_MODELS` | ML-Modelle beim lokalen Build vorladen | `0` |
| `PROTOKOLL_BUILD_NO_CACHE` | Docker-Build ohne Cache erzwingen | `false` |
| `PROTOKOLL_IMAGE_TAG` | veröffentlichte App-Images auf einen Release-Tag pinnen | leer |
| `PROTOKOLL_PULL_POLICY` | Pull-Verhalten für Images (`missing`, `always`, `never`) | `missing` |
| `FRONTEND_IMAGE`, `BACKEND_IMAGE`, `BACKEND_GPU_IMAGE` | explizite Image-Referenzen verwenden | leer |

`build` verwendet immer die lokalen Image-Namen
`ki-protokollierung-frontend:local`, `ki-protokollierung-backend:local` bzw.
`ki-protokollierung-backend:gpu-local`. Die Image-Variablen sind vor allem für
manuelle Compose-Aufrufe oder veröffentlichte Images relevant.

## Datenschutz und Sprecherprofile

Die Anwendung verarbeitet Audio, Transkripte, TOPs, Zusammenfassungen und
Sprecherinformationen lokal in der Docker-Umgebung. Persistente Daten liegen in:

- `./data/sessions.sqlite3` für Sitzungen, Jobs, TOPs, Zuordnungen, Sprecherprofile und Export-Metadaten
- `./uploads` für gespeicherte Audiodateien zur Wiedergabe und Sitzungswiederherstellung
- Docker-Volume `ollama_data` für lokale Ollama-Modelle
- Docker-Volumes `backend_hf_cache` und `backend_torch_cache` für lokal geladene WhisperX-, HuggingFace- und Torch-Modelle

Das dauerhafte Merken von Sprechern ist standardmäßig ausgeschaltet. Ohne Opt-in
werden keine globalen Sprecherprofile vorgeschlagen oder automatisch dauerhaft
gespeichert. Lokale Sprecher können weiterhin nur für die aktuelle Sitzung
benannt werden.

Dauerhafte Sprecherprofile und globale Embeddings entstehen erst nach einer
ausdrücklichen Aktion, zum Beispiel:

- "Sprecher dauerhaft merken" aktivieren
- "Vorschlag übernehmen"
- "Neues Profil merken"
- "Bestehendem Profil zuordnen"

In der Sprecherprüfung können Profile umbenannt, archiviert und gespeicherte
Embeddings gelöscht werden. Archivierte Profile werden standardmäßig nicht mehr
für Vorschläge verwendet.

Für dauerhafte Profile werden keine Roh-Audiodaten gespeichert. Nach einer
expliziten Bestätigung werden nur qualitätsgefilterte Segment-Embeddings aus
nicht überlappenden, ausreichend langen Sprecherabschnitten übernommen. Pro
Profil und Embedding-Modell bleibt die Anzahl begrenzt; beim Matching werden
mehrere Referenzen inklusive Profilmittel verglichen. Falls kein Vorschlag
entsteht, liefert die Sprecherprüfung Diagnosegründe wie fehlende
Profil-Embeddings, zu wenig Sprecher-Audio, ein nicht verfügbares
Embedding-Modell oder Scores unterhalb des Schwellwerts.

`pyannote/embedding` ist ein gated HuggingFace-Modell. Der verwendete
`HF_TOKEN` oder `HUGGINGFACE_TOKEN` muss Leserechte haben und die Nutzungsbedingungen auf der
Modellseite müssen akzeptiert sein. Wenn dieses Modell nicht geladen werden
kann, versucht das Backend standardmäßig
`pyannote/wespeaker-voxceleb-resnet34-LM` als Fallback. Bereits manuell
zugeordnete Profile ohne Embeddings können in der Sprecherprüfung über
"Embeddings nachholen" nachträglich befüllt werden, sofern die Audiodatei noch
vorhanden ist.

Der Laufzeitstatus ist unter `/api/speaker-embeddings/diagnostics` sichtbar.
Mit `?session_id=<id>` zeigt der Endpoint zusätzlich lokale Sprecher pro
Sitzung, qualifizierte Sekunden, Segmentfilterung, lokale
`job_speaker_embeddings`, globale `speaker_embeddings` und Counts pro Profil.

## Konfiguration

Die wichtigsten Laufzeitvariablen können in `.env` gesetzt werden.

| Variable | Beschreibung | Standard |
| --- | --- | --- |
| `HF_TOKEN` / `HUGGINGFACE_TOKEN` | HuggingFace-Token für PyAnnote, wenn Modelle nicht vorinstalliert sind | leer |
| `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` | PyTorch-2.6+-Kompatibilität für WhisperX/pyannote-Checkpoints | `1` |
| `WHISPER_MODEL` | Whisper-Modell | `large-v2` |
| `WHISPER_DEVICE` | Gerät für WhisperX (`cpu`, `cuda`, `auto`) | Compose: `cpu` |
| `WHISPER_BATCH_SIZE` | Batch-Größe für Transkription | `16` |
| `WHISPER_CPU_THREADS` | CPU-Threads pro WhisperX/CTranslate2-Worker | `4` |
| `TORCH_NUM_THREADS` | optionale PyTorch-CPU-Threads, falls CPU-Server getunt wird | leer |
| `TORCH_NUM_INTEROP_THREADS` | optionale PyTorch-Inter-op-Threads | leer |
| `OMP_NUM_THREADS` / `MKL_NUM_THREADS` | optionale OpenMP/MKL-Threadlimits | leer |
| `WHISPER_LANGUAGE` | Sprache | `de` |
| `LLM_BASE_URL` | OpenAI-kompatibler LLM-Endpunkt | Compose: `http://ollama:11434/v1`, lokale Backend-Entwicklung: `http://localhost:11434/v1` |
| `LLM_MODEL` | Modell für Zusammenfassungen und TOP-Extraktion | `qwen3:8b` |
| `LLM_REASONING_EFFORT` | Reasoning für alle LLM-Aufgaben: leer = bisheriges Verhalten, `none` = aus, `low`/`medium`/`high`/`max` = an (modell-/serverabhängig) | leer |
| `LLM_TIMEOUT_SECONDS` | Timeout für Zusammenfassungen, PDF-Extraktion und LLM-Diagnose (nicht TOP-Erkennung im Transkript) | `120` |
| `LLM_CHUNK_CHARS` | Chunk-Größe für lange TOP-Texte | `12000` |
| `SPEAKER_EMBEDDING_ENABLED` | lokale Sprecher-Embeddings für prüfbare Matches erzeugen | `true` |
| `SPEAKER_EMBEDDING_MODEL` | primäres Embedding-Modell | `pyannote/embedding` |
| `SPEAKER_EMBEDDING_FALLBACK_MODELS` | kommagetrennte Fallback-Modelle, falls das primäre Modell nicht lädt | `pyannote/wespeaker-voxceleb-resnet34-LM` |
| `SPEAKER_MATCH_AUTO_THRESHOLD` | Schwelle für starke Sprecherprofil-Matches | `0.82` |
| `SPEAKER_MATCH_SUGGEST_THRESHOLD` | Mindestschwelle für Vorschläge | `0.72` |
| `SPEAKER_MATCH_TOP_K` | Anzahl der besten Profilreferenzen für Top-k-Matching | `3` |
| `SPEAKER_EMBEDDING_MIN_SECONDS` | Mindestmenge qualifiziertes Audio pro lokalem Sprecher | `8.0` |
| `SPEAKER_EMBEDDING_MIN_SEGMENT_SECONDS` | Mindestdauer eines Referenzsegments | `1.5` |
| `SPEAKER_EMBEDDING_MAX_SEGMENT_SECONDS` | maximale Segmentdauer vor dem Cropping | `12.0` |
| `SPEAKER_EMBEDDING_MAX_SEGMENTS` | maximale lokale Segmente pro Sprecher für die Extraktion | `8` |
| `SPEAKER_PROFILE_MAX_EMBEDDINGS_PER_MODEL` | maximale globale Referenz-Embeddings je Profil und Modell | `16` |
| `AGENDA_DETECTION_USE_LLM` | Serverstandard für LLM-TOP-Erkennung; explizite API-Entscheidung hat Vorrang | `false` |
| `AGENDA_DETECTION_TIMEOUT_SECONDS` | Positiver, endlicher Timeout in Sekunden pro Agenda-Anfrage/Chunk; keine SDK-Retries | `8` |
| `AGENDA_DETECTION_CHUNK_LINES` | Zeilen pro Chunk bei unbekannter Agenda | `160` |
| `AGENDA_DETECTION_CHUNK_OVERLAP_LINES` | Überlappende Zeilen zwischen Chunks | `12` |
| `AGENDA_DETECTION_CONTEXT_WINDOW_BEFORE` / `AGENDA_DETECTION_CONTEXT_WINDOW_AFTER` | Kontextzeilen um Grenzen bei langer bekannter Agenda | `4` / `8` |
| `PERSISTENCE_DB_PATH` | SQLite-Pfad im Backend-Container | `/app/data/sessions.sqlite3` |
| `MAX_UPLOAD_BYTES` | maximale Uploadgröße | `524288000` |
| `TRANSCRIPTION_CONCURRENCY` | parallele Transkriptionsjobs | `1` |
| `PIPELINE_CONCURRENCY` | parallele End-to-End-Pipelinejobs | `1` |
| `SUMMARY_CONCURRENCY` | parallele, manuell bestätigte TOP-Regenerierungsjobs | `1` |

Weitere Optionen stehen in `.env.example`. In Docker Compose sollte
`LLM_BASE_URL` normalerweise nicht gesetzt werden; der Backend-Container nutzt
dann automatisch den internen Ollama-Service. Ein `localhost`-Wert ist nur für
lokale Backend-Entwicklung außerhalb von Docker sinnvoll. Externe
OpenAI-kompatible Endpunkte müssen explizit mit vollständiger `/v1`-URL
konfiguriert werden.

### LLM-Nutzung für automatische TOP-Zuordnung

`LLM_REASONING_EFFORT` wird als `reasoning_effort` an die OpenAI-kompatible
Chat-API übergeben: bei PDF-Extraktion, TOP-Zuordnung und allen
Zusammenfassungsaufrufen einschließlich Teilzusammenfassungen und Fallbacks.
Ein leerer Wert lässt das API-Feld weg; die PDF-Extraktion behält dann ihre
bisherige `/no_think`-Promptanweisung. Bei einem expliziten Wert steuert die API
das Reasoning, ohne automatisch hinzugefügte `/no_think`-Anweisung.
Ungültige Werte werden abgewiesen. Der Modellserver muss den gewählten Wert
unterstützen; die Einstellung wird bei Fehlern nicht stillschweigend entfernt.
Die unterstützten Werte sind in der [Ollama-API-Dokumentation](https://docs.ollama.com/api/openai-compatibility)
beschrieben. Reasoning-Stufen können je nach Modell gleich behandelt werden.

Für einen CPU-Test mit Qwen3.5 bietet sich `LLM_MODEL=qwen3.5:9b` mit
`LLM_REASONING_EFFORT=none`, `AGENDA_DETECTION_USE_LLM=true` und großzügigen
Timeouts an, beispielsweise jeweils `1800` Sekunden für
`LLM_TIMEOUT_SECONDS` und `AGENDA_DETECTION_TIMEOUT_SECONDS`.
Das sind Timeouts je Aufruf, keine Gesamtlaufzeitgrenze. Aktiviertes Reasoning
kann zusätzliche Ausgabetokens benötigen; die vorhandenen `max_tokens`-Limits
werden durch diese Variable nicht erhöht. `LLM_MAX_RETRIES=0` deaktiviert
Wiederholungen bei Zusammenfassung und PDF-Extraktion. Die TOP-Zuordnung
verwendet unabhängig davon keine SDK-Wiederholungen.

`POST /api/agenda-detection` akzeptiert `use_llm`: `true` aktiviert das LLM,
`false` erzwingt Heuristik, fehlend oder `null` übernimmt
`AGENDA_DETECTION_USE_LLM` (Standard `false`). Der Serverwert ist ein Standard,
keine administrative Sperre. Modellname und Prompt aktivieren das LLM niemals.

Für `POST /api/pipeline/start` heißt die Option `agenda_use_llm`: entweder als
boolescher Wert bzw. `null` im JSON-Feld `options` oder als Formularfeld
`true`/`false`. Ein gesetztes Formularfeld hat Vorrang vor JSON-Optionen.
`skip_agenda_detection=true` überspringt weiterhin die gesamte TOP-Zuordnung.
Die React-App übernimmt ohne explizite Client-Option den Serverstandard;
der TypeScript-Client unterstützt `useLlm` bzw. `agendaUseLlm`.
PDF-Extraktion und Zusammenfassungen haben ihre eigenen Modellaufrufe und werden
von diesem Agenda-Schalter nicht deaktiviert.

**Migration:** Bestehende Request-Felder, Python-Argumente und `strategy`-Werte
bleiben erhalten. Aufrufer, die bisher allein mit `model` oder `system_prompt`
einen Agenda-Modellaufruf ausgelöst haben, müssen jetzt explizit aktivieren
oder den Serverstandard auf `true` setzen. Vorhandene Python-Aufrufer können
zusätzlich `use_llm=True`/`False` als Schlüsselwort übergeben.

Die Antwort enthält zusätzlich `llm` mit wirksamer Aktivierung, Quelle
(`request`/`server_default`), Timeout, Status (`disabled`, `skipped`, `success`,
`fallback`, `partial_fallback`), Aufruf-/Fehleranzahl und festen Ursachencodes
(`timeout`, `connection_error`, `request_error`, `invalid_response`, `empty_response`).
`warnings` erläutert Ersatzverarbeitung; die Pipeline speichert diese Angaben,
übernimmt die Warnungen und zeigt sie in der Prüfung an. Bestehende gespeicherte
Ergebnisse ohne diese Angaben bleiben lesbar (`llm=null`). Reparierte Grenzen
bleiben außerdem über `strategy`, Segmentbegründung und Unsicherheit erkennbar.
Logs enthalten bei LLM-Ausfällen nur feste Ursachencodes und Aufrufnummern,
keine Prompts, Transkripte, Provider-Antworten oder Zugangsdaten.

Der Agenda-Timeout beträgt tatsächlich standardmäßig 8 Sekunden pro
SDK-Anfrage/Chunk, ohne automatische SDK-Wiederholungen und ohne frühere
30-Sekunden-Untergrenze. Er ist kein Zeitlimit für die gesamte Pipeline;
mehrere Chunks und die übrigen Verarbeitungsschritte können länger dauern.
Langsame Modelle benötigen gegebenenfalls einen höheren Agenda-Timeout.
Ungültige Serverwerte für den LLM-Schalter (erlaubt: `true`/`false`) oder
ein nicht positiver bzw. nicht endlicher Timeout verhindern den Backend-Start.
Die Einstellungen werden beim Backend-Start geladen. Nach Änderungen Backend
neu starten bzw. Container neu erstellen; CPU-Compose und GPU-Override reichen
alle obigen Agenda-Variablen durch. Kubernetes verwendet die Backend-ConfigMap.

## GPU-Modus

Für NVIDIA-GPUs unter Windows oder Linux:

1. NVIDIA-Treiber installieren.
2. NVIDIA Container Toolkit installieren.
3. Docker neu starten.
4. Setup ausführen und GPU-Modus auswählen, wenn das Skript danach fragt.

Der GPU-Override nutzt `docker-compose.gpu.yml` und setzt das Backend auf
`WHISPER_DEVICE=cuda`. macOS unterstützt diesen NVIDIA-GPU-Modus nicht.

## Fehlerbehebung

### Docker läuft nicht

Docker Desktop bzw. Docker Engine starten und das Setup erneut ausführen.

### Nach Änderungen im Repository neu bauen

```bash
./setup.sh build
```

Windows:

```powershell
.\setup.ps1 build
```

Das Skript entfernt vorhandene Container automatisch. Modell-Volumes werden
standardmäßig behalten, damit Ollama-, HuggingFace- und Torch-Modelle nicht bei
jedem Backend-/Frontend-Build erneut heruntergeladen werden.

### Vorhandene Container nur starten

```bash
./setup.sh start
```

Windows:

```powershell
.\setup.ps1 start
```

Dieser Befehl baut keine Images. Wenn noch keine Container vorhanden sind,
führen Sie zuerst `build` aus.

### Backend meldet fehlenden HuggingFace-Token

Wenn lokale Images ohne vorinstallierte Modelle gebaut wurden, setzen Sie in
`.env`:

```text
HF_TOKEN=hf_...
```

### Zusammenfassung meldet LLM-/Ollama-Fehler

Docker Compose startet einen internen Ollama-Dienst und lädt standardmäßig
`LLM_MODEL=qwen3:8b`. Prüfen Sie die LLM-Diagnose mit:

```bash
curl http://localhost:8010/api/llm/diagnostics
```

Wenn das Modell fehlt, laden Sie es nach:

```bash
docker compose exec ollama ollama pull qwen3:8b
```

Für lokale Backend-Entwicklung ohne Docker muss Ollama lokal laufen und
`LLM_BASE_URL=http://localhost:11434/v1` gesetzt sein.

Danach neu starten:

```bash
./setup.sh restart
```

oder unter Windows:

```powershell
.\setup.ps1 restart
```

### Anwendung ist langsam

- CPU-Transkription ist deutlich langsamer als GPU-Transkription.
- Der erste Lauf lädt Modelle und kann länger dauern.
- Docker sollte ausreichend RAM erhalten, empfohlen sind mindestens 8 GB.

### Logs ansehen

```bash
docker compose logs -f
```

oder über das Setup:

```bash
./setup.sh logs
```

### Cleanup

Das entfernt Container und Docker-Volumes, insbesondere heruntergeladene
Ollama-, HuggingFace- und Torch-Modelle. Die lokalen Bind-Mounts `uploads/` und
`data/` bleiben bestehen; löschen Sie diese Ordner nur bewusst, wenn auch
hochgeladene Dateien und gespeicherte Sitzungen entfernt werden sollen.

```bash
./setup.sh cleanup
```

Windows:

```powershell
.\setup.ps1 cleanup
```

## Entwicklung

### Projektstruktur

```text
ki-protokollierung/
├── app/
│   ├── frontend/          React + TypeScript + Vite
│   └── backend/           FastAPI, WhisperX, PyAnnote, Export, Persistenz
├── scripts/               Hilfs- und Research-Skripte
├── k8s/                   Kubernetes-Manifeste
├── docker-compose.yml     lokale Compose-Umgebung
├── docker-compose.gpu.yml GPU-Override
├── setup.ps1              Windows-Setup
└── setup.sh               macOS/Linux-Setup
```

### Backend lokal

```bash
cd app/backend
uv sync
HF_TOKEN=hf_... uv run uvicorn main:app --port 8010
```

Das Backend läuft dann unter `http://localhost:8010`.

### Frontend lokal

```bash
cd app/frontend
npm install
npm run dev
```

Das Frontend läuft dann unter `http://localhost:5173`.

### Tests

Backend:

```bash
cd app/backend
uv run pytest
```

Frontend:

```bash
cd app/frontend
npm test
npm run typecheck
npm run lint
```

### Lokale Docker-Images bauen

Der empfohlene Weg für lokale Repo-Änderungen ist:

```bash
./setup.sh build
```

Windows:

```powershell
.\setup.ps1 build
```

Die folgenden Docker-Befehle sind nur für manuelle Spezialfälle gedacht.

CPU:

```bash
docker build --build-arg PRECACHE_MODELS=0 -t ki-protokollierung-backend:local ./app/backend
docker build -t ki-protokollierung-frontend:local ./app/frontend
```

GPU:

```bash
docker build -f app/backend/Dockerfile.gpu --build-arg PRECACHE_MODELS=0 -t ki-protokollierung-backend:gpu-local ./app/backend
```

Wenn Modelle im Image vorinstalliert werden sollen, `PRECACHE_MODELS=1` setzen
und `HF_TOKEN` als BuildKit-Secret bereitstellen.

Für NVIDIA-Blackwell-GPUs (z. B. RTX 50xx, B100/B200) `Dockerfile.gpu-blackwell`
verwenden, da `Dockerfile.gpu` PyTorch-Wheels ohne Blackwell-Kernel (CUDA 12.6)
installiert:

```bash
docker build -f app/backend/Dockerfile.gpu-blackwell --build-arg PRECACHE_MODELS=0 -t ki-protokollierung-backend:gpu-blackwell-local ./app/backend
```

## TOP-Nummern und Unterpunkte

TOPs behalten Originalnummern, Unterpunkte und Sitzungsabschnitte. Details zu
Kompatibilität und unterstützten Verweisen: [TOP-Nummerierung](docs/top-nummerierung.md).

## API-Auszug

| Endpoint | Methode | Zweck |
| --- | --- | --- |
| `/health` | GET | Healthcheck |
| `/api/pipeline/start` | POST | End-to-End-Verarbeitung starten |
| `/api/pipeline/{pipeline_id}` | GET | Pipeline-Status abrufen |
| `/api/pipeline/{pipeline_id}/cancel` | POST | Pipeline abbrechen |
| `/api/pipeline/{pipeline_id}/result` | GET | fertiges Review-Ergebnis laden |
| `/api/sessions` | POST | Sitzung speichern/anlegen |
| `/api/sessions/{session_id}` | GET/PUT | Sitzung laden/speichern |
| `/api/sessions/{session_id}/summary-jobs` | POST | selektive TOP-Regenerierung einreihen |
| `/api/summary-jobs/{summary_job_id}` | GET | Fortschritt einer Regenerierung abrufen |
| `/api/summary-jobs/{summary_job_id}/cancel` | POST | Regenerierung abbrechen |
| `/api/sessions/{session_id}/summaries/{top_id}/accept` | POST | bestehende Zusammenfassung ohne LLM übernehmen |
| `/api/transcribe` | POST | Legacy-Transkriptionsjob starten |
| `/api/audio/{job_id}` | GET | Audiodatei streamen |
| `/api/extract-tops` | POST | TOPs aus PDF extrahieren |
| `/api/agenda-detection` | POST | TOPs und Segmentgrenzen erkennen |
| `/api/summarize` | POST | Zusammenfassung erzeugen |
| `/api/llm/diagnostics` | GET | LLM-Endpunkt und Modell prüfen |
| `/api/export` | POST | Protokoll als TXT, DOCX oder PDF exportieren |
| `/api/speaker-profiles` | GET/POST | Sprecherprofile listen/anlegen |
| `/api/speaker-profiles/{profile_id}` | PUT/DELETE | Profil ändern oder archivieren |
| `/api/speaker-profiles/{profile_id}/embeddings` | DELETE | gespeicherte Embeddings löschen |
| `/api/sessions/{session_id}/speaker-observations` | GET | Sprecherprofil-Vorschläge abrufen |

Der Pipeline-Start trennt `summary_system_prompt`, `agenda_system_prompt` und
`pdf_system_prompt` (Formularfelder oder JSON-Feld `options`). Das bisherige
`system_prompt` bleibt als Fallback ausschließlich für Zusammenfassungen erhalten;
gespeicherte KI-Einstellungen bleiben kompatibel. Fachliche Ergänzungen lassen
die verbindlichen Ausgabeformate und Aufgabenregeln bestehen.

## Kubernetes

Unter `k8s/` liegen Manifeste für ein GPU-beschleunigtes Deployment. Die
Kubernetes-Variante nutzt standardmäßig eine externe OpenAI-kompatible LLM-API
statt des lokalen Ollama-Containers. Details stehen in `k8s/README.md`.

## Ursprung und Förderung

Die ursprüngliche Protokollierungsassistenz entstand im Umfeld des AI Service
Centre Berlin Brandenburg. Dieses Repository enthält die darauf aufbauende
Weiterentwicklung durch Keule-Services und die Stadt
Doberlug-Kirchhain.

<p>
  <a href="http://hpi.de/kisz">
    <img src="app/frontend/public/logos/logo_aisc_150dpi.png" alt="AI Service Centre Berlin Brandenburg Logo" height="48">
  </a>
  &nbsp;&nbsp;
  <img src="app/frontend/public/logos/logo_bmftr_de.png" alt="BMFTR Logo" height="48">
</p>

Das AI Service Centre Berlin Brandenburg wird durch das Bundesministerium für
Forschung, Technologie und Raumfahrt unter dem Förderkennzeichen 16IS22092
gefördert.
