# Gemma 4 und Protokoll-LoRA

Das Standard-Setup verwendet `google/gemma-4-31B-it` als Q4_K_M-GGUF von Unsloth
und `aihpi/gemma-4-31b-protokoll` (Revision `153460cf9a7c566df4a809a2f454c7b854c2405c`).
Der LoRA wird in F16-GGUF konvertiert und separat geladen. Q4_K_M ist eine
Speicherentscheidung für 32 GB RAM / 8 GB VRAM; die Ausgabe kann von der
BNB-4bit-Trainingsbasis abweichen.

## Vorbereitung und Start

`setup.ps1 build` bzw. `./setup.sh build` stellt Python 3.12, Git und die gepinnten
Konvertierungsabhängigkeiten automatisch in einem eigenen Docker-Image bereit.
Python muss auf dem Host weder installiert noch umgestellt werden; vorhandenes
Python 3.14 bleibt unverändert. Unter Windows und macOS wird wie für die übrige
Anwendung Docker Desktop mit Linux-Containern verwendet.

Das Image `gremienpilot-gemma-converter:python3.12-b11429` wird aus
`scripts/Dockerfile.gemma-converter` gebaut. Der Build installiert nur Werkzeuge,
keine Modellgewichte, und erfolgt vor dem Stoppen bestehender App-Container.
Beim ersten Build ist Internetzugang für das Python-Image, Debian-Pakete, GitHub
und Python-Pakete nötig; der Docker-Cache wird bei weiteren Builds wiederverwendet.
Die Speicherplanung berücksichtigt zusätzlich ungefähr 3 GB für den Konverter.
Ein vorhandenes `.certs/custom-ca.crt` wird für Git-, Paket- und Modelldownloads
in den Zertifikatsspeicher des Konverter-Images aufgenommen.

Ein temporärer Container lädt anschließend feste Modellrevisionen mit Prüfsummen,
verwendet llama.cpp `b11429` und schreibt die Gewichte und `checksums.sha256` nach
`data/gemma4/`. Nur dieses Verzeichnis wird eingebunden; Python und Konverter
bleiben im Image. Unter Linux/macOS entstehen Dateien mit der Benutzer-ID des
Setup-Aufrufs. Teil-Downloads sind fortsetzbar; nach einem Fehler genügt erneut
`setup build`. Frühere lokale Konvertierungsumgebungen bleiben unangetastet.
Die Dateien bleiben lokal und werden nicht eingecheckt. Beim Start prüft llama.cpp
diese Prüfsummen. Eine vorhandene, nicht leere `checksums.sha256` überspringt die
Modellvorbereitung einschließlich des Konverter-Builds im Setup.
Für das lokale Setup müssen die Anwendungimages lokal gebaut werden.

Für die manuelle Vorbereitung ohne Docker bleibt
`python scripts/prepare_gemma4.py --bootstrap` verfügbar. Dafür müssen Python
3.11–3.13 und Git bereits installiert sein; die virtuelle Umgebung entsteht unter
`data/gemma4/converter-venv/`. Das gepinnte
[NumPy 2.2.6](https://numpy.org/doc/stable/release/2.2.6-notes.html) unterstützt
Python 3.14 nicht. Das normale Setup übernimmt diese Versionswahl automatisch.

Docker Compose startet den Alias `gemma-4-31b` unter `http://llama:8080/v1`.
Der Hostport 8080 ist nur an Loopback gebunden. CPU-Betrieb verwendet das
`server-b11429`-Image, der GPU-Override `server-cuda-b11429`.

## Aufgaben und Prompts

- **Landtags-Stil mit LoRA:** Adapter 0 mit Stärke 1, unveränderter HPI-Trainingsprompt,
  `Name: Wortbeitrag` ohne Zeitstempel/Quellen-IDs, benachbarte Beiträge derselben
  Person zusammengeführt. System- und Nutzervorgabe stehen gemeinsam im
  Nutzerturn, entsprechend dem Unsloth-Produktionsvertrag. Temperatur 0,3,
  top_p 0,9, kein Thinking und keine Wiederholungsstrafe; `<turn|>` beendet die Ausgabe.
- **Eigener Stil ohne LoRA:** dasselbe Gemma-Modell mit Adapterstärke 0.
  In den KI-Einstellungen ist ein eigener Prompt für Gliederung, Umfang, Sprachton
  und Beschlussdarstellung bearbeitbar. Fachliche Treue bleibt verbindlich;
  erfundene Aussagen und Beschlüsse sind auch bei eigenen Stilvorgaben untersagt.
  Der eigene Stil verwendet denselben Protokolltext-Ablauf, keinen strukturierten
  JSON-Zusammenfassungspfad. Beide Stile nutzen dieselbe Markdown-Leseansicht
  und dieselben TXT-, DOCX- und PDF-Exporte.
- **Inhaltsprüfung, PDF und TOP-Zuordnung:** derselbe Modellbestand mit Adapterstärke 0.
  JSON-Schemata und bestehende Quellenverträge bleiben erhalten. PDF-Seiten
  verwenden den F16-Bildprojektor; maximal 1.120 Bildtokens je Seite.
- **Fast:** ausschließlich Protokolltext im gewählten Stil, als ungeprüfter Entwurf.
  **Slow:** zwei unabhängige Prüfungen je Teiltext gegen dessen vollständiges
  Originaltranskript; konkrete Hinweise separat. Beide Modi verzichten auf eine
  Quellen- und Kategorienzuordnung der Absätze. Text und Überschriften bleiben
  unverändert. Lange TOPs werden ohne stilles Abschneiden geteilt; bei nicht
  durchführbarer Prüfung bleibt der Entwurf ausdrücklich unvollständig geprüft.

Ohne Stilauswahl bleibt `LLM_SUMMARY_STYLE=gemma4-lora` der bisherige Standard;
das Frontend zeigt dafür den festen Prompt nur lesbar. Auswahl und eigener Prompt
werden im Browser und in der Sitzung gespeichert. Ein Wechsel erhält den eigenen
Prompt, vorhandene Texte und Prüfstände und wirkt erst auf neue Generierungen.
Pipeline und einzelne TOP-Aufträge speichern eigene Kopien der Einstellungen.
Die API-Felder `summary_style` (`gemma4-lora`/`gemma4-custom`) und
`custom_summary_prompt` gelten nur für Gemma-Protokollprofile. Der bestehende
`system_prompt`/`summary_system_prompt` bleibt für andere Profile erhalten.
Adapterstärke, Stil und Prompt sind Teil der Cache- und Checkpointkennung;
die Umschaltung erfolgt pro Anfrage, ohne globale llama.cpp-Änderung.
Eigene Stilvorgaben gelangen nicht in die fachlichen Inhaltsprüfungen.

`structured` aktiviert den
bisherigen JSON-Zusammenfassungspfad. Explizite andere Modellnamen benötigen
einen passenden Server und verwenden nicht automatisch den Adapter.

Zur Aktivierung nach dem Update Backend und Frontend dieses Branches im gewohnten
Deployment aktualisieren, sobald laufende Aufträge beendet sind. Es sind keine neuen
Modellgewichte oder Änderungen am llama.cpp-Dienst nötig. Die Sitzungsdatenbank
erhält beim Backendstart ausschließlich zusätzliche Einstellungsspalten. Anschließend
die Seite neu laden und den Zusammenfassungsstil in den KI-Einstellungen wählen.
Jobs aus älteren Codeversionen unterliegen weiterhin der vorhandenen Versionsprüfung;
gespeicherte Texte und Prüfhinweise bleiben lesbar. Ein echter Sitzungstest folgt manuell.

## Speicher und Transport

Das lokale Profil nutzt 16.384 Kontexttokens, 0 GPU-Layer im CPU-Modus bzw.
12 im GPU-Override und automatische llama.cpp-Threadwahl bei leerem `LLM_CPU_THREADS`,
Q8_0 für beide KV-Caches, einen Slot und einen CPU-Bildprojektor. Batch und Microbatch
umfassen je 1.120 Tokens, damit Gemma die Bildauflösung nicht begrenzt. Mehr GPU-Layer
oder Kontext erst nach Speichermessung einstellen. `LLM_CONTEXT_TOKENS`,
`LLM_GPU_LAYERS` und `LLM_CPU_THREADS` wirken bei llama.cpp beim Containerstart;
Änderungen erfordern dessen Neuerstellung. Antworten und Eingabe müssen gemeinsam
in den Kontext passen. Der Backend-Transport überprüft das Serverbudget; er kürzt
keine Quellen und akzeptiert keine abgeschnittene Ausgabe.

`GPU_MODEL_SWITCHING=true` koordiniert Whisper und den dedizierten lokalen
llama.cpp-Dienst. Vor Whisper wartet es auf den bestätigten Schlafzustand
(`--sleep-idle-seconds 30`), danach gibt Whisper RAM/VRAM frei. Ein Backend-Prozess
ist erforderlich. Andere GPU-Anwendungen werden dadurch nicht gesteuert.

`LLM_LOAD_TIMEOUT_SECONDS` begrenzt Modellladen/erste Ausgabe,
`LLM_READ_TIMEOUT_SECONDS` spätere Inaktivität. `LLM_TOTAL_TIMEOUT_SECONDS=0`
erlaubt lange Streams; Abbruch bleibt möglich. Modell-, Adapter- und Promptrevision
gehen in die Cache-/Jobidentität ein. Unvollständige Prüfungen erteilen kein Zertifikat.
Explizite Ollama- und OpenAI-kompatible Konfigurationen bleiben im Transport
unterstützt; sie verwenden keine llama.cpp-spezifischen Anfragefelder.

### Konservative Startwerte für CPU-Server

Q4_K_M bleibt vorerst fest voreingestellt. Es gibt noch keine automatische Wahl
anderer Quantisierungen oder vermessene Hardwareprofile. Die vorhandene RAM-/GPU-Erkennung
im Setup bleibt bestehen; GPU-Betrieb wird weiterhin bestätigt. Für Gemma sollten
mindestens ungefähr 32 GB RAM mit ausreichender freier Reserve und passendem
Docker-Speicherlimit eingeplant werden; die bisherige 8-GB-Empfehlung gilt nicht mehr.

Ohne explizite `.env`- oder Shell-Overrides gelten bei `setup build`:

| Einstellung | CPU | GPU-Override |
| --- | --- | --- |
| `LLM_GPU_LAYERS` | 0 | 12 |
| `LLM_CPU_THREADS` | automatisch | automatisch |
| `LLM_TIMEOUT_SECONDS` (Inaktivität nach Ausgabe) | 3600 s | 600 s |
| `LLM_LOAD_TIMEOUT_SECONDS` (Laden/Prompt bis erste Ausgabe) | 14400 s | 3600 s |
| `LLM_CONNECT_TIMEOUT_SECONDS` | 60 s | 60 s |
| `LLM_TOTAL_TIMEOUT_SECONDS` | 0 (keine Gesamtlaufzeitgrenze) | 0 |
| `LLAMA_HTTP_TIMEOUT_SECONDS` | 14400 s | 14400 s |

`LLM_READ_TIMEOUT_SECONDS` überschreibt das Inaktivitätslimit, falls gesetzt.
Die `.env.example` lässt die profilabhängigen Zeitlimits leer, damit Compose den
passenden Wert einsetzt. Ohne Compose nutzt llama.cpp im Backend die konservativen
CPU-Standardzeitlimits. Andere Provider behalten ihre bisherigen Backend-Fallbacks.
Der Web-Proxy wartet bis zu 24 Stunden auf eine Antwort; langlebige Aufträge werden
weiterhin über die persistente Warteschlange mit Statusabfragen verarbeitet.

Beide Setup-Skripte warten standardmäßig bis zu vier Stunden auf Bereitschaft.
Die Shell-Variable `SETUP_WAIT_SECONDS` überschreibt dies (`0` wartet unbegrenzt).
Ein Ende der Warteanzeige beendet keine Container oder Downloads. Healthchecks
gewähren vier Stunden Startzeit, melden aber sofort Bereitschaft, wenn sie vorliegt.
Diese Werte sind konservative Annahmen, keine auf dem CPU-Server gemessenen Garantien.

**Upgrade von Qwen/Ollama:** Vor `setup build` die vorhandene `.env` prüfen.
Das Setup überschreibt sie bewusst nicht. Alte Modellnamen, Provider, Basis-URLs,
Tokenizer, Revisionen, Kontextgrößen und Zeitlimits können die neuen Vorgaben sonst
weiterhin übersteuern. Die dort bereits erprobten CPU-Werte zuerst sichern und dann
gezielt auf Gemma/llama.cpp übertragen; Secrets, Ports und Datenpfade erhalten.
Auch zusätzliche Reverse-Proxys können eigene kürzere Zeitlimits besitzen.

Der mitgelieferte Tokenizer wird nur für den passenden Modellalias gewählt.
Andere Modelle brauchen passende lokale Tokenizer oder verwenden die konservative
UTF-8-Bytezählung. Ohne feste Modellrevision wird kein persistenter Cache genutzt.
Bestehende Sitzungen werden nicht gelöscht; Jobs mit alter Modell-/Codeversion
müssen neu eingereicht werden.

## Quellen

- [Adapter und Promptvertrag](https://huggingface.co/aihpi/gemma-4-31b-protokoll)
- [Training und Datenaufbereitung](https://github.com/aihpi/pilotproject-automatic-protocols)
- [HPI-Anwendung und Prompt](https://github.com/aihpi/pilotproject-protokollierungsassistenz)
- [Quantisierte Basis](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF)
- [llama.cpp Serververtrag](https://github.com/ggml-org/llama.cpp/blob/b11429/tools/server/README.md)
