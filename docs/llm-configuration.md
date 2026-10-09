# Gemma 4 und Protokoll-LoRA

Dieser Branch verwendet `google/gemma-4-31B-it` als Q4_K_M-GGUF von Unsloth
und `aihpi/gemma-4-31b-protokoll` (Revision `153460cf9a7c566df4a809a2f454c7b854c2405c`).
Der LoRA wird in F16-GGUF konvertiert und separat geladen. Q4_K_M ist eine
Speicherentscheidung für 32 GB RAM / 8 GB VRAM; die Ausgabe kann von der
BNB-4bit-Trainingsbasis abweichen.

## Vorbereitung und Start

`setup.ps1 build` bzw. `./setup.sh build` bereitet fehlende Gewichte mit Python
3.11+ und Git vor. Separat: `python scripts/prepare_gemma4.py --bootstrap`.

Das Skript erzeugt eine eigene Konvertierungsumgebung unter `data/gemma4/`,
lädt feste Revisionen mit Prüfsummen, verwendet llama.cpp `b11429` und schreibt
`checksums.sha256`. Teil-Downloads sind fortsetzbar. Die Dateien bleiben lokal
und werden nicht eingecheckt. Beim Start prüft llama.cpp diese Prüfsummen.
Ein vorhandenes Modellmanifest überspringt die Vorbereitung im Setup.
Für diesen Branch müssen die Anwendungimages lokal gebaut werden.

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

Das lokale Profil nutzt 16.384 Kontexttokens, 12 GPU-Layer, 12 CPU-Threads,
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
