# Modell- und Transportvertrag

Stand der Prüfung: 22. September 2026. Neue Installationen verwenden den
Qualitätskandidaten `gemma4:31b-it-q4_K_M`; explizite Modellnamen, einschließlich
Browser-/API-Überschreibungen, bleiben unverändert. Die Quantisierung muss zum
Speicher des Betreibers passen. Es gibt keinen automatischen Modellwechsel,
keine automatische Kontextverkleinerung und keine Schema-/Thinking-Fallbacks
im Transport. Fachliche Reparaturen und bestehende manuelle Sitzungsdaten
bleiben erhalten; eine Datenbankmigration ist nicht erforderlich.

## Provider und Thinking

`LLM_PROVIDER=ollama` verwendet `/api/chat`; `openai-compatible` verwendet
`/chat/completions` unter `LLM_BASE_URL`. Leer erhält die bisherige Erkennung
über Endpunkt und `LLM_OLLAMA_NATIVE`. Explizite Providerwahl hat Vorrang.
Ein externer Ollama-Server sollte ausdrücklich `ollama` setzen. Ein lokaler
vLLM-/anderer OpenAI-kompatibler Server setzt `openai-compatible`.

Gemma 4 31B Instruct unterstützt Text/Bilder und optionales Thinking; 256K ist
die Architekturgrenze, keine Zusage über freien Laufzeitspeicher. Die dokumentierte
Sampling-Empfehlung ist Temperatur 1.0, top_p 0.95, top_k 64. Diese Werte lassen
sich mit `LLM_TEMPERATURE`, `LLM_TOP_P`, `LLM_TOP_K` setzen; leere Werte erhalten
die bestehenden fachlichen Temperaturvorgaben bzw. Providerdefaults.
[Offizielle Modellbeschreibung](https://ollama.com/library/gemma4),
[Google-Modellübersicht](https://ai.google.dev/gemma/docs/core).

`LLM_THINKING=true|false` ist der native Ollama-Schalter. Alternativ bleibt
`LLM_REASONING_EFFORT=none|low|medium|high|max` kompatibel: Gemma erhält daraus
einen booleschen Schalter, andere Ollama-Modelle ihre benannte Stufe.
OpenAI-kompatible Server erhalten `reasoning_effort` ausschließlich, wenn
konfiguriert. Sie müssen die gewählte Stufe unterstützen. Beide Variablen
zusammen werden abgewiesen. Leere Einstellungen lassen den Provider entscheiden.
Die vorhandenen fachlichen `*_THINK`-Überschreibungen betreffen weiterhin nur
Ollama. Qwen-Steuerzeilen werden aus Systemprompts entfernt. Denktext wird
weder als Ergebnis interpretiert noch in normale Logs oder Audits geschrieben.
[Ollama Thinking](https://docs.ollama.com/capabilities/thinking).

`LLM_OUTPUT_TOKENS` überschreibt ausdrücklich die bisherigen aufgabenspezifischen
Ausgabebudgets; leer behält sie bei. `LLM_THINKING_TOKENS` ist eine zusätzliche
Reservierung, **kein separat durchsetzbares Denklimit**. Ollama bietet hier nur
`num_predict`; OpenAI-kompatible Reasoning-Server können den gemeinsamen Deckel
über `LLM_OUTPUT_PARAMETER=max_completion_tokens` erhalten. Standard bleibt
`max_tokens` für bestehende kompatible Server. Bei Ollamas schemaabhängigem
zweiphasigem Generieren wird konservativ zweimal das kombinierte Budget
reserviert, auch wenn ein Runner mit einer Phase auskommt. `length`, fehlender
Abschluss, leerer Antwortkanal und ausgeschöpftes natives Generierungslimit gelten
als unvollständig. Der Transport repariert keine fachlichen Inhalte.
[Chat-Vertrag](https://docs.ollama.com/api/chat),
[geprüfter Servercode v0.34.2](https://github.com/ollama/ollama/blob/v0.34.2/server/routes.go).

## Kontext, Bilder und Identität

`LLM_CONTEXT_TOKENS` umfasst System-/Nutzertexte, Chat-Template-Reserve,
Bildreservierungen, JSON-Schema und sämtliche Generierungsphasen. Native Anfragen
setzen `num_ctx`, `truncate=false`, `shift=false`. Vor der Generierung werden
Modellgrenze und Fähigkeiten aus `/api/show` geprüft; anschließend werden Kontext
und Digest des geladenen Runners aus `/api/ps` kontrolliert. Ein zu kleiner oder
nicht belegbarer Kontext führt zu einem Fehler. Dieser Vertrag verlangt Ollama
ab 0.34.2; ältere Versionen werden mit einem Konfigurationsfehler abgewiesen.
[API-Typen v0.34.2](https://github.com/ollama/ollama/blob/v0.34.2/api/types.go).

OpenAI-kompatible APIs standardisieren keine Abfrage der tatsächlichen
Server-Kontextgröße. Dort muss der Betreiber `LLM_CONTEXT_TOKENS` passend zum
Server setzen; gemeldete Tokenverwendung wird zusätzlich geprüft. Die Provenienz
kennzeichnet den effektiven Kontext deshalb als nicht verifiziert. Ein Provider,
der Eingaben intern still kürzt, erfüllt diesen Vertrag nicht.

Optional: `LLM_TOKENIZER_PATH` zeigt auf einen **lokalen**, modellpassenden
Hugging-Face-`tokenizer.json` (Bibliothek `tokenizers`, bereits Teil der ML-Abhängigkeiten); `LLM_TOKENIZER_MODEL` muss
mit dem tatsächlichen Modellnamen übereinstimmen. Bei Modellüberschreibungen
wird ein unpassender Tokenizer abgewiesen. Kein Download und kein fremder
Python-Code im Job. Die modellpassende Text-/Schema-Zählung wird um 32 Tokens je Nachricht und
512 Tokens für Chat-Template/Steuerzeichen ergänzt; Provider-Templates können abweichen. Ohne Tokenizer gilt die konservative
UTF-8-Bytezahl plus 32 Tokens je Nachricht und 512 globale Reservetokens.
Das ist ein bewusst großzügiges Ersatzverfahren, keine universelle mathematische
Garantie für beliebige Tokenizer; die native Laufzeitprüfung bleibt maßgeblich.

Nachrichten akzeptieren Text, OpenAI-Content-Parts (`text`, `image_url`) und
native `images`. Ollama benötigt Base64, externe APIs dürfen Bild-URLs erhalten;
der Backend-Transport lädt solche URLs nicht selbst. `LLM_IMAGE_TOKENS=0`
verweigert Bilder, bis eine passende obere Schranke pro Bild konfiguriert wurde.
Gemma dokumentiert bis 1120 visuelle Tokens je Bild; zusätzliche Marker/Crops
und abweichende Providerprozessoren müssen berücksichtigt werden. Der gewählte
Wert ist eine **Reservierung**, kein Befehl zur Bildskalierung.
[Gemma-Bildverarbeitung](https://ollama.com/library/gemma4).

Pro Fachoperation wird die Umgebung einmal aufgelöst. Diagnose und
Zusammenfassungs-/Agenda-Provenienz enthalten die wirksame Konfiguration,
Modellquelle (`environment`/`request`), Konfigurations-ID und bei Generierungen
Digest, geprüften Kontext und Transportversuche. API-Schlüssel sind ausgenommen.
Vorhandene Browsermodelle werden nicht automatisch überschrieben; für den
Serverstandard muss das Modellfeld leer sein. Die Diagnose akzeptiert denselben
`model`-Parameter wie die Verarbeitung.

Cacheversion 3 trennt bisherige Ergebnisse von diesem Vertrag. Der Schlüssel
enthält Konfiguration, Eingaben, Zweck und Modellidentität. Native Tags werden
über ihren Digest aufgelöst; externe Modelle benötigen `LLM_MODEL_REVISION`
als unveränderliche Betreiberzusage. Ohne Digest/Revision gibt es keinen
persistenten Cachetreffer. Alte Cachedateien und Sitzungen werden nicht gelöscht.
`LLM_AUDIT_DIR` bleibt optional und privat (Dateien 0600); die Dateien enthalten
Anfrage, Endergebnis und Provenienz, keinen separaten Denktext.

## Laufzeiten und Ressourcen

- `LLM_CONNECT_TIMEOUT_SECONDS`: Verbindungsaufbau (Standard 10 s).
- `LLM_READ_TIMEOUT_SECONDS`: Inaktivität beim Lesen; leer übernimmt
  `LLM_TIMEOUT_SECONDS` (Standard 120 s). Gilt einheitlich für alle Aufgaben.
- `LLM_TOTAL_TIMEOUT_SECONDS=0`: kein Gesamtlaufzeitlimit; positive Werte begrenzen
  Generierung einschließlich Metadaten und Wiederholungen, nach Erwerb des Slots.
- `AGENDA_DETECTION_TIMEOUT_SECONDS` bleibt als veraltete Konfigurationsangabe
  lesbar, begrenzt aber keine eigenen Netzwerkaufrufe mehr.
- `OLLAMA_LOAD_TIMEOUT`: Watchdog des **Servers** zum Modellladen, in Compose
  weitergereicht. Ein entfernter Server muss ihn separat konfigurieren.
- `LLM_KEEP_ALIVE`: native Anfrage überschreibt `OLLAMA_KEEP_ALIVE`;
  sonst dessen Wert bzw. 5m. Null ist mit der nachträglichen Runnerprüfung
  unvereinbar und wird abgewiesen. Negative Dauer hält das Modell geladen.
- `LLM_CPU_THREADS` und `LLM_GPU_LAYERS`: leer überlässt die Auswahl Ollama;
  `LLM_GPU_LAYERS=0` erzwingt CPU. Keine Maschinenwerte im Anwendungscode.
- `OLLAMA_NUM_PARALLEL`, `OLLAMA_MAX_LOADED_MODELS`, `OLLAMA_FLASH_ATTENTION` und
  `OLLAMA_KV_CACHE_TYPE` sind Serverparameter. Quantisierte KV-Caches erfordern
  unterstützte Flash Attention; F16 bleibt ohne diese Annahme nutzbar.

Interne NDJSON-/SSE-Streams sammeln ausschließlich vollständige Antworten.
Ein aktiver Stream darf beliebig lange laufen; ein stummer Stream bleibt durch
Read-Timeout, optionales Gesamtlimit und kooperativen Abbruch begrenzt. Der
Abbruch schließt den Netzwerkstream auch beim Modellladen. Persistierte Jobs
melden wartenden/denkenden/generierenden Status ohne Textinhalt. Einmal pro
Sekunde wird aktualisiert; die GPU-/Inferenzwarteschlange prüft ebenfalls Abbruch.
Verbindungsfehler, Inaktivität, abgerissene Verbindungen und vorübergehende
HTTP-Fehler werden bis `LLM_MAX_RETRIES` mit exponentiellem
`LLM_RETRY_BACKOFF_SECONDS` wiederholt. Teilantworten werden verworfen.
Konfigurations-, Speicher-, Kontextfehler und Abbrüche werden nicht wiederholt.
[Ollama Ressourcen-FAQ](https://docs.ollama.com/faq),
[Load-Watchdog](https://github.com/ollama/ollama/blob/v0.34.2/envconfig/config.go).

Die lokale CPU-Auswahl und gemessene Ressourcen gehören ausschließlich in die
unversionierte `.env`. Vor einer späteren Aktivierung sind ein kontrollierter
Ladeversuch, die tatsächliche KV-/RSS-Belegung, Durchsatz, Bildtokenisierung und
fachliche Qualität zu prüfen. Modellgewichte allein belegen keinen passenden
Kontext. Änderungen dieser Arbeit starten keine Dienste und laden kein Modell.
