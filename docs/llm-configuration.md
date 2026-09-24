# Modell- und Transportvertrag

Stand der Prüfung: 24. September 2026. Neue Installationen verwenden den
Standard `qwen3.5:9b`; explizite Modellnamen, einschließlich
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

Qwen3.5:9b unterstützt Text/Bilder und optionales Thinking. Das Ollama-Modell ist
etwa 6,6 GB groß und unterstützt 256K Kontext. Die Anwendung reserviert bewusst
131.072 Tokens: lange Sitzungen bleiben vollständig durch Abschnittsverarbeitung
zugänglich, ohne überall das größte Fenster zu belegen. Das Fenster ist keine
Garantie für reine GPU-Ausführung; auf kleineren GPUs wird RAM mitbenutzt.
[Offizielle Ollama-Modellbeschreibung](https://ollama.com/library/qwen3.5/tags).

Leere Sampling-Variablen verwenden für genau `qwen3.5:9b` unter Ollama Temperatur
1.0, top_p 0.95 und top_k 20, entsprechend dem lokal geprüften Profil und den
Qwen-Empfehlungen für Thinking beziehungsweise Reasoning ohne Thinking.
Explizite Werte haben Vorrang. Andere Modelle und externe Provider erben dieses
Profil nicht; dort bleiben bisherige Aufgaben-/Providerwerte maßgeblich.
[Qwen-Modellkarte](https://huggingface.co/Qwen/Qwen3.5-9B).

Die Standardinstallation verwendet die bereits lokal geprüfte Ollama-Version
0.34.4. Setup-Skripte, Compose und Kubernetes verwenden denselben Stand.
Der Ollama-Ladewatchdog erlaubt 30 Minuten, entsprechend der bestehenden
Backend-Grenze für Laden/erstes Token. Aktive Generierung hat keine pauschale
Gesamtlaufzeitgrenze; die Inaktivitätsgrenze bleibt 120 Sekunden.

Die Thinking-Einstellung wird pro Verarbeitungsmodus aufgelöst:

```dotenv
LLM_FAST_REASONING_EFFORT=
LLM_SLOW_REASONING_EFFORT=
```

Leer, nur Leerzeichen oder nicht gesetzt bedeutet **Fast: `none` (aus)** und
**Slow: `medium` (an)**. Explizit erlaubt sind `none`, `low`, `medium`, `high`,
`max`. Gemma 4 und Qwen 3/3.5 erhalten einen booleschen Ollama-Schalter: alle
Stufen außer `none` schalten Thinking ein; sie sind bei diesen Modellen keine
unterschiedlichen Denkintensitäten. Andere Modelle/Provider müssen die gewählte
Stufe unterstützen. OpenAI-kompatible Server erhalten `reasoning_effort`.

Die alten globalen Variablen `LLM_THINKING` und `LLM_REASONING_EFFORT` sind für
Fast-/Slow-Aufgaben abgelöst und beeinflussen auch die leeren Modusdefaults nicht.
Alle Phasen einer Aufgabe erben den Modus, einschließlich PDF, Agenda und
Zusammenfassung. Job-Snapshots lösen denselben Modus bereits beim Einreichen auf.
Die wirksame Konfiguration wird pro Operation eingefroren, protokolliert und in
Cache-/Checkpoint-Identitäten aufgenommen. Parallele Aufgaben beeinflussen ihre
Moduseinstellungen nicht. Bei `none` entfällt auch die Thinking-Tokenreserve.
Qwen-Steuerzeilen werden aus Systemprompts entfernt. Denktext wird nicht als
Endergebnis interpretiert.
[Ollama Thinking](https://docs.ollama.com/capabilities/thinking).

`LLM_OUTPUT_TOKENS` überschreibt ausdrücklich die bisherigen aufgabenspezifischen
Ausgabebudgets; leer behält sie bei. `LLM_THINKING_TOKENS` ist eine zusätzliche
Reservierung, **kein separat durchsetzbares Denklimit**. Leer/fehlend bedeutet
bei Ollama 4.096 Tokens, bei externen APIs 0. Ohne Thinking ist die Reserve immer
0. Agenda (Fast und Slow), Zusammenfassungen und PDF verwenden standardmäßig je
8.192 Ausgabetokens. Das sind Obergrenzen, keine Mindestlängen oder zusätzlichen
Modellaufrufe. Ollama bietet hier nur
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
Bildreservierungen, JSON-Schema und sämtliche Generierungsphasen. Der Standard
beträgt **131072 Tokens** und gilt gemeinsam für **Fast und Slow**. Auch lokale
`.env`-Dateien müssen auf diesen Wert angepasst werden, da bestehende Werte die
Deployment-Defaults überschreiben. Das größere Fenster gibt TOP-Liste,
Verlaufsnotizen und Rekonstruktionen einschließlich der Slow-Vergleiche mehr
Platz; die Anzahl der Inhaltsprüfungen wird dadurch nicht erhöht.
Modell und Server müssen diese Kontextgröße unterstützen. Der größere
Kontext kann mehr Arbeitsspeicher beziehungsweise VRAM beanspruchen.
Native Anfragen
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

Die Docker-Images enthalten den offiziellen Qwen3.5:9b-Tokenizer. Beim Image-Bau
wird nur die öffentliche JSON-Datei (rund 13 MB) aus der in `llm_assets.py`
festgelegten Revision geladen und gegen ihre SHA-256 geprüft. Es werden keine
Modellgewichte und kein fremder Python-Code geladen; kein HF-Token ist nötig.
Ohne explizite Tokenizer-Einstellungen wird diese Datei ausschließlich für
`qwen3.5:9b` unter Ollama automatisch ausgewählt. Ein Modellwechsel verwendet
diesen Tokenizer nicht versehentlich weiter.

Optional: `LLM_TOKENIZER_PATH` zeigt auf einen eigenen **lokalen**, modellpassenden
Hugging-Face-`tokenizer.json` (Bibliothek `tokenizers`, bereits Teil der ML-Abhängigkeiten); `LLM_TOKENIZER_MODEL` muss
mit dem tatsächlichen Modellnamen übereinstimmen. Bei Modellüberschreibungen
wird ein ausdrücklich konfigurierter unpassender Tokenizer abgewiesen. Kein Download und kein fremder
Python-Code im Job. Die modellpassende Text-/Schema-Zählung wird um 32 Tokens je Nachricht und
512 Tokens für Chat-Template/Steuerzeichen ergänzt; Provider-Templates können abweichen. Ohne Tokenizer gilt die konservative
UTF-8-Bytezahl plus 32 Tokens je Nachricht und 512 globale Reservetokens.
Das ist ein bewusst großzügiges Ersatzverfahren, keine universelle mathematische
Garantie für beliebige Tokenizer; die native Laufzeitprüfung bleibt maßgeblich.

Nachrichten akzeptieren Text, OpenAI-Content-Parts (`text`, `image_url`) und
native `images`. Ollama benötigt Base64, externe APIs dürfen Bild-URLs erhalten;
der Backend-Transport lädt solche URLs nicht selbst. Leeres `LLM_IMAGE_TOKENS`
reserviert für Qwen3.5:9b/Ollama **17.408 Tokens je Bild**, für andere Modelle 0.
Die Qwen-Reserve basiert auf maximal 16.777.216 Pixeln bei 16er-Patches und
2-facher räumlicher Zusammenfassung (16.384 Positionen), plus 1.024 Tokens
Spielraum. Der native Laufzeitcheck bleibt maßgeblich. Explizites `0` verweigert
Bilder; ein expliziter anderer Wert überschreibt die Reserve. Andere Modelle
benötigen weiterhin eine passende Einstellung. Der Wert ist eine **Reservierung**,
kein Befehl zur Bildskalierung.
[Offizielle Qwen-Bildkonfiguration](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/preprocessor_config.json).

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

Im Fast-Modus erlaubt `AGENDA_FAST_OUTPUT_TOKENS=8192` mehr Platz für die
TOP-Modellantworten. Slow verwendet ebenfalls `AGENDA_OUTPUT_TOKENS=8192`.
Bei kleineren Kontextfenstern wird das Fast-Fachbudget so begrenzt, dass
einschließlich Denkreserve höchstens die Hälfte für die Ausgabe reserviert wird.
Ein explizites `LLM_OUTPUT_TOKENS` überschreibt beide Fachbudgets und bleibt von
dieser Begrenzung ausgenommen. Die Planung berücksichtigt das wirksame
Ausgabebudget einschließlich Denkreserve im
konfigurierten Kontextfenster und bildet bei Bedarf kleinere Eingabeabschnitte;
das Kontextfenster wird nicht automatisch vergrößert und keine Quelle gekürzt.
Fast-Kontextnotizen enthalten höchstens acht repräsentative Quellenverweise.
Der vollständige Quellenbereich bleibt separat erhalten und die Originalzeilen
bleiben für die spätere Zuordnung verfügbar. Fast behält einen Versuch je
Abschnitt ohne zusätzliche unabhängige Inhaltsprüfung oder Reparaturrunde.

Kompakte Zuordnungen sind mit `AGENDA_COMPACT_ASSIGNMENTS=true` der Standard.
Das Modell gibt zusammenhängende Abschnitte mit gemeinsamer Zuordnung, kurzer
Begründung und Originalbelegen aus. Der Server prüft lückenlose, überlappungsfreie
Abdeckung und bildet jeden Abschnitt wieder auf die einzelnen Zeilen ab.
Die unabhängige Gegenprüfung und die Klärung von Abweichungen bleiben bestehen.
`AGENDA_OUTPUT_TOKENS_PER_LINE=40` und `AGENDA_DETECTION_CHUNK_LINES=80` sind die
passenden Standardwerte. Ohne kompakte Ausgabe und ohne expliziten Planungswert
reserviert die Anwendung weiterhin 256 Tokens je Zeile. Fast teilt nur vor der Generierung zu große Eingaben;
fehlerhafte oder abgeschnittene Modellantworten lösen keine Reparaturaufteilung aus.
Slow behält seine begrenzten Reparaturversuche.
Diese Planungswerte garantieren weder eine bestimmte Qualität noch Laufzeit.
`LLM_FAST_REASONING_EFFORT=none` bzw. `LLM_SLOW_REASONING_EFFORT=none`
deaktiviert Thinking und die zusätzliche Reserve im jeweiligen Modus.

Ein bereits gespeicherter Transkript-Checkpoint verhindert bei Wiederanläufen
eine erneute Audioverarbeitung. Normale Jobs verweigern weiterhin geänderte
Modell-/Codekonfigurationen. Für die eng begrenzte Umstellung eines unterbrochenen
Auftrags in der Phase `agenda_detect` gibt es die Offline-Wartung
`python resume_pipeline.py JOB_ID`: zunächst nur Prüfung, mit `--apply` nach
Datenbanksicherung anwenden. Der Backendserver muss gestoppt sein. Sie erlaubt
nur die beschriebenen Denk-/Zuordnungsparameter sowie die zugehörigen Codeänderungen,
behält bereits abgeschlossene PDF-Schritte samt ursprünglicher Provenienz und
protokolliert die alte und neue Konfiguration. Modellwechsel, geänderte PDF-Logik
oder bereits abgeschlossene Zuordnungen werden abgewiesen. `--fork-session`
führt den Auftrag in einer neuen Sitzung fort und erhält zwischenzeitliche
Editoränderungen an der bisherigen Sitzung. Die Modellidentität wird beim
Wiederanlauf weiterhin kontrolliert.

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
Kontext. Die lokale Qwen-Umstellung und ihre Messungen stehen im
[Verifikationsbericht vom 24. September](mode-pipeline-validation-2026-09-24.md).

Die verpflichtende Quellenprüfung, Offline-Bewertung und Grenzen sehr großer Notizinventare sind in [quality-verification.md](quality-verification.md) beschrieben.
