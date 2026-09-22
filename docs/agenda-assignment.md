# Bekannte Agenda: Zuordnung und Nachweis

Der bekannte-Agenda-Pfad verarbeitet jede Transkriptzeile. Ein erfolgreich
verarbeitetes Fenster ist **kein fachlicher Richtigkeitsnachweis**. Nicht
ausgewertete technische Lücken, begründete semantische Lücken und unsichere
Zuordnungen werden getrennt in `llm`, `gaps` und `segments` gespeichert.

## Identitäten

SQLite und Sitzungs-API behalten unabhängige `top_ids` (UUIDs). Modell-IDs
verwenden Abschnitt und unveränderte Originalnummer, etwa `public:07`,
`nonpublic:07`, `unspecified:02.10`. Wiederholungen erhalten ausdrücklich
`~occurrence-N`; nummernlose Einträge heißen `unnumbered`. Aus Listenpositionen
wird keine Originalnummer abgeleitet. `llm.provenance.identities` dokumentiert
die Zuordnung von Modell-ID, Originalnummer, Titel, Index und (in einer Sitzung
bzw. bei mitgesendeten `top_ids`) stabiler TOP-UID.

## Kontext und Reparaturen

`agenda_context.py` erkennt konservative Originalbelege. Abschnitt und letzter
TOP-Aufruf werden getrennt fortgeführt; indirekte neue TOPs bleiben möglich.
Ein `previous_top` ist ausdrücklich eine unbestätigte Vorhersage. Reviews
erhalten keine früheren Nachbarlabels. Ein ursprünglicher Aufruf samt
Nachbarzeilen bleibt auch nach einem Reparatursplit verfügbar.
Fortsetzungen werden zusätzlich gespeichert und ersetzen den ursprünglichen
Aufruf nicht. Ein vermuteter indirekter Themenbeginn wird mit Originalzeilen
als unbestätigte Hypothese mitgeführt. Bereits belegte Abschnittsgrenzen und
fehlende Schließungsbelege begrenzen auch die pro Zeile erlaubten Schema-IDs;
`null` bleibt möglich. Damit muss eine strukturell gültige Antwort nicht erst
nachträglich wegen eines verbotenen Abschnitts verworfen werden.

`AGENDA_DETECTION_CONTEXT_WINDOW_BEFORE/AFTER` gelten auch für die bekannte
Agenda. Fehlen diese Einstellungen, dient die Hälfte von
`AGENDA_DETECTION_CHUNK_OVERLAP_LINES` als kompatibler Standard. Beim konservativen
UTF-8-Bytebudget werden zuerst entferntere Kontextzeilen reduziert; Zieltext
und eigentliche Anker werden nicht gekürzt. `context_budget` verzeichnet
ausgelassene Indizes. Das Budget enthält 512 Tokens Chatreserve, Nachrichten,
2048 Ausgabetokens (bei Thinking ggf. zwei Phasen) und Platz für Reparatur.

Ein Validierungsfehler erhält zunächst einen Reparaturversuch im selben
Fenster mit denselben Originalbelegen. Bei fehlenden Gap-Begründungen fordert
ein eigenes kleines Schema ausschließlich die erforderlichen Gründe an;
die Labels werden unverändert übernommen. Danach sind begrenzte Splits möglich;
erschöpfte Fehler erzeugen technische Lücken, keine geratenen Ersatzlabels.
Reviews werden atomar übernommen und bei Fehlern als offene Prüfung markiert.

## Frische Antworten und Cache

- `POST /api/pipeline/start`: Multipart-Feld `agenda_fresh=true`.
- `POST /api/agenda-detection`: JSON `fresh: true`; optional `top_ids`.
- Oberfläche: **Frische TOP-Berechnung** erzeugt neue Antworten.
- Für eine genaue Wiederholung `fresh` weglassen und den zurückgegebenen
  `llm.provenance.cache_namespace` als `cache_namespace` mitsenden.

Bestehende Caches werden nicht gelöscht. Der Schlüssel enthält Modell-Digest,
Kontextgröße, Generierungsparameter, Prompt-/Schemaversion, Namensraum und
Nachrichten. Bei nicht auflösbarem lokalem Digest wird die Wiederverwendung
zwischen Läufen deaktiviert. Cache-Einträge bewahren Reparatur- und
Split-Historie; `chunks` weist aktuelle Treffer und Parent-Schlüssel aus.
`LLM_AUDIT_DIR` aktiviert private lokale Provider-Anfrage-/Antwortartefakte.
Diese enthalten Sitzungsinhalte und gehören nicht ins Git-Repository.

## Audiozeiten und Kompatibilität

WhisperX-Segmente werden nicht mehr über lange Sprecherbeiträge verschmolzen.
`timing.words` enthält Zeichenbereiche und Alignment-Zeiten,
`timing.segments` die ursprünglichen Segment-IDs und Intervalle. Satzsplits
verwenden ausgerichtete Wortgrenzen; ohne vollständige Randwörter bleibt eine
explizit markierte Zeichenschätzung. Alignment kann selbst ungenau sein.

Die additive SQLite-Spalte `timing_json` existiert in Job- und
Sitzungstranskripten. Altdaten erhalten keine erfundenen Wortzeiten. Alte
Clients dürfen das optionale Feld weglassen; bei unverändertem Text/Zeitbereich
bleibt es erhalten. Textänderungen entwerten alte Wortzuordnungen; der Editor
kennzeichnet neue Grenzen als Schätzungen. Zusammenführen erhält Wortzeiten
mit angepassten Zeichenpositionen. Proposals berücksichtigen Zeitprovenienz
bei der Prüfung auf veraltete Ergebnisse.

## Prüfungen

Backend: `cd app/backend; python -m pytest`. Frontend:
`cd app/frontend; npm test; npm run build`.

`test_agenda_evidence.py` deckt Identitäten, Sitzungsabschnitte,
Niederschriftsrückblicke, offene Informationspunkte, nichtmonotone Aufrufe,
Erhalt von Originalbelegen in Splits, strukturelle Reparatur, fehlerhafte
Reviews, Cache-Isolation und Zeitpersistenz ab. Die bestehenden Vertrags-,
API-, Persistenz- und UI-Tests ergänzen diese Tests.

## Optionales separates lokales TOP-Modell

Unter **KI-Einstellungen → Modell für TOP-Zuordnung** kann ein separates Modell
aktiviert werden. Standard ist `gemma4:31b-it-q4_K_M`; der Download erfolgt nicht
automatisch. Die Option ist zunächst aus. **TOP-Einstellungen speichern** schreibt
atomar `agenda-model.json` neben die aktive SQLite-Datenbank (alternativer Pfad:
`AGENDA_MODEL_SETTINGS_PATH`). Keine Datenbankmigration, keine Änderung vorhandener
Sitzungen, keine automatische Neuberechnung. Die Einstellung gilt serverweit für
neue Pipeline-Verarbeitungen und `/api/agenda-detection`, auch für alte API-Clients.
Ein explizites `use_llm=false` bleibt ein bewusster Aufruf ohne LLM.

`GET/PUT /api/settings/agenda-model` liest/speichert Konfiguration und lokalen
Verfügbarkeitsstatus. Aktiviert ersetzt sie die bisherige Modellwahl ausschließlich
innerhalb der TOP-Verarbeitung. Zusammenfassungen einschließlich Reviews und
Neugenerierung sowie PDF-Extraktion verwenden weiter ihre bisherigen Einstellungen.
Ein einzelner TOP-Lauf verwendet eine unveränderliche Konfiguration; es werden
keine gemeinsamen Umgebungsvariablen zur Laufzeit umgeschrieben.

Konfiguration: `enabled`, `model`, `context_tokens` (anfangs 32768),
`output_tokens` (4096), `timeline_output_tokens` (4096), `timeout_seconds` (1800),
`cpu_threads` (8), `temperature` (0.1), `seed` (42 oder null), `thinking` (false).
Die Startwerte sind Testbudgets, keine Zusage der Ladbarkeit oder Geschwindigkeit.
Gemma 31B unterstützt laut offizieller Modellkarte maximal 256K; RAM/VRAM,
KV-Speicher, Betriebssystem, Docker-Limit und übrige Prozesse begrenzen den Betrieb.
Das Modellpaket allein benötigt laut Registry rund 19,87 GB. Offizielle Quellen:
[Ollama-Tag](https://ollama.com/library/gemma4:31b-it-q4_K_M) und
[Google-Modellkarte](https://ai.google.dev/gemma/docs/core/model_card_4).

Der separate Modus benötigt die native lokale Ollama-API. Cloud-Tags und externe
Endpoints sind ausgeschlossen. Fehlende Modelle werden vor der Verarbeitung
erkannt. Modell-/Speicherfehler erzeugen technische Lücken und sichtbare Diagnosen;
es gibt keinen Qwen-Ersatzlauf. Nach echten TOP-Provideraufrufen wird der ausgewählte
Runner explizit entladen und die Freigabe geprüft. Reine Cache-Wiederholungen
entladen Qwen nicht. Die gemeinsame Inferenzsperre hält einen vollständigen
TOP-Lauf zusammen; mit `GPU_MODEL_SWITCHING=true` bleibt auch die vorhandene
Transkriptionskoordination wirksam. Nachfolgende Zusammenfassungen laden Qwen;
wiederholte Neugenerierungen benutzen unverändert seinen normalen Lebenszyklus.

In Docker stellt `ollama-agenda` einen getrennten lokalen Runner bereit und liest
dasselbe Modellvolume schreibgeschützt. Er lädt beim Start kein Modell. Ausschließlich
dieser Dienst begrenzt die llama.cpp-Kontextcheckpoints auf 2 und deaktiviert dessen
zusätzlichen RAM-Promptcache (`LLAMA_ARG_CTX_CHECKPOINTS=2`, `LLAMA_ARG_CACHE_RAM=0`).
Qwens ursprünglicher Ollama-Dienst und dessen Parameter bleiben erhalten.
`AGENDA_LLM_BASE_URL` wählt diesen Endpoint; ohne diese Variable verwendet eine
lokale Installation den bisherigen Ollama-Endpoint. Beide Dienste sind über dieselbe
Backend-Inferenz-/GPU-Sperre koordiniert. Vor TOP-Inferenz wird Qwen freigegeben,
vor verwalteter Transkription werden bei aktiver Option beide Dienste freigegeben.
Die Diagnose-Schnittstelle des TOP-Dienstes ist nur an `127.0.0.1:11435` gebunden.

### Themenverlauf und Budgets

`agenda_timeline.py` liest zunächst Agenda und möglichst viele vollständige
Originalzeilen. Ist der Gesamttext zu groß, werden maximal passende, überlappende
Fenster verarbeitet; **jede** Fenstergrenze erhält einen separaten Review. Vorherige
Hypothesen werden samt ursprünglichen Belegen weitergeführt. Struktur-/Budgetfehler
werden begrenzt mit überlappenden Splits repariert; unlösbare Fälle brechen sichtbar
ab. Keine Originalzeile wird still abgeschnitten. Auch eine einzelne übergroße
Zeile wird als Budgetfehler ausgewiesen.

Ereignisse enthalten globale Zeilenindizes, wortgetreue Belege, Ereignistyp,
Abschnitt, TOP-ID, Unsicherheit, Begründung und Gegenbelege. Die API ergänzt die
Zuordnung zu stabilen Zeilen-IDs, Zeiten und ursprünglichen Segment-IDs.
Eine ausschließlich abweichende Groß-/Kleinschreibung eines Zitats wird nur bei
einem eindeutigen Treffer in derselben Originalzeile korrigiert. Das Modellzitat
bleibt als `model_quote` erhalten, die Korrektur wird in `evidence_repairs`
protokolliert und die Hypothese als unsicher markiert. Wörter, Satzzeichen,
Zeilenreferenzen und TOP-Hypothesen werden dabei nicht ersetzt. Andere ungültige
Belege bleiben Fehler; auch reparaturbedingte Abschnittssplits erhalten einen
expliziten Übergangsreview.
Der vollständige Verlauf einschließlich Herkunft und Abdeckung wird in
`llm.provenance.timeline` gespeichert. Die nachfolgende bestehende präzise
Zuordnung und ihre Reviews erhalten relevante Hypothesen **und Originaltext**.
Diese Hypothesen begrenzen weder Schema-IDs noch Validatoren. Lokale Belege dürfen
sie korrigieren; es gibt keine monotone TOP-Reihenfolge.

Die Budgetprüfung nutzt einen lokal registrierten, zum Modell-Digest passenden
Tokenizer, wenn vorhanden: `512 + Σ(Tokens(Nachricht) + 32) + Tokens(Schema)`.
Die Zusatzreserve berücksichtigt Chat-Template und Providerdetails; dies ist
keine exakte Zählung des fertig gerenderten Ollama-Templates. Der offizielle
Google-Tokenizer wurde unter `data/tokenizers/gemma4-31b/` abgelegt, mit Revision,
SHA-256 und zugehörigem Ollama-Digest in `metadata.json`. Ein Modell- oder
Tokenizer-Digestwechsel erfordert eine neue Prüfung. Alternativer Verzeichnispfad:
`AGENDA_TOKENIZER_DIR`. Andere Installationen ohne registrierten Tokenizer verwenden
die ausdrücklich konservative UTF-8-Byteobergrenze
`512 + Σ(UTF8(Nachricht) + 32) + UTF8(Schema)`.
Dazu kommen das separat konfigurierte Ausgabelimit und 768 Tokens Reparaturreserve.
Bei aktiviertem Thinking wird für strukturierte Antworten vorsorglich das doppelte
Ausgabelimit reserviert. Im separaten Modus wird Thinking als boolescher Ollama-
Parameter übergeben. Providerzählungen (`prompt_eval_count`, `eval_count`) und
der per `/api/ps` verifizierte Kontext dienen der nachträglichen Gegenprüfung.
Bei lokalem Tokenizer wird zusätzlich geprüft, ob der Provider mindestens die
Originalinhalt-Tokens (mit kleiner Toleranz für Template-Ränder) verarbeitet hat;
eine auffällig niedrigere Zählung ist ein möglicher Trunkierungs-/Tokenizerfehler.
256K-Praxisnutzung wird nicht behauptet.

### Nachtläufe und Wiederaufnahme (Entwicklungsbranch)

Der separate TOP-Dienst liest native Ollama-Streams. `timeout_seconds` begrenzt
jetzt das Warten auf die **erste Ausgabe**, einschließlich Modellladen und
Promptverarbeitung. Diese beiden Vorgänge lassen sich während des Wartens mit
der Ollama-API nicht getrennt beobachten; nach Abschluss stehen getrennte
Providerzeiten zur Verfügung. `connect_timeout_seconds` (15 s) begrenzt den
Verbindungsaufbau, `idle_timeout_seconds` (300 s) ausbleibende Ausgabe nach dem
ersten Inhalt, `total_timeout_seconds` (43200 s) den gesamten einzelnen Aufruf.
Aktive Generierung kann somit länger als 1800 Sekunden laufen. Die Grenzen
sind getrennt in den KI-Einstellungen editierbar. Qwens Transport und gemeinsame
Umgebungsparameter werden dadurch nicht verändert.

Nur ein abgeschlossenes `done=true` mit vollständiger, budgetgeprüfter und
fachlich strukturell validierter Antwort wird übernommen. Teil-JSON gilt nie
als Zuordnung. Abbruch schließt den HTTP-Stream; der bestehende Modell-Lebenszyklus
prüft anschließend die Freigabe. Verbindungs-/Ladefehler werden nicht durch ein
Ersatzmodell kaschiert. Zeit-/Ausgabe-/Validierungsfehler des Themenverlaufs
erhalten höchstens drei Teilungsebenen mit Überlappung und Grenzprüfung.

Validierte Fenster werden atomar mit `fsync` im vorhandenen persistenten Cache
gesichert. Themenverlaufsteilungen erhalten vor dem ersten Kind einen eigenen
gespeicherten Teilungsplan. Bei Wiederaufnahme werden erfolgreiche Kinder erneut
validiert, ohne den bereits fehlgeschlagenen großen Elternaufruf zu wiederholen.
Cacheidentität umfasst Eingabe, Modell/Digest, Parameter, Tokenizer, Promptversion,
Schema und Namensraum. Es gibt keine Wiederverwendung bei abweichender Identität.
Unvollständige Antworten werden nicht als erfolgreiche Cacheeinträge gespeichert.

Im Sitzungsverlauf startet **TOPs und Zusammenfassungen in neuer Sitzung berechnen**
einen dauerhaften Pipeline-Auftrag auf einer Kopie des bestehenden Transkripts.
Die Ursprungssitzung bleibt unverändert; IDs, Zeitdaten und Audioverknüpfung bleiben
erhalten. Der Auftrag läuft unabhängig vom geöffneten Browser weiter. Nach Fehler
oder Backend-Neustart kann **TOP-Auftrag fortsetzen** denselben Cache-Namensraum
verwenden. Änderungen an Transkript, Agenda oder TOP-Einstellungen erfordern einen
neuen Auftrag. Aktuelle Pipeline-API:

- `POST /api/sessions/{session_id}/agenda-jobs`, JSON `{}`; optional `tops` mit
  einer neu aus dem Original-PDF extrahierten Agenda. Antwort enthält neue Sitzung
  und Pipeline-ID. Kein erneuter Transkriptionsauftrag.
- `POST /api/pipeline/{pipeline_id}/resume`: unterbrochenen Auftrag fortsetzen.
- `GET /api/pipeline/{pipeline_id}`: Aktivität, Laufzeit, Ausgabezeichen,
  technische Zeilenabdeckung und abgeschlossene Fenster in `agenda_progress`.
- `GET /api/pipeline/{pipeline_id}/result`: auch gespeicherte Teilergebnisse
  fehlgeschlagener/abgebrochener Aufträge. Der Status bleibt ausdrücklich fehlgeschlagen.

Im separaten Modus werden sämtliche ermittelten Grenz-/Lückenprüfbereiche in einem
endlichen Durchgang geprüft; das bisherige Limit von drei/vier Reviews bleibt im
Ein-LLM-Modus erhalten. Fehlgeschlagene Reviews, technische Lücken und fachlich
begründete Unklarheit werden getrennt dokumentiert. Technische Fehler sperren
Zusammenfassungen und führen zu `failed`, nicht `completed/ready_for_review`.
Begründete semantische Lücken dürfen bestehen bleiben und bleiben sichtbar.

### PDF-Quellenabgleich

Die strukturierte Qwen-Extraktion verlangt nun JSON auch im Providerformat.
Die Antwort bleibt zusammen mit nummerierten Originalzeilen, verworfenen
Dokumentartefakten und Konflikten in `provenance` erhalten; bei Pipeline-PDFs
zusätzlich unter `result_refs.pdf_extraction`. Ungültiges JSON wird nicht als
nummerierte Freitextliste interpretiert. Die alte Regel „längere Liste gewinnt“
entfällt: Originalnummern, Abschnitte und einzelne Vorkommen werden abgeglichen.
Unbelegte Modellzusätze werden als Konflikte gespeichert, nicht still ergänzt.
Fehlende/mehrdeutige Modellbestätigung bleibt prüfpflichtig. Die heuristische
Originaltextrekonstruktion ist kein unabhängiger fachlicher Goldstandard.

Allgemeine Signatur-/Amtsbezeichnungs-, Trennlinien- und Seitenfußregeln ersetzen
die früheren namensbezogenen Filter. Gedrehte Randstempel einer sonst überwiegend
aufrechten PDF-Seite werden geometrisch ausgefiltert; vollständig gedrehte Seiten
werden dadurch nicht geleert. Original-PDFs für eigene Diagnosezwecke aufbewahren.

### Eindeutiger Ein-LLM- und Entwicklungsstart

Ein Git-Wechsel verändert weder `.env`, gespeicherte Einstellungen, Datenbank noch
Images. `main` wurde nicht zurückgesetzt und bleibt auf dem bisherigen Ein-LLM-
Quellstand. Die Weiterentwicklung liegt auf `feature/dual-llm-agenda`; kein neuer
Commit wurde für diese Arbeit angelegt. GPU-Compose liest `BACKEND_GPU_IMAGE`.

Mit den neuen lokal gebauten Images und der vorhandenen `.env`:

```powershell
# Ein LLM, unabhängig von einer gespeicherten aktivierten Gemma-Option:
docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.single-llm.yml up -d --pull never

# Entwicklungsbetrieb, gespeicherte TOP-Option gilt wieder:
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --pull never
```

Der Ein-LLM-Override setzt `AGENDA_MODEL_FORCE_DISABLED=true` nur im Container.
Er überschreibt die JSON-Einstellungen nicht und lehnt das Aktivieren der
separaten Option in diesem Startmodus ab. Dies ist der Ein-LLM-Betrieb des neuen
Codes; für exakt den historischen `main`-Build sind eigene, zu dessen Quellstand
passende Image-Tags erforderlich. Niemals aus dem Branchnamen auf den laufenden
Code schließen. Vor jedem Neustart aktive Pipeline-, Transkriptions- und
Zusammenfassungsjobs prüfen. Die TLS-CA-Variablen und persistenten Torch-/HF-
Volumes bleiben Bestandteil des regulären Compose-Starts.

Öffentliche Transportreferenzen: [Ollama Chat](https://docs.ollama.com/api/chat),
[HTTPX Streaming](https://www.python-httpx.org/async/),
[HTTPX Timeoutarten](https://www.python-httpx.org/advanced/timeouts/).

Die genaue Zuordnung behält ihre begrenzten Ausgabefenster: Ein großes
Eingabefenster des Themenverlaufs ist kein Budget für hunderte JSON-Zeilenlabels.
Cache-Schlüssel trennen Modell, Digest, Parameter, Kontext, Prompt-/Schemaversion,
Verarbeitungsschritt, Eingabe und frischen Namensraum. Der Verlauf erhält eine
eigene Identität; davon abhängige Zuordnungen verweisen auf diese Herkunft.
`chunks` enthält Cache-/Providerstatus, Aufrufmetriken und Parent-/Reparaturhistorie.

Originalhinweise auf eine nachträglich geänderte Tagesordnung werden auch in
späten Zuordnungs- und Reparaturfenstern mitgegeben. In diesem Fall gilt eine
gesprochene Nummer nicht als harter Schlüssel zur gedruckten PDF-Nummer. Es wird
weder ein Nummernversatz berechnet noch die Annahme eines Änderungsantrags
behauptet. Inhalt und Originalbelege einschließlich möglicher Ablehnung bleiben
maßgeblich. Ein neu eingefügter Punkt ohne vorhandene Agendaidentität bleibt eine
begründete fachliche Lücke. Nach einem belegten Wechsel des Sitzungsteils beginnt
diese Prüfung für den neuen Abschnitt erneut.

PDF-Daten werden mit einem eigenen strukturierten Ausgabeschema angefordert;
Qwen-Zusammenfassungen verwenden weiterhin ihre bisherigen Parameter. Der
manuelle PDF-Upload zeigt verbleibende Quellenkonflikte an und bietet den lokalen
Download des vollständigen Quellenabgleichs einschließlich Originalantwort an.
Der Pipeline-PDF-Pfad speichert diesen Abgleich dauerhaft beim Auftrag.
Reine Typographie und eine nur im Originaltitel enthaltene Vorlagennummer gelten
bei sonst identischem Titel als erklärbarer Abgleich. Widersprechende
Vorlagennummern und fehlende Titelteile bleiben Konflikte. Originaltitel werden
dabei nicht überschrieben. Nummerierte Unterpunkte bleiben separate Einträge;
unnummerierte Aufzählungen bleiben als `subpoints` beim Quellenkandidaten erhalten.

Die Schließungsprüfung trennt den tatsächlich geschlossenen Sitzungsteil von
einer nachfolgenden Ankündigung des anderen Teils in derselben Zeile. Eine höflich
formulierte gegenwärtige Schließung wird nur zusammen mit der unmittelbar
folgenden Verabschiedung als Originalbeleg akzeptiert; Zukunft, Rückblick,
Negation und bloße Ankündigung des Schließungs-TOPs reichen nicht aus.

Im separaten Modellmodus löst auch eine unsichere Segmentgrenze oder ein
Widerspruch zwischen Verlaufshypothese und lokaler Zuordnung eine gezielte
unabhängige Nachprüfung aus. Überlappende Prüfbereiche werden vereinigt. Es gibt
einen endlichen Nachprüfdurchgang; weiterhin fachlich unsichere Antworten bleiben
sichtbar unsicher. Prüfgründe stehen in `provenance.review_triggers`. Eine globale
Hypothese schränkt die zulässigen lokalen TOPs nicht ein.

Fortschrittsspeicherung und Abbruch teilen eine Sperre für das Lesen,
Zusammenführen und Speichern des Auftragszustands im Backendprozess. Ein
Fortschrittsupdate kann so keinen Abbruchwunsch durch einen alten Stand ersetzen.
Abbruch wird bis zum Pipeline-Worker weitergereicht und löst keine Modellreparatur
aus. Beim Fortsetzen werden Warnungen des vorigen Versuchs in
`result_refs.previous_attempts` aufbewahrt; sie erscheinen nicht als Fehler des
neu laufenden Versuchs.
