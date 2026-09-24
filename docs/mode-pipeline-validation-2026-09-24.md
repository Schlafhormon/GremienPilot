# Fast/Slow-Umstellung und lokale Qwen-Verifikation

Stand: 24. September 2026. Implementierung: `91dd7da`, Promptbereinigung: `3c793e1`,
begrenzte Rekonstruktionsquellen: `3b3e99a`.
Die gezielte Ergänzungssuche wurde mit `c18b1dc` präzisiert.
Die Pflichtschlüssel für Statusentscheidungen wurden mit `0dcd2b8` eingeführt.
Die Begrenzung der vorbereitenden Quellenabrufe ist `078e2f9`.
Die Budgetanpassung der Fenster ist `031358f`.
Das Listenformat mit Anfangsentscheidung und Wechseln wurde mit `46caf8e` eingeführt.
Die anschließende native Verifikation führte zum geschlossenen Wechselobjekt v11
und dem schlankeren Fast-Detailkontext (`177240a`).
v12 (`8b607e1`) verkürzt ausschließlich die Fast-Detailausgabe: kurze technische TOP-IDs,
höchstens 96 Zeichen Begründung und ein Quellenbeleg je Entscheidung.
Die nachfolgenden Messungen beziehen sich auf die lokale RTX 3070 Ti mit 8 GB
VRAM und Ollama 0.34.4. Sie sind keine allgemeine Modellrangliste.

**Abschließender Fast-Gesamttest:** 1.774/1.774 Zeilen technisch verarbeitet,
kein technischer Fehlversuch, **30:02 Minuten** statt ursprünglich 69:53 Minuten
mit nur 640 verarbeiteten Zeilen. 1.665 Zeilen sind zugeordnet, 109 ausdrücklich
nicht zugeordnet. Fast bleibt ein fachlich ungeprüfter Entwurf. 808 Backendtests
und 63 relevante Frontendtests sind bestanden. Qwen3.5:9b ist lokal aktiv;
leere Modusvariablen ergeben Fast ohne Thinking und Slow mit Thinking.

## Verifizierter Ausgangsfehler

Auftrag `9965f5c7-648f-43e3-9c1c-dac4b1c7d3d9`, 1.774 Originalzeilen:

- 32 Modellaufrufe, alle mit `stop`; maximal 2.412 generierte Tokens je Aufruf
  bei 8.192 erlaubten Ausgabetokens. Kein Beleg für abgeschnittene Antworten.
- 23 Detailblöcke: acht gültig, 15 verworfen. Acht Antworten mit ungültiger
  Abdeckung, sieben mit gleichzeitigem Ergebnis und Quellenanforderung;
  zwei der gemischten Antworten hatten zusätzlich überlappende Bereiche.
- Konkrete nullbasierte Grenzen: `80–130, 130–159` überlappt bei 130;
  `720–736, 737–750, 751–776, 778–799` lässt 777 aus;
  im ersten Block kommt `33–10` vor.
- Sechs der sieben gemischten Antworten forderten ausschließlich schon
  mitgelieferte Zielquellen an. Eine weitere forderte zusätzlich Index 640 an.
- Die 640 verarbeiteten Zeilen sind acht verteilte 80er-Blöcke:
  160–239, 480–559, 640–719, 800–879, 880–959, 1440–1519,
  1520–1599 und 1680–1759. Davon 599 zugeordnet, 41 fachlich nicht zugeordnet.
- Laufzeit 4.193,39 Sekunden (69:53); Modell-Wandzeit 4.184,8 Sekunden,
  davon Eingabeverarbeitung 898,57 Sekunden und Generierung 3.262,65 Sekunden.
  692.870 Eingabetokens und 24.472 generierte Tokens über alle Aufrufe.
  23 Detailaufrufe benötigten zusammen 2.917,65 Sekunden.
- Kontext 131.072, Thinking aus. Gemma4:12b lagerte 29/49 Schichten auf die
  GPU aus; die Logs zeigen 27 vollständige erneute Promptverarbeitungen.
  Die reine Generierung erreichte ungefähr 7,5 Tokens/s.

Die gespeicherten Rohantworten und Diagnosen bestätigen damit den Fehler im
Antwortvertrag. Redundante inklusive Start-/Endgrenzen erfordern vermeidbare
Grenzarithmetik. Ergebnis und Quellenanforderung waren gleichzeitig darstellbar.
Die strenge Verwerfung verhinderte widersprüchliche Zuordnungen, machte aber
jeweils den gesamten Block unbrauchbar. Alte Browserlimits, SQLite-Leases und
das Ausgabelimit erklären diesen konkreten Lauf nicht.

## Implementierte Entscheidung

Das Modell gibt `initial` für die erste Zielzeile und ein verpflichtendes Objekt
`changes` mit Quellen-IDs als Schlüsseln und neuen Zuordnungen als Werten aus.
Der Server sortiert eindeutige Wechsel nach Quellenreihenfolge und berechnet
ausschließlich technische Enden: vor dem nächsten Wechsel beziehungsweise am
Ende des Zielblocks. Jede Entscheidung gilt laut Vertrag bis zu diesem Ende.
Fehlende Pflichtfelder, doppelte JSON-Schlüssel, doppelte oder unbekannte
Wechselquellen bleiben Fehler. Bestehende widersprüchliche Start-/Endantworten
werden nicht repariert oder in diesen Vertrag umgedeutet.

Quellenanforderung und Ergebnis sind getrennte, geschlossene Varianten, auch in
den Vorbereitungsphasen. TOP-Statusantworten enthalten feste Pflichtfelder je
angeforderter TOP-ID; der Server ergänzt keine fehlenden Statuswerte.
Vorbereitung und kompakte Detailzuordnung erlauben nur technische Quellenfenster
mit tatsächlich fehlenden Originalen, höchstens drei Fenster mit jeweils 80 Zeilen
pro Abruf. Fast behält einen Versuch ohne zusätzliche Inhaltsprüfung und ohne
Reparaturaufteilung für abgeschnittene Antworten. Kürzere Begründungen und
weniger Auditmetadaten im Prompt begrenzen den Generierungsaufwand. Slow behält
unabhängige Bewertung und Klärung. Fachliche Abschnittsgrenzen und TOP-Auswahl
bleiben Modellentscheidungen; das Format beweist keine inhaltliche Richtigkeit.
Fast-Details erhalten nur den Verlaufstext der Rekonstruktion; ihre ungeprüften
Episodengrenzen und TOP-Statuszuordnungen bleiben im Archiv und in Slow erhalten,
werden aber nicht als konkurrierende Vorgaben in Fast-Details wiederholt.
Originalzeilen haben im Prompt ausdrücklich Vorrang. Gleiche aufeinanderfolgende
Zuordnungen sollen einen Abschnitt bilden; ein Sprecher- oder Belegwechsel allein
rechtfertigt keinen weiteren Eintrag.
Im Fast-Detailprompt und Antwortschema ersetzen kurze Kennungen (`T1`, `T2`, …)
die langen TOP-UUIDs. Der Server übersetzt sie vor Cache, Checkpoint und öffentlichem
Ergebnis in die unveränderten kanonischen IDs zurück. Die konkrete Zuordnung ist
Teil der Cacheidentität und der Ergebnisprovenienz. Die Kurzkennungen kollidieren
nicht mit vorhandenen IDs; wiederholte Checkpointvalidierung bleibt idempotent.
Dies reduziert Ausgabevolumen, ohne eine fachliche Entscheidung vorwegzunehmen.
Die Kontextnotizen verweisen phasenneutral auf das jeweils angebotene
Quellenschema. Leere Abruffelder abgeschlossener Vorphasen werden nicht in neue
Prompts übernommen; sie könnten sonst wieder das alte Antwortformat nahelegen.
Bei Quellenfenster-Protokollen werden außerdem die konkurrierenden numerischen
Zeilenindizes und Zielbereichsgrenzen aus Modellprompts entfernt. Quellen-IDs und
Originaltexte bleiben vorhanden; interne Indizes für Validierung und Speicherung
bleiben unverändert.

Bei kleinen Kontexten werden die technischen Fenster weiter unterteilt und die
zulässige Zahl gleichzeitig angeforderter Fenster reduziert. Das benötigt keine
Modellaufrufe. Wenn zusätzliche Originale nicht mehr passen, kann ein bereits
passendes Ergebnis trotzdem fertiggestellt werden. Tatsächliche Quellenabrufe
bleiben budgetgeprüft; zu große Einzelquellen oder ausgeschöpfte Abrufrunden sind
weiterhin ausdrückliche Fehler. Es gibt keine stille Kürzung von Originaltext.

Eine Entscheidung pro Originalzeile wäre ebenfalls eindeutig, erzeugte aber
wesentlich mehr Ausgabe. Das nachträgliche Reparieren alter Start-/Endpaare
könnte unbemerkt fachliche Aussagen verändern. Zusätzliche Modellreparaturen
würden Fast verlangsamen. Diese Alternativen wurden deshalb nicht gewählt.

Objekte mit einer verpflichtenden letzten Endquelle erwiesen sich im Qwen-Test
als zu grob: Der erste Block wurde vollständig TOP 1 zugeordnet, obwohl bei den
nullbasierten Indizes 13, 18, 22 und 33 ausdrücklich TOPs 2 bis 5 aufgerufen werden.
Der v8-Gesamttest wurde deshalb kontrolliert gestoppt. Der native
[Grammatikgenerator](https://github.com/ggml-org/llama.cpp/blob/master/common/json-schema-to-grammar.cpp)
stellt Pflichtfelder vor optionale Felder; dies ist für eine letzte Endquelle
als erstes Ausgabefeld ungünstig. Das allein erklärt das Modellverhalten nicht:
Auch eine Anfangskarte und kurze TOP-Kürzel führten zu groben Zusammenfassungen.
Eine ausdrücklich nach der ersten Zuordnung verlangte Wechselliste funktionierte
in der gezielten nativen Probe besser. Lange bestehende TOP-IDs bleiben erhalten;
es wurde keine heuristische Zuordnung oder TOP-Umnummerierung eingeführt.

Öffentliche zeilenweise Ergebnisse, gespeicherte Vorschläge, manuelle
Zuordnungen und Quellenidentitäten bleiben kompatibel. Neue Prompt-/Modusversionen
trennen Cache- und Checkpointidentitäten. Ein alter Auftrag wird nicht unter
geänderter Modellkonfiguration still fortgesetzt.

Betroffene Schnittstellen:

| Bereich | Dateien und Änderung |
| --- | --- |
| Modellantworten | `app/backend/agenda_llm.py`: Anfangsentscheidung und Wechsel, disjunkte Antwortvarianten, verfügbare Quellenfenster, Promptprojektion, Fast-Fehlerbehandlung |
| Quellenidentitäten | `app/backend/source_contract.py` übersetzt weiterhin zwischen kurzen Modellreferenzen und stabilen Original-IDs; zusätzlich kompakte Schemagrenzen für vorhandene Quellen, keine Änderung des Speicherformats |
| Moduskonfiguration | `llm_config.py`, `llm_transport.py`, `processing_mode.py`: getrennte Auflösung und native Thinking-Schalter |
| Dauerhafte Aufträge | `durable_jobs.py`: modusabhängiger Konfigurationssnapshot; `resume_pipeline.py`: eng begrenzte bestehende Offline-Migration erkennt auch `reasoning_effort=none` |
| Deployment | `.env.example`, `app/backend/.env.example`, `docker-compose.yml`, `k8s/backend/configmap.yaml`; lokale unversionierte `.env` mit Qwen, Tokenizer und Imageversionen |
| Regressionen | Backendtests für Quellenvertrag, Moduskonfiguration, Transport, Promptfixtures und Offline-Wiederaufnahme |

## Konfiguration

Lokal aktiviert: `qwen3.5:9b`, Q4_K_M, rund 6,6 GB Modelldatei,
Digestpräfix `6488c96fa5fa`, Kontext 131.072 und passender lokaler Qwen-Tokenizer.
Backendimage `gremienpilot-backend:mode-pipeline-20260924`.
Die aktive Backenddatei und der Arbeitsstand haben denselben SHA-256:
`341893debe153c57546c9069b781406ef511ba87d197cc3bc442f16900dce465`.
Der Gesundheitscheck nach Aktivierung ist `healthy`.

```dotenv
LLM_FAST_REASONING_EFFORT=
LLM_SLOW_REASONING_EFFORT=
```

Leer/fehlend bedeutet Fast `none`, Slow `medium`. Bei Qwen3.5 und Gemma4 ist
Thinking ein boolescher Schalter; `low`, `medium`, `high`, `max` sind dort keine
unterschiedlichen Denkstufen. `none` schaltet aus. Die alten globalen
Thinking-Variablen überschreiben diese Modusdefaults nicht.

| Verhalten | Fast | Slow |
| --- | --- | --- |
| Zweck | Schneller, ungenauer Entwurf | Unabhängig geprüfte Zuordnung |
| Thinking bei leerer Modusvariable | Aus | Ein |
| Ungültige Zuordnungsantwort | Ein Versuch, ausdrücklicher Fehler | Bestehende begrenzte Reparatur und Klärung |
| Fachliche Zweitbewertung | Keine | Unabhängiger Leser, Abweichungsklärung |
| Detailausgabe | Kurze TOP-Kennungen, 96 Zeichen, ein Beleg | Kanonische IDs, ausführlichere Begründungen und Belege |
| Gespeicherte IDs und manuelle Bearbeitung | Bestehendes Format | Bestehendes Format |

Fast reserviert keine Denktokens. Slow verwendet lokal 2.048 zusätzliche Tokens
neben dem fachlichen Ausgabelimit von 4.096. Ollama kann diese Reserve nicht als
separates Denklimit durchsetzen. Die tatsächliche Thinking-Laufzeit schwankt.
Qwen liegt bei diesem Kontext nur mit 18/34 Schichten auf der GPU; ein kleinerer
Modellname bedeutet auf dieser Hardware keine vollständige GPU-Ausführung.

Die lokale `.env` und Datenbanksicherung sind unversioniert unter `data/backups`
gesichert. Globale Installationsdefaults und explizite Browser-/API-Modellwahl
wurden nicht automatisch auf einen anderen Modellnamen umgeschrieben.

## Automatisierte Verifikation

- Vollständige Backend-Suite der abschließenden Fassung: **808 bestanden**, 298,75 Sekunden,
  isolierter Linux-Testcontainer ohne Live-Datenbankvolume.
- Frontend-Lebenszyklus, Zuordnungsansicht und App: **63 bestanden**, drei Dateien.
- Vorbereitender gezielter Durchgang vor der Umstellung auf das Wechselobjekt:
  **159 bestanden**, 93,37 Sekunden.
  Er enthält auch TOP-IDs wie `evidence`, `grounding` und `uncertain`, die als
  Identitäten behandelt und nicht mit gleichnamigen Metadaten verwechselt werden.
  Hinzu kommen begrenzte Originalabrufe und adaptive Quellenfenster bei kleinen
  Kontextbudgets, einschließlich des Erhalts beider unabhängiger Slow-Leser.
- Regressionen enthalten die Bereichsgeometrien aller 15 tatsächlich verworfenen
  Antworten, doppelte JSON-Schlüssel, fehlende Anfangsentscheidungen,
  doppelte/fremde Wechselquellen und die Sortierung gültiger Wechselquellen,
  gemischte Antwortvarianten, erneute Quellenanforderung, Abrufgrenzen,
  Fast-Abbruch ohne versteckte Reparaturschleife, leere/fehlende/gesetzte
  Modusvariablen, Konfigurationssnapshots und parallele Modusauflösung.
  Ein vorheriger vollständiger Durchgang hatte unter gleichzeitiger nativer
  Inferenz einen Timeout im Worker-Test zur vorübergehenden Speicherausnahme;
  beide Varianten bestanden isoliert. Die abschließenden 808 Tests liefen ohne
  parallele Inferenz und ohne Fehler. Der Worker-Code wurde dafür nicht geändert.
- Weitere Regressionen prüfen die verlustfreie Fast-Kurzkennung, Kollisionen mit
  bestehenden TOP-IDs, idempotente erneute Validierung von Checkpoints,
  unterschiedliche Cacheidentitäten bei gleichen Titeln und verschiedenen IDs
  sowie unveränderte TOP-IDs und Belegbudgets in Slow. Gezielt: 85 Tests bestanden.
- Separater gespeicherter Daten-Replay ohne Modellaufruf: Die acht gültigen
  Altantworten sind für alle 640 Zeilen verlustfrei als Anfangsentscheidung mit Wechseln darstellbar;
  TOPs, Begründungen, Konfidenz und Belege stimmen mit den gespeicherten
  Ergebnissen überein. Alle 15 ungültigen Altantworten werden abgewiesen.
  Keine Umwandlung wurde in einer Sitzung gespeichert.
- Vergleich mit der Linux-SQLite-Sicherung: Inhalte und manuelle Zuordnungen
  der ursprünglichen Sitzung über zwölf Tabellen unverändert. Der Dienststart
  aktualisierte ausschließlich `pipeline_jobs.updated_at`. Der alte Auftrag
  einschließlich 66 Artefakten, 37 Checkpoints und 32 Metrikdatensätzen ist
  vollständig unverändert. SQLite wurde ausschließlich im Linux-Container geöffnet.

## Reale Modellläufe

Die Verifikation verwendet neue Aufträge ohne `session_id`. Ergebnisse werden
deshalb nicht in die ursprüngliche Sitzung oder deren manuelle Zuordnungen
übernommen. Warteschlangenzeit wird bei Laufzeitvergleichen separat behandelt.

### Abschließender Gesamtlauf mit v12

Auftrag `b8beb577-35a2-4d20-9856-573db787b30e`, kompletter ursprünglicher Input,
neuer Auftrag ohne Sitzungsbindung, keine Antwortübernahme aus alten Aufträgen:

- **1.774/1.774 Zeilen technisch verarbeitet**, Ergebnisstatus `success`,
  `processing_complete=true`, `review_status=skipped`. 1.665 zugeordnete und
  109 ausdrücklich nicht zugeordnete Zeilen. Der Auftragszustand bleibt
  `review_required`; die fachliche Inhaltsprüfung ist nicht erfolgt.
- **1.802,02 Sekunden (30:02)** Ausführungszeit. Gegenüber dem ursprünglichen
  Fehlerlauf etwa 57 % kürzer; unterschiedliche Modelle, Prompts und Cachezustände
  erlauben keine isolierte Aussage über die Ursache dieses Geschwindigkeitsgewinns.
- 39 Aufrufe, **kein Fehlversuch**, alle mit `stop` beendet, Thinking durchgehend
  aus. 23 erfolgreiche Zuordnungsantworten und sechs kurze Quellenanforderungen
  in den Details; keine technische Reparatur und keine fachliche Zweitprüfung.
- 553.080 Eingabetokens, 11.043 generierte Tokens. Modellzeit 1.789,86 Sekunden:
  Eingabeverarbeitung 349,98, Generierung 1.424,73, Laden 0,05 Sekunden.
  Die aggregierte Generierungsrate liegt bei 7,75 statt zuvor 7,50 Tokens/s.
  Die Detailausgabe sank von 17.193 auf 4.887 Tokens. Die Messung zeigt deutlich
  weniger Ausgabe bei ähnlicher mittlerer Generierungsrate; sie belegt keinen
  allgemeinen Geschwindigkeitsvorteil des Modellnamens Qwen gegenüber Gemma.
- Vorbereitung: zehn Aufrufe, 1.033,66 Sekunden (17:14). Details: 29 Aufrufe,
  756,20 Sekunden (12:36), 4.887 generierte Tokens. Die Vorbereitung ist damit
  der größte verbleibende Zeitanteil; die Pipeline wurde nicht um weitere
  Inhaltsprüfungen erweitert.
- Technische Nachprüfung des gespeicherten Ergebnisses: alle ursprünglichen
  Quellen-IDs genau einmal in unveränderter Reihenfolge, Indizes 0–1773,
  ausschließlich kanonische TOP-IDs und gültige Quellenbelege. Diese Prüfung
  beurteilt weder Themenzuordnung noch Beleginhalt.
- Erneuter Vergleich mit der Sicherung: ursprünglicher Auftrag vollständig
  unverändert; Sitzungsinhalte und manuelle Zuordnungen unverändert.

### Abschließender nativer Slow-Detailtest mit v12

Sechs synthetische Zeilen mit zwei eindeutigen Themen wurden im endgültigen
Antwortformat **6/6 korrekt** als `[0,0,0,1,1,1]` zugeordnet. Ein Modellaufruf,
kein Fehlversuch, `effective_thinking=true`, `finish_reason=stop`.
Laufzeit **460,28 Sekunden (7:40)**; 1.549 Eingabe- und 4.314 generierte Tokens.
Die Probe lief nach dem Fast-Gesamttest ohne konkurrierende Testlast.

Dies bestätigt das endgültige native Antwortformat mit Thinking. Es ist ein
einzelner Detailtest, kein vollständiger neuer Slow-Gesamtlauf und kein Nachweis
allgemeiner fachlicher Überlegenheit. Die unabhängigen Leser und die Klärung von
Abweichungen sind zusätzlich durch die abschließende Backend-Suite abgesichert.
Thinking kann auf der vorhandenen Hardware bereits bei kurzem Input mehrere
Minuten benötigen; Fast bleibt deshalb standardmäßig ohne Thinking.

### Vorbereitende Kurztests und Zwischenstände

Fast-Kurztest `e7f1f70f-cbdb-4a3d-9448-6e5c6456d15e`: sechs synthetische Zeilen,
zwei bekannte Themen, **6/6 verarbeitet**, vier Aufrufe, keine technischen Fehler,
Thinking aus, Inhaltsprüfung ausdrücklich `skipped`. Ausführungszeit 161,72 Sekunden
einschließlich initialem Modellladen. 1.052 generierte Tokens in 106,66 Sekunden
(rund 9,9 Tokens/s). Die Themenzuordnung ist in diesem einfachen Beispiel richtig,
aber das Modell erzeugte beide vorhandenen TOPs nochmals als zusätzliche Einträge.
Dies zeigte eine fachliche Schwäche des damaligen Ermittlungs-Prompts; keine
heuristische Zusammenführung und kein zusätzlicher Bewertungsdurchgang wurden
eingeführt. Die spätere Präzisierung und ihr erneuter Kurztest stehen unten.
Technische Vollständigkeit bedeutet nicht fehlerfreie TOP-Identität.

Slow-Kurztest `7dcda6ef-8ef2-47a0-8f52-4865b7dca046`: **6/6 korrekt zugeordnet**,
zwei TOPs ohne Duplikate, alle sechs unabhängigen Detailbewertungen `agreed`,
Verarbeitung und Prüfung vollständig, keine unsicheren Zeilen. Acht Aufrufe,
keine technischen Fehler oder Reparaturversuche. Beide unabhängigen Inventare
erkennen, dass keine zusätzlichen TOPs vorliegen. Gesamtausführungszeit
**2.412,19 Sekunden (40:12 Minuten)**, 22.963 generierte Tokens. Der Test verwendet
das vorherige Endquellen-Listenformat. Er fand vor der Umstellung auf Pflichtschlüssel
und getrennte Vorbereitungsantworten statt; er ist deshalb kein vollständiger
Slow-End-to-End-Nachweis des abschließenden Drahtformats. Einzelne Thinking-Aufrufe
benötigen dennoch mehrere Minuten;
die unabhängige Inventarprüfung allein 560,21 Sekunden und 5.331 generierte Tokens.
Der Test fand vor der letzten Präzisierung der Ermittlungsfrage statt. Er zeigt
den möglichen Thinking-Aufwand, beweist aber keinen Qualitätsvorsprung gegenüber
dem aktuellen Fast-Prompt oder gegenüber Gemma.

Der erste große Qwen-Test (`df9cae8e-34a6-4f9d-805b-bc5c79e1ee34`) endete nach
502,55 Sekunden bereits in der Verlaufsrekonstruktion: `invalid_episode_range`.
Das Modell erfand Bereiche wie `L1820–L2100` und `L3550–L3600` außerhalb der 1.774
Quellen. Die Anwendung verwarf sie, keine Zeile wurde als verarbeitet markiert.
Das zuvor freie Zeichenkettenfeld für Rekonstruktionsgrenzen wird deshalb nun
durch eine kompakte Grammatik auf `L1..L1774` begrenzt, ergänzt um den expliziten
Quellenrahmen im Prompt. Die Regel wächst mit der Zahl der Dezimalstellen statt
mit der Zahl der Transkriptzeilen. Der strenge Identitäts-/Bereichsvalidator bleibt
zusätzlich bestehen. Kein nachträgliches Abschneiden oder Erraten einer gültigen
Referenz findet in der Anwendung statt.

Eine separate native Schema-Probe mit Ollama 0.34.4 akzeptiert die Grammatik
und liefert eine gültige Referenz, obwohl eine ungültige verlangt wurde.
Das ist ein Strukturtest, keine Aussage über fachliche Quellenwahl.
Die verwendeten Konstrukte gehören zum unterstützten
[JSON-Schema-Grammatikumfang von llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md).

Im nachfolgenden großen Test (`102faf10-f391-4d94-8b49-5edcfb65c2a9`) kopierte
Qwen alle 27 bekannten TOPs einschließlich identischem Titel, Nummer und
Sitzungsteil in die Ergänzungsliste. Dieser Test wurde daraufhin kontrolliert
abgebrochen. Die Ermittlungsfrage unterscheidet nun explizit zwischen einer
Differenzliste und einem vollständigen Inventar; die bestehende kurze Begründung
wird im Antwortschema vor der Liste ausgegeben. Es gibt keinen zusätzlichen
Modellprüfschritt und keine heuristische Zusammenführung identischer Titel.
Der erneute Fast-Kurztest (`70e0fa68-ebad-4be8-b809-8cf3f806e3ff`) mit dem
damaligen Stand v6 liefert **6/6 korrekte Zuordnungen, genau zwei TOPs ohne Duplikate,
keine unsicheren Zeilen und keine technischen Fehler**. Vier Aufrufe,
809 generierte Tokens, 82,61 Sekunden Ausführungszeit (1:23), Thinking aus und
Prüfstatus ausdrücklich `skipped`. Das ist ein einfacher Regressionstest,
keine allgemeine fachliche Bewertung realer Sitzungen.

Der große v6-Test (`5b894101-38a7-4207-9a9f-bc30d1295a46`) erzeugte keine
Agenda-Duplikate und rekonstruierte gültige Quellenbereiche. Anschließend fehlte
jedoch einer von sechs angeforderten TOP-Statuswerten. Auch dieser Entwurf wurde
verworfen; kein Status wurde ergänzt. Das neue Objekt `agenda_states_by_id`
macht jede angeforderte TOP-ID zu einem Pflichtschlüssel. Die damals verwendete
Endquellenkarte erzwang ebenfalls eine letzte Entscheidung, wurde aber nach
inhaltlicher Prüfung durch Anfangsentscheidung und Wechsel ersetzt.
Gemeinsame Definitionen halten das Schema trotz der Pflichtfelder kompakt.

Ein nativer Ollama-Test des zunächst vorgesehenen Endquellenformats mit optionalen
Endquellen, Pflicht-Endquelle, `$defs` und alternativer Quellenanforderung lieferte
zunächst eine strukturell gültige Zuordnung (`stop`, Thinking aus). Die späteren
inhaltlichen Stichproben führten zur Wahl von `initial` und `changes`.
Der zusätzliche native Slow-Test
des v8-Schemas liefert alle sechs Status-Pflichtfelder, reguläres `stop`,
`effective_thinking=true` und 1.776 generierte Tokens bei einem Limit von 6.144.
Er benötigte 189,81 Sekunden; zeitgleich lief die isolierte Backend-Testsuite.
Dieser Test belegt den nativen Schema-/Thinking-Vertrag, keine fachliche Qualität.

Der integrierte native Fast-Blocktest mit `agenda-changes-v10` verarbeitet alle
80 Originalzeilen in einem Aufruf, ohne technische Fehler. An den vier expliziten
Aufrufen bei Index 13, 18, 22 und 33 stimmen die TOP-Zuordnungen. Laufzeit 122,65
Sekunden unter paralleler Testlast. Einzelne Grenzen lagen zu früh, und eine
Vorbereitung auf einen späteren TOP wurde zugeordnet: Das ist ein verbleibender
fachlicher Fehler. Anschließend wurden die redundanten numerischen Zeilenindizes
aus den ID-basierten Modellprompts entfernt; der große End-to-End-Lauf prüft diese
Fassung. Keine der Kontrollzuordnungen wird im Anwendungscode hinterlegt.

Der anschließende große v10-Lauf `888fab25-3c0a-439c-b675-81876d2793fc` zeigt in
derselben Stichprobe eine deutliche fachliche Schwäche: Seine erste Detailantwort
ordnet alle 80 Zeilen TOP 1 zu, einschließlich der vier ausdrücklich aufgerufenen
anderen TOPs. Die vorherige Rekonstruktion enthält bereits falsche grobe
Episodengrenzen. Dass diese Notizen die Detailantwort beeinflussen, ist eine
plausible Erklärung, noch kein isoliert nachgewiesener Kausalzusammenhang.
Der kurze native Test ohne diese Rekonstruktion war kein ausreichender Nachweis
für den vollständigen Pipelinekontext.

In demselben v10-Lauf brach der Block 560–639 nach **1.091,74 Sekunden (18:12)**
bei **8.192 Tokens** mit `IncompleteResponseError` ab. Die im Transportartefakt
gespeicherte Teilantwort enthält 58 Wechsel-Einträge für nur 46 verschiedene
Quellen. Unter anderem wurden `L579` und `L605` doppelt sowie `L624` dreimal
ausgegeben. Zahlreiche weitere Einträge wiederholen dieselbe Zuordnung.
Thinking war nachweislich aus. Das ist ein neu beobachteter Fehler dieses
Zwischenstands, keine nachträgliche Erklärung des ursprünglichen Gemma-Laufs.
Der Test wurde nach 640 akzeptierten Zeilen kontrolliert abgebrochen; der
fehlerhafte 80er-Block wurde weder übernommen noch erneut angefordert.

Deshalb verwendet v11 `changes` als geschlossenes Objekt mit optionalen,
eindeutigen Quellen-Schlüsseln statt einer Liste wiederholbarer Quellenfelder.
`initial` steht weiterhin separat davor; die erste Zielzeile ist kein zulässiger
Wechselschlüssel. Doppelte JSON-Schlüssel alternativer Provider bleiben Parserfehler.
Die Zahl gewählter TOPs ist außerdem strukturell auf die Zahl vorhandener TOPs
begrenzt. Keine der Grenzen ersetzt eine fachliche Entscheidung.

Die native Probe mit Quellen-Objekt und expliziter Quellenhierarchie, aber noch
vollständiger Rekonstruktion, blieb fachlich zu grob (0/4 Kontrollaufrufe exakt,
112,38 Sekunden). Mit ausschließlich dem globalen Rekonstruktionstext statt
geschätzter Episodengrenzen und Statuszuordnungen wurden anschließend **4/4
Kontrollaufrufe** korrekt erkannt (93,10 Sekunden). Die Eröffnungszeilen wurden
weiterhin falsch zugeordnet. Der Ansatz verbessert diese Stichprobe, beweist
aber keine vollständige inhaltliche Richtigkeit oder einen isolierten Kausaleffekt.

Eine zweite Probe verwendet dieselbe gespeicherte Vorbereitung und den zuvor
abgebrochenen Block 560–639: **80/80 technisch verarbeitet, vier Abschnitte,
3/3 ausdrücklich aufgerufene TOP-Wechsel korrekt, ein Aufruf ohne Fehlversuch,
95,80 Sekunden**. Hier stimmen die Wechsel bei 582, 604 und 623 trotz der
eingeschobenen Tagesordnungsposition mit den vorhandenen TOP-Identitäten überein.
Die Vorbereitung wurde technisch aus Artefakten wiedergegeben, nicht neu vom
Modell erzeugt. Diese Proben sind gezielte Offline-Verifikation; die Fast-Pipeline
erhält keinen zusätzlichen Bewertungsdurchgang.

Ein nativer Slow-Detailtest des Zwischenstands v10 ordnet die sechs synthetischen
Zeilen korrekt als `[0,0,0,1,1,1]` zu. Ein Aufruf, kein Fehlversuch,
`effective_thinking=true`, 546,24 Sekunden unter zeitweiser paralleler Testlast.
Damit wurde die Anfangsentscheidung mit Wechseln und Thinking praktisch geprüft;
die später eingeführten eindeutigen Wechselschlüssel sind in dieser Slow-Probe
noch nicht enthalten. Die abschließende Slow-Orchestrierung und unabhängige
Mehrfachprüfung werden durch die Backendtests abgesichert.

Der v7-Lauf (`96dc292f-4c31-4815-a6d2-376640d2f61d`) lieferte drei vollständige
Statusgruppen mit jeweils sechs TOPs. Anschließend forderte das Modell mit
`source_ranges=[{start:0,end:1773}]` das gesamte Transkript erneut an. Die lokale
Budgetprüfung verhinderte den zu großen Folgeaufruf (`ContextBudgetError`), acht
Modellaufrufe waren bis dahin regulär beendet. Seit v8 verwenden auch die
Vorbereitungsphasen die begrenzten Fenster-IDs. Die noch angebotenen Quellen und
der Höchstwert von drei Fenstern stehen im Schema; alte Gesamtabrufe werden
abgewiesen. Das ist eine technische Abrufbegrenzung, keine inhaltliche Reparatur.

## Grenzen der Aussagekraft

Fast ist ausdrücklich ein schneller, ungenauer Entwurf. Einzelne falsche
Zuordnungen oder unvollkommene Zusammenfassungen lösen weder weitere fachliche
Prüfungen noch Reparaturschleifen aus. Maßgeblich für den abschließenden Fast-Test
sind die technische Stabilität und die gesamte Laufzeit. Die gründliche
Inhaltsprüfung bleibt Aufgabe von Slow; die nativen Stichproben werden nicht
zu zusätzlichen Produktionsschritten.

Der weitere vollständige Fast-Test verwendete Auftrag
`cb392f5d-6084-4595-8549-6c2779549665` mit `agenda-change-map-v11`, denselben
1.774 Originalzeilen und ohne Sitzungsbindung. Der Block 80–159 erreichte trotz
eindeutiger Wechselquellen erneut das Ausgabelimit von 8.192 Tokens nach
1.014,06 Sekunden (16:54). Die Teilantwort enthielt 54 unterschiedliche Wechsel,
viele davon mit wiederholter gleicher TOP-Zuordnung. Vor dem kontrollierten
Abbruch waren 640 Zeilen verarbeitet. Es gab keinen Reparaturaufruf.

Die Teilantwort wiederholte 105-mal eine von sechs langen TOP-UUIDs. Ein rein
technischer Vergleich derselben Zeichenfolge mit dem lokalen Qwen-Tokenizer
ergab 8.134 Tokens vor und 5.030 nach Ersetzung durch kurze Kennungen, etwa 38 %
weniger. Das ist eine Volumenmessung, keine gemessene Laufzeitverbesserung.
v12 übernimmt deshalb kurze technische TOP-Kennungen ausschließlich für Fast
und reduziert Begleittext auf 96 Zeichen und einen Beleg. Zusätzliche fachliche
Prüfungen oder Reparaturschleifen wurden nicht eingeführt.

Gezielte native Nachprüfung des Blocks 80–159 mit v12 und derselben gespeicherten
Vorbereitung: **80/80 technisch verarbeitet, kein Fehlversuch, 119,79 Sekunden**.
Zwei reguläre Quellenabrufe mit 40 und 33 Tokens gingen der Zuordnungsantwort
mit 539 Tokens voraus. Alle drei Antworten endeten mit `stop`, Thinking war aus.
Die einzelne Zuordnungsantwort dauerte 84,24 Sekunden. Es wurde keine zusätzliche
fachliche Qualitätsbewertung durchgeführt. Diese Probe belegt die Behebung des
konkreten Abbruchs, aber noch keine Laufzeitgarantie für andere Blöcke.

Technische Vollständigkeit ist kein Qualitätsurteil über die fachlichen Zuordnungen.
Fast kennzeichnet die unabhängige Inhaltsprüfung weiterhin als `skipped`; Slow
behält zwei unabhängige Rekonstruktionen und die Klärung von Abweichungen.
Thinking kann dafür hilfreich sein, garantiert aber keine richtige Entscheidung.
Die bisherigen Stichproben beweisen keinen allgemeinen Qualitätsvorteil von
Qwen3.5:9b gegenüber Gemma4:12b und keinen generellen Vorteil von Slow gegenüber Fast.

Für einen belastbaren fachlichen Vergleich wären manuell bestätigte Abschnitte mit
Themenwechseln, gemeinsamen Beratungen, Wiederaufnahmen, Vertagungen und Wechseln
zwischen öffentlichen/nichtöffentlichen Teilen erforderlich. Beide Modi müssten
gegen dieselben bestätigten Zuordnungen bewertet werden. Diese Inhaltsbewertung
ist eine separate Testaufgabe und kein zusätzlicher Durchgang in der Fast-Pipeline.

Die Laufzeitmessung vergleicht den ursprünglichen Fehlerlauf mit der neuen lokalen
Konfiguration einschließlich anderem Modell, Tokenizer und Prompt. Sie isoliert
keinen einzelnen Geschwindigkeitsfaktor. Der abschließende Fast-Lauf nutzt den
bereits für die Proben verwendeten Ollama-Dienst; Modell- und Promptcaches wurden
nicht geleert. Er ist deshalb kein kontrollierter Kaltstartvergleich.
CPU-Auslagerung bei 131.072 Kontexttokens
und die zusätzlichen Thinking-Tokens begrenzen weiterhin den Durchsatz.
