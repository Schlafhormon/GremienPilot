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
