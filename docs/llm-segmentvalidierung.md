# Validierung von LLM-Segmenten bei bekannter Tagesordnung

Die öffentliche API behält `tops`, `assignments` und die bisherigen Segmentfelder.
Persistierte `top_ids` bleiben unverändert. Die zustandslose Erkennung bekommt nur
Titel; deshalb verwendet ihr interner LLM-Vertrag eigene, nur für diesen Aufruf
gültige Referenzen `agenda:0`, `agenda:1`, … auf die bereinigte TOP-Liste. Diese
Referenzen sind weder Originalnummern noch Session-IDs. Die Antwortposition hat
keine Bedeutung für die Zuordnung.

## Vertrag

Der Prompt enthält die Agenda als Objekte mit `top_id` und `top_title`. Nummern,
Unterpunkte und Abschnittspräfixe bleiben im Titel erhalten. Die Antwort lautet:

```json
{"tops": [{
  "top_id": "agenda:2",
  "top_title": "3. Schulbau",
  "start_index": 4,
  "end_index": 5,
  "confidence": 0.9,
  "evidence_index": 4,
  "evidence_text": "Kommen wir zu TOP 3 Schulbau.",
  "uncertain": false
}]}
```

Ohne bekannte Agenda wird `top_id` weggelassen. Für ältere Modellantworten bleiben
`segments`, direkte Arrays und `title` als Aliase lesbar. Ohne ID muss der Titel
nach Normalisierung exakt und eindeutig passen. Angegebene Nummern und Abschnitte
müssen übereinstimmen; Titelähnlichkeit allein reicht nicht. Auch ein exakter
Gesamttitel darf einen zweiten Kandidaten mit gleichem Kerntitel nicht verdecken.
Eine vorhandene, ungültige ID wird niemals über den Titel gerettet. Bei gültiger
ID ist der Titel optional; ein vorhandener Titel muss dazu passen.

## Übernahme und Ersatz

| Fall | Verhalten |
| --- | --- |
| Eindeutige Identität und gültiger Bereich | Unverändert übernehmen und chronologisch sortieren. |
| Fehlender, zusätzlicher oder umsortierter Eintrag | Keine Verschiebung anderer Identitäten. Unbekannte IDs/Titel verwerfen. |
| Wiederaufnahme desselben TOPs | Mehrere getrennte Bereiche sind erlaubt. |
| Überlappung, einschließlich exakter Duplikate | Alle beteiligten gültigen Vorschläge verwerfen, keinen nach Reihenfolge bevorzugen. |
| Fehlende, negative, zu große oder umgekehrte Grenzen | Segment verwerfen. Keine Interpolation, Kürzung oder Verschiebung. |
| Nicht ganzzahliger JSON-Index, String oder Boolean | Segment verwerfen; auch `1.0` wird nicht als Integer akzeptiert. |
| Lücke | Unzugeordnet lassen; keine Verlängerung benachbarter Segmente. |
| Fehlende/verworfene TOP-Identität | Unabhängigen Heuristikbereich nur vollständig und ohne Überlappung ergänzen; immer unsicher, Konfidenz höchstens 0,5. |
| Vollständig unbrauchbare Segmentliste | Dieselbe konservative Heuristikergänzung; ohne heuristische Evidenz bleibt alles unzugeordnet. |
| Transportfehler, leere Antwort oder ungültige JSON-Struktur | Bestehender vollständiger Heuristikfallback samt Fehlerdiagnose. |

Heuristikergänzungen betreffen nur TOPs, für die kein LLM-Segment übernommen wurde.
Es werden keine zusätzlichen Bereiche eines bereits übernommenen TOPs erfunden.
Eine ausgelassene Wiederaufnahme kann deshalb unzugeordnet bleiben.

`evidence_text` muss ein wörtlicher Ausschnitt der angegebenen Transkriptzeile
innerhalb des Segments sein. Ohne `evidence_index` wird für alte Antworten nur ein
eindeutiger Fund innerhalb des Bereichs akzeptiert. Unbelegte Angaben werden aus
den Evidenzfeldern entfernt; das Segment bleibt höchstens ein unsicherer Vorschlag
mit Konfidenz 0,5. Es wird kein vermeintlicher Beleg am Segmentanfang erfunden.

Eine sichere Übernahme benötigt zusätzlich einen eindeutigen aktuellen TOP-Aufruf
am Anfang und in der vollständigen Belegzeile, keinen erkannten widersprechenden
Aufruf innerhalb des Bereichs sowie Modellkonfidenz mindestens 0,7 und
`uncertain=false`. Ein eindeutiger Aufruf eines anderen TOPs am Anfang oder in der
Belegzeile verwirft das Segment. Schwächere oder mehrdeutige Signale markieren es
als unsicher. Die vollständige Zeile verhindert, dass ein Zitat Negationen oder
Vorschauen abschneidet. Die vorhandene Nummernprüfung bleibt zusätzlich aktiv.

## Diagnostik und Grenzen

`llm.validation_reasons` und `warnings` machen verworfene, unsichere oder ergänzte
Ergebnisse sichtbar, ohne Modelltexte in Diagnosen auszugeben. Die Codes sind
`invalid_identity`, `invalid_bounds`, `overlapping_segments`,
`unverified_evidence`, `weak_boundary_evidence`, `contradictory_evidence` und
`heuristic_supplement`. Der kompatible Strategie-Suffix `_repaired` umfasst auch
Verwerfen und Unsicherheitsmarkierung; er behauptet keine reparierte Semantik.
`llm.status` beschreibt weiterhin den Erfolg der Modellaufrufe, nicht die
inhaltliche Validität ihrer Ergebnisse.

IDs sichern die referenzierte Agendaidentität, aber keine inhaltliche Wahrheit.
Auch ein echtes Zitat beweist weder den gesamten Bereich noch eine exakte
Endgrenze. Indirekte Themenwechsel, unvollständige Transkripte und die gekürzte
LLM-Ansicht langer Transkripte können zu Fehlern führen. Die Prüfung bleibt eine
konservative, regelbasierte Plausibilitätskontrolle; manuelle Prüfung bleibt nötig.
Die strengere Bereichs- und Evidenzvalidierung hier betrifft die bekannte Agenda;
die freie TOP-Erkennung verwendet weiterhin ihren bisherigen Materialisierer.
