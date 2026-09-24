# Quellengebundene Qualitätsprüfung

Die folgenden Prüfabläufe beschreiben **Slow**, den Standardmodus. **Fast** nutzt
einen verkürzten Ablauf ohne unabhängige Inhaltsprüfung; Unterschiede und
Ergebnisstatus stehen unter [Verarbeitungsmodi](processing-modes.md).

## Verbindlicher Ablauf

`summary_grounding.py` ersetzt Stichwortfilter, Ähnlichkeitssuche und fachliche Python-Korrekturen. Jede Notiz enthält Kategorie, zeitliche Rolle (`current`, `proposal`, `retrospective`, `quoted_prior`, `unclear`) sowie originale Quellen-IDs und exakte Zitate. Die bisherigen Text- und Listenfelder sowie API-Endpunkte bleiben erhalten; Belege und Prüfmetadaten sind zusätzliche Felder.

1. Vollständige Quellzeilen werden technisch in adressierbare Zeichenabschnitte zerlegt. Keine Wörter, Seiten oder Zwischenbeiträge werden nach fachlichen Regeln ausgesiebt.
2. Zwei getrennte Modellaufrufe inventarisieren die Originalquellen. Der zweite sieht den ersten Entwurf nicht.
3. Ein unabhängiger Prüfaufruf liest jeden Originalausschnitt mit sämtlichen Entwurfsnotizen und sucht unbelegte Aussagen, falsche Quellenbezüge, Auslassungen, zeitliche Verwechslungen und Widersprüche.
4. Das Modell konsolidiert beide Inventare mit ihren wörtlichen Belegen. Danach prüfen zwei getrennte Durchgänge die tatsächlich gerenderte strukturierte Endfassung gegen alle Originalausschnitte, einschließlich fehlender Beschlüsse/Abstimmungen und Widersprüchen zwischen Notizen.
5. Beanstandungen lösen gezielte Folgeaufrufe mit Originalquellen aus. Jede Änderung führt zu erneuter vollständiger Endprüfung. Nach der konfigurierten Zahl von Klärungsrunden bleiben konkrete Prüffragen mit Quellnavigation sichtbar. Es gibt keine Freigabe anhand selbstberichteter Konfidenz.

Die Unabhängigkeit bezeichnet getrennte, beim Inventar blinde Aufrufe desselben konfigurierten Modells. Sie garantiert keine Fehlerfreiheit und ersetzt keine extern bewerteten Referenzsitzungen.

Die Modellaufrufe müssen vollständige Quellen-/Notiz-ID-Listen zurückgeben. Python prüft Schema, exakte Zitate, Referenzen, lückenlose Zeichenabdeckung und Revisionen. Eine technische Lücke, ein fehlender Prüfschritt oder Kontextüberschreitung bricht die Verarbeitung ab; kein unstrukturierter Ersatztext wird als erfolgreiches Ergebnis veröffentlicht. Erfolgreiche Teilschritte bleiben fortsetzbar. Fachlich unaufgelöste Fragen und technische Fehler haben unterschiedliche Jobzustände.

## Kontext und Konfiguration

Alle Phasen erben Modell, Provider, Thinking, Zeitlimits und Kontext aus `llm_config.py`. `SUMMARY_OUTPUT_TOKENS=4096`, `SUMMARY_MODEL_ATTEMPTS=2`, `SUMMARY_RECONCILIATION_ROUNDS=2` steuern Ausgabe, Schema-Reparaturen und fachliche Folgeprüfung. `LLM_CHUNK_CHARS` begrenzt die anfängliche Quellenzerlegung. Die alten `LLM_SUMMARY_*`-Schalter, `LLM_STRUCTURED_FALLBACK` und `LLM_REPAIR_SPLIT_DEPTH` haben keine Wirkung mehr auf Zusammenfassungen; Pflichtprüfungen lassen sich damit nicht abschalten.

Originalquellen werden über mehrere Aufrufe vollständig gelesen. Sämtliche Endnotizen müssen neben einem Originalabschnitt in den konfigurierten Kontext passen. Sehr umfangreiche Notizinventare können beim Konsolidieren oder Prüfen das Budget überschreiten: der Job schlägt dann nachvollziehbar fehl, statt Notizen oder Quellen abzuschneiden. Vor Freigabe ist deshalb ein repräsentativer Langsitzungstest erforderlich. Ein größeres Kontextfenster darf erst nach Ressourcenmessung konfiguriert werden; es gibt keine automatische Verkleinerung, andere Quantisierung oder Cloud-Ausweichverarbeitung.

Modellantworten und Schritte sind an tatsächliches Modell, Digest/Revision, Prompt, Schema, Quellen und Prüfkonfiguration gebunden. Eine Browserüberschreibung wird sichtbar angezeigt und in Jobs berücksichtigt. Eine geänderte Konfiguration beendet alte Jobs technisch erfolglos; eine neue Verarbeitung ist nötig. Externe Provider müssen für eine sichere Wiederaufnahme eine unveränderliche `LLM_MODEL_REVISION` anbieten. Ohne sie gibt es keine vertrauenswürdige Wiederaufnahme persistierter Modellschritte.

Gemeinsam mehreren TOPs zugeordnete Originalbeiträge fließen in alle betroffenen Zusammenfassungen ein. Manuelle Einzelzuordnungen haben Vorrang. Die gespeicherten Originalzeilenindizes ermöglichen Quellenzugriff im Frontend. Änderungen an Quelltext, Sprecher, Zeit, Zuordnung, Titel oder Ergebnis dürfen laufende Berechnungen nicht über eine konkurrierende manuelle Fassung schreiben.

Bestehende/manuelle Texte bleiben erhalten. Eine fachliche Übernahme durch einen Menschen ist kein neuer automatischer Prüfnachweis. Veränderte Texte/Quellen erben keine gültigen Modellbelege. Export bleibt eine Formatierung vorhandener Kategorien: er entscheidet nicht selbst über fachliche Einordnung und erhält Unsicherheiten und konkrete Prüffragen in TXT, DOCX und PDF.

## Reproduzierbare Bewertung ohne Modellaufruf

`scripts/verify_llm_snapshot.py` kontaktiert weder Produktions-API noch Modellserver und setzt keine Modell-, Kontext-, Thinking- oder Timeoutwerte. Alte Echtlauf-CLI-Optionen sind durch eine reine Offline-Prüfung ersetzt.

```sh
python scripts/verify_llm_snapshot.py --inventory testdata --output /private/quality/inventory.json
python scripts/verify_llm_snapshot.py --reference /private/reference.json \
  --candidate /private/candidate.json --adjudication /private/human-review.json \
  --output /private/quality/evaluation.json
```

Ausgabedateien werden exklusiv neu angelegt; Originale bleiben unverändert. Die Referenz und fachliche Zuordnung müssen jeweils `origin: "human"`, `approved: true` und `approved_by` enthalten. Diese Felder dokumentieren die verantwortliche Freigabe, sie beweisen nicht selbst die Urheberschaft. Modellantworten dürfen nicht als Referenz oder menschliche Bewertung deklariert werden.

Referenzvertrag:

- `source_files`: Liste `{path, sha256}`; Originalpfade relativ zur Referenzdatei. Die CLI prüft Dateihashes vor Bewertung.
- `agenda`: Liste `{id, number, section}` mit menschlich festgelegten Originalnummern und Sitzungsteilen (`public`, `nonpublic`, `null`).
- `transcript`: vollständige Liste `{id, top_ids}`; gemeinsame Beratungen können mehrere Referenz-TOPs tragen.
- `facts`: Liste `{id, kind, top_id}`; `kind` ist `decision`, `vote`, `action`, `open_point` oder `discussion`. Fachlich freigegebene Inhalte und Originalbelege zusätzlich in dieser Liste dokumentieren.

Kandidatenvertrag: dieselben `source_files`, `agenda` mit Kandidaten-IDs, `assignments` als Zeilen-ID → Liste von Kandidaten-TOP-IDs, `claims` als `{id, kind, top_id, text, ...}`, `review_questions` und `processing_complete`. Diese Form normalisiert die gespeicherten PDF-/Agenda-/Zusammenfassungsergebnisse; sie erfindet keine Referenzdaten.

Fachliche Bewertung: `reference_sha256` und `candidate_sha256` binden die Freigabe an die kanonischen JSON-Digests (`sha()` im Skript). `top_matches` bildet jede Kandidaten-TOP-ID auf eine Referenz-ID oder `null` ab. `claim_matches` bildet jede Kandidatennotiz auf belegte Referenz-Fakten-IDs ab; eine leere Liste kennzeichnet eine unbelegte Aussage. `claim_source_validity` bewertet für jede Kandidatennotiz unabhängig den Quellenbezug (`true`/`false`). `remaining_review_questions` enthält den menschlich festgestellten restlichen Prüfbedarf; die vom Modell gemeldeten Fragen werden getrennt gezählt. Jede Notiz benötigt eine menschliche Bewertung. Auslassungen werden über nicht abgedeckte Referenz-Fakten berechnet.

Ausgegeben werden TOP-Vollständigkeit, zusätzliche/doppelte TOPs, Originalnummer-/Sitzungsteilfehler, Zeilen- und Quellenzuordnungsfehler, fehlende/zusätzliche Grenzpositionen, unbelegte und fehlende Beschlüsse/Abstimmungen sowie offene Prüffragen und technische Vollständigkeit. Die Tests des Auswerters verwenden ausdrücklich synthetische Fälle und sind keine reale Qualitätsmessung.

## Aktivierung und Rückkehr

Dieser Umbau führt keine Neustarts/Deployments aus. Vor späterer Aktivierung:

1. Aktuelle Images/Commit, `.env`, SQLite einschließlich WAL mittels konsistenter SQLite-Sicherung und erhaltene Dokumente sichern; aktive Jobs auslaufen lassen.
2. Gewähltes Gemma-Modell auf einem separaten Modellserver bereitstellen. RAM-Spitzen für Modell, Thinking, nutzbaren Kontext, maximale PDF-Bilder und parallele Basisprozesse messen. Freigegebene Referenzen fachlich annotieren; lange Sitzungen, gescannte PDFs und abweichende Nummerierungen bewerten.
3. Bei erfüllter Abnahme lokal bauen und in einem geplanten Wartungsfenster Backend/Frontend aktivieren. Keine alten Arbeitsbäume in laufende Prozesse kopieren. Ältere Jobs mit anderer Konfiguration werden nicht als erfolgreich umgedeutet; neue Jobs anlegen. Provider-/Browserüberschreibungen prüfen.
4. Bei Rückkehr bisherige Images/Commit und vorherige `.env` verwenden. Die neue Zusammenfassungsstruktur ist additiv, es gibt keine destruktive Datenmigration. Falls die alte Version zusätzliche Metadaten nicht verträgt, einen konsistenten Datenbankstand nur nach Sicherung zwischenzeitlicher manueller Änderungen wiederherstellen. Keine Nutzerdaten pauschal zurücksetzen und keine Quellen/Caches löschen.
