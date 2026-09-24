# Fast und Slow

Der Schalter vor dem Upload gilt für eine einzelne Sitzung. Neue und bestehende
Sitzungen ohne gespeicherte Auswahl verwenden `slow`. Die Auswahl liegt in SQLite
(`sessions.processing_mode`), im Sitzungsentwurf des Browsers und in den Eingaben
jedes gestarteten Jobs. Sie wird beim Wiederöffnen wiederhergestellt und ist während
laufender Verarbeitung gesperrt. Neue Sitzungen starten unabhängig von der letzten
Auswahl mit Slow.

Ein späterer Wechsel steuert neue Berechnungen, einschließlich TOP-Neuzuordnung und
selektiver Zusammenfassungen. Bestehende Texte, manuelle Änderungen und ihr ursprünglicher
Prüfstand bleiben erhalten. Ein Fast-PDF kann keine geprüfte PDF-Agenda für einen
neuen Slow-Lauf liefern; es muss in Slow erneut ausgewertet werden. Ein geprüftes
Slow-PDF darf dagegen in Fast wiederverwendet werden.

## Abläufe

| Bereich | Slow | Fast |
| --- | --- | --- |
| PDF | Bilder und Text aller Seiten; Inventare, Zusammenführung, unabhängige Seiten- und Beziehungsprüfungen, gezielte Korrekturen | Textlayer bevorzugt; Seiten ohne Text als Bild; Seiten gemeinsam bis zum Kontextlimit auswerten, mehrere Inventare einmal zusammenführen; keine Audits oder Reparaturaufrufe |
| TOP-Zuordnung | Zwei unabhängige Rekonstruktionen und Detailzuordnungen, Klärung von Abweichungen | Ein Durchlauf für Agenda, Verlauf, TOP-Status und kompakte Zuordnungsbereiche; keine unabhängige zweite Bewertung oder Klärung |
| Zusammenfassung | Zwei Inventare, Entwurfsprüfung, Konsolidierung, zwei Abschlussprüfungen und begrenzte Korrekturen | Eine Generierung pro passendem Quellblock; bei mehreren Blöcken höchstens eine Konsolidierung, sofern sie ins Kontextbudget passt; sonst alle Blockergebnisse erhalten |

Fast verwendet einen Versuch pro fachlichem Modellauftrag; ungültige Antworten
werden nicht durch weitere fachliche Reparaturaufrufe korrigiert. Begrenzte
Transportwiederholungen, Abbruch, technische Schema-/ID-/Quellenkontrollen und
die kontextabhängige Aufteilung bleiben aktiv. Es werden keine Seiten oder
Transkriptteile zur Beschleunigung abgeschnitten. Leere oder beschädigte
PDFs werden nicht durch eine erfundene Agenda ersetzt.

Modell, Thinking-Einstellungen, Tokenbudgets, WhisperX und Sprecherverarbeitung
bleiben unverändert. Die zentrale serielle Warteschlange bleibt ebenfalls erhalten:
Fast bedeutet weniger Arbeit im Job, keine höhere Priorität gegenüber laufenden Jobs.
Laufzeit und Qualität müssen für das eingesetzte Modell mit Referenzsitzungen gemessen
werden; die automatisierten Tests verwenden synthetische Quellen und Modellantworten.

## API und Persistenz

- `POST /api/pipeline/start`: Multipart-Feld `processing_mode` oder
  `options.processing_mode`; das explizite Feld hat Vorrang. Ohne Angabe gilt
  der Modus der vorhandenen Sitzung, ansonsten `slow`.
- `POST /api/extract-tops` und `/api/extract-tops/jobs`: Multipart-Feld
  `processing_mode`, Standard `slow`.
- `POST /api/agenda-detection`, `/api/agenda-detection/jobs`,
  `/api/assignment-suggestions` und `/api/summarize`: JSON-Feld `processing_mode`.
- Sitzungsantworten und Verlauf enthalten `processing_mode`. Sitzungen lassen
  sich über die bestehenden Save-Endpunkte umstellen. Alte Clients, die das Feld
  beim Speichern auslassen, behalten die bestehende Auswahl.
- Zusammenfassungsjobs übernehmen den Sitzungsmodus in ihre unveränderlichen
  Eingaben; sie lesen ihn während der Ausführung nicht erneut aus der Sitzung.

Erlaubt sind ausschließlich `fast` und `slow`. Die SQLite-Migration ergänzt das Feld
mit `slow`, ohne vorhandene Ergebnisse umzuschreiben. `processing_mode.py` enthält
die pro Aufruf isolierte Policy; globale Umgebungsvariablen werden nicht verändert.
Modus und Policyversion gehören zu den Jobversionen und zu den Modellcache-Policies.
PDF-Checkpoints trennen `fast-extraction-v1` vom geprüften `page-evidence-v3`-Vertrag.
Die Wiederaufnahme nutzt den ursprünglichen Modus. Geänderte Code-/Policyversionen
bleiben wie bisher ein Grund, einen neuen Job zu starten.

## Ergebnisstatus und Bearbeitung

Ein vollständiges Fast-Ergebnis trägt `processing_complete: true`,
`processing_mode: "fast"` und `review_status: "skipped"`. TOP-Zuordnungen und
Zusammenfassungen behalten `review_complete: false`; `review_required` bleibt wahr.
Der dauerhafte Job darf damit als `review_required` enden, während die Pipeline
technisch erfolgreich ist. Fehlende Quellenabdeckung oder fehlerhafte Modellantworten
bleiben technische Fehler. Fast setzt keine Prüfkennzeichen künstlich auf erfolgreich.

Die Oberfläche zeigt „Fast – ohne automatische Inhaltsprüfung“. Erfolgreiche
Fast-Pipelines führen direkt zum bearbeitbaren Protokoll. Quellen bleiben zugänglich;
manuelle Änderungen entfernen nicht mehr passende Belege und bleiben als manuell
bearbeitet dokumentiert. Auch ein später auf Slow gestellter Sitzungsschalter macht
ältere Fast-Ergebnisse nicht zu geprüften Ergebnissen. Exporte mit Fast-Zusammenfassungen
enthalten die Kennzeichnung im Titel.

Die Tests in `app/backend/tests/test_processing_modes.py` prüfen insbesondere
Modellisolation, Text-/Scan-PDFs, ausbleibende Review-Aufrufe, Quellenabdeckung,
Migration, Cachetrennung, Jobwiederaufnahme, PDF-Wiederverwendung und selektive
Neugenerierung. Frontendtests prüfen Weitergabe, Speicherung, Wiederherstellung,
Sperrung und die Anzeige ungeprüfter Ergebnisse.
