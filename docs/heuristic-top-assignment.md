# Deterministische TOP-Zuordnung

Die Ersatzverarbeitung benötigt kein LLM. `assignment_suggestions.py` folgt den
beobachteten Aufrufen im Transkript. Die frühere Suche nach genau einer Grenze
pro TOP in Tagesordnungsreihenfolge entfällt: Sie bevorzugte frühe Erwähnungen,
setzte TOP 1 pauschal auf Zeile 0 und erfand fehlende Grenzen aus der Zeilenanzahl.

## Evidenz und Ablauf

1. Zunächst wird der Sprechakt geprüft: gegenwärtiger Aufruf, mögliche
   TOP-Überschrift, bloße Erwähnung/Vorschau/Rückverweis, Beendigung oder gemischte
   Aussage. Die Regeln beschreiben gemeinsame sprachliche Merkmale (Präsens,
   Negation, Zeitbezug, Modalität, Zitate), keine besonderen Haushalt-Fälle.
   Eine TOP-Nummer allein innerhalb einer Aussage ist kein Aufruf.
2. Erst danach wird die Identität anhand der Originalnummer einschließlich
   Abschnitt oder des Titelabgleichs bestimmt. Listenpositionen sind niemals
   TOP-Nummern. Bei unnummerierten Agenden kann ein eindeutiger Titel den TOP
   identifizieren; die im Transkript genannte Nummer wird damit nicht bestätigt.
   Konkurrierende Titel, widersprüchliche Nummern/Titel und mehrdeutige Nummern
   begründen keine sichere Grenze.
3. Ein eindeutiger Aufruf eröffnet ein Segment. Es läuft bis zum nächsten Aufruf,
   Abbruch oder konkurrierenden Hinweis. Ein unbekannter/mehrdeutiger Aufruf
   unterbricht auch die Fortführung des bisherigen TOPs. Ein Stichworttreffer
   allein ergibt höchstens einen unsicheren Vorschlag für diese einzelne Zeile.
   Fehlende Evidenz wird weder auf TOP 1 noch auf einen erwarteten Folgetop verteilt.

Die Bewertung belohnt weder frühe Fundstellen noch häufige Sprecher. Titelabgleich
misst die Abdeckung der Titelbegriffe; bei fast gleichwertigen Kandidaten wird
keiner gewählt. Die Werte sind feste Evidenzstufen, keine statistisch kalibrierten
Wahrscheinlichkeiten: 0,90 für einen aktuellen nummerierten Aufruf, 0,80 für einen
Titelaufruf bzw. eine passende bekannte Überschrift, 0,75 für Titelauflösung bei
nicht hinterlegter Nummer und 0,50 für lokale Stichworttreffer. Ohne bekannte
Agenda bleiben bloße Überschriften mit höchstens 0,65 unsicher.

Im gemeldeten Fünf-Zeilen-Beispiel ergibt sich für `Begrüßung, Haushalt`:
`[null, null, null, 1, 1]`. Der Haushalt beginnt erst in Zeile 4; die reine
Sitzungseröffnung wird nicht automatisch zum TOP „Begrüßung“ erklärt.

## Datenmodell und Integration

`assignments` enthält pro Zeile einen TOP-Index oder `null`; `segments` kann
mehrere getrennte Bereiche mit demselben Index enthalten. Damit sind vorgezogene,
ausgelassene und explizit wiederaufgenommene TOPs ohne Schemaänderung möglich.
Gleichzeitige Beratung mehrerer TOPs in einer Zeile lässt sich damit nicht
verlustfrei abbilden und bleibt bei mehrdeutiger Evidenz unzugeordnet.

`agenda_detection.py` übernimmt die heuristischen Bereiche unverändert, auch bei
LLM-Ausfall. LLM-Ergebnisse für bekannte Agenden werden über aufruflokale IDs
(bei älteren Antworten über eindeutige Titel) und Transkriptposition validiert.
Die Regeln für Verwerfen, Evidenz und unsichere Heuristikergänzungen stehen in
[LLM-Segmentvalidierung](llm-segmentvalidierung.md). Fehlende Grenzen
werden nicht interpoliert; widersprüchliche Sprechakte können durch eine hohe
LLM-Confidence nicht zu sicheren Aufrufen werden. Ohne bekannte Agenda werden
identische wiederholte Titel zusammengeführt. Der zusätzliche Pipeline-Fallback
in `main.py` erhält bei bekannten TOPs ebenfalls offene Zuordnungen.

Die Review-Anzeige zeigt offene Zeilen und TOPs ohne Segmentnachweis zusätzlich
zu unsicheren Segmenten. `uncertain_count` zählt weiterhin nur Segmente, nicht
Lücken. „Alle sicheren übernehmen“ lässt unsichere Vorschläge und offene Zeilen
unverändert. Fehlender Segmentnachweis beweist weder eine Absetzung noch, dass ein
TOP tatsächlich nicht behandelt wurde.

## Verbleibende Grenzen

- Die Erkennung ist sprachlich begrenzt: freie Umschreibungen, Transkriptionsfehler,
  indirekte Rede ohne erkennbare Marker und komplexe Negationsbezüge bleiben schwierig.
  Ganze Zeilen werden konservativ bewertet; eine beiläufige Negation kann dadurch
  auch einen echten Aufruf verdecken. Gemischte Aufrufe erfordern ggf. manuelles Teilen.
- Ein erkannter Aufruf begründet die Fortführung bis zum nächsten erkannten Signal.
  Unangekündigte Themenwechsel können deshalb innerhalb eines Segments übersehen
  werden. Der Evidenzwert gilt für den Anfang, nicht für jede folgende Zeile.
- Ohne bekannte Agenda sind Titel aus dem Wortlaut abgeleitet. Unterschiedliche
  Benennungen desselben TOPs werden nicht semantisch zusammengeführt; wiederholte
  Nummern mit unterschiedlichen Titeln/Abschnitten können unauflösbar bleiben.
- Das Datenmodell hat keinen eigenen Status „abgesetzt“, „vertagt“ oder
  „abschließend behandelt“. Aufrufe und Unterbrechungen können sichtbar werden,
  ein vollständiges Sitzungsablaufmodell wird daraus nicht rekonstruiert.

Kontrasttests prüfen echte Aufrufe gegen Vorschauen, Negationen, Rückverweise,
Zitate, Bedingungen und Fragen. Weitere Tests prüfen lokale Stichwortevidenz,
lange Abstände, Lücken, andere Reihenfolgen, Wiederaufnahmen, Konflikte sowie
LLM-Ausfälle, API/Pipeline und das Übernehmen von Vorschlägen in der Review-Anzeige.

## Lebenszyklus im Zuordnungsschritt

Automatische Vorschläge werden als `agenda_proposals` zusammen mit der Sitzung
und im lokalen Entwurf gespeichert. Der versionierte Datensatz enthält den
unveränderten Erkennungsstand (`result`, einschließlich Unsicherheiten,
Evidenz und Warnungen) sowie dessen Eingaben (`source`): geordnete TOP-Titel und
stabile TOP-IDs sowie Transkriptzeilen mit stabilen IDs, Sprecher, Text und Zeiten.
Die automatischen Zuordnungen bleiben von den aktuell bearbeiteten Zuordnungen
getrennt. Speichern übernimmt sie nicht erneut.

Der Editor erlaubt die Übernahme nur bei identischen Eingaben und gültigen
Indexbereichen. Einfügen, Löschen, Zusammenlegen und Umbenennen von TOPs sowie
Änderungen an Transkripttext, Zeilenstruktur, Sprechern oder Zeiten sperren die
bisherigen Vorschläge. Reine Änderungen der Sprecher-Anzeigenamen oder manuelle
Zeilenzuordnungen lassen die Vorschläge gültig. Ein exaktes Zurücksetzen der
Eingaben einschließlich ihrer IDs macht den ursprünglichen Stand wieder gültig.
Alte Evidenz und Unsicherheiten bleiben bei einer Sperre sichtbar; die früheren
Zeilennummern werden nicht mehr zur Markierung aktueller Transkriptzeilen genutzt.

„TOP-Erkennung erneut berechnen“ startet genau einen expliziten Aufruf für die
vorhandenen TOPs. Die serverseitige LLM-Konfiguration und das ausgewählte Modell
gelten weiterhin. Beim Öffnen, Bearbeiten oder Speichern wird keine Erkennung
angestoßen. Der Aufruf nutzt `preserve_transcript_structure: true`: Bereits
bearbeitete Zeilen werden nicht erneut geteilt. Übergänge innerhalb einer solchen
Zeile können deshalb nur auf Zeilenebene vorgeschlagen werden. Aufteilen ist
weiterhin manuell möglich; danach kann erneut berechnet werden.

Neue Ergebnisse ersetzen nur die Vorschläge, niemals aktuelle Zuordnungen.
Zwischenzeitliche Änderungen der Eingaben oder ein Sitzungswechsel verwerfen
verspätete Ergebnisse. Manuelle Zuordnungsänderungen während der Erkennung bleiben
erhalten. Anschließend ist die bewusste Übernahme einzelner, sicherer oder aller
gültigen Vorschläge weiterhin möglich und ersetzt die entsprechenden manuellen
Zuordnungen. Fehler lassen den bisherigen Vorschlagsstand erhalten.

Die SQLite-Migration ergänzt `sessions.agenda_proposals_json` ohne bestehende
Daten zu verändern. Vorschläge werden in derselben Transaktion und unter derselben
Revisionsprüfung wie die übrigen Sitzungsdaten gespeichert. Ältere Clients, die
das Feld auslassen, behalten den vorhandenen Vorschlagsstand. Alte Pipeline-Artefakte
ohne unveränderlichen Quellenstand werden nur als historische, gesperrte Evidenz
angezeigt (`source: null`); ihre Indizes werden nie an die inzwischen bearbeitete
Sitzung gebunden. Ohne vorhandene Evidenz weist die Oberfläche auf unbekannte
Unsicherheiten hin. Eine explizite Neuberechnung ersetzt diesen historischen Stand.
