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
LLM-Ausfall. LLM-Ergebnisse für bekannte Agenden werden nach Titelidentität und
Transkriptposition validiert, nicht nach ihrer Listenposition. Fehlende Grenzen
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
