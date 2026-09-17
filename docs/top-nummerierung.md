# TOP-Nummerierung

## Datenmodell und Kompatibilität

- `top_ids` (SQLite: `top_uid`) bleiben stabile Identitäten, unabhängig von
  Nummer, Titel und Reihenfolge.
- Originalnummern sind optionale Strings, keine Listenpositionen. Beim Vergleich
  gilt `02.01` als `2.1`, aber `2.10` bleibt verschieden von `2.1`.
- Die sichtbare Bezeichnung enthält Nummer, Titel und gegebenenfalls
  `[Öffentlich]` oder `[Nichtöffentlich]`. Der Inhaltsabgleich nutzt den reinen Titel.

API und Persistenz behalten `tops: string[]` und `top_ids`; eine Migration ist
nicht nötig. `agenda_labels.AgendaLabel` trennt die Angaben intern. Frontend und
Export ergänzen keine Positionsnummern. Alte Sitzungen bleiben lesbar; verlorene
Nummern werden nicht aus der Reihenfolge rekonstruiert. Manuelle Ergänzungen im
Titelfeld ändern die ID nicht. Beginnt ein Titel selbst mit einer Zahl, empfiehlt
sich eine eindeutige Formulierung wie `Haushalt 2026` statt `2026 Haushalt`.

Die interne PDF-LLM-Antwort verwendet `number` (String oder null), `title` und
`section` (`public`, `nonpublic` oder null). Alte String-Arrays und Standardlisten
bleiben akzeptiert. Der PDF-Fallback kann Nummern bei exakt und eindeutig gleichem
Titel ergänzen; enthält er mehr Einträge, wird seine Liste übernommen.

## Unterstützte Schreibweisen

Listen: `2. Titel`, `2) Titel`, `02 Titel`, `TOP 2: Titel`, `2.1 Titel`,
`2.1. Titel`, tiefere Hierarchien sowie `IV. Titel` und `a) Titel`.
PDF-Abschnittsüberschriften gelten für die folgenden TOPs und sind keine eigenen
Einträge. Unnummerierte Bullet-Unterpunkte bleiben ausgeschlossen.

Explizite Transkriptverweise beginnen mit `TOP`, `Tagesordnungspunkt` oder `Punkt`,
gefolgt von einer vollständigen Nummer oder einem deutschen Kardinalzahlwort
von null bis neunundneunzig, einschließlich `ue`/`oe`/`ss`-Umschriften.
Zahlwörter werden als arabische Zahlen angezeigt; geschriebene Nummern behalten
führende Nullen.

## Mehrdeutigkeit und Grenzen

`TOP 2` trifft weder `2.1` noch `2.2`. Wiederholte Nummern benötigen einen
eindeutigen Abschnittsbezug in derselben Äußerung. „Nichtöffentlich“,
„nicht öffentlich“ und „nicht-öffentlich“ sind gleichwertig. Unbekannte Abschnitte,
doppelte Nummern innerhalb eines Abschnitts, mehrere TOP-Verweise und Zahlenbereiche
können mehrdeutig bleiben. Abschnittswechsel werden nicht aus früheren Äußerungen
fortgeschrieben.

Mehrdeutige Verweise werden auch durch Titelabgleich oder LLM-Verfeinerung nicht
zu sicheren Treffern. Geschätzte Grenzen bleiben mit `uncertain=true` markiert.
Die Segmentierung nimmt eine geordnete Behandlung der TOPs an; Rückverweise können
wie Ankündigungen aussehen. PDF-Layout und Modellantworten können unvollständig sein.

Bewusst keine explizite Nummernauflösung für `2a`, `2/1`, `2,1`, `2 . 1`,
`zwei Punkt eins`, Ordinalzahlen (`zweiter`), römische oder Buchstabenverweise,
Zahlwörter ab hundert und Dialektformen (`zwo`). Numerische Sonderformen werden
nicht auf die führende Zahl verkürzt. Inhaltsabgleich und manuelle Zuordnung
bleiben verfügbar.

## Verifikation

- Backend: **340 Tests bestanden**, zwei wegen fehlender optionaler PDF-Datei
  übersprungen. Abgedeckt sind auch ein erzeugtes PDF, API-/SQLite-Rundreisen,
  stabile IDs und Mehrdeutigkeit mit und ohne LLM.
- Frontend: **78 bestanden**, drei bestehende Fehler in `AssignmentStep.test.tsx`
  am unveränderten Ausgangsstand bestätigt (Transkriptbearbeitung und Sprecherzusammenführung).
- Produktionsbuild und `git diff --check` erfolgreich; ESLint ohne Fehler,
  mit einer bestehenden Hook-Warnung in `App.tsx`.
