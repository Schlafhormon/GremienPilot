# Modellgestützte Agenda und Transkriptzuordnung

Die folgenden Prüfabläufe beschreiben **Slow**, den Standardmodus. **Fast** nutzt
einen verkürzten Ablauf ohne unabhängige Inhaltsprüfung; Unterschiede und
Ergebnisstatus stehen unter [Verarbeitungsmodi](processing-modes.md).

## Erneute Verarbeitung im Editor

Zwei getrennte Aktionen verwenden den gewählten Sitzungsmodus:

- **TOPs aus PDF neu extrahieren** wertet die gespeicherte Einladung erneut aus.
  Die neue Liste erscheint zur Übernahme. Eindeutig unveränderte TOPs behalten
  dabei ihre IDs, Zuordnungen und Zusammenfassungen.
- **TOP-Zuordnung neu berechnen** erzeugt frische Zuordnungsvorschläge aus der
  aktuellen TOP-Liste und dem vollständigen Transkript. Manuelle Zuordnungen
  bleiben bis zur Übernahme erhalten.

Beide Aktionen haben eigene Statusanzeigen mit Phase, Laufzeit, verfügbarem
Fortschritt und Abbruch. Aufträge sind an die Sitzung gebunden und werden beim
erneuten Öffnen wieder angezeigt. Eine unbekannte Restdauer wird nicht als
geschätzter Prozentwert ausgegeben. Ohne gespeicherte Einladung ist die
PDF-Aktion deaktiviert. Fast erhält keine zusätzlichen Modellprüfungen.

Die Hintergrundaufträge laufen über `/api/sessions/{session_id}/pdf-jobs`
und `/api/agenda-detection/jobs`; ihr Status über `/api/model-jobs/{job_id}`.
Die frühere Browsergrenze für synchrone Modellanfragen gilt nicht für die
Zuordnung vollständiger Sitzungen über diese Hintergrundverarbeitung.

## Modellablauf

Bekannte Agenden und Erkennung ohne Einladung verwenden denselben Ablauf.
Zwei getrennte Modellaufrufe ermitteln zuerst belegbare Punkte bzw. zusätzliche
Punkte einer bekannten Agenda; Unterschiede werden anhand der Quellen
modellgestützt geklärt. Bestehende TOPs bleiben mit ihren Identitäten erhalten.
Originalnummern werden nur aus Modellnachweisen übernommen, nie aus Positionen.

Danach rekonstruieren zwei unabhängige Durchgänge den Sitzungsverlauf und den
Status jedes TOPs (`treated`, `deferred`, `removed`, `not_evidenced`). Beide
ordnen jede Transkriptzeile zu. Der zweite Durchgang sieht keine Zuordnungen
des ersten. Sämtliche abweichenden oder unsicheren Entscheidungen erhalten
eine begrenzte Klärung gegen die Originalquellen. Fachlich offene Ergebnisse
bleiben offen; technisch fehlgeschlagene Prüfungen heißen `technical_pending`.

## Quellen, Kontext und technische Verträge

Persistierte TOP-IDs und Transkript-IDs bleiben unverändert. Alte Aufrufer ohne
IDs erhalten deterministische, an den Quellstand gebundene Transkriptverweise.
Die API bindet fehlende IDs vor dem Speichern eines Jobs. Zeitwerte werden
übernommen, niemals vom Modell neu erzeugt. Geprüft werden Identitäten,
Abdeckung, Bereiche, Originalzitate und Quellenzeiten. Sprachmuster begrenzen
weder Schema-IDs noch Antworten und erzeugen keine Ergänzungen.

Wenn möglich enthält jeder Detailauftrag das vollständige Transkript und die
Agenda. Größere Quellen werden in zwei unabhängigen Lesedurchgängen vollständig
modellgestützt verdichtet. Die gesamte Quellenabdeckung und alle verdichteten
Knoten bleiben im Prüfprotokoll erhalten. Detailaufträge behalten den globalen
Kontext und können weitere Originalbereiche anfordern. Kontext, Schema,
Thinking-Reserve und Ausgabebudget bestimmen die Aufteilung. Ein nicht mehr
passender Einzelauftrag oder eine erfolglose Verdichtung erzeugt einen
sichtbaren technischen Fehler; Quellen werden nicht still gekürzt.

Verdichtung kann fachliche Details verlieren. Quellenzitate und unabhängige
Modelldurchgänge reduzieren dieses Risiko, beweisen aber keine Richtigkeit.

## Kompatible Speicherung und Darstellung

`llm.line_results` enthält je `line_id` den Status `assigned`, `unassigned`
(fachlich begründet) oder `not_processed` (technisch), `top_ids`, Begründung,
Originalbelege und `review_status`. Mehrere IDs bedeuten gemeinsame Beratung.
Die bisherige skalare `assignments`-Liste bildet nur Einzelzuordnungen ab;
gemeinsame Beratungen erhalten dort `null` und keinen willkürlichen Einzel-TOP.
Die Oberfläche zeigt die vollständigen Entscheidungen gesondert. Bestehende
Einzel-TOP-Zusammenfassungen verwenden Mehrfachvorschläge noch nicht; der Job
weist darauf ausdrücklich hin. Übernahme
von Einzelvorschlägen bleibt eine explizite Nutzeraktion.

`llm.agenda_states`, `reconstructions`, `processing_complete`, `review_complete`
und `review_required` ergänzen die bestehende API. Sie werden im vorhandenen
`agenda_proposals_json` gespeichert; Altdaten benötigen keine Umschreibung.
Manuelle Zuordnungen und gespeicherte Vorschläge sind getrennt. Quellenänderungen
sperren die Übernahme alter Vorschläge; Revisionen und Job-Leases verhindern
überholte automatische Veröffentlichungen.

Erfolgreiche Modellschritte werden einzeln im persistenten Job checkpointed.
Abbruch greift auch während Modellaufrufen. Fehlerantworten werden nicht als
erfolgreiche Teilschritte gecacht. `fresh` trennt neue Antworten vom bisherigen
Cache. Konfiguration und Codeversion gehören zum Job- und Cachevertrag.

Technische Vollständigkeit bedeutet vollständige Verarbeitung bzw. Prüfung.
Modellübereinstimmung ist **kein nachgewiesener fachlicher Qualitätsmaßstab**.

## Kompakter Quellenvertrag ab September 2026

Fast und optional kompaktes Slow verwenden `agenda-end-sources-v3`. Jede
Zuordnung enthält nur `end_line_id` und die fachlichen Felder `top_ids`,
`reason`, `evidence`, `uncertain`, `confidence`. Ein Abschnitt gilt ausdrücklich
für jede noch nicht zugeordnete Zielzeile bis einschließlich seiner Endquelle.
Die Anwendung berechnet den Anfang aus der Quellenreihenfolge. Endquellen
müssen im Zielblock liegen und streng aufsteigen; die letzte Endquelle muss
explizit die letzte Zielzeile sein. Ein fehlendes Ende wird niemals ergänzt.
Alte widersprüchliche Start-/Endantworten werden nicht in dieses Format repariert.

`response` ist entweder `{kind: "assignments", spans: [...]}` oder
`{kind: "source_request", source_window_ids: [...]}`. Beide Varianten sind in
Schema und Validator getrennt. Quellenfenster sind technische Adressen von
höchstens 80 Originalzeilen, keine Themenabschnitte. Nur Fenster mit noch nicht
bereitgestellten Originalzeilen werden angeboten. Originale aus dem Zielblock,
dem vollständigen Kontext und vollständigen Belegzitaten gelten bereits als
vorhanden. Weitere Abrufe liefern nur fehlende Originale. Doppelte oder unbekannte
Anforderungen sind Fehler; ein ausgeschöpftes Abrufbudget erzeugt keine neue
Aufteilung in Modellaufträge.

Fast begrenzt Abschnittsbegründungen auf 240 Zeichen und drei Beleg-IDs. Es
bleibt bei einem Zuordnungsdurchgang ohne automatische Inhaltsprüfung und ohne
Reparatur fehlerhafter Antworten. Slow behält unabhängige Prüfung und Klärung.
Prompts enthalten weniger technische Auditmetadaten; Originaltexte, Belege,
fachliche Fragen und Widerspruchsstatus bleiben erhalten. Archive bleiben vollständig.

Die öffentliche zeilenweise API und gespeicherte Vorschläge ändern sich nicht.
Manuelle Zuordnungen, Quellen-IDs und Originalzeiten bleiben unverändert.
Neue Code-/Promptversionen verwenden neue Checkpoints; bestehende Ergebnisse
bleiben lesbar und werden nicht automatisch neu interpretiert oder übernommen.
