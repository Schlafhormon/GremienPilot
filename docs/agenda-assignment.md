# Modellgestützte Agenda und Transkriptzuordnung

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
