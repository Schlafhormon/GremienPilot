# Modellgestützte PDF-Auswertung

Die folgenden Prüfabläufe beschreiben **Slow**, den Standardmodus. **Fast** nutzt
einen verkürzten Ablauf ohne unabhängige Inhaltsprüfung; Unterschiede und
Ergebnisstatus stehen unter [Verarbeitungsmodi](processing-modes.md).

Der Browser sendet die tatsächliche Datei als Multipart-Feld `pdf`. Ein aktivierter
PDF-Modus ohne Datei wird vor dem Pipeline-Start abgewiesen (auch im Backend).
Manuelle TOPs haben weiterhin Vorrang. Ein angehängtes PDF ohne manuelle TOPs
wird auch bei ausgeschaltetem Automatik-Flag ausgewertet; ein fehlgeschlagener
Sofort-Upload darf nicht still in die Transkript-Erkennung wechseln. Ohne PDF ist Transkriptverarbeitung nach
Ausschalten der PDF-Erkennung möglich. Ein beschädigtes, fehlendes oder nicht
vollständig geprüftes vorgesehenes PDF löst niemals eine Transkript-Ersatzagenda aus.

Der Upload bleibt unter einem zufälligen Jobpfad erhalten. Der dauerhafte Job
speichert seine SHA-256-Identität; vor Wiederaufnahme und Quellenabruf wird diese
geprüft. Alle Seiten werden zunächst registriert. Pro Seite werden ein PNG mit
konfigurierter Auflösung, dessen Hash, Abmessungen und der unveränderte Textlayer
als Checkpoint gespeichert. Fehlender Text bei Scans ist zulässig. Ein Fehler beim
Textauslesen wird festgehalten; ein Renderfehler ist ein Verarbeitungsfehler.
Es gibt keine OCR-Wortlisten oder fachlichen Textkorrekturen.

Jede Seite erhält einen eigenen multimodalen Modellaufruf mit verbindlichem
JSON-Schema. Danach führt das Modell die Seiteninventare einschließlich Fortsetzungen
zusammen. TOPs und Abschnittsüberschriften haben getrennte Arten, IDs, Originalnummer
(String oder null), Titel, Sitzungsteil, Eltern-ID und Seitenbelege. Sitzungstag,
Uhrzeit, Gremium, Ort und Titel haben ebenfalls Seitenbelege. Technische Validatoren
prüfen ausschließlich Typen, Referenzen, Seitenabdeckung, ID-Eindeutigkeit und Zyklen.
Sie vergleichen keine Titel/Zitate mit dem möglicherweise beschädigten Textlayer.

Ein separater Modellaufruf je Originalseite prüft das Gesamtergebnis visuell ohne
Extraktionsdialog. Er meldet fehlende/doppelte Einträge, Unterordnung, Nummerierung,
Abschnitte, Fortsetzungen und Metadatenwidersprüche. Befunde führen zu gezielten
Modellreparaturen mit den betroffenen Originalbildern. Danach werden **alle** Seiten
neu geprüft. Nur ein nichtleeres Ergebnis ohne offene Befunde wird übernommen.
Die unabhängigen Aufrufe verwenden derzeit dieselbe konfigurierte Modellinstanz;
das ist keine Unabhängigkeit unterschiedlicher Modellfamilien.

Alle Modellantworten (auch ungültige), reparierten Kandidaten und Seitenprüfungen
bleiben in `durable_steps`. Nach Prozessunterbrechung nutzt der Job vollständige
Checkpoints; Abbruch wird vor Seiten und während Modelltransport geprüft. Bei
veränderter Modell-/Code-/PDF-Konfiguration verweigert die vorhandene Jobversionierung
eine gemischte Wiederaufnahme. Dann ist ein neuer Lauf erforderlich. Nach ausgeschöpften
Reparaturbudgets endet der Job sichtbar fehlerhaft; das Original bleibt erhalten.
Die Oberfläche kann eine gespeicherte eigenständige PDF-Auswertung erneut öffnen.

Die bestehenden `tops`-Strings und Metadatenfelder bleiben als API-Projektion bestehen;
zusätzliche Felder sind `items`, `metadata_sources`, `document`, `pages`, `audits`,
`processing_complete` und `review_required`. Alte unstrukturierte Modellantworten
werden nicht mehr durch Heuristiken interpretiert. Legacy-Textaufrufe nutzen dasselbe
Schema, können aber keine visuelle Vollständigkeit bescheinigen.

Geprüfte IDs werden bei unveränderter Agenda zu Sitzungs-TOP-IDs. Die ursprüngliche
PDF-Auswertung wird in `agenda_proposals.source.pdf_extraction` erhalten und unabhängig
von manuellen Änderungen angezeigt. Wiederverwendung einer eigenständigen Auswertung
nutzt `pdf_source_job_id`; nur ein abgeschlossener, geprüfter Serverjob mit passender
Dokumentidentität ist zulässig. Sitzungsübernahme verwendet die bestehende atomare
Revision-/Lease-Prüfung. Quellenabruf erfolgt über Job-ID und Hash, niemals über einen
vom Client gelieferten Dateipfad. Die vorhandenen Zugriffsregeln der Anwendung gelten
auch für diese Route. PDF- und Job-Daten gehören in private, gesicherte Datenvolumes.

## Grenzen und Qualitätsprüfung

Die neuen `PDF_*`-Optionen stehen im README-Konfigurationsabschnitt und beiden
Umgebungsbeispielen. Auflösung, Pixel-/Seitenlimits, Ausgabebudget und Reparaturanzahl
sind hostunabhängig. Vision und `LLM_IMAGE_TOKENS` müssen zum betriebenen Modell passen.
Es gibt keine automatische Bildverkleinerung oder Seitenkürzung. Sehr große Inventare
oder viele widersprüchliche Seiten können das konfigurierte Kontextbudget überschreiten;
der Lauf schlägt dann fehl. Die lokale CPU-Konfiguration wird nicht verändert.

Tests erzeugen ausschließlich synthetische digitale, gescannte und gemischte PDFs.
Sie prüfen echten Textgewinn/Rendering sowie Schema, vollständigen Bildtransport,
Reparatursteuerung, Quellen, fehlende Uploads, Abbruch, Wiederaufnahme und Revisionen
mit simulierten Modellantworten. **Das ist kein Qualitätsnachweis des Modells.**
Vor Freigabe sind echte Aufrufe gegen die gewählte Vision-Modellrevision nötig:

1. Freigegebene Einladungen mit handgeprüfter Referenzagenda und Metadaten verwenden;
   digitale, gescannte, gemischte und beschädigte Textlayer einschließen.
2. Kleine Schrift, Tabellen, Seitenumbrüche, Unterpunkte, kurze Titel, führende Nullen
   und wiederholte Nummern in unterschiedlichen Sitzungsteilen prüfen.
3. Fehlende/zusätzliche TOPs, falsche Quellen, Hierarchie, Datum/Uhrzeit und Fehlbestätigungen
   der Nachprüfung getrennt erfassen. Auch bewusst fehlerhafte Kandidaten prüfen.
4. Gewählte Modellrevision, Auflösung, Kontext-/Bildbudget, Befunde und Laufzeiten
   dokumentieren. CPU-Messungen nur in einem ausdrücklich freigegebenen Wartungsfenster
   oder einer isolierten Instanz durchführen; keine produktiven Nutzerjobs verdrängen.

Für diese Änderung wurden keine Produktivmodelle aufgerufen oder Dienste neu gestartet.
