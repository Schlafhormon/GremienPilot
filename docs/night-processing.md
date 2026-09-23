# Nächtliche Verarbeitung und Wiederaufnahme

## PDF-Prüfvertrag `page-evidence-v3`

Die Seiteninventare und die Zusammenführung bleiben quellengebundene Entwürfe. Ein
Ergebnis ist erst bestätigt, wenn alle Originalseiten und die Zusammenhänge
unabhängig geprüft sind. Alte `pdf:v2:*:review:*`-Checkpoints sind keine Nachweise
für den neuen Vertrag.

Eine Seitenprüfung erhält ein Originalbild, dessen unveränderten Textlayer und
nur die Kandidatenteile mit Quellen auf dieser Seite. Mehrseitige Einträge
erscheinen als lokale Quellenfragmente. Ihre vollständige Formulierung sowie
Elternbeziehungen, Sitzungsteile und Metadaten werden nicht lokal beurteilt.
Auch eine Seite mit leerer Kandidatenprojektion wird vollständig auf Auslassungen
geprüft. Der unabhängige Zusammenhangsprüfer erhält den vollständigen Kandidaten
und alle Originalseiten: Fortsetzungen, Hierarchie, echte Dubletten, Metadaten,
falsche Quellen und Widersprüche müssen dort gegen die Bilder bestehen.

Befunde enthalten `kind`, `item_ids`, `metadata_fields`, `pages`, `evidence` und
eine konkrete Prüffrage in `description`. Unbekannte IDs und ungesehene Seiten
werden technisch zurückgewiesen. Eine Reparatur darf nur beanstandete Einträge
ändern/löschen, fehlende Einträge an einer expliziten Position ergänzen und
beanstandete Metadaten korrigieren. Sie erhält auch die Originalquellen der
betroffenen Eltern. Unveränderte Einträge bleiben erhalten. Jede Änderung geht
wieder durch die unabhängigen Prüfungen; unveränderte Seitenprojektionen können
ihre exakt an Inhalt, Bild-Hashes und Prüfvertrag gebundenen Checkpoints nutzen.

Ein unveränderter Kandidat oder identische wiederkehrende Befunde beenden die
Reparaturschleife. Der Entwurf samt Originalzugriff und Prüffragen wird gespeichert,
nicht als Erfolg umgedeutet. `processing_complete=false` sperrt die Übernahme als
geprüfte Agenda. Ein Pipeline-Lauf, der hier endet, ist kein exportierbares Protokoll.
Bei sehr großen PDFs kann die globale Zusammenhangsprüfung das Kontextbudget
überschreiten; dann wird kein Vollständigkeitsnachweis ausgestellt. Eine weitere
Skalierung muss alle betroffenen Originalseiten und Beziehungen abdecken.

## Checkpoints und Betrieb

Die laufende SQLite-Datenbank liegt im Docker-Volume `backend_state` unter
`/app/state/sessions.sqlite3`, nicht in einem Windows-Bind-Mount. Niemals eine
laufende Linux-SQLite-Datei gleichzeitig mit Windows-SQLite öffnen: Die beiden
Dateisystemzugriffe können unterschiedliche Sperrsemantik haben. Auch das
inhaltsfreie Monitoring liest über `docker exec` innerhalb desselben Containers.
Sicherungen werden mit der SQLite-Backup-API im Container zunächst im Linux-Volume
erzeugt (kleine Seitenschritte, z. B. `source.backup(target, pages=256)`). Erst die
geschlossene Sicherung ins lokale `data/backups/` kopieren. Eine vollständige
Live-Sicherung direkt auf den langsameren Windows-Bind-Mount kann Schreibzugriffe
länger als den SQLite-Timeout blockieren und einen vermeidbaren Job-Retry auslösen.
Bei vorhandenen Installationen muss die bisherige Datenbank vor Umstellung von
`PERSISTENCE_DB_PATH` gesichert und bei gestopptem Backend vollständig in das
Volume kopiert werden; eine leere neue Datenbank ist keine Migration.

Der langlebige Worker besitzt einen exklusiven Prozess-Lock. Transaktionale Leases
sichern Checkpoint-Schreibvorgänge und Veröffentlichungen zusätzlich ab.
Session-Revisionen verhindern, dass ein Ergebnis zwischenzeitliche Bearbeitungen
überschreibt. Die Veröffentlichung und ihr Marker liegen in derselben Transaktion.
Ein bereits gespeichertes `pipeline:transcript` wird nach einem Neustart wiederverwendet,
auch wenn der kurzlebige Transkriptionsjob nicht mehr im Arbeitsspeicher vorhanden ist.

Vor Weiterverwendung werden die unveränderliche Modellidentität, Code- und
Konfigurationsversionen sowie die Hashes der behaltenen Eingaben geprüft. Neue
Checkpoints tragen zusätzlich einen SHA-256-Integritätsnachweis. Neue Aufträge
binden PDF und Audio sowie die Transkriptionskonfiguration. Ein Unterschied
veranlasst keine automatische Neutranskription, sondern einen sichtbaren Abbruch.

Temporäre SQLite- und Dateisystemfehler beim Queue-Abruf und Heartbeat beenden den
Worker nicht. Fehler innerhalb eines Jobs behalten Checkpoints und verwenden das
begrenzte Retry-Budget. Bei längerem Ausfall oder ausgeschöpften Versuchen bleibt
der Fehler sichtbar. Ein gestorbener Worker führt zu einem negativen Healthcheck.

## Kontrollierte Migration eines alten PDF-Abbruchs

1. Eine konsistente SQLite-Sicherung mit der SQLite-Backup-API und eine separat
   geprüfte Kopie aller Uploads erstellen. Laufzeitdaten und Sicherungen bleiben
   ausschließlich unter ignorierten lokalen Datenverzeichnissen.
2. Tests bestehen lassen, Images lokal bauen und bisherige Images für Rückkehr
   erhalten. Keine Volumes löschen und keine Datenbank durch eine leere ersetzen.
3. Backend stoppen. Die Migration in derselben Container-/Dateisystemumgebung
   ausführen; Ollama bleibt für die Digest-Prüfung erreichbar.
4. Mit `python resume_pipeline.py JOB_ID --pdf-contract --fork-session` zunächst
   prüfen. `--apply` erstellt vor der Auftragsänderung eine weitere SQLite-Sicherung
   und zeichnet alte/neue Versionen, Checkpoint-Hashes und Sitzungsherkunft auf.
   Bei bearbeiteter Zielsitzung ist eine neue Sitzung zwingend; die alte bleibt erhalten.
5. Das Backend starten. Modell bleibt unverändert. Neue Prüf-Checkpoints müssen
   unter `pdf:page-evidence-v3:*` entstehen. Zeitstempel und Hash des ursprünglichen
   Transkript-Checkpoints müssen unverändert bleiben.

Die Migration unterstützt gezielt einen abgebrochenen Agenda-Schritt vor dessen
Veröffentlichung. Sie übernimmt keine alten Gesamt-Erfolgscheckpoints. Bei älteren
Aufträgen ohne ursprünglichen Audio-Hash prüft sie das gespeicherte Transkript
gegen den abgeschlossenen Transkriptionsjob und bindet die erhaltene Audioquelle
jetzt. Diese historische Nachweislücke wird ausdrücklich in der Provenienz vermerkt;
ein früherer Audio-Hash wird nicht erfunden.

## Messung und Abnahme

`python scripts/monitor_pipeline.py JOB_ID` schreibt ausschließlich lokale,
inhaltsfreie Stichproben nach `data/measurements/`. Erfasst werden Pipelinephase,
Checkpoint-Anzahl, GPU-Auslastung/VRAM/Leistung und CPU-/RAM-Nutzung der Container.
Das Monitoring überlebt vorübergehende Docker-/Datenbankausfälle. Vollständige
Modellaufrufe stehen in `durable_metrics`: Phase, Modell-Digest, Eingabe-/Ausgabetokens,
Ladezeit, Prompt-Verarbeitungszeit, Generierungszeit und Wall-Clock-Zeit.
Abgebrochene Aufrufe liefern möglicherweise keine abschließenden Tokenzähler;
Gesamtlaufzeit und Stichproben dürfen deshalb nicht durch die Summe erfolgreicher
Aufrufe ersetzt werden. Transkriptionsdauer stammt bei einer Wiederaufnahme aus
dem früheren Lauf und muss separat ausgewiesen werden.

Die Abnahme von 3–4 Stunden benötigt einen vollständigen repräsentativen GPU-Lauf
einschließlich aller Qualitätsprüfungen. Ein CPU-Nachtlauf ist eine eigene Abnahme.
Ein bestandener synthetischer Test oder eine Hochrechnung ersetzt keine dieser
Messungen. Zu prüfen sind vollständige Quellenabdeckung, offene Prüffragen,
Beschlüsse, Stimmenzahlen, Aufträge, gemeinsame und wiederaufgenommene TOPs sowie
korrekte Links zu Originalquellen. Die Konfiguration darf nicht zur Erreichung
einer scheinbaren Laufzeit stillschweigend Modell oder Qualitätskontrollen ändern.

Die kompakte Zuordnung hält weiterhin zwei unabhängige Durchgänge mit vollständiger
Zeilenabdeckung. Ein blindes Fakteninventar kann bei identischen Quellgruppen für
mehrere TOPs wiederverwendet werden. Primärentwurf, Entwurfsprüfung und beide
abschließenden Prüfungen bleiben TOP-spezifisch. Teilweise überlappende Quellgruppen
werden derzeit nicht pauschal zusammengelegt: Kontext, Quellenidentität und
TOP-spezifische Auslassungen müssen weiterhin vollständig prüfbar bleiben.

## Abgestufter Quellenvertrag

`graded-sources-v1` trennt technische Vollständigkeit, fachlichen Status und
Veröffentlichung. `processing_complete`/`review_complete` bezeichnen gelesene
Quellen und ausgeführte Pflichtprüfungen, nicht die Richtigkeit jeder Aussage.
Eine Veröffentlichung kann `review_draft` sein. `incomplete_draft` bleibt lokal
prüfbar, erzeugt aber keine erfolgreichen Agenda-/Zusammenfassungscheckpoints
und keinen regulären Protokollexport. Die Fehlerphase bleibt sichtbar.

- `exact`: unveränderte, eindeutige Originalstelle und abgeschlossene fachliche
  Modellprüfung ohne offene Befunde für diese Aussage.
- `source_range`: Originalbereich zugänglich, aber genauer Beleg oder fachliche
  Stützung offen. Konkrete Prüffragen stehen am Inhalt.
- `unsupported`: kein belastbarer Beleg; ausschließlich unbestätigter Vorschlag.

`reference_status` ist nur die technische Referenzprüfung. `content_status`
unterscheidet `unreviewed`, `supported`, `unclear` und `contradicted`. Ein exaktes
Zitat allein bestätigt keine Aussage. Widerlegte Kandidaten stehen getrennt als
`rejected_candidates` und erscheinen ausschließlich mit „VERWORFEN“ unter den
Unsicherheiten. Beschlüsse, Stimmenzahlen und Aufträge erhalten dieselben direkten
Kennzeichnungen. Speichern und Umformulieren erteilen keine Bestätigung; manuell
geänderte Fassungen verlieren ihren Modellnachweis. Eine menschliche Bestätigung
ist in diesem Vertrag noch nicht als gesonderte Funktion vorgesehen.

Agenda-Antworten wählen kurze `L1`-Kennungen. Der gespeicherte Quellenkatalog bindet
jede Kennung an ursprüngliche Zeilen-ID, Originalstellen-Hash und gesamten
Quellen-Hash. Die Anwendung übernimmt Originaltext. Unbekannte IDs werden nicht
erraten. Alte freie Zitate werden nur bei eindeutigen reinen Leerraumabweichungen
auf Originalstellen zurückgeführt; sonst bleibt die Frage offen. Unlesbare
Antworten, fehlende Abdeckung, fehlende Pflichtprüfungen, Modellwechsel und
Integritätsfehler bleiben technische Fehler.

`durable_artifacts` speichert Rohversuche, Quellenkataloge, strukturierte Diagnosen
und unvollständige Entwürfe ausschließlich in der lokalen Datenbank. Diese
Artefakte sind keine wiederverwendbaren Erfolgsmarker. Wiederholte fehlerhafte
Antworten und wirkungslose Reparaturen begrenzen teure Wiederholungen. Offene
Entwürfe durchlaufen die abschließenden Originalquellenprüfungen ebenfalls.

## Wiederaufnahme nach dem Kontext-Belegfehler

Nach Sicherung und Tests unterstützt `python resume_pipeline.py JOB_ID
--source-contract` eine trockene Prüfung; mit `--apply` wird eine neue Sitzung
und ein neuer Auftrag angelegt. Backend dafür stoppen, Ollama erreichbar lassen
und ausschließlich die Linux-Containerumgebung verwenden. Die Migration lässt
keine Konfigurations-/Modelländerung zu, prüft alle Original-Hashes, Audioabschluss,
Transkriptgleichheit, Checkpointintegrität und die an Originalbilder/Kandidat/
Prüfprompt gebundenen PDF-Seiten- und Zusammenhangsprüfungen. Alte Aufträge,
Sitzungen, Veröffentlichungen und Checkpoints bleiben unverändert erhalten.
Nur bestätigte Transkription, deterministische Aufteilung und verifizierte
PDF-Arbeit werden kopiert. Der Kindauftrag ist bei gleicher Codefassung idempotent.

Die Backup-API schreibt in Schritten von 128 Seiten zunächst ins Linux-Volume.
Erst nach Schließen und Integritätsprüfung die Sicherung nach `data/backups`
kopieren. Die alte Hostdatei `data/sessions.sqlite3` bleibt unbenutzt.

`scripts/validate_context_block.py` führt vor einem langen Wiederanlauf genau den
ersten realen Kontextblock und eine unabhängige Prüfung aller darin enthaltenen
Originalquellen aus. Es läuft im Linux-Backend mit separater Datenbank unter
`/app/state/validations/`, verweigert entfernte Modellendpunkte und prüft die
unveränderte Modellidentität/Konfiguration. Ausgabe: nur Kennzahlen. Rohantworten
bleiben lokal. Dieser Versuch ist keine End-to-End-Abnahme und wird nicht als
bestandener Verarbeitungsschritt in den eigentlichen Auftrag übernommen.

`setup.ps1 build` entfernt vorhandene Container, baut Images und zieht außerdem
das Ollama-Image neu. Für eine kontrollierte Fehlerbehebung können stattdessen
die gleichen lokalen Backend-/Frontend-Builds ausgeführt und anschließend nur
diese beiden Dienste mit Compose neu erstellt werden. So bleiben Ollama-Image,
Modell und Volumes unverändert. `setup.ps1 start` baut geänderten Code nicht neu.
