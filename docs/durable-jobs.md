# Dauerhafte Modellarbeiten

SQLite (`PERSISTENCE_DB_PATH`) und das Upload-Verzeichnis müssen einen Neustart überleben. Unterstützt wird ein Backend-Prozess mit einer gemeinsamen seriellen Arbeitswarteschlange, auf einem lokalen Dateisystem. `flock` auf einer Datei neben SQLite verweigert einen zweiten Prozess. Mehrere Hosts/Repliken, NFS und mehrere unabhängig kopierte Datenbanken sind kein unterstützter Worker-Cluster. Kubernetes verwendet deshalb `Recreate`; CPU/GPU und Provider bleiben separat konfigurierbar.

Neue Starts: `POST /api/extract-tops/jobs` (Multipart wie bisher), `POST /api/agenda-detection/jobs` (bisheriges JSON). Die Antwort ist `202` mit `job_id`. `GET /api/model-jobs/{id}` liefert kurze Statusantworten, `POST /api/model-jobs/{id}/cancel` bricht ab. PDF-Clients können alternativ `Prefer: respond-async` am bisherigen Endpunkt verwenden. Alte Endpunkte behalten ihren synchronen Ergebnisvertrag, laufen intern jedoch ebenfalls dauerhaft; alte Clients können weiterhin am Proxy-Timeout scheitern. Der neue Browser-Client verwendet Statusabfragen. Pipeline- und Zusammenfassungsantworten behalten ihre alten Statuswerte und ergänzen `execution` mit dem genauen Jobzustand.

| Zustand | Bedeutung |
| --- | --- |
| `queued` | Wartet auf Ressourcen/Verarbeitung |
| `running` | Übernommen; Details in `progress` |
| `retry_wait` | Vorübergehender Transportfehler, begrenzte Wiederholung geplant |
| `review_required` | Fachliche Prüfung oder geänderte Modell-/Promptkonfiguration |
| `failed` | Technischer Fehler oder erschöpftes Versuchsbudget |
| `completed` | Technisch vollständig verarbeitet |
| `cancelled` / `superseded` | Abgebrochen / durch neuere Eingaben überholt |

Heartbeat und Lease bestätigen nur Worker-Besitz. `last_delta_at` wird ausschließlich bei echter Text-/Reasoning-Ausgabe aktualisiert; leere Stream-Ereignisse zählen nicht. `LLM_LOAD_TIMEOUT_SECONDS` begrenzt das Warten auf die erste Ausgabe (Provider geben Laden und Prompt-Auswertung nicht durchgehend getrennt an). Danach begrenzt `LLM_READ_TIMEOUT_SECONDS` die Ausgabepause. Es gibt keinen Nachtlauf-Deadline; `LLM_TOTAL_TIMEOUT_SECONDS=0` deaktiviert auch das optionale Gesamtlimit einer einzelnen Modellanfrage. Transportversuche (`LLM_MAX_RETRIES`) und Jobübernahmen (`MODEL_JOB_MAX_ATTEMPTS`) sind begrenzt. Bei unbekanntem Providerstillstand endet damit ein Versuch, statt unbegrenzt Heartbeats als Fortschritt auszugeben.

Eingaben, Dokumenthashes, öffentliche Modellkonfiguration, Code-/Promptversionen, geprüfte Teilantworten und Veröffentlichungsmarken liegen in SQLite. API-Schlüssel werden nicht gespeichert. Erfolgreiche Schritte werden nach Neustart wiederverwendet. Eine unterbrochene, noch nicht geprüfte Generierung muss erneut beginnen; Token-genaues Fortsetzen unterstützt der Provider nicht. Geänderte Modell-/Promptversionen halten einen alten Job zur Prüfung an. Ollama-Digests werden je Job gebunden; externe Provider benötigen eine verlässliche `LLM_MODEL_REVISION`, da ein stiller Wechsel hinter einem Provider-Alias sonst nicht sicher erkennbar ist. Die Transkription wird als kompletter Schritt gespeichert; ein unterbrochener Transkriptionsschritt beginnt neu. Abbruch der Transkription erfolgt an deren vorhandenen Fortschrittsgrenzen, Abbruch von Modellstreams und Ressourcenwarten kooperativ innerhalb kurzer Prüfintervalle.

Veröffentlichungen prüfen Lease und Abbruch in derselben SQLite-Transaktion wie die Sitzungsänderung. Pipeline-Ergebnisse werden gegen die Startrevision übernommen; selektive Zusammenfassungen prüfen TOP-Identität, Eingabe- und Bearbeitungsfingerabdruck. Konflikte erhalten manuelle Änderungen. Abgeschlossene Zusammenfassungen besitzen eine atomare Veröffentlichungsmarke, sodass auch ein Absturz unmittelbar nach dem Speichern keinen erfolgreichen TOP erneut generiert. Ältere Daten und Statuswerte bleiben lesbar; aktive alte Jobs werden beim Start in die neue Warteschlange aufgenommen.

PDFs werden nicht beim Jobende gelöscht. `MODEL_DOCUMENT_RETENTION_DAYS=0` bewahrt sie unbegrenzt auf. Bei einem positiven Wert sind nur abgeschlossene oder abgebrochene Jobs nach Ablauf bereinigbar; aktive, gestörte, fehlgeschlagene und prüfbedürftige Jobs bleiben gesperrt. Aus dem Backend-Verzeichnis zeigt `python -m durable_jobs --cleanup-documents` die Kandidaten, erst `--apply` löscht diese Dateien und speichert den Löschzeitpunkt mit dem Dokumenthash. Jobdaten, geprüfte Ergebnisse und die Nachweisdaten bleiben erhalten. Es erfolgt keine automatische Bereinigung beim Start oder im Worker.
