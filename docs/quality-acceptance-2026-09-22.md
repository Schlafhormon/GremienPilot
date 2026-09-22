# Abnahmeprotokoll vom 22. September 2026

**Stand: implementiert und automatisiert geprüft; reale Qualitäts- und Ressourcenfreigabe ausstehend.** Ausgangspunkt war Repository-Commit `93e5a37` bei sauberem Arbeitsbaum. `AGENTS.md` war im Repository und seinen übergeordneten Verzeichnissen nicht vorhanden. Die laufenden Container enthalten einen älteren Stand. Keine Veröffentlichung, Neustarts, Modell-Downloads oder produktiven Datenänderungen wurden ausgeführt. Die während der Arbeit vom Nutzer ergänzten lokalen `testdata` und deren Git-Ignore-Regel blieben erhalten.

## Implementiert

- Modellgestützte Notizinventare mit exakten Quellen-IDs/Zitaten, blindem Zweitinventar, vollständigen Quellenprüfungen, Konsolidierung, Endprüfungen und gezielter Widerspruchsklärung. Keine fachlichen Stichwortfilter, Ähnlichkeitsbelege, nachträglichen Regelkorrekturen oder ungesicherten Freitext-Fallbacks im Zusammenfassungspfad.
- Verbindlicher technischer Abschluss, fortsetzbare validierte Teilschritte, Modell-/Prompt-/Konfigurationsbindung einschließlich Browsermodell und Modellrevision. Unvollständige Verarbeitung ist kein erfolgreicher Job. Schutz vor konkurrierenden Text-, Sprecher-, Zeit- und Zuordnungsänderungen.
- Gemeinsame TOP-Beratungen als Quellen aller betroffenen Zusammenfassungen; manuelle Zuordnungen haben Vorrang. Alle offenen Prüffragen sind sichtbar, Belegstellen einzeln erreichbar. Veraltete Modellbelege verlieren ihren Prüfnachweis. Sammelübernahme von TOP-Vorschlägen benötigt vollständige unabhängige Prüfung statt Konfidenzschwelle.
- Unsicherheiten und Prüffragen bleiben in TXT, DOCX und PDF erhalten. Bestehende editierbare Texte, API-Felder und Sitzungsdaten bleiben kompatibel; neue Prüffelder sind additiv.
- Historische Research-Inferenzprogramme durch Adapter zum gemeinsamen Workflow ersetzt. Alte experimentelle Python-Klassen/CLI-Formate sind damit ausdrücklich abgelöst; die Anwendungs-APIs bleiben erhalten.
- Konfigurationsbeispiele und Compose vereinheitlicht. Kubernetes hat zum lokalen Gemma-Default einen passenden optionalen Ollama-Dienst; externe Provider bleiben explizit konfigurierbar. Setup prüft das tatsächliche Compose-Modell statt beliebiger Modelldaten und verwendet eine konfigurierbare Speicherplatzreserve.
- Offline-Auswerter mit Originalhashes, freigegebenen externen Referenzen und ergebnisgebundener menschlicher Bewertung. Keine Modellantwort als eigene Referenz; keine erzwungenen alten Inferenzwerte.

## Tatsächlich ausgeführte Prüfungen

| Prüfung | Ergebnis |
|---|---|
| Vollständige Backend-Suite | **522 bestanden**, 1 bestehende Starlette/AnyIO-Deprecation-Warnung; 69,92 s |
| Nachprüfung zuletzt geänderter Integrationspfade | **117 bestanden**, gleiche Warnung; 45,77 s |
| Nachprüfung Modell-/Konfigurationsbindung und Quellenaufteilung | **57 bestanden**, gleiche Warnung; 8,36 s |
| Offline-Auswerter nach Erweiterung der menschlichen Quellenbewertung | **4 bestanden** |
| Frontend Vitest | **125 bestanden**, 13 Dateien; 13,08 s |
| Frontend TypeScript/Vite-Produktionsbuild | erfolgreich, 46 Module; letzter Build 1,69 s |
| Frontend ESLint | erfolgreich |
| Python-Kompilierung Backend/Skripte | erfolgreich |
| Compose-Konfigurationsprüfung, Bash-Syntax, Diff-Whitespace | erfolgreich |
| Kubernetes-YAML | Nicht geheime Manifeste erfolgreich geparst; kein Cluster-Apply/Admission-Test |

Tests verwenden isolierte temporäre Datenbanken und simulierte Modelle; Frontend-Prüfungen liefen in einer Kopie unter `/tmp/gremienpilot-quality-frontend`. Exporttests prüfen alle drei Formate. Abgedeckt sind unter anderem Prozessabsturz/Neustart, validierte Wiederaufnahme, simulierte sehr lange Generierung, Abbruch, Modell-/Schema-/Kontextfehler und konkurrierende manuelle Änderungen. Kein Docker-Image-Build und keine PowerShell-Ausführung; PowerShell-Änderungen wurden statisch geprüft.

Lokale Prüfprotokolle: `/tmp/gremienpilot-quality-backend-final.log`, `gremienpilot-quality-integration-final.log`, `gremienpilot-quality-frontend-accepted.log`, `gremienpilot-quality-build-accepted.log` und `gremienpilot-quality-lint-accepted.log` unter `/tmp`.

## Tatsächliche Qualität und Ressourcen

Auf Nutzeranweisung **kein echter Modelllauf**. `testdata` enthält eine MP3 (76.221.952 Bytes) und `PDF_Anlage.pdf` (178.434 Bytes, drei lesbare Seiten mit 1.101 / 1.838 / 739 extrahierten Textzeichen). Das ist eine technische Bestandsaufnahme, keine modellgeprüfte PDF-Extraktion. Das aus ausschließlich gelesenen Originalen erzeugte Inhaltsinventar liegt unter `/tmp/gremienpilot-quality-acceptance/inventory.json`.

Eine fachlich freigegebene Transkription/Referenzannotation wurde dort nicht gefunden. TOP-Vollständigkeit, Nummerierung/Sitzungsteile, Zuordnungs-/Grenzfehler, unbelegte/fehlende Ergebnisse und tatsächlicher manueller Aufwand sind deshalb **nicht gemessen**. Der Auswerter und seine synthetischen Tests liefern hierfür keine Ersatzquote.

Bestandsmessung: 32 vCPU (Xeon Gold 5218 unter VMware), 62.338 MiB RAM, davon zum Messzeitpunkt 53.808 MiB verfügbar; 8.191 MiB Swap, 290 MiB belegt. Containerbelegung: Backend rund 3,65 GiB, Ollama im Leerlauf 21,35 MiB, Frontend 28,85 MiB. Kein Modell geladen. Installiert sind Qwen 3.5 9B/Q4_K_M und Qwen 3 8B/Q4_K_M; **Gemma 4 ist nicht installiert**.

Keine Messung des Gemma-Gewichtsspeichers, KV-Caches, Thinking-Verbrauchs, Bildbedarfs oder maximal nutzbaren Kontextes. Es wurde daher keine Quantisierung oder Kontextgrenze als ausreichend zertifiziert und keine stillschweigende Qualitätsreduzierung vorgenommen.

## Lokales CPU-Profil ohne Secrets

`.env` behält den zuvor gewählten Kandidaten `gemma4:31b-it-q8_0`, natives Ollama, CPU (`LLM_GPU_LAYERS=0`), 16 LLM-Threads, 16.384 Kontexttokens, Thinking an, 4.096 Ausgabetokens, Bildreserve 2.048, Lese-/Ladefrist 1.800 s und kein Gesamtzeitlimit. Ollama bleibt bei einem parallelen Aufruf, einem geladenen Modell und f16-KV-Cache; Lade-Watchdog 30 Minuten. Whisper bleibt `large-v3`, CPU, 24 Threads. Das Profil ist vorbereitet, **nicht als Gemma-lauffähig gemessen**.

Ergänzt: `SUMMARY_OUTPUT_TOKENS=4096`, `SUMMARY_MODEL_ATTEMPTS=2`, `SUMMARY_RECONCILIATION_ROUNDS=2`, `LLM_LOAD_TIMEOUT_SECONDS=1800`. Überholte Zusammenfassungs-Prüfschalter entfernt. Vorherige `.env` liegt lokal in `.env.pre-quality.backup` (0600, ignoriert, keine Secrets ausgegeben/committet). Die generischen Installationsdefaults bleiben Q4_K_M; die lokale Q8-Auswahl wird ausdrücklich nicht überschrieben.

## Offene Punkte und spätere Aktivierung

1. Freigegebene Originaldaten menschlich referenzieren, anschließend nach gesonderter Beauftragung auf isolierten Ressourcen echte Modellläufe und Quellenbewertung durchführen. Derzeit gibt es keine empirische Modellqualitätsfreigabe.
2. Gemma bereitstellen und Q8 sowie Kontext/Bilder/Thinking zusammen mit den Basisprozessen vermessen. Sehr große vollständige Notizinventare können das konfigurierte Kontextbudget überschreiten; der Job wird dann technisch erfolglos beendet, statt Quellen oder Ergebnisse zu kürzen. Diese Grenze muss im Langsitzungstest bewertet werden.
3. Vor späterer Aktivierung aktuelle Datenbank konsistent sichern, Images und `.env` festhalten, aktive Jobs auslaufen lassen; neue Images bauen und erst im Wartungsfenster gezielt aktivieren. Browsermodellüberschreibungen sowie Image-/Provider-Versionen prüfen. Keine automatische Cloud-Ausweichverarbeitung.
4. Rückkehr: vorherige Images und `.env.pre-quality.backup` verwenden. Für den Arbeitsbaum ist `93e5a37` der Ausgangsstand; eigene spätere Änderungen getrennt sichern. Keine pauschale Datenbank-Rücksetzung über neue manuelle Bearbeitungen. Genaue Schritte und Konfigurationsverträge: [quality-verification.md](quality-verification.md).

Aktuell laufende Image-IDs als Rückkehranker:

- Backend: `sha256:46babbe554b205442fa08c6bb848c4f6eea74ce430bdf4f986ed1e471764dd67`
- Frontend: `sha256:3148466314372fb83d4354b1f643ccf0d453ea913413bff7f99196ac93bd4d68`
- Ollama: `sha256:da6e0dc5651df159e45686fd663c4dbe1624a52c44d7280eeac1551d8f865532`
