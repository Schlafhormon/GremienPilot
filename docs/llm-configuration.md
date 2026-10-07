# Zentrale Modellkonfiguration

Dieser Branch verwendet [Aleph-Alpha/Kolibri-1](https://huggingface.co/Aleph-Alpha/Kolibri-1)
mit der vollständigen [Sakura IQ2_XS-Mischquantisierung](https://huggingface.co/webmp3/Sakura-MicroQuality-Kolibri-1-GGUF)
(20,94 GiB, 78B Parameter, keine entfernten Experten). Die Quantisierung ist eine
Community-Konvertierung; ihre starke Kompression kann die Ergebnisqualität senken.

Stock Ollama/llama.cpp unterstützt diese Architektur derzeit nicht. `scripts/kolibri/`
verwendet die gepatchte Laufzeit [hob-b11434](https://github.com/Seraphiel102/llama.cpp/releases/tag/hob-b11434).
Laufzeit, Modell und offizieller Tokenizer sind auf Revision und SHA-256 festgelegt.
Das Modell wird beim ersten Start im Volume `kolibri_data` gespeichert; Downloads
sind fortsetzbar. Keine Modellgewichte oder Zugangsdaten werden eingecheckt.

## Lokales Profil und Start

`setup.ps1 build` / `./setup.sh build` baut die lokalen Images. Compose verbindet
das Backend mit `http://kolibri:8080/v1`, lokale Backend-Entwicklung mit
`http://localhost:8080/v1`. `LLM_PROVIDER=llama-cpp` aktiviert den Serververtrag.
Der Serveralias ist `Aleph-Alpha/Kolibri-1`; alte Browser-Modellüberschreibungen
über „Servermodell verwenden“ zurücksetzen.

Für den i9-12900KF, 32 GB RAM und RTX 3070 Ti (8 GB) verwendet die lokale `.env`
32.768 Kontexttokens, 8 CPU-Threads, 10 GPU-Layer und GPU-Modellwechsel.
Der Server nutzt eine parallele Anfrage, Flash Attention und Q8-KV-Cache.
`KOLIBRI_GPU_LAYERS=0` ermöglicht CPU-Betrieb; das GPU-Compose-Profil setzt sonst 10.
CPU-Offloading und erneutes Laden brauchen Zeit. Kontext-/Layer-Änderungen erfordern
das Neuerstellen des Containers, z.B. `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d`.

Mit `GPU_MODEL_SWITCHING=true` teilen Kolibri und Whisper eine GPU. Der Server
entlädt Modell und KV-Cache nach 15 Sekunden Inaktivität. Das Backend wartet vor
Whisper auf `is_sleeping=true` aus `/props`; die nächste LLM-Anfrage lädt Kolibri
nach der Whisper-Freigabe wieder. Ein Backend-Prozess und exklusive Servernutzung
sind erforderlich. Der Gesundheitscheck weckt das Modell nicht auf.

## Reasoning und Transport

Fast verwendet `LLM_FAST_REASONING_EFFORT=none`, Slow standardmäßig
`LLM_SLOW_REASONING_EFFORT=medium`. Leere Werte wählen diese Standards.
Kolibri unterstützt `none`, `low`, `medium`, `high`; `max` wird abgewiesen.
Der Transport setzt `chat_template_kwargs.reasoning_effort` und `enable_thinking`.
Getrennte Reasoning-Streams werden niemals als Ergebnistext übernommen.

Leere Sampling-Werte wählen für Kolibri Temperatur `1.0`, `top_p=0.97`, `top_k=128`
gemäß Modellkarte; explizite Werte haben Vorrang. `LLM_THINKING_TOKENS` reserviert
in Slow 4.096 zusätzliche Tokens, in Fast immer 0. Das gemeinsame `max_tokens`
ist eine Obergrenze für Denken und Antwort, kein separates Denklimit. Die
aufgabenspezifischen Ausgabebudgets bleiben erhalten.

Eingabe, Schema, Template-Reserve und Ausgabe müssen gemeinsam in
`LLM_CONTEXT_TOKENS` passen. Der Server startet mit derselben Grenze und
`--no-context-shift`; `/props` muss ausreichend Kontext bestätigen. Lange
Transkripte werden durch die vorhandene Pipeline abschnittsweise verarbeitet.
Unvollständige Streams, `finish_reason=length` und leere Antworten sind Fehler.
Die vorhandenen begrenzten Reparaturen und Quellenprüfungen bleiben aktiv.

Der offizielle Tokenizer wird beim Backend-Bau geladen und geprüft (9,5 MB).
Explizite `LLM_TOKENIZER_PATH`/`LLM_TOKENIZER_MODEL` müssen zum Modell passen.
Ohne lokalen Tokenizer gilt die konservative UTF-8-Bytezählung. Während Jobs
erfolgen keine Tokenizer-Downloads. `LLM_MODEL_REVISION` enthält die GGUF-SHA-256
für die Identität von Cache und dauerhaften Jobs; bei anderen Gewichten anpassen.

## PDF und andere Provider

Kolibri verarbeitet nur Text (`LLM_IMAGE_TOKENS=0`). Slow liest jede PDF-Seite
über Textlayer und lokale OCR und prüft beide Darstellungen unabhängig gegen
den Kandidaten. Fast nutzt Textlayer, bei Scans OCR. Originaldokument und
Quellreferenzen bleiben erhalten. Das ersetzt keine visuelle Prüfung; insbesondere
Tabellen und schlecht lesbare Scans können manuelle Kontrolle brauchen.

`ollama` und `openai-compatible` bleiben als explizite Provider verfügbar.
Externe Endpunkte benötigen den passenden Modellnamen, Kontext, Tokenizer und
eine eigene unveränderliche Modellrevision. Ollama verlangt mindestens 0.34.2
für den bestehenden Vertrag (`truncate=false`, `shift=false`, verifizierter
Kontext und Digest). Normale OpenAI-kompatible APIs bieten keine standardisierte
Kontextabfrage. Es gibt keinen automatischen Modell- oder Providerwechsel.

`LLM_LOAD_TIMEOUT_SECONDS` begrenzt Laden/erstes Token, `LLM_READ_TIMEOUT_SECONDS`
die Inaktivität danach; `LLM_TOTAL_TIMEOUT_SECONDS=0` erlaubt lange aktive Streams.
Diagnose: `/api/llm/diagnostics`; Startfortschritt: `docker compose logs -f kolibri backend`.