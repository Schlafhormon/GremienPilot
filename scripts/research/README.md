# Research adapters

`model_workflow.py` uses the production PDF, agenda and source-verified summary workflows. The former segmentation/minutes/archive commands delegate to it; their old heuristic classes and command-line formats are retired. Existing research output files are untouched. The application APIs are unchanged.

Use the backend Python environment:

```sh
python scripts/research/model_workflow.py --transcript /private/transcript.json --pdf /private/agenda.pdf --output /private/new-result.json
```

The transcript is a complete JSON array of `{speaker, text, start?, end?}`. `--tops` accepts original agenda labels as a JSON array. `--model` is optional; provider, thinking, context and timeouts come from central configuration. Never pass moderator-only excerpts as a complete meeting. `extract_moderator_transcript.py` is a separate format conversion utility, not an inference input selector.

Model runs require an isolated endpoint and resources. Nothing here is automatically run or deployed. `llama-70b/` remains a separate, explicitly selected deployment experiment. For offline evaluation use `../verify_llm_snapshot.py`; see `../../docs/quality-verification.md`.
