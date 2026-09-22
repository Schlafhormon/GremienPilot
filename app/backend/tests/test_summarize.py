import summarize
import pytest


@pytest.mark.parametrize('content', [
    '{"discussion":["Erster Beitrag"],"discussion":["Zweiter Beitrag"]}',
    '{"discussion":["Erster Beitrag"],"Diskussion":["Zweiter Beitrag"]}',
])
def test_summary_rejects_fields_that_would_silently_overwrite_content(content):
    with pytest.raises(summarize.StructuredOutputError):
        summarize.parse_structured_summary(content)


def test_added_source_check_reuses_completed_primary_and_fact_calls(fake_openai_module, monkeypatch, tmp_path):
    import json
    monkeypatch.setenv('LLM_CACHE_DIR', str(tmp_path / 'cache'))
    monkeypatch.setenv('LLM_SUMMARY_FACT_REVIEW_MAX_CALLS', '3')
    source = ('MOD: In der letzten Sitzung wurde beschlossen: Die Verwaltung erstellt einen Entwurf. '
              'Ich lese wortwörtlich vor.')
    draft = structured_response(action_items=['Die Verwaltung erstellt einen Entwurf.'])
    fake_openai_module.responses = [draft, draft]
    first = summarize.summarize_segment('Bericht', source, model='test-model')
    assert first.llm_usage['fact_review_calls'] == 1
    monkeypatch.setenv('LLM_SUMMARY_GROUNDING_MAX_CALLS', '1')
    fake_openai_module.responses = [json.dumps({'0:action_items:0': {
        'status': 'reported', 'evidence_id': 'target:0', 'scope_id': 'target:0'}})]
    checked = summarize.summarize_segment('Bericht', source, model='test-model')
    assert not checked.structured.action_items
    assert checked.llm_usage['grounding_calls'] == 1
    assert not checked.llm_usage.get('fact_review_calls')
    assert len(fake_openai_module.instances[-1].calls) == 1
    replay = summarize.summarize_segment('Bericht', source, model='test-model')
    assert replay.llm_usage['cached_summary']
    assert not fake_openai_module.instances[-1].calls


def test_actual_frontend_prompt_preserves_summary_contract(fake_openai_module, frontend_summary_prompt):
    fake_openai_module.content = structured_response()
    result = summarize.summarize_segment(
        "Haushalt", "MOD: Der Haushalt wird beraten.",
        model="test-model", system_prompt=frontend_summary_prompt,
    )
    prompt = fake_openai_module.instances[0].calls[0]["messages"][0]["content"]
    assert summarize.DEFAULT_SYSTEM_PROMPT in prompt
    assert frontend_summary_prompt in prompt
    assert "das JSON-Ausgabeformat hat Vorrang" in prompt
    assert not result.fallback_used


def structured_response(**overrides):
    payload = {
        "discussion": ["Die Vorsitzende erlaeuterte den Sachverhalt."],
        "decisions": [],
        "votes": [],
        "action_items": [],
        "open_points": [],
        "uncertainties": [],
    }
    payload.update(overrides)
    import json

    return json.dumps(payload)


def test_resolve_llm_base_url_uses_local_default_outside_docker(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setattr(__import__("llm_config"), "is_docker_runtime", lambda: False)

    base_url, source = summarize.resolve_llm_base_url()

    assert base_url == "http://localhost:11434/v1"
    assert source == "local_development_default"


def test_resolve_llm_base_url_uses_internal_default_in_docker(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setattr(__import__("llm_config"), "is_docker_runtime", lambda: True)

    base_url, source = summarize.resolve_llm_base_url()

    assert base_url == "http://ollama:11434/v1"
    assert source == "internal_docker_default_from_local_value"


def test_resolve_llm_base_url_keeps_explicit_external_url(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setattr(__import__("llm_config"), "is_docker_runtime", lambda: True)

    base_url, source = summarize.resolve_llm_base_url()

    assert base_url == "https://llm.example.test/v1"
    assert source == "external_configured"


def test_summarize_segment_uses_structured_output_and_returns_duration(
    fake_openai_module,
    monkeypatch,
):
    fake_openai_module.responses = [
        structured_response(
            decisions=["Der Ausschuss empfahl die Annahme der Vorlage."],
            votes=["Die Empfehlung erfolgte einstimmig."],
        )
    ]
    times = iter([100.0, 102.5])
    monkeypatch.setattr(summarize.time, "time", lambda: next(times))

    result = summarize.summarize_segment(
        "Haushalt",
        "SPEAKER_00: Wir beraten den Haushalt.",
        model="test-model",
        system_prompt="Formal als JSON zusammenfassen",
    )

    assert "Diskussion:" in result.summary
    assert "Die Vorsitzende erlaeuterte den Sachverhalt." in result.summary
    assert "Beschluss:" in result.summary
    assert "Abstimmung:" in result.summary
    assert result.duration_seconds == 2.5
    assert result.structured is not None
    assert result.structured.decisions == [
        "Der Ausschuss empfahl die Annahme der Vorlage."
    ]
    assert result.fallback_used is False
    assert result.chunks_processed == 1

    client = fake_openai_module.instances[0]
    assert client.kwargs["base_url"]
    assert client.kwargs["timeout"].read == summarize.LLM_TIMEOUT_SECONDS
    request = client.calls[0]
    assert request["model"] == "test-model"
    assert request["temperature"] == 0.2
    assert request["max_tokens"] == 1400
    assert request["messages"][0]["role"] == "system"
    assert "Gib ausschließlich valides JSON" in request["messages"][0]["content"]
    assert "Formal als JSON zusammenfassen" in request["messages"][0]["content"]
    assert "TOP: Haushalt" in request["messages"][1]["content"]
    assert "SPEAKER_00: Wir beraten den Haushalt." in request["messages"][1]["content"]


def test_summarize_segment_preserves_all_partial_notes_for_long_transcripts(
    fake_openai_module,
    monkeypatch,
):
    monkeypatch.setattr(summarize, "LLM_CHUNK_CHARS", 80)
    fake_openai_module.responses = [
        structured_response(discussion=["Teil 1 wurde beraten."]),
        structured_response(discussion=["Teil 2 wurde beraten."]),
        structured_response(
            discussion=["Die Beratung wurde zusammengefuehrt."],
            open_points=["Die Verwaltung liefert Zahlen nach."],
        ),
    ]

    transcript = "\n".join(
        [
            "SPEAKER_00: " + ("Haushaltsansatz und Begruendung. " * 2),
            "SPEAKER_01: " + ("Nachfrage zu Kosten und Fristen. " * 2),
        ]
    )

    result = summarize.summarize_segment("Haushalt", transcript, model="test-model")

    assert result.chunks_processed == 2
    assert "Teil 1 wurde beraten." in result.summary
    assert "Teil 2 wurde beraten." in result.summary

    calls = fake_openai_module.instances[0].calls
    assert len(calls) == 2
    assert "Haushaltsansatz" in calls[0]["messages"][1]["content"]
    assert "Nachfrage" in calls[1]["messages"][1]["content"]


def test_summarize_segment_falls_back_to_freetext_on_malformed_structured_response(
    fake_openai_module,
):
    fake_openai_module.responses = [
        "Das ist kein JSON.",
        "Die Vorlage wurde beraten. Ein Beschluss wurde nicht gefasst.",
    ]

    result = summarize.summarize_segment(
        "Baugebiet",
        "SPEAKER_00: Die Vorlage wird beraten.",
        model="test-model",
    )

    assert result.summary == "Die Vorlage wurde beraten. Ein Beschluss wurde nicht gefasst."
    assert result.structured is None
    assert result.fallback_used is True
    assert result.chunks_processed == 1

    calls = fake_openai_module.instances[0].calls
    assert len(calls) == 2
    assert calls[1]["temperature"] == 0.3
    assert "Zusammenfassung:" in calls[1]["messages"][1]["content"]


def test_summarize_segment_retries_transient_llm_errors(
    fake_openai_module,
    monkeypatch,
):
    class ServerError(Exception):
        status_code = 503

    monkeypatch.setattr(summarize, "LLM_RETRY_BACKOFF_SECONDS", 0)
    monkeypatch.setenv("LLM_MAX_RETRIES", "2")
    fake_openai_module.responses = [
        ServerError("Service unavailable"),
        structured_response(discussion=["Die Beratung wurde fortgesetzt."]),
    ]

    result = summarize.summarize_segment(
        "Gebuehren",
        "SPEAKER_00: Wir beraten die Gebuehren.",
        model="test-model",
    )

    assert result.fallback_used is False
    assert "Die Beratung wurde fortgesetzt." in result.summary
    assert len(fake_openai_module.instances[0].calls) == 2


def test_summarize_segment_checks_model_before_chat(fake_openai_module):
    fake_openai_module.models_response = ["other-model"]

    with pytest.raises(summarize.LLMCallError) as error:
        summarize.summarize_segment(
            "Haushalt",
            "SPEAKER_00: Wir beraten den Haushalt.",
            model="missing-model",
        )

    assert error.value.category == "model_missing"
    assert "missing-model" in str(error.value)
    assert "LLM_MODEL=missing-model" in str(error.value)
    assert fake_openai_module.instances[0].calls == []


def test_summarize_segment_reports_unreachable_ollama_before_chat(
    fake_openai_module,
):
    fake_openai_module.models_response = ConnectionError("Connection error")

    with pytest.raises(summarize.LLMCallError) as error:
        summarize.summarize_segment(
            "Haushalt",
            "SPEAKER_00: Wir beraten den Haushalt.",
            model="test-model",
        )

    assert error.value.category == "network"
    assert "LLM_BASE_URL=" in str(error.value)
    assert "LLM_MODEL=test-model" in str(error.value)
    assert fake_openai_module.instances[0].calls == []


def test_summary_review_links_structured_items_to_transcript_lines():
    structured = summarize.StructuredSummary(
        discussion=["Die Vorlage zum Haushalt wurde beraten."],
        decisions=["Der Ausschuss beschloss die Vorlage zum Haushalt."],
        votes=["Die Abstimmung erfolgte einstimmig."],
    )
    lines = [
        {
            "speaker": "SPEAKER_00",
            "text": "Wir beraten die Vorlage zum Haushalt.",
            "start": 5,
            "end": 9,
        },
        {
            "speaker": "SPEAKER_01",
            "text": "Der Ausschuss beschloss die Vorlage einstimmig.",
            "start": 10,
            "end": 14,
        },
    ]

    review = summarize.build_summary_review(
        structured=structured,
        summary=summarize.render_structured_summary(structured),
        lines=lines,
    )

    decision_link = next(
        link
        for link in review.source_links
        if link.section == "decisions" and link.item_index == 0
    )
    vote_link = next(
        link
        for link in review.source_links
        if link.section == "votes" and link.item_index == 0
    )

    assert decision_link.missing_source is False
    assert 1 in decision_link.line_indices
    assert decision_link.start == 5
    assert "beschloss" in decision_link.excerpt
    assert vote_link.missing_source is False
    assert 1 in vote_link.line_indices


def test_summary_review_warns_when_decision_keyword_is_missing_from_summary():
    lines = [
        {
            "speaker": "SPEAKER_00",
            "text": "Der Antrag wurde abgelehnt.",
            "start": 20,
            "end": 24,
        }
    ]

    review = summarize.build_summary_review(
        structured=summarize.StructuredSummary(
            discussion=["Der Antrag wurde diskutiert."]
        ),
        summary="Der Antrag wurde diskutiert.",
        lines=lines,
    )

    warning = next(
        warning
        for warning in review.warnings
        if warning.kind == "missing_decision_signal"
    )
    assert warning.keyword == "abgelehnt"
    assert warning.line_indices == [0]
    assert warning.start == 20
    assert "abgelehnt" in warning.message


def test_failed_partial_repair_does_not_repeat_successful_summary(fake_openai_module, monkeypatch, tmp_path):
    monkeypatch.setenv('LLM_CACHE_DIR', str(tmp_path))
    monkeypatch.setenv('LLM_REPAIR_SPLIT_DEPTH', '1')
    monkeypatch.setattr(summarize, 'LLM_MAX_RETRIES', 0)
    monkeypatch.setattr(summarize, 'LLM_CHUNK_CHARS', 250)
    text = 'A: ' + 'Haushaltsberatung. ' * 10 + '\nB: ' + 'Schulbaufinanzierung. ' * 10
    fake_openai_module.responses = [structured_response(discussion=['Erfolgreicher Teil.']),
                                    TimeoutError('offline'), TimeoutError('offline')]
    with pytest.raises(summarize.LLMCallError, match='Teilzusammenfassung unvollständig'):
        summarize.summarize_segment('Haushalt', text)
    first_calls = fake_openai_module.instances[0].calls
    assert len(first_calls) == 3
    assert len(list(tmp_path.iterdir())) == 1
    fake_openai_module.responses = [structured_response(discussion=['Zweiter Teil.'])]
    result = summarize.summarize_segment('Haushalt', text)
    assert 'Erfolgreicher Teil.' in result.summary
    assert 'Zweiter Teil.' in result.summary
    assert result.llm_usage['cached_calls'] == 1
    assert result.llm_usage['attempted_calls'] == 1


def test_summary_budget_splits_without_losing_text(fake_openai_module, monkeypatch):
    monkeypatch.setenv('LLM_CONTEXT_TOKENS', '8192')
    monkeypatch.setattr(summarize, 'LLM_CHUNK_CHARS', 10000)
    fake_openai_module.content = structured_response(discussion=['Beratung.'])
    text = 'A: ' + 'ä Haushaltsberatung. ' * 200
    result = summarize.summarize_segment('Haushalt', text, system_prompt='Fachkontext')
    calls = fake_openai_module.instances[0].calls
    assert len(calls) > 1
    from llm_transport import fits
    assert all(fits(c['messages'], c['max_tokens']) for c in calls)
    inputs = ''.join(c['messages'][-1]['content'].split('\nRandkontext')[0] for c in calls)
    assert inputs.count('Haushaltsberatung.') == 200
    assert result.chunks_processed == len(calls)


def test_successful_summary_repair_is_reused_without_failed_parent(fake_openai_module, monkeypatch, tmp_path):
    monkeypatch.setenv('LLM_CACHE_DIR', str(tmp_path))
    monkeypatch.setenv('LLM_REPAIR_SPLIT_DEPTH', '1')
    monkeypatch.setattr(summarize, 'LLM_MAX_RETRIES', 0)
    text = 'A: ' + 'Beratung. ' * 30 + '\nB: ' + 'Abstimmung. ' * 30
    fake_openai_module.responses = [TimeoutError('offline'),
        structured_response(discussion=['Beratung.']), structured_response(votes=['Einstimmig.'])]
    result = summarize.summarize_segment('Haushalt', text)
    assert result.llm_usage['attempted_calls'] == 3
    fake_openai_module.responses = []
    resumed = summarize.summarize_segment('Haushalt', text)
    assert resumed.summary == result.summary
    assert resumed.llm_usage.get('attempted_calls', 0) == 0
    assert resumed.llm_usage['cached_summary']
    assert resumed.llm_usage['original_usage']['attempted_calls'] == 3


def test_requested_off_record_passage_remains_visible_for_manual_review():
    review = summarize.build_summary_review(
        structured=None, summary='Sachdebatte.',
        lines=[{'text': 'Das bitte außerhalb des Protokolls besprechen.', 'start': 12, 'end': 15}])
    warning = next(w for w in review.warnings if w.kind == 'recording_scope')
    assert warning.line_indices == [0]
    assert warning.start == 12
    assert 'außerhalb des Protokolls' in warning.excerpt


def test_missing_vote_gets_bounded_fact_review_and_is_cached(fake_openai_module, monkeypatch, tmp_path):
    monkeypatch.setenv('LLM_SUMMARY_FACT_REVIEW_MAX_CALLS', '1')
    monkeypatch.setenv('LLM_CACHE_DIR', str(tmp_path))
    fake_openai_module.responses = [structured_response(discussion=['Rederecht wurde erörtert.']),
        structured_response(decisions=['Das Rederecht wird erteilt.'], votes=['Einstimmige Zustimmung.'])]
    text = 'Rederecht für den Gast. Bitte Handzeichen. Danke, das ist einstimmig.'
    result = summarize.summarize_segment('Eröffnung', text, system_prompt='Anonymisierte Darstellung.')
    assert result.structured.votes == ['Einstimmige Zustimmung.']
    assert result.llm_usage['fact_review_calls'] == 1
    assert result.llm_usage['attempted_calls'] == 2
    assert 'Anonymisierte Darstellung.' in fake_openai_module.instances[0].calls[1]['messages'][0]['content']
    resumed = summarize.summarize_segment('Eröffnung', text, system_prompt='Anonymisierte Darstellung.')
    assert resumed.summary == result.summary
    assert resumed.llm_usage.get('attempted_calls', 0) == 0
    assert resumed.llm_usage['cached_summary']
    assert resumed.llm_usage['original_usage']['fact_review_calls'] == 1


def test_fact_review_may_confirm_only_a_retrospective_report(fake_openai_module, monkeypatch):
    monkeypatch.setenv('LLM_SUMMARY_FACT_REVIEW_MAX_CALLS', '1')
    fake_openai_module.responses = [structured_response(decisions=['Die Satzung wurde beschlossen.']),
        structured_response(discussion=['Bericht über den früheren Satzungsbeschluss der Verbandsversammlung.'])]
    result = summarize.summarize_segment('Informationen', 'In der letzten Sitzung der Verbandsversammlung wurde die Satzung beschlossen.')
    assert result.structured.decisions == []
    assert result.structured.votes == []
    assert 'früheren' in result.summary


def test_fact_review_limit_is_visible_without_unbounded_calls(fake_openai_module, monkeypatch, tmp_path):
    monkeypatch.setenv('LLM_SUMMARY_FACT_REVIEW_MAX_CALLS', '1')
    monkeypatch.setenv('LLM_CACHE_DIR', str(tmp_path))
    monkeypatch.setattr(summarize, 'split_transcript_into_chunks', lambda text: ['Einstimmig zu A.', 'Einstimmig zu B.'])
    fake_openai_module.responses = [structured_response(discussion=['Beratung A.']),
        structured_response(votes=['Einstimmige Zustimmung A.']), structured_response(discussion=['Beratung B.'])]
    result = summarize.summarize_segment('Anträge', 'Einstimmig zu A. Einstimmig zu B.')
    assert result.llm_usage['fact_review_calls'] == 1
    assert result.llm_usage['attempted_calls'] == 3
    assert result.llm_usage['fact_review_limit_reached']
    assert any('ausgeschöpft' in item for item in result.structured.uncertainties)
    resumed = summarize.summarize_segment('Anträge', 'Einstimmig zu A. Einstimmig zu B.')
    assert resumed.llm_usage['cached_summary']
    assert resumed.llm_usage['attempted_calls'] == 0
    assert resumed.summary == result.summary


def test_omitted_negative_result_gets_source_review(fake_openai_module, monkeypatch):
    monkeypatch.setenv('LLM_SUMMARY_FACT_REVIEW_MAX_CALLS', '1')
    fake_openai_module.responses = [structured_response(discussion=['Die Fragestunde wird eröffnet.']),
        structured_response(discussion=['Es gibt keine Wortmeldungen der Einwohner.'])]
    result = summarize.summarize_segment('Einwohnerfragestunde', 'Gibt es Wortmeldungen? Das ist nicht der Fall.')
    assert 'keine Wortmeldungen' in result.summary
    assert result.llm_usage['fact_review_calls'] == 1


def test_person_mentioned_in_speech_is_not_automatically_its_speaker():
    review = summarize.build_summary_review(
        structured=summarize.StructuredSummary(discussion=['Frau Muster (SPEAKER_01) bestätigt die Prüfung.']),
        summary='Frau Muster (SPEAKER_01) bestätigt die Prüfung.',
        lines=[{'speaker': 'SPEAKER_01', 'text': 'Frau Muster wird das prüfen.', 'start': 1, 'end': 3}])
    assert any(w.kind == 'speaker_reference' for w in review.warnings)


def test_conflicting_centuries_remain_source_warnings_without_date_correction():
    review = summarize.build_summary_review(
        structured=None, summary='Ein Termin wird diskutiert.', lines=[
            {'text': 'Vorgeschlagen ist 2028.', 'start': 1, 'end': 2},
            {'text': 'Dann wäre das 1928.', 'start': 3, 'end': 4}])
    warning = next(w for w in review.warnings if w.kind == 'date_conflict')
    assert warning.line_indices == [0, 1]
    assert '1928' in warning.message and '2028' in warning.message
    assert '1928' in warning.excerpt


def test_summary_boundary_keeps_quotation_context_without_duplicating_targets(fake_openai_module, monkeypatch):
    first = 'A: Ich lese jetzt den Auftrag der früheren Sitzung vor.'
    second = 'A: Der Vorsitzende beauftragt die Verwaltung.'
    monkeypatch.setattr(summarize, 'LLM_CHUNK_CHARS', len(first)+1)
    fake_openai_module.content = structured_response(discussion=['Bericht über einen früheren Auftrag.'])
    result = summarize.summarize_segment('Informationen', first+'\n'+second)
    calls = fake_openai_module.instances[0].calls
    assert len(calls) == 2
    user = calls[1]['messages'][-1]['content']
    target, context = user.split('\nRandkontext')
    assert first not in target and second in target
    assert first in context and 'kein Auftrag der heutigen Sitzung' in context
    assert result.llm_usage['source_parts'] == [
        {'start_char': 0, 'end_char': len(first)},
        {'start_char': len(first)+1, 'end_char': len(first)+1+len(second)}]


def test_fact_budget_subdivision_does_not_require_failure_retries(fake_openai_module, monkeypatch):
    monkeypatch.setenv('LLM_REPAIR_SPLIT_DEPTH', '0')
    monkeypatch.setenv('LLM_SUMMARY_FACT_REVIEW_MAX_CALLS', '3')
    fake_openai_module.content = structured_response(discussion=['Beratung mit Abstimmung.'])
    real_fits = summarize.fits
    def limited_fact_budget(messages, output, config=None):
        if messages[0]['content'].startswith('Lies die Quelle unabhängig'):
            target = messages[-1]['content'].split('Vollständiger Quellausschnitt:\n')[1].split('\nRandkontext')[0]
            return len(target) <= 500
        return real_fits(messages, output, config)
    monkeypatch.setattr(summarize, 'fits', limited_fact_budget)
    text = '\n'.join('A: Einstimmig. ' + 'Beratung. ' * 20 for _ in range(4))
    result = summarize.summarize_segment('Anträge', text)
    assert result.llm_usage['budget_splits'] >= 1
    assert result.llm_usage.get('repair_splits', 0) == 0
    assert result.llm_usage['fact_review_calls'] <= 3
    parts = result.llm_usage['source_parts']
    normalized = '\n'.join(line.rstrip() for line in text.strip().splitlines())
    assert parts[0]['start_char'] == 0 and parts[-1]['end_char'] == len(normalized)
    assert all(a['end_char'] == b['start_char'] for a, b in zip(parts, parts[1:]))


@pytest.mark.parametrize('status,category,transient', [(429, 'rate_limit', True), (503, 'server', True), (400, 'client', False)])
def test_native_http_status_has_same_retry_classification(status, category, transient):
    import httpx
    response = httpx.Response(status, request=httpx.Request('POST', 'http://example.test/api/chat'))
    error = httpx.HTTPStatusError('test', request=response.request, response=response)
    result = summarize.classify_llm_error(error)
    assert result.category == category
    assert result.transient is transient


def test_future_protocol_approval_does_not_trigger_missing_current_decision_review(fake_openai_module, monkeypatch):
    monkeypatch.setenv('LLM_SUMMARY_FACT_REVIEW_MAX_CALLS', '3')
    fake_openai_module.content = structured_response(discussion=['Das Protokoll ist in einer späteren Sitzung zu behandeln.'])
    result = summarize.summarize_segment('Verfahrensfragen',
        'Das Protokoll muss in der nächsten Sitzung beschlossen werden.')
    assert result.summary
    assert not result.llm_usage.get('fact_review_calls')
    assert len(fake_openai_module.instances[0].calls) == 1


def test_conditional_closing_does_not_trigger_missing_negative_review(fake_openai_module, monkeypatch):
    monkeypatch.setenv('LLM_SUMMARY_FACT_REVIEW_MAX_CALLS', '3')
    fake_openai_module.content = structured_response(discussion=['Die Sitzung wurde um 19:21 Uhr geschlossen.'])
    result = summarize.summarize_segment('Schließung',
        'Wenn das nicht der Fall ist, dann schließe ich die Sitzung um 19:21 Uhr. Danke allen.')
    assert result.summary
    assert not result.llm_usage.get('fact_review_calls')
    assert len(fake_openai_module.instances[0].calls) == 1
