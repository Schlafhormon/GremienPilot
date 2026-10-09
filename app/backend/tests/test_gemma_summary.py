import json
from types import SimpleNamespace

import pytest
import gemma_summary as gemma
import summary_grounding
import summarize
from llm_config import LLMConfig
from processing_mode import processing_scope


def test_exact_training_framing_and_merged_speakers():
    messages = gemma.protocol_messages('2. Haushalt', ['Müller: Wir beraten.', 'Müller: Die Kosten betragen 5 Euro.',
                                                     'SPEAKER_07: Ich stimme zu.'])
    assert len(messages) == 1 and messages[0]['role'] == 'user'
    assert messages[0]['content'] == gemma.PROMPT + (
        '\n\nErstelle eine Zusammenfassung für folgenden Tagesordnungspunkt:\n\n'
        'TOP: 2. Haushalt\n\nTranskript:\n'
        'Müller: Wir beraten. Die Kosten betragen 5 Euro.\nSPEAKER_07: Ich stimme zu.\n\nZusammenfassung:')
    assert gemma.transcript_turns(['A: eins', 'B: zwei', 'A: drei']).count('A:') == 2


def test_displayed_prompt_matches_inference_prompt():
    from pathlib import Path
    import re
    source = (Path(__file__).resolve().parents[2] / 'frontend/src/components/LLMSettingsPanel.tsx').read_text(encoding='utf-8')
    assert re.search(r'export const GEMMA_SYSTEM_PROMPT = `([^`]+)`;', source)[1] == gemma.PROMPT


DRAFT = '## Zu TOP 2:\n\n**Aus der Beratung**\n\nMüller erkläre, die Kosten betrügen 9 Euro.\n\nBeschlüsse und Festlegungen:\nKein Beschluss.\n'
ORIGINAL = ['Müller: Die Kosten betragen 5 Euro.', 'A: Es gibt  keinen Beschluss.']
FINDING = dict(kind='contradiction', question='Im Entwurf stehen 9 Euro, im Original 5 Euro. Bitte prüfen.', excerpts=['9 Euro', '5 Euro'])


def config(mode='slow', context=16384):
    return LLMConfig(base_url='http://llama:8080/v1', model='gemma-4-31b', api_key='local',
        provider='llama-cpp', summary_style='gemma4-lora', context_tokens=context,
        processing_mode=mode, reasoning_effort='none')


def response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def setup_model(monkeypatch, mode='slow', review=None, draft=DRAFT, style='gemma4-lora'):
    cfg = config(mode).with_summary_style(style)
    calls = []
    monkeypatch.setattr(summarize, 'get_llm_config', lambda model=None: cfg)
    monkeypatch.setattr(summarize, '_load_openai_client', lambda _: None)
    monkeypatch.setattr(summarize, 'check_llm_availability', lambda **_: None)
    def complete(client, settings, **kwargs):
        calls.append((settings, kwargs))
        if 'response_format' not in kwargs:
            return response(draft if isinstance(draft, str) else draft(kwargs))
        body = json.loads(kwargs['messages'][1]['content'])
        return response(json.dumps(review(body) if review else dict(complete=True, issues=[FINDING])))
    monkeypatch.setattr(gemma, 'complete', complete)
    return calls


def generate(mode='slow', lines=None, **preferences):
    lines = ORIGINAL if lines is None else lines
    return summarize.summarize_segment('2. Kosten', '\n'.join(lines), source_lines=lines, processing_mode=mode, **preferences)


@pytest.mark.parametrize('mode', ['fast', 'slow'])
def test_only_protocol_and_direct_reviews_preserve_entire_prose(monkeypatch, mode):
    calls = setup_model(monkeypatch, mode)
    result = generate(mode)
    assert result.summary == result.structured.protocol_text == DRAFT
    assert result.structured.discussion == result.structured.decisions == result.structured.evidence == []
    assert result.llm_usage['required_checks'] == ['protocol'] + ([] if mode == 'fast' else ['final_review', 'consolidated_review'])
    assert result.llm_usage['processing_complete'] is True
    assert result.llm_usage['review_complete'] is (mode == 'slow')
    assert len(calls) == (1 if mode == 'fast' else 3)
    assert calls[0][0].lora_scale == 1
    assert 'response_format' not in calls[0][1]
    assert calls[0][1]['messages'] == gemma.protocol_messages('2. Kosten', ORIGINAL)
    for settings, request in calls[1:]:
        assert settings.lora_scale == 0
        body = json.loads(request['messages'][1]['content'])
        assert body['draft'] == DRAFT
        assert body['original_transcript'] == '\n'.join(ORIGINAL)
        assert set(body) == {'phase', 'draft', 'original_transcript'}
        assert 'source_id' not in json.dumps(request['response_format'])
        assert 'section' not in json.dumps(request['response_format'])
    review = summarize.build_summary_review(structured=result.structured, summary=result.summary, lines=[dict(speaker=line.split(': ', 1)[0], text=line.split(': ', 1)[1]) for line in ORIGINAL])
    assert review.source_links == []
    assert not any(w.kind in ('missing_source', 'open_evidence') for w in review.warnings)
    if mode == 'fast':
        assert result.structured.review_questions == []
        assert [w.kind for w in review.warnings] == ['review_skipped']
    else:
        assert all(w.message == FINDING['question'] for w in review.warnings)
        assert all(w.excerpt == '9 Euro … 5 Euro' for w in review.warnings)


def test_slow_without_findings_is_checked_but_does_not_invent_confirmation(monkeypatch):
    setup_model(monkeypatch, review=lambda _: dict(complete=True, issues=[]))
    result = generate()
    assert result.llm_usage['review_complete'] is True
    assert result.structured.review_questions == []
    assert summarize.build_summary_review(structured=result.structured, summary=result.summary, lines=[dict(speaker=line.split(': ', 1)[0], text=line.split(': ', 1)[1]) for line in ORIGINAL]).warnings == []


@pytest.mark.parametrize('mode', ['fast', 'slow'])
@pytest.mark.parametrize('style', ['gemma4-lora', 'gemma4-custom'])
def test_long_top_retains_all_parts_and_all_original_characters(monkeypatch, style, mode):
    lines = ['Müller: ' + ('Satz mit Zahl 5. ' * 1500), 'B: Ende.']
    serial = []
    def draft(request):
        text = f'## Teil {len(serial)}\n\nUnveränderter Absatz.\n'
        serial.append((text, request['messages'][0]['content']))
        return text
    calls = setup_model(monkeypatch, mode, style=style, draft=draft, review=lambda _: dict(complete=True, issues=[]))
    result = generate(mode, lines)
    assert len(serial) > 1
    assert result.summary == '\n\n'.join(text for text, _ in serial)
    assert result.chunks_processed == len(serial)
    assert len(calls) == len(serial) * (1 if mode == 'fast' else 3)
    reviews = [json.loads(k['messages'][1]['content']) for c, k in calls if 'response_format' in k]
    if mode == 'slow':
        for phase in ('final_review', 'consolidated_review'):
            windows = [r for r in reviews if r['phase'] == phase]
            assert [r['draft'] for r in windows] == [text for text, _ in serial]
            assert sum(r['original_transcript'].count('Satz mit Zahl 5.') for r in windows) == 1500
            assert windows[-1]['original_transcript'].endswith('B: Ende.')
    # Internal identities account for every original byte; none are emitted as paragraph evidence.
    with processing_scope(mode):
        w = gemma.GemmaWorkflow(None, config(mode).with_summary_style(style), '2. Kosten', '', {})
        w.run(lines)
    for index, line in enumerate(lines):
        assert ''.join(r['text'] for r in w.rows if r['line_index'] == index) == line


@pytest.mark.parametrize('failure', ['timeout', 'incomplete', 'malformed', 'invented_excerpt'])
@pytest.mark.parametrize('style', ['gemma4-lora', 'gemma4-custom'])
def test_failed_second_review_retains_verbatim_draft_and_prior_findings(monkeypatch, style, failure):
    def review(body):
        if body['phase'] == 'final_review':
            return dict(complete=True, issues=[FINDING])
        if failure == 'timeout':
            raise TimeoutError('private backend error')
        if failure == 'incomplete':
            return dict(complete=False, issues=[])
        if failure == 'invented_excerpt':
            return dict(complete=True, issues=[dict(FINDING, excerpts=['Erfundener Ausschnitt'])])
        return dict(issues=[])
    setup_model(monkeypatch, style=style, review=review)
    with pytest.raises((summarize.LLMCallError, summarize.StructuredOutputError)) as caught:
        generate()
    partial = caught.value.partial_result
    assert partial.summary == partial.structured.protocol_text == DRAFT
    assert partial.llm_usage['generation_complete'] is True
    assert partial.llm_usage['review_complete'] is False
    assert partial.llm_usage['processing_complete'] is False
    assert partial.llm_usage['review_status'] == 'incomplete'
    assert len(partial.structured.review_questions) == 1
    review = summarize.build_summary_review(structured=partial.structured, summary=partial.summary, lines=[dict(speaker=line.split(': ', 1)[0], text=line.split(': ', 1)[1]) for line in ORIGINAL])
    assert review.source_links == []
    assert [w.kind for w in review.warnings] == ['contradiction', 'technical_incomplete']


@pytest.mark.parametrize('style', ['gemma4-lora', 'gemma4-custom'])
def test_oversized_draft_is_retained_without_truncated_review(monkeypatch, style):
    text = 'Überschrift\n\n' + 'Wort ' * 4000
    calls = setup_model(monkeypatch, style=style, draft=text)
    with pytest.raises(gemma.ContextBudgetError) as caught:
        generate()
    assert caught.value.partial_result.summary == text
    assert len(calls) == 1
    assert caught.value.partial_result.llm_usage['review_complete'] is False


def test_partial_utterance_retains_speaker():
    w = gemma.GemmaWorkflow(None, config(), 'TOP 1', '', {})
    w.lines = ['Müller: Ein langer Beitrag.']
    assert w.source_text([{'line_index': 0, 'start_char': 15, 'text': 'Beitrag.'}]) == ['Müller: Beitrag.']


@pytest.mark.parametrize('style', ['gemma4-lora', 'gemma4-custom'])
def test_resume_reuses_draft_and_successful_review_only(monkeypatch, style):
    import threading
    import durable_jobs as jobs
    calls = setup_model(monkeypatch, style=style)
    job = jobs.submit('test', {})
    current = jobs.claim('test-worker', 60)
    token = jobs.CURRENT.set(jobs.Runtime(current, 'test-worker', threading.Event()))
    try:
        original_complete = gemma.complete
        def fail_second(client, cfg, **kwargs):
            if 'response_format' in kwargs and json.loads(kwargs['messages'][1]['content'])['phase'] == 'consolidated_review':
                raise TimeoutError()
            return original_complete(client, cfg, **kwargs)
        monkeypatch.setattr(gemma, 'complete', fail_second)
        with pytest.raises(summarize.LLMCallError):
            generate()
        assert len(calls) == 2
        monkeypatch.setattr(gemma, 'complete', original_complete)
        result = generate()
        assert result.summary == DRAFT and result.llm_usage['review_complete'] is True
        assert len(calls) == 3
        assert sum('response_format' not in k for _, k in calls) == 1
        assert len(result.structured.verification['completed_checks']) == 2
    finally:
        jobs.CURRENT.reset(token)


def test_version_changes_cache_and_checkpoint_identity(monkeypatch):
    calls = setup_model(monkeypatch, mode='fast')
    keys, steps = [], []
    monkeypatch.setattr(gemma, 'cache_read', lambda key: keys.append(key))
    monkeypatch.setattr(gemma.durable, 'checkpoint', lambda step, operation: steps.append(step) or operation())
    generate('fast')
    monkeypatch.setattr(gemma, 'VERSION', 'gemma4-protokoll-v2')
    generate('fast')
    assert len(calls) == 2
    assert keys[0] != keys[1] and steps[0] != steps[1]


def test_re_rendering_old_annotated_results_preserves_protocol_style():
    structured = summarize.StructuredSummary(discussion=['Müller erläutere den Plan.'],
        decisions=['Einstimmig beschlossen.'], verification={'summary_style': 'gemma4-lora'})
    assert summarize.render_structured_summary(structured) == gemma.render_protocol(structured)
    assert 'Beschlüsse und Festlegungen:\nEinstimmig beschlossen.' in gemma.render_protocol(structured)


def test_streaming_progress_preserves_actual_summary_phase():
    import threading
    import durable_jobs as jobs
    job = jobs.submit('test', {})
    current = jobs.claim('phase-worker', 60)
    token = jobs.CURRENT.set(jobs.Runtime(current, 'phase-worker', threading.Event()))
    try:
        jobs.progress({'phase': 'summary_final_review'})
        jobs.progress({'phase': 'generating', 'elapsed_seconds': 5})
        assert jobs.load(job['job_id'])['progress']['summary_phase'] == 'summary_final_review'
        jobs.progress({'phase': 'summary_protocol'})
        jobs.progress({'phase': 'loading', 'elapsed_seconds': 1})
        assert jobs.load(job['job_id'])['progress']['summary_phase'] == 'summary_protocol'
    finally:
        jobs.CURRENT.reset(token)


@pytest.mark.parametrize('style', ['gemma4-lora', 'gemma4-custom'])
def test_generation_failure_after_first_long_top_part_keeps_partial_draft(monkeypatch, style):
    generated = []
    def draft(_):
        if generated:
            raise TimeoutError()
        generated.append(DRAFT)
        return DRAFT
    calls = setup_model(monkeypatch, style=style, draft=draft)
    with pytest.raises(summarize.LLMCallError) as caught:
        generate(lines=['A: ' + 'Langer Satz. ' * 2000])
    partial = caught.value.partial_result
    assert partial.summary == DRAFT
    assert partial.llm_usage['generation_complete'] is False
    assert partial.llm_usage['review_complete'] is False
    assert all('response_format' not in k for c, k in calls)
    review = summarize.build_summary_review(structured=partial.structured, summary=partial.summary,
        lines=[dict(speaker='A', text='Langer Satz. ' * 2000)])
    assert any('Protokollerzeugung' in w.message for w in review.warnings)
