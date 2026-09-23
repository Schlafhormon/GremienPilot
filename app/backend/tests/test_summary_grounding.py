"""Contract/integration tests with scripted replies, not measured model accuracy."""
import json
import threading
from copy import deepcopy

import pytest
import summarize
import summary_grounding as grounding
import durable_jobs as jobs
import persistence
from llm_transport import ContextBudgetError, LLMCancelledError


def generate():
    return summarize.summarize_segment('Beratung', 'A: Sachverhalt.\nB: Fortsetzung.', model='test-model')


def test_every_source_and_final_claim_is_independently_reviewed(summary_model):
    result = generate()
    phases = [body['phase'] for body, _ in summary_model.calls]
    assert phases == ['generate', 'blind_inventory', 'draft_review', 'consolidate', 'final_review', 'consolidated_review']
    blind = summary_model.calls[1][0]
    assert 'candidate' not in blind
    assert [r['source_id'] for r in blind['source']] == ['T:0:0', 'T:1:0']
    assert result.llm_usage['processing_complete']
    assert result.structured.evidence[0]['sources'][0]['quote'] == 'A: Sachverhalt.'
    review = summarize.build_summary_review(structured=result.structured, summary=result.summary,
        lines=[dict(speaker='A', text='Sachverhalt.', start=0, end=2), dict(speaker='B', text='Fortsetzung.', start=2, end=4)])
    assert not review.warnings
    assert review.source_links[0].line_indices == [0]
    assert review.source_links[0].source_ids == ['T:0:0']


@pytest.mark.parametrize('phase', ['generate', 'blind_inventory', 'draft_review', 'consolidate', 'final_review', 'consolidated_review'])
def test_missing_required_call_never_finishes(phase, summary_model):
    summary_model.overrides[phase] = ValueError('broken schema')
    with pytest.raises(summarize.StructuredOutputError):
        generate()


@pytest.mark.parametrize('phase', ['draft_review', 'final_review', 'consolidated_review'])
def test_coverage_cannot_be_asserted_with_missing_ids(phase, summary_model):
    summary_model.overrides[phase] = dict(checked_claim_ids=[], considered_source_ids=['T:0:0'], issues=[])
    with pytest.raises(summarize.StructuredOutputError):
        generate()


def test_unknown_quote_cannot_become_source_link(summary_model):
    summary_model.overrides['generate'] = dict(considered_source_ids=['T:0:0', 'T:1:0'], claims=[dict(
        section='decisions', text='Eine erfundene Entscheidung.', scope='current',
        evidence=[dict(source_id='T:0:0', quote='Nicht in der Quelle')])])
    with pytest.raises(summarize.StructuredOutputError):
        generate()


@pytest.mark.parametrize('scope', ['proposal', 'retrospective', 'quoted_prior', 'unclear'])
def test_model_temporal_scope_must_agree_with_outcome_category(scope, summary_model):
    summary_model.overrides['generate'] = dict(considered_source_ids=['T:0:0', 'T:1:0'], claims=[dict(
        section='decisions', text='Die frühere Entscheidung.', scope=scope,
        evidence=[dict(source_id='T:0:0', quote='A: Sachverhalt.')])])
    with pytest.raises(summarize.StructuredOutputError):
        generate()


def question(body):
    return dict(checked_claim_ids=[c['claim_id'] for c in body['candidate']],
        considered_source_ids=[r['source_id'] for r in body['source']], issues=[dict(
            kind='scope', question='Wurde heute entschieden oder nur der frühere Stand berichtet?',
            claim_ids=['C:0'], evidence=[dict(source_id=body['source'][0]['source_id'], quote=body['source'][0]['text'])])])


def test_disagreement_gets_targeted_followup_and_rechecks_changed_final(summary_model):
    summary_model.overrides['final_review'] = question
    def repair(body):
        summary_model.overrides.pop('final_review')
        claims = deepcopy(body['candidate'])
        claims[0]['text'] = 'Präzisierter Sachverhalt.'
        return dict(claims=claims, considered_source_ids=body['source_catalog'])
    summary_model.overrides['reconcile'] = repair
    result = generate()
    assert result.llm_usage['reconciliation_rounds'] == 1
    assert not result.llm_usage['review_required']
    phases = [body['phase'] for body, _ in summary_model.calls]
    assert phases[-3:] == ['reconcile', 'final_review', 'consolidated_review']


def test_unresolved_question_is_concrete_and_has_source_access(summary_model):
    summary_model.overrides['final_review'] = question
    result = generate()
    assert result.llm_usage['processing_complete'] and result.llm_usage['review_required']
    assert result.llm_usage['reconciliation_rounds'] == 1
    review = summarize.build_summary_review(structured=result.structured, summary=result.summary,
        lines=[dict(speaker='A', text='Sachverhalt.'), dict(speaker='B', text='Fortsetzung.')])
    assert review.warnings[0].message.endswith('berichtet?')
    assert review.warnings[0].line_indices == [0]


def test_manual_changes_and_source_edits_cannot_reuse_certificate(summary_model):
    result = generate()
    for summary, lines in [(result.summary+' Manuell.', ['Sachverhalt.', 'Fortsetzung.']),
                           (result.summary, ['Neue Quelle.', 'Fortsetzung.'])]:
        review = summarize.build_summary_review(structured=result.structured, summary=summary,
            lines=[dict(speaker=speaker, text=text) for speaker, text in zip(['A', 'B'], lines)])
        assert not review.source_links
        assert review.warnings[0].kind == 'verification_required'


def test_multiline_utterance_retains_logical_line_identity(summary_model):
    lines = ['A: Erste Zeile.\nFortsetzung.', 'B: Antwort.']
    result = summarize.summarize_segment('Beratung', '\n'.join(lines), source_lines=lines)
    review = summarize.build_summary_review(structured=result.structured, summary=result.summary,
        lines=[dict(speaker='A', text='Erste Zeile.\nFortsetzung.'), dict(speaker='B', text='Antwort.')])
    assert not review.warnings
    assert result.llm_usage['source_line_count'] == 2


def test_context_shortfall_fails_without_freetext_or_silent_truncation(monkeypatch, summary_model):
    monkeypatch.setattr(grounding, 'fits', lambda messages, *args: json.loads(messages[1]['content'])['phase'] != 'consolidate')
    with pytest.raises(ContextBudgetError):
        generate()


def test_long_sources_are_reviewed_without_lost_characters(summary_model, monkeypatch):
    original_fits = grounding.fits
    monkeypatch.setattr(grounding, 'fits', lambda messages, *args: original_fits(messages, *args)
        and len(json.loads(messages[1]['content']).get('source', [])) <= 2)
    text = 'A: ' + 'Langer Beitrag. '*220
    result = summarize.summarize_segment('Beratung', text)
    rows = result.structured.verification['sources']
    assert ''.join(row['text'] for row in rows) == text
    for phase in ['generate', 'blind_inventory', 'final_review', 'consolidated_review']:
        seen = [row['source_id'] for body, _ in summary_model.calls if body['phase'] == phase for row in body['source']]
        assert set(seen) == {row['source_id'] for row in rows}


def test_restart_resumes_only_finished_validated_model_calls(summary_model):
    job = jobs.submit('summary', {})
    summary_model.overrides['final_review'] = jobs.WorkerStopped()
    def runner(_):
        return generate().llm_usage, 'completed'
    manager = jobs.Manager(runner)
    manager.execute(jobs.claim('first', 60))
    assert jobs.load(job['job_id'])['state'] == 'queued'
    summary_model.calls.clear()
    summary_model.overrides.clear()
    with persistence.connect() as db:
        db.execute('UPDATE durable_jobs SET available_at=0')
    manager.execute(jobs.claim('second', 60))
    assert jobs.load(job['job_id'])['state'] == 'completed'
    assert [body['phase'] for body, _ in summary_model.calls] == ['final_review', 'consolidated_review']


def test_cancelled_final_review_cannot_publish(summary_model):
    job = jobs.submit('summary', {})
    def cancel(body):
        jobs.cancel(job['job_id'])
        jobs.check()
    summary_model.overrides['final_review'] = cancel
    manager = jobs.Manager(lambda _: (generate().llm_usage, 'completed'))
    manager.execute(jobs.claim('worker', 60))
    assert jobs.load(job['job_id'])['state'] == 'cancelled'
    assert jobs.load(job['job_id'])['result'] is None


def test_legacy_metadata_has_no_invented_similarity_links():
    review = summarize.build_summary_review(structured=summarize.StructuredSummary(
        decisions=['Haushalt beschlossen.']), summary='Haushalt beschlossen.',
        lines=[dict(speaker='A', text='Haushalt beschlossen.')])
    assert not review.source_links
    assert review.warnings[0].kind == 'verification_required'


def test_joint_model_sources_are_included_but_manual_assignment_wins():
    import main
    lines = [dict(line_id='l0', speaker='A', text='Gemeinsame Beratung.', start=0, end=1)]
    session = dict(transcript=lines, tops=['A', 'B'], top_ids=['a', 'b'], assignments=[None],
        agenda_proposals=dict(source=dict(transcript=deepcopy(lines), top_ids=['a', 'b']),
            result=dict(assignments=[None], llm=dict(provenance=dict(identities=[
                dict(top_id='model:a', top_index=0), dict(top_id='model:b', top_index=1)]),
                line_results=[dict(index=0, top_ids=['model:a', 'model:b'])]))))
    assert main.summary_line_indices(session, 0) == [0]
    assert main.summary_line_indices(session, 1) == [0]
    session['assignments'] = [1]
    assert main.summary_line_indices(session, 0) == []
    assert main.summary_line_indices(session, 1) == [0]
    session['assignments'] = [None]
    session['transcript'][0]['text'] = 'Manuell geändert'
    assert main.summary_line_indices(session, 0) == []


def test_cache_binds_prompt_configuration_and_source(summary_model, monkeypatch, tmp_path):
    monkeypatch.setenv('LLM_CACHE_DIR', str(tmp_path))
    first = generate()
    summary_model.calls.clear()
    assert generate().summary == first.summary
    assert not summary_model.calls
    monkeypatch.setenv('SUMMARY_RECONCILIATION_ROUNDS', '3')
    generate()
    assert len(summary_model.calls) == 6
    summary_model.calls.clear()
    summarize.summarize_segment('Beratung', 'A: Geänderte Quelle.', model='test-model')
    assert len(summary_model.calls) == 6


def test_job_snapshot_records_actual_browser_model_override(monkeypatch):
    job = jobs.submit('summary', dict(legacy_snapshot=dict(refs=dict(model='browser-model'))))
    snapshot = job['payload']['versions']
    assert snapshot['overrides']['model']['model'] == 'browser-model'
    assert 'api_key' not in json.dumps(snapshot)


def test_model_revision_change_prevents_resuming_old_checkpoints(summary_model, monkeypatch):
    job = jobs.submit('summary', {})
    monkeypatch.setenv('LLM_MODEL_REVISION', 'new-weights')
    manager = jobs.Manager(lambda _: (generate().llm_usage, 'completed'))
    manager.execute(jobs.claim('worker', 60))
    assert jobs.load(job['job_id'])['state'] == 'failed'
    assert not summary_model.calls


def test_omission_without_signal_words_is_repaired_by_model(summary_model):
    # Scripted disagreement proves every source is reviewed, even without old keywords.
    def omission(body):
        value = question(body)
        value['issues'][0].update(kind='omission', question='Fehlt die festgelegte Frist zur Rückmeldung?')
        return value
    summary_model.overrides['final_review'] = omission
    def repair(body):
        summary_model.overrides.pop('final_review')
        claim = dict(section='action_items', text='Die Rückmeldung erfolgt morgen.', scope='current',
                     evidence=[dict(source_id='T:1:0', quote='B: Fortsetzung.')])
        return dict(claims=[*body['candidate'], claim], considered_source_ids=body['source_catalog'])
    summary_model.overrides['reconcile'] = repair
    result = generate()
    assert result.structured.action_items == ['Die Rückmeldung erfolgt morgen.']
    final = [body for body, _ in summary_model.calls if body['phase'] == 'consolidated_review'][-1]
    assert len(final['candidate']) == 2
    assert result.llm_usage['processing_complete']


def test_whitespace_is_preserved_in_source_certificate(summary_model):
    lines = ['A:   Original  ', 'B: \nweiter\n']
    result = summarize.summarize_segment('Beratung', '\n'.join(lines), source_lines=lines)
    review = summarize.build_summary_review(structured=result.structured, summary=result.summary,
        lines=[dict(speaker='A', text='  Original  '), dict(speaker='B', text='\nweiter\n')])
    assert not review.warnings


def test_shared_blind_facts_inventory_keeps_independent_reviews_for_each_top(summary_model):
    from test_durable_jobs import claimed
    job = jobs.submit('test', {})
    with claimed(job):
        first = summarize.summarize_segment('TOP A', 'S: Gemeinsame Beratung.', model='test-model')
        second = summarize.summarize_segment('TOP B', 'S: Gemeinsame Beratung.', model='test-model')
    phases = [body['phase'] for body, _ in summary_model.calls]
    assert phases.count('blind_inventory') == 1
    for phase in ['generate', 'draft_review', 'final_review', 'consolidated_review']:
        assert phases.count(phase) == 2
    assert first.llm_usage['processing_complete'] and second.llm_usage['processing_complete']


def test_unchanged_summary_repair_preserves_review_question_without_repeat(summary_model):
    summary_model.overrides['final_review'] = question
    result = generate()
    assert result.llm_usage['review_required']
    assert result.llm_usage['stop_reason'] == 'unchanged_candidate'
    assert [body['phase'] for body, _ in summary_model.calls].count('reconcile') == 1
    assert result.structured.review_questions
