"""Synthetic adversarial source contracts, not claims about measured model accuracy."""
import json
import re
from copy import deepcopy
import pytest

from source_contract import SourceCatalog, reviewed
import agenda_llm
import durable_jobs as durable
import persistence
from test_agenda_llm import run
from test_summary_grounding import generate, question


@pytest.mark.parametrize('count', [1,2,9,10,11,99,100,101,999,1000,1774,10000])
def test_source_alias_grammar_has_exact_decimal_bounds_without_linear_schema(count):
    catalog = SourceCatalog([{'line_id':str(i), 'text':'Original'} for i in range(count)])
    schema = catalog.alias_schema()
    pattern = re.compile(schema['pattern'])
    assert len(json.dumps(schema)) < 500
    assert [i for i in range(count*2+2) if pattern.fullmatch(f'L{i}')] == list(range(1,count+1))
    assert not any(pattern.fullmatch(s) for s in ['L01','L-1','L1x',' L1','L1\n'])


def test_short_ids_are_lossless_even_with_repeated_text():
    rows = [dict(line_id='long-original-identity-' + str(i), text='Nicht angenommen.', index=i) for i in range(2)]
    catalog = SourceCatalog(rows)
    assert catalog.translate(catalog.translate(rows), decode=True) == rows
    selected = catalog.prepare(catalog.translate({'evidence': [{'line_id': 'L2'}]}, decode=True))
    assert selected['evidence'] == [{'line_id': rows[1]['line_id'], 'quote': 'Nicht angenommen.'}]
    assert selected['grounding']['evidence_status'] == 'source_range'
    assert selected['grounding']['content_status'] == 'unreviewed'
    assert reviewed(selected['grounding'], supported=True)['evidence_status'] == 'exact'
    assert catalog.manifest()['references'][0]['original_sha256'] != catalog.manifest()['references'][1]['original_sha256']


@pytest.mark.parametrize('quote', ['Angenommen mit 12:3.', 'Nicht angenommen mit 13:3.', 'angenommen mit 12:3.'])
def test_changed_numbers_and_negation_are_never_copy_repairs(quote):
    catalog = SourceCatalog([dict(line_id='original', text='Nicht angenommen mit 12:3.')])
    evidence, grounding = catalog.inspect([dict(line_id='original', quote=quote)])
    # A substring may be verbatim while still reversing the meaning. Structural
    # validity alone NEVER makes it supported; the semantic reviewer must decide.
    assert grounding['evidence_status'] == 'source_range'
    assert grounding['content_status'] == 'unreviewed'
    if quote.startswith(('Angenommen', 'Nicht')):
        assert not evidence and grounding['diagnostics'][0]['code'] == 'invalid_source_quote'


def test_whitespace_only_unique_copy_repair_and_ambiguous_match():
    catalog = SourceCatalog([dict(line_id='a', text='12\n gegen 3. Ja  nein. Ja\n nein.')])
    evidence, g = catalog.inspect([dict(line_id='a', quote='12 gegen 3.')])
    assert evidence[0]['quote'] == '12\n gegen 3.'
    assert g['diagnostics'][0]['code'] == 'format_copy_error'
    evidence, g = catalog.inspect([dict(line_id='a', quote='Ja nein.')])
    assert not evidence and g['reference_status'] == 'source_range'
    evidence, g = catalog.inspect([dict(line_id='unknown', quote='12 gegen 3.')])
    assert not evidence and g['evidence_status'] == 'unsupported'


def test_model_cannot_supply_its_own_certificate():
    catalog = SourceCatalog([dict(line_id='a', text='Quelle')])
    result = catalog.prepare(dict(evidence=[dict(line_id='missing')], grounding=dict(evidence_status='exact')))
    assert result['grounding']['evidence_status'] == 'unsupported'


def test_bad_context_keeps_both_original_readers_and_detail_checks(agenda_model, monkeypatch):
    monkeypatch.setenv('LLM_CONTEXT_TOKENS', '16384')
    texts = [f'Quelle {i}. ' + 'Beratung und Wiederaufnahme. '*10 for i in range(90)]
    def bad(body):
        return dict(narrative='Unbestätigter Kontextentwurf.', evidence=[dict(line_id='does-not-exist', quote='Erfunden')])
    agenda_model.overrides['primary:context'] = bad
    # Scripted reconstruction requests originals instead of trusting this note.
    def reconstruct(body):
        if 'requested_originals' not in body:
            return dict(source_ranges=[dict(start=0, end=1)], narrative='Originale benötigt', episodes=[], agenda_states=[])
        modified = deepcopy(body)
        modified['context'] = {'original_transcript': body['requested_originals']}
        return agenda_model.answer(modified)
    agenda_model.overrides['primary:reconstruct:trajectory:v1'] = reconstruct
    agenda_model.overrides['primary:reconstruct:states:v1'] = reconstruct
    result = run(texts)
    assert result.llm.processing_complete and result.llm.review_complete
    notes = result.llm.provenance['context_archive']
    assert any(n['grounding']['evidence_status'] == 'unsupported' for n in notes)
    for role in ['primary', 'independent']:
        assert [r['index'] for body, _ in agenda_model.calls if body['phase'] == role+':detail' for r in body['target_lines']] == list(range(90))
    assert any(body.get('requested_originals') for body, _ in agenda_model.calls)


def test_invalid_evidence_stays_open_if_adjudication_does_not_fix_it(agenda_model):
    def bad(body):
        answer = agenda_model.answer(body)
        for row in answer['lines']:
            row['evidence'] = [dict(line_id='missing')]
        return answer
    agenda_model.overrides.update({'primary:detail': bad, 'resolve:detail': bad})
    result = run(['Beschluss unklar.'])
    assert result.llm.processing_complete and result.llm.review_complete and result.llm.review_required
    row = result.llm.line_results[0]
    assert row['grounding']['evidence_status'] == 'unsupported' and row['review_status'] == 'unresolved'
    assert row['evidence'] == []


def test_exact_quote_does_not_override_semantic_contradiction(summary_model):
    def contradiction(body):
        answer = question(body)
        answer['issues'][0].update(kind='contradiction', question='Die Quelle verneint den Beschluss. Muss dieser Kandidat verworfen werden?')
        return answer
    summary_model.overrides['final_review'] = contradiction
    result = generate()
    assert result.llm_usage['processing_complete']
    assert result.structured.discussion == []
    assert result.structured.rejected_candidates[0]['grounding']['content_status'] == 'contradicted'
    assert '[VERWORFEN' in result.summary
    assert result.llm_usage['reconciliation_rounds'] == 1


def test_unknown_reference_survives_rewording_and_export(summary_model):
    summary_model.overrides['generate'] = dict(considered_source_ids=['T:0:0','T:1:0'], claims=[dict(
        section='votes', text='Antrag mit 12:3 angenommen.', scope='current', evidence=[dict(source_id='missing')])])
    def consolidate(body):
        claim = deepcopy(body['candidate'][0])
        claim['text'] = 'Annahme mit zwölf Stimmen.'
        return dict(claims=[claim], considered_source_ids=body['source_catalog'])
    summary_model.overrides['consolidate'] = consolidate
    result = generate()
    assert result.llm_usage['review_required']
    assert all(c['grounding']['evidence_status'] != 'exact' for c in result.structured.evidence)
    from export_protocol import build_protocol_document, ProtocolMetadata, render_protocol
    for text in [result.summary, 'Beschluss:\nAntrag angenommen.']:
        document = build_protocol_document(metadata=ProtocolMetadata(), tops=['TOP'], summaries={0:text},
            summary_reviews={0:dict(structured=result.structured.to_dict())})
        output = render_protocol(document, 'txt').decode()
        assert 'Prüfentwurf' in output and 'UNBESTÄTIGT' in output and 'Prüffrage:' in output


def test_repair_preserves_untargeted_claim(summary_model):
    def draft(body):
        return dict(considered_source_ids=[r['source_id'] for r in body['source']], claims=[dict(
            section='discussion', text=text, scope='current', evidence=[dict(source_id='T:0:0')]) for text in ['Offen', 'Gültiger Teil']])
    summary_model.overrides['generate'] = draft
    summary_model.overrides['final_review'] = question
    def repair(body):
        claims = deepcopy(body['candidate'])
        claims[0]['text'] = 'Korrigiert'
        claims[1]['text'] = 'Unzulässig geändert'
        summary_model.overrides.pop('final_review')
        return dict(claims=claims, considered_source_ids=body['source_catalog'])
    summary_model.overrides['reconcile'] = repair
    result = generate()
    assert 'Gültiger Teil' in result.summary and 'Unzulässig geändert' not in result.summary


def test_draft_checkpoint_and_attempts_are_not_completed_steps():
    from test_durable_jobs import claimed
    job = durable.submit('agenda', {})
    with claimed(job):
        assert durable.draft_checkpoint('test', lambda: {'complete':False}, lambda v:v['complete']) == {'complete':False}
        durable.artifact('test', 'model_attempt', {'raw':'private synthetic answer'})
    with persistence.connect() as db:
        assert not db.execute("select 1 from durable_steps where step_key='test'").fetchone()
        assert db.execute("select count(*) from durable_artifacts where step_key='test'").fetchone()[0] == 2
    assert 'private synthetic answer' not in json.dumps(durable.public(durable.load(job['job_id'])))


def test_last_delta_age_advances_after_failure(monkeypatch):
    monkeypatch.setattr(durable.time, 'time', lambda: 130)
    result = durable.public(dict(kind='agenda', state='failed', progress=dict(last_delta_at=100,silence_seconds=0)))
    assert result['progress']['silence_seconds'] == 30


def test_failed_independent_summary_retains_marked_partial(summary_model):
    import summarize
    summary_model.overrides['final_review'] = ValueError('synthetic malformed response')
    with pytest.raises(summarize.StructuredOutputError) as error:
        generate()
    partial=error.value.partial_result
    assert not partial.llm_usage['processing_complete']
    assert '[UNBESTÄTIGT' in partial.summary
    assert all(e['grounding']['evidence_status'] != 'exact' for e in partial.structured.evidence)
