"""Full-coverage contract: no heuristic input selection or silent fallback."""
import json
from types import SimpleNamespace

import pytest

from agenda_detection import segment_known_agenda
from assignment_suggestions import TranscriptUtterance
import agenda_llm


def row(start, end, identity='agenda:0', text='Beratung.', **extra):
    return dict(top_id=identity, start_index=start, end_index=end, evidence_index=start,
                evidence_text=text, reason='Inhaltliche Beratung', confidence=0.9, uncertain=False, **extra)


def classify(fake, rows, texts=None, tops=None):
    fake.content = json.dumps({'tops': rows})
    return segment_known_agenda([TranscriptUtterance('M', t) for t in (texts or ['Beratung.'])],
                                tops or ['1 Haushalt'], use_llm=True)


@pytest.mark.parametrize('value', [True, False, None, 1.2, '0', -1, 9])
@pytest.mark.parametrize('field', ['start_index', 'end_index', 'evidence_index'])
def test_strict_global_indices(fake_openai_module, monkeypatch, value, field):
    monkeypatch.setenv('LLM_REPAIR_SPLIT_DEPTH', '0')
    proposal = row(0, 0)
    proposal[field] = value
    result = classify(fake_openai_module, [proposal])
    assert result.assignments == [None]
    assert result.llm.status == 'failed'
    assert result.llm.processed_lines == []
    assert result.llm.gaps[0]['kind'] == 'technical'


@pytest.mark.parametrize('rows', [[], [row(0, 0), row(0, 1)], [row(1, 1)],
                                   [row(0, 0)], [row(0, 1, 'agenda:9')]])
def test_missing_overlapping_or_unknown_identity_is_technical(fake_openai_module, monkeypatch, rows):
    monkeypatch.setenv('LLM_REPAIR_SPLIT_DEPTH', '0')
    result = classify(fake_openai_module, rows, ['Beratung.'] * 2)
    assert result.assignments == [None, None]
    assert result.llm.status == 'failed'


def test_semantic_gap_is_successfully_processed_and_reasoned(fake_openai_module):
    result = classify(fake_openai_module, [row(0, 0, None)])
    assert result.llm.status == 'success'
    assert result.llm.processed_lines == [0]
    assert result.llm.gaps == [{'start_index': 0, 'end_index': 0, 'kind': 'semantic', 'reason': 'Inhaltliche Beratung'}]
    assert result.assignments == [None]
    assert result.llm.warnings


def test_duplicate_line_key_is_not_silently_overwritten(fake_openai_module, monkeypatch):
    monkeypatch.setenv('LLM_REPAIR_SPLIT_DEPTH', '0')
    fake_openai_module.content = ('{"assignments":{"0":null,"0":"agenda:0"},'
                                  '"uncertain_lines":[],"gaps":[]}')
    result = segment_known_agenda([TranscriptUtterance('M', 'Beratung.')], ['1 Haushalt'], use_llm=True)
    assert result.llm.status == 'failed'
    assert result.llm.processed_lines == []
    assert 'duplicate_response_key' in result.llm.validation_reasons


def test_repeated_numbers_and_revisits_use_stable_ids(fake_openai_module):
    tops = ['[Öffentlich] 01 Haushalt', '[Nichtöffentlich] 01 Vergabe']
    result = classify(fake_openai_module, [row(0, 0), row(1, 1, 'agenda:1'), row(2, 2)], ['Beratung.'] * 3, tops)
    assert result.assignments == [0, 1, 0]
    assert result.llm.processed_lines == [0, 1, 2]


def test_all_lines_are_sent_with_overlap_and_disjoint_ownership(monkeypatch, fake_openai_module):
    monkeypatch.setenv('AGENDA_DETECTION_CHUNK_LINES', '3')
    monkeypatch.setenv('AGENDA_DETECTION_CHUNK_OVERLAP_LINES', '2')
    seen = []
    def complete(client, config, **kwargs):
        prompt = json.loads(kwargs['messages'][1]['content'])
        seen.append(prompt)
        proposal = row(prompt['target_start'], prompt['target_end'])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({'tops': [proposal]})))])
    monkeypatch.setattr(agenda_llm, 'complete', complete)
    result = classify(fake_openai_module, [], ['Beratung.'] * 8)
    assert result.assignments == [0] * 8
    assert result.llm.processed_lines == list(range(8))
    assert [(p['target_start'], p['target_end']) for p in seen] == [(0, 2), (3, 5), (6, 7)]
    assert [r['index'] for r in seen[1]['context_before'] + seen[1]['target_lines'] + seen[1]['context_after']] == [2, 3, 4, 5, 6]
    assert seen[1]['previous_top']['top_id'] == 'agenda:0'


def test_failed_chunk_is_split_once_successful_chunks_are_cached(monkeypatch, fake_openai_module, tmp_path):
    monkeypatch.setenv('LLM_CACHE_DIR', str(tmp_path))
    monkeypatch.setenv('LLM_REPAIR_SPLIT_DEPTH', '1')
    fake_openai_module.responses = [TimeoutError('SECRET'), json.dumps({'tops': [row(0, 0)]}),
                                    json.dumps({'tops': [row(1, 1)]})]
    result = classify(fake_openai_module, [], ['Beratung.'] * 2)
    assert result.assignments == [0, 0]
    assert result.llm.status == 'success'
    assert result.llm.attempted_calls == 3
    assert result.llm.failed_calls == 1
    assert 'SECRET' not in str(result.llm)
    assert len(list(tmp_path.iterdir())) == 3
    fake_openai_module.responses = []
    resumed = classify(fake_openai_module, [], ['Beratung.'] * 2)
    assert resumed.assignments == [0, 0]
    assert resumed.llm.attempted_calls == 0


def test_exhausted_failure_is_never_heuristic_success(monkeypatch, fake_openai_module):
    monkeypatch.setenv('LLM_REPAIR_SPLIT_DEPTH', '1')
    fake_openai_module.responses = [TimeoutError('secret')] * 3
    result = classify(fake_openai_module, [], ['Kommen wir zu TOP 1 Haushalt.'] * 2)
    assert result.assignments == [None, None]
    assert result.llm.attempted_calls == result.llm.failed_calls == 3
    assert result.llm.status == 'failed'
    assert all(g['kind'] == 'technical' for g in result.llm.gaps)


def test_line_map_has_exact_global_id_set(fake_openai_module):
    fake_openai_module.content = json.dumps({'assignments': {'0': None, '1': 'agenda:0'},
        'uncertain_lines': [], 'gaps': [{'line_index': 0, 'reason': 'Isolierte Dankesformel'}]})
    result = segment_known_agenda([TranscriptUtterance('A', 'Danke.'), TranscriptUtterance('M', 'Begrüßung.')],
                                  ['1 Eröffnung'], use_llm=True)
    assert result.assignments == [None, 0]
    assert result.llm.processed_lines == [0, 1]
    assert result.llm.gaps[0]['reason'] == 'Isolierte Dankesformel'
    assert result.segments[0].evidence_index == 1
    assert result.segments[0].evidence_text == 'Begrüßung.'


@pytest.mark.parametrize('mapping,gaps', [
    ({'1': 'agenda:0'}, []), ({'0': 'agenda:0', '1': 'agenda:0'}, []),
    ({'0': None}, []), ({'0': 'agenda:99'}, []),
    ({'0': None}, [{'line_index': 0, 'reason': ''}]),
])
def test_line_map_rejects_shifted_extra_missing_or_unexplained_lines(fake_openai_module, mapping, gaps):
    fake_openai_module.content = json.dumps({'assignments': mapping, 'uncertain_lines': [], 'gaps': gaps})
    result = segment_known_agenda([TranscriptUtterance('M', 'Beratung.')], ['1 Haushalt'], use_llm=True)
    assert result.llm.status == 'failed'
    assert result.llm.processed_lines == []


@pytest.mark.parametrize('text', ['Später kommen wir zu TOP 1.', 'Wir behandeln TOP 1 nicht.',
                                  'Gestern sagte er: „Kommen wir zu TOP 1.“'])
def test_noncurrent_evidence_remains_reviewable(fake_openai_module, text):
    fake_openai_module.content = json.dumps({'assignments': {'0': 'agenda:0'}, 'uncertain_lines': [], 'gaps': []})
    result = segment_known_agenda([TranscriptUtterance('M', text)], ['1 Haushalt'], use_llm=True)
    assert result.segments[0].uncertain
    assert result.segments[0].confidence <= 0.5


def test_legacy_prompt_builder_never_filters_by_heuristic_hits():
    from agenda_detection import _indexed_transcript_for_llm
    transcript = [TranscriptUtterance('M', f'Sachbeitrag {i}.') for i in range(558)]
    text, compacted = _indexed_transcript_for_llm(transcript, tops=['1 Haushalt'], heuristic_segments=[])
    assert not compacted
    assert text.splitlines() == [f'{i}: M: Sachbeitrag {i}.' for i in range(558)]


def test_semantic_gap_keeps_last_nonpublic_context(monkeypatch, fake_openai_module):
    monkeypatch.setenv('AGENDA_DETECTION_CHUNK_LINES', '1')
    fake_openai_module.responses = [
        json.dumps({'assignments': {'0': 'agenda:1'}, 'uncertain_lines': [], 'gaps': []}),
        json.dumps({'assignments': {'1': None}, 'uncertain_lines': [], 'gaps': [{'line_index': 1, 'reason': 'Pause'}]}),
        json.dumps({'assignments': {'2': 'agenda:1'}, 'uncertain_lines': [], 'gaps': []}),
    ]
    result = segment_known_agenda([TranscriptUtterance('M', t) for t in ['Beratung.', 'Pause.', 'Fortsetzung.']],
                                  ['[Öffentlich] 1 Informationen', '[Nichtöffentlich] 1 Informationen'], use_llm=True)
    assert result.assignments == [1, None, 1]
    third = fake_openai_module.instances[0].calls[2]
    assert json.loads(third['messages'][1]['content'])['previous_top']['top_id'] == 'agenda:1'


def test_substantial_semantic_gaps_get_bounded_llm_second_opinion(monkeypatch, fake_openai_module):
    monkeypatch.setenv('AGENDA_DETECTION_GAP_REVIEW_MAX_CALLS', '1')
    fake_openai_module.responses = [
        json.dumps({'assignments': {'0': 'agenda:0', '1': None, '2': None, '3': None},
                    'uncertain_lines': [], 'gaps': [{'line_index': i, 'reason': 'Anderes Sachthema'} for i in [1, 2, 3]]}),
        json.dumps({'assignments': {'1': 'agenda:0', '2': 'agenda:0', '3': 'agenda:0'}, 'uncertain_lines': [], 'gaps': []}),
    ]
    result = segment_known_agenda([TranscriptUtterance('M', 'Information.') for _ in range(4)],
                                  ['[Öffentlich] 7 Anfragen und Informationen - Satzung'], use_llm=True)
    assert result.assignments == [0, 0, 0, 0]
    assert result.llm.processed_lines == [0, 1, 2, 3]  # No double-counted coverage.
    assert result.llm.attempted_calls == 2
    assert result.llm.gaps == []
    assert result.llm.chunks[-1]['phase'] == 'gap_review'
    request = fake_openai_module.instances[0].calls[1]
    assert 'offenen TOP' in request['messages'][0]['content']
    assert json.loads(request['messages'][1]['content'])['context_assignments'] == {'0': 'agenda:0'}


def test_failed_second_opinion_keeps_semantic_gap_and_successful_assignments(monkeypatch, fake_openai_module):
    monkeypatch.setenv('AGENDA_DETECTION_GAP_REVIEW_MAX_CALLS', '1')
    fake_openai_module.responses = [
        json.dumps({'assignments': {'0': 'agenda:0', '1': None, '2': None, '3': None},
                    'uncertain_lines': [], 'gaps': [{'line_index': i, 'reason': 'Unklar'} for i in [1, 2, 3]]}),
        TimeoutError('secret'),
    ]
    result = segment_known_agenda([TranscriptUtterance('M', 'Information.') for _ in range(4)], ['7 Informationen'], use_llm=True)
    assert result.assignments == [0, None, None, None]
    assert all(gap['kind'] == 'semantic' for gap in result.llm.gaps)
    assert result.llm.attempted_calls == 2
    assert result.llm.failed_calls == 1
    assert result.llm.status == 'success'  # First-pass coverage is intact; warning reports second-opinion failure.
    assert result.llm.warnings


def test_open_topic_scope_is_context_not_input_filter(monkeypatch, fake_openai_module):
    monkeypatch.setenv('AGENDA_DETECTION_CHUNK_LINES', '1')
    fake_openai_module.responses = [
        json.dumps({'assignments': {'0': 'agenda:0'}, 'uncertain_lines': [], 'gaps': []}),
        json.dumps({'assignments': {'1': 'agenda:0'}, 'uncertain_lines': [], 'gaps': []}),
    ]
    result = segment_known_agenda([TranscriptUtterance('M', 'Anfragen.'), TranscriptUtterance('M', 'Anderes Sachthema.')],
                                  ['[Nichtöffentlich] 3 Anfragen und Informationen - Untertitel'], use_llm=True)
    request = json.loads(fake_openai_module.instances[0].calls[1]['messages'][1]['content'])
    assert request['target_lines'][0]['text'] == 'Anderes Sachthema.'
    assert 'keine abschließende Themenbeschränkung' in request['active_topic_scope']
    assert result.assignments == [0, 0]


def test_explicit_current_number_cannot_be_assigned_to_different_top(fake_openai_module):
    fake_openai_module.content = json.dumps({'assignments': {'0': 'agenda:0'}, 'uncertain_lines': [], 'gaps': []})
    result = segment_known_agenda([TranscriptUtterance('M', 'Kommen wir zu TOP 3 Schulbau.')],
                                  ['2 Haushalt', '3 Schulbau'], use_llm=True)
    assert result.assignments == [None]
    assert result.llm.status == 'failed'
    assert 'contradictory_current_call' in result.llm.validation_reasons
    assert result.llm.gaps[0]['kind'] == 'technical'


def test_minutes_context_distinguishes_protocol_corrections_from_new_questions(monkeypatch, fake_openai_module):
    monkeypatch.setenv('AGENDA_DETECTION_CHUNK_LINES', '1')
    fake_openai_module.responses = [
        json.dumps({'assignments': {'0': 'agenda:0'}, 'uncertain_lines': [], 'gaps': []}),
        json.dumps({'assignments': {'1': 'agenda:1'}, 'uncertain_lines': [], 'gaps': []}),
    ]
    result = segment_known_agenda([TranscriptUtterance('M', 'Niederschrift.'), TranscriptUtterance('M', 'Neue Sachfrage.')],
                                  ['[Nichtöffentlich] 2 Niederschrift', '[Nichtöffentlich] 3 Anfragen'], use_llm=True)
    request = json.loads(fake_openai_module.instances[0].calls[1]['messages'][1]['content'])
    assert 'nicht automatisch alle folgenden Sachdebatten' in request['minutes_topic_scope']
    assert result.assignments == [0, 1]


@pytest.mark.parametrize('limit, fail', [(0, False), (1, False), (1, True)])
def test_boundary_review_changes_only_validated_target_lines(monkeypatch, fake_openai_module, limit, fail):
    monkeypatch.setenv('AGENDA_DETECTION_BOUNDARY_REVIEW_MAX_CALLS', str(limit))
    texts = ['Beratung.', 'Keine weiteren Fragen.', 'Ich schließe die öffentliche Sitzung.', 'Guten Heimweg.']
    initial = {'assignments': {str(i): 'agenda:0' for i in range(4)}, 'uncertain_lines': [], 'gaps': []}
    reviewed = {'assignments': {'1': 'agenda:0', '2': 'agenda:1', '3': 'agenda:1'}, 'uncertain_lines': [], 'gaps': []}
    fake_openai_module.responses = [json.dumps(initial), TimeoutError('offline') if fail else json.dumps(reviewed)]
    result = segment_known_agenda([TranscriptUtterance('M', t) for t in texts],
                                  ['[Öffentlich] 1 Informationen', '[Öffentlich] 2 Schließung'], use_llm=True)
    assert result.llm.processed_lines == list(range(4))
    assert result.assignments == ([0, 0, 1, 1] if limit and not fail else [0]*4)
    assert result.llm.attempted_calls == 1 + limit
    if limit and not fail:
        assert result.llm.chunks[-1]['changes'] == [
            {'line_index': 2, 'before': 0, 'after': 1}, {'line_index': 3, 'before': 0, 'after': 1}]
        prompt = json.loads(fake_openai_module.instances[0].calls[1]['messages'][1]['content'])
        assert set(prompt['context_assignments']) == {'0'}
        assert all(s.uncertain for s in result.segments if s.top_index == 1)
    assert result.llm.status == 'success'


def test_cross_section_jump_triggers_llm_review_not_heuristic_reassignment(monkeypatch, fake_openai_module):
    monkeypatch.setenv('AGENDA_DETECTION_BOUNDARY_REVIEW_MAX_CALLS', '1')
    initial = {'assignments': {str(i): 'agenda:0' if i < 3 else 'agenda:1' for i in range(5)},
               'uncertain_lines': [], 'gaps': []}
    reviewed = {'assignments': {str(i): 'agenda:0' for i in range(2, 5)}, 'uncertain_lines': [], 'gaps': []}
    fake_openai_module.responses = [json.dumps(initial), json.dumps(reviewed)]
    result = segment_known_agenda([TranscriptUtterance('M', 'Weitere Sachfrage.')]*5,
        ['[Nichtöffentlich] 3 Anfragen', '[Öffentlich] 3 Verpflichtung'], use_llm=True)
    assert result.assignments == [0]*5
    assert result.llm.attempted_calls == 2
    prompt = json.loads(fake_openai_module.instances[0].calls[1]['messages'][1]['content'])
    assert prompt['section_review']['previous_section'] == 'nonpublic'
    assert prompt['section_review']['same_section_top_ids'] == ['agenda:0']


def test_preview_cannot_silently_switch_to_mentioned_topic(monkeypatch, fake_openai_module):
    monkeypatch.setenv('LLM_REPAIR_SPLIT_DEPTH', '0')
    fake_openai_module.content = json.dumps({'assignments': {'0': 'agenda:0', '1': 'agenda:1'},
                                            'uncertain_lines': [], 'gaps': []})
    result = segment_known_agenda([TranscriptUtterance('M', 'Wir beraten den Haushalt.'),
        TranscriptUtterance('M', 'In der nächsten Sitzung behandeln wir TOP 2 Schulbau.')],
        ['1 Haushalt', '2 Schulbau'], use_llm=True)
    assert result.llm.status == 'failed'
    assert 'noncurrent_reference_cannot_change_topic' in result.llm.validation_reasons
