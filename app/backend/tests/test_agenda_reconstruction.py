"""Bounded reconstruction, incomplete drafts and blind reviews with synthetic sources."""
import json
import re
from copy import deepcopy

import pytest
import agenda_llm
import durable_jobs as durable
import persistence
from agenda_context import model_agenda
from agenda_detection import AgendaLLMUsage
from test_agenda_llm import transcript, run
from test_durable_jobs import claimed


def workflow():
    work = agenda_llm.Workflow(transcript(['Gemeinsam beraten.', 'Spätere Wiederaufnahme.']),
        AgendaLLMUsage(True, 'test'), None, None, None, '')
    agenda = model_agenda([f'{i+1} Punkt' for i in range(36)])
    return work, agenda, {'original_transcript': work.rows, 'coverage': [0, 1]}


def test_seven_of_36_are_retained_as_unreviewed_drafts(agenda_model):
    work, agenda, context = workflow()
    body = dict(phase='primary:reconstruct:states:v1', agenda=agenda, context=context)
    states = work.catalog.prepare(agenda_model.answer(body))['agenda_states'][:7]
    with pytest.raises(agenda_llm.IncompleteAgendaStates) as error:
        work.state_entries(states, [t['top_id'] for t in agenda])
    assert error.value.states == states
    assert error.value.missing == [t['top_id'] for t in agenda[7:]]
    assert all(s['grounding']['content_status'] == 'unreviewed' for s in error.value.states)


@pytest.mark.parametrize('defect', ['missing', 'duplicate', 'unknown', 'invalid_status', 'excessive_evidence'])
def test_only_missing_or_invalid_states_are_requested_again(agenda_model, defect):
    work, agenda, context = workflow()
    def partial(body):
        answer = agenda_model.answer(body)
        if 'technical_repair' not in body:
            if defect == 'missing':
                answer['agenda_states'] = answer['agenda_states'][:-1]
            elif defect == 'duplicate':
                answer['agenda_states'].append(deepcopy(answer['agenda_states'][-1]))
            elif defect == 'unknown':
                answer['agenda_states'][-1]['top_id'] = 'foreign'
            elif defect == 'invalid_status':
                answer['agenda_states'][-1]['status'] = 'unknown'
            else:
                answer['agenda_states'][-1]['evidence'] *= 61
        return answer
    agenda_model.overrides['independent:reconstruct:states:v1'] = partial
    result = work.reconstruction('independent:reconstruct', context, agenda)
    assert [s['top_id'] for s in result['agenda_states']] == [t['top_id'] for t in agenda]
    calls = [b for b, _ in agenda_model.calls if 'expected_top_ids' in b]
    assert len(calls) == 12
    for initial, repair in zip(calls[::2], calls[1::2]):
        assert len(initial['expected_top_ids']) == 6
        if defect == 'duplicate':
            assert repair['expected_top_ids'] == initial['expected_top_ids']
            assert repair['technical_repair']['code'] == 'invalid_agenda_states'
        else:
            assert repair['expected_top_ids'] == initial['expected_top_ids'][-1:]
            assert repair['technical_repair']['missing_top_ids'] == repair['expected_top_ids']
        assert repair['agenda'] == initial['agenda']
    assert all(b['opinions'] is None for b in calls)
    assert all(s['grounding']['content_status'] == 'unreviewed' for s in result['agenda_states'])


def test_failed_group_keeps_drafts_without_success_checkpoint(agenda_model):
    work, agenda, context = workflow()
    def incomplete(body):
        result = agenda_model.answer(body)
        # The first group succeeds, the second cannot complete its final TOP.
        if 'agenda:11' in body['expected_top_ids']:
            result['agenda_states'] = [s for s in result['agenda_states'] if s['top_id'] != 'agenda:11']
        return result
    agenda_model.overrides['primary:reconstruct:states:v1'] = incomplete
    job = durable.submit('test', {})
    with claimed(job):
        with pytest.raises(agenda_llm.IncompleteAgendaStates):
            work.reconstruction('primary:reconstruct', context, agenda)
    with persistence.connect() as db:
        drafts = [json.loads(r[0]) for r in db.execute("SELECT value FROM durable_artifacts WHERE kind='incomplete_draft'")]
        states = [json.loads(r[0]) for r in db.execute("SELECT value FROM durable_steps WHERE step_key LIKE 'agenda:%'")]
    assert len(drafts[-1]['agenda_states']) == 11
    assert len(drafts[-1]['missing_top_ids']) == 25
    assert drafts[-1]['processing_complete'] is False
    saved_groups = [s['response']['result']['agenda_states_by_id'] for s in states
                    if 'agenda_states_by_id' in s.get('response',{}).get('result',{})]
    assert saved_groups and all(len(group)==6 for group in saved_groups)
    assert len([b for b, _ in agenda_model.calls if 'expected_top_ids' in b]) == 1 + work.attempts


def test_actual_qwen_five_of_six_states_require_all_keys_and_are_not_filled(agenda_model):
    from processing_mode import processing_scope
    with processing_scope('fast'):
        work,agenda,context=workflow(); agenda=agenda[:6]
        def missing(body):
            data=agenda_model.answer(body)
            data['agenda_states']=data['agenda_states'][:5]
            return data
        agenda_model.overrides['fast:reconstruct:states:v1']=missing
        with pytest.raises(agenda_llm.IncompleteAgendaStates) as error:
            work.reconstruction_states('fast:reconstruct',context,agenda,{})
    assert len(error.value.states)==5 and error.value.missing==[agenda[-1]['top_id']]
    assert len(agenda_model.calls)==1
    wire=agenda_model.calls[0][1]['response_format']['json_schema']['schema']
    states=wire['properties']['response']['anyOf'][0]['properties']['result']['properties']['agenda_states_by_id']
    assert states['required']==[t['top_id'] for t in agenda]
    assert all(v=={'$ref':'#/$defs/agenda_state'} for v in states['properties'].values())


def test_preparation_result_and_source_request_are_disjoint_even_if_provider_ignores_schema(agenda_model):
    from processing_mode import processing_scope
    with processing_scope('fast'):
        work,agenda,context=workflow()
        agenda_model.overrides['fast:discover']={'response':{'kind':'result',
            'result':{'reason':'Entwurf','items':[]},'source_ranges':[{'start':0,'end':1}]}}
        with pytest.raises(agenda_llm.AgendaValidationError,match='mixed_source_request'):
            work.discover('fast:discover',context,known_agenda=agenda)
    assert len(agenda_model.calls)==1


@pytest.mark.parametrize('identity',['evidence','grounding','source_ranges','uncertain','agenda_states_by_id','spans_by_end'])
def test_keyed_state_ids_remain_identities_even_when_named_like_metadata(agenda_model,identity):
    result=run(['Beratung.'],tops=['1 Haushalt'],top_ids=[identity],processing_mode='fast')
    assert result.llm.processing_complete
    assert result.llm.agenda_states[0]['top_id']==identity
    assert result.llm.agenda_states[0]['grounding']['reference_status']=='exact'


def test_truncated_json_never_becomes_a_state_or_success(agenda_model, monkeypatch):
    from types import SimpleNamespace
    work, agenda, context = workflow()
    monkeypatch.setattr(agenda_llm, 'complete', lambda *a, **k:
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"agenda_states":['))]))
    with pytest.raises(ValueError):
        work.reconstruction_states('independent:reconstruct', context, agenda, {})
    assert work.usage.attempted_calls == 2  # identical malformed output stops ineffective repairs


def test_joint_and_resumed_episodes_survive_grouping_with_original_access(agenda_model):
    work, agenda, context = workflow()
    context = {'model_notes':[{'evidence':[{'line_id':work.rows[0]['line_id'],'quote':'Gemeinsam'}]}], 'coverage':[0,1]}
    def trajectory(body):
        result = agenda_model.answer(body)
        result['episodes'][0]['top_ids'] = ['agenda:0', 'agenda:8']
        resumed = deepcopy(result['episodes'][0])
        resumed.update(start_line_id='L2', end_line_id='L2')
        result['episodes'].append(resumed)
        return result
    def states(body):
        if 'requested_originals' not in body:
            return dict(source_ranges=[dict(start=0, end=1)], agenda_states=[])
        return agenda_model.answer(body)
    agenda_model.overrides['independent:reconstruct:trajectory:v1'] = trajectory
    agenda_model.overrides['independent:reconstruct:states:v1'] = states
    result = work.reconstruction('independent:reconstruct', context, agenda)
    assert [e['top_ids'] for e in result['episodes']] == [['agenda:0', 'agenda:8']] * 2
    assert result['episodes'][1]['start_line_id'] == work.rows[1]['line_id']
    originals = [b for b, _ in agenda_model.calls if 'requested_originals' in b]
    assert len(originals) == 6 and all(len(b['requested_originals']) == 2 for b in originals)
    assert all(b['opinions'] is None for b in originals)


def test_reconstruction_bounds_anchors_without_restricting_source_access(agenda_model):
    work, agenda, context = workflow()
    work.reconstruction('primary:reconstruct', context, agenda)
    for body, request in agenda_model.calls:
        wire_schema = request['response_format']['json_schema']['schema']
        variants = wire_schema['properties']['response']['anyOf']
        schema = variants[0]['properties']['result']['properties']
        entries = (schema['episodes']['items']['properties'] if 'episodes' in schema else
                   wire_schema['$defs']['agenda_state']['properties'])
        assert entries['evidence']['maxItems'] == 3
        if 'episodes' in schema:
            assert entries['start_line_id'] == work.catalog.alias_schema()
            assert entries['end_line_id'] == work.catalog.alias_schema()
            assert body['source_bounds'] == {'count': 2, 'first_line_id': 'L1', 'last_line_id': 'L2'}
        if 'agenda_states_by_id' in schema:
            assert schema['agenda_states_by_id']['required'] == body['expected_top_ids']
            assert not schema['agenda_states_by_id']['additionalProperties']
            assert 'top_id' not in entries
        assert 'source_ranges' not in schema
        assert len(variants)==1  # All originals already supplied: no redundant retrieval.
        assert body['context']['coverage'] == [0, 1]


@pytest.mark.parametrize('start,end', [('L1820','L2100'), ('L3550','L3600')])
def test_qwen_full_run_hallucinated_episode_sources_are_bounded_and_rejected(agenda_model, start, end):
    from processing_mode import processing_scope
    with processing_scope('fast'):
        work = agenda_llm.Workflow(transcript(['Original']*1774),
            AgendaLLMUsage(True, 'test'), None, None, None, '')
        agenda = model_agenda(['1 Haushalt'])
        def invalid(body):
            return {'narrative':'Entwurf', 'source_ranges':[], 'episodes':[
                {'start_line_id':start, 'end_line_id':end, 'top_ids':[agenda[0]['top_id']],
                 'section':'public', 'reason':'Entwurf', 'evidence':[{'line_id':'L1'}]}]}
        agenda_model.overrides['fast:reconstruct:trajectory:v1'] = invalid
        with pytest.raises(agenda_llm.AgendaValidationError, match='invalid_episode_range'):
            work.reconstruction('fast:reconstruct', {'model_notes':[], 'coverage':[0,1773]}, agenda)
    assert len(agenda_model.calls) == 1
    properties = agenda_model.calls[0][1]['response_format']['json_schema']['schema']['properties']['response']['anyOf'][0]['properties']['result']['properties']['episodes']['items']['properties']
    for key in ('start_line_id','end_line_id'):
        pattern = re.compile(properties[key]['pattern'])
        assert all(pattern.fullmatch(f'L{i}') for i in range(1,1775))
        assert not pattern.fullmatch(start) and not pattern.fullmatch(end)


def test_excessive_episode_anchors_are_rejected_even_if_provider_ignores_schema(agenda_model):
    work, agenda, context = workflow()
    def excessive(body):
        data = agenda_model.answer(body)
        data['episodes'][0]['evidence'] *= 61
        return data
    agenda_model.overrides['primary:reconstruct:trajectory:v1'] = excessive
    with pytest.raises(agenda_llm.AgendaValidationError, match='too_many_episode_anchors'):
        work.reconstruction('primary:reconstruct', context, agenda)


def test_missing_independent_check_preserves_primary_and_blocks_completion(agenda_model):
    agenda_model.overrides['independent:reconstruct:states:v1'] = dict(source_ranges=[], agenda_states=[])
    result = run(['Gemeinsame Beratung.'])
    assert len(result.llm.reconstructions) == 1
    assert not result.llm.processing_complete and not result.llm.review_complete
    assert all(g['kind'] == 'technical' for g in result.llm.gaps)


def test_adjudication_keeps_global_timeline_but_only_compares_target_states(agenda_model):
    work, agenda, context = workflow()
    opinions = [work.reconstruction(role + ':reconstruct', context, agenda) for role in ('primary', 'independent')]
    agenda_model.calls.clear()
    result = work.reconstruction('resolve:states', context, agenda, opinions)
    assert len(result['agenda_states']) == 36
    timeline = agenda_model.calls[0][0]
    assert all('agenda_states' not in opinion for opinion in timeline['opinions'])
    assert all(opinion['episodes'] for opinion in timeline['opinions'])
    for body, _ in agenda_model.calls[1:]:
        assert body['agenda'] == agenda
        assert body['context']['coverage'] == [0, 1]
        assert body['trajectory']['episodes'] == timeline['opinions'][0]['episodes']
        assert [len(opinion['agenda_states']) for opinion in body['opinions']] == [6, 6]
        assert all({s['top_id'] for s in opinion['agenda_states']} == set(body['expected_top_ids'])
                   for opinion in body['opinions'])
