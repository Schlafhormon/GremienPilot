"""Initial decision/change protocol and sanitized legacy failures from job 9965f5c7.

Only ranges/diagnostic codes are retained from the private job, never transcript
content or inferred corrections. Scripted answers do not establish model quality.
"""
from copy import deepcopy
from dataclasses import replace
import json
import pytest
from types import SimpleNamespace

from agenda_llm import Workflow, AgendaValidationError, parse_response
from source_contract import SourceCatalog
from processing_mode import processing_scope
from llm_transport import IncompleteResponseError
from test_agenda_llm import run


def decision(start, **kwargs):
    return dict(start_line_id=start, top_ids=['top'], reason='Beratung',
                evidence=[{'line_id': 'L1'}], uncertain=False, confidence=0.8, **kwargs)


def answer(*spans):
    result={'kind':'assignments','changes':{s['start_line_id']:
        {k:v for k,v in s.items() if k!='start_line_id'} for s in spans[1:]}}
    if spans: result['initial']={k:v for k,v in spans[0].items() if k!='start_line_id'}
    return {'response':result}


def workflow(responses, n=160):
    work = Workflow.__new__(Workflow)
    work.provenance = {}
    work.usage = SimpleNamespace(provenance={})
    work.rows = [dict(line_id=f'id-{i}', index=i, text=f'Original {i}') for i in range(n)]
    work.by_id = {r['line_id']: r for r in work.rows}
    work.catalog = SourceCatalog(work.rows)
    from llm_config import get_llm_config
    from agenda_llm import BASE
    work.config=replace(get_llm_config(),context_tokens=131072)
    work.reserve=4096
    work.system=BASE
    work.retrieval_rounds = 2
    calls = []
    def call(phase, instruction, body, schema, validate):
        calls.append((deepcopy(body), deepcopy(schema)))
        raw = responses.pop(0)
        if callable(raw): raw = raw(body)
        data = work.catalog.prepare(work.catalog.translate(raw, decode=True))
        validate(data)
        validate(data)  # Durable checkpoint reads validate canonical IDs again.
        return data
    work.call = call
    return work, calls


@pytest.mark.parametrize('start,end', [(0,0), (0,79), (80,159), (1760,1773)])
def test_exact_starts_expand_without_changing_source_identities(start, end):
    work, calls = workflow([answer(decision(f'L{start+1}'))], n=end+1)
    rows = work.compact_details('fast:detail', {}, [{'top_id':'top'}], {}, start, end)
    assert [r['line_id'] for r in rows] == [f'id-{i}' for i in range(start, end+1)]
    assert all(r['top_ids'] == ['top'] for r in rows)
    assert len(calls) == 1


def test_start_order_is_source_order_not_lexicographic_and_preserves_joint_and_gap():
    joint = decision('L9'); joint['top_ids'] = ['top','other']
    gap = decision('L10'); gap['top_ids'] = []
    work, _ = workflow([answer(decision('L1'),gap,joint)], n=10)
    rows = work.compact_details('detail', {}, [{'top_id':'top'},{'top_id':'other'}], {}, 0,9)
    assert [r['top_ids'] for r in rows] == [['top']]*8 + [['top','other'],[]]


@pytest.mark.parametrize('ends', [['L1'], ['L4'], ['missing'], [False]])
def test_unknown_or_first_change_is_not_repaired(ends):
    work, calls = workflow([answer(decision('L1'),*(decision(e) for e in ends))], n=3)
    with pytest.raises(AgendaValidationError, match='incomplete_source_coverage'):
        work.compact_details('detail', {}, [{'top_id':'top'}], {}, 0,2)
    assert len(calls) == 1


@pytest.mark.parametrize('raw',[
    '{"response":{"kind":"assignments","initial":{},"initial":{},"changes":{}}}',
    '{"response":{"kind":"assignments","initial":{},"changes":{"L579":{},"L579":{}}}}',
])
def test_duplicate_start_keys_are_rejected_before_any_assignment_can_be_overwritten(raw):
    with pytest.raises(AgendaValidationError,match='duplicate_response_key'):
        parse_response(raw)


def test_schema_requires_initial_and_unique_change_keys_and_reuses_shared_definition():
    work,calls=workflow([answer(decision('L1'))],n=80)
    work.compact_details('detail',{},[{'top_id':'top'}],{},0,79)
    schema=calls[0][1]
    response=schema['properties']['response']['anyOf'][0]
    assert response['required']==['kind','initial','changes'] and not response['additionalProperties']
    assert response['properties']['initial']=={'$ref':'#/$defs/assignment'}
    changes=response['properties']['changes']
    assert changes['type']=='object' and changes['required']==[] and not changes['additionalProperties']
    assert list(changes['properties'])==[f'L{i}' for i in range(2,81)]
    assert all(value=={'$ref':'#/$defs/assignment'} for value in changes['properties'].values())
    assert 'end_line_id' not in schema['$defs']['assignment']['properties']


def test_missing_initial_decision_is_not_filled():
    work,_=workflow([{'response':{'kind':'assignments','changes':{}}}],n=3)
    with pytest.raises(AgendaValidationError,match='invalid_compact_response'):
        work.compact_details('detail',{},[{'top_id':'top'}],{},0,2)


def test_previous_change_list_cannot_silently_overwrite_repeated_sources():
    repeated={'start_line_id':'L2','assignment':{k:v for k,v in decision('L2').items() if k!='start_line_id'}}
    raw=answer(decision('L1'))
    raw['response']['changes']=[repeated,repeated]
    work,calls=workflow([raw],n=3)
    with pytest.raises(AgendaValidationError,match='incomplete_source_coverage'):
        work.compact_details('detail',{},[{'top_id':'top'}],{},0,2)
    assert len(calls)==1


@pytest.mark.parametrize('mode',['fast','slow'])
def test_fast_detail_uses_global_narrative_without_prior_source_labels(mode):
    work,calls=workflow([answer(decision('L1'))],n=3)
    reconstruction={'narrative':'Thema wurde wieder aufgenommen.',
        'episodes':[{'start_line_id':'id-0','end_line_id':'id-2','top_ids':['wrong']}],
        'agenda_states':[{'top_id':'wrong','status':'discussed'}]}
    before=deepcopy(reconstruction)
    from processing_mode import processing_scope
    with processing_scope(mode):
        rows=work.compact_details('detail',{'model_notes':['Globaler Verlauf']},[{'top_id':'top'}],reconstruction,0,2)
    body=calls[0][0]
    assert body['reconstruction']==({'narrative':reconstruction['narrative']} if mode=='fast' else reconstruction)
    assert body['target_lines']==work.rows and body['context']=={'model_notes':['Globaler Verlauf']}
    assert reconstruction==before and len(rows)==3 and len(calls)==1


@pytest.mark.parametrize('identities', [['uuid-a','uuid-b'], ['T1','TT2'], []])
def test_fast_short_top_ids_roundtrip_without_collisions_or_mutating_agenda(identities):
    agenda=[{'top_id':t,'title':f'Topic {i}'} for i,t in enumerate(identities)]
    original=deepcopy(agenda)
    def response(body):
        raw=answer(decision('L1'))
        raw['response']['initial']['top_ids']=[t['top_id'] for t in body['agenda']]
        assert not set(raw['response']['initial']['top_ids']) & set(identities)
        return raw
    work,calls=workflow([response],n=3)
    with processing_scope('fast'):
        rows=work.compact_details('fast:detail',{},agenda,{},0,2)
    assert agenda==original and all(r['top_ids']==identities for r in rows)
    mapping=work.provenance['model_top_aliases']
    assert list(mapping.values())==identities and work.usage.provenance['model_top_aliases']==mapping
    schema=calls[0][1]['$defs']['assignment']['properties']
    assert schema['reason']['maxLength']==96 and schema['evidence']['maxItems']==1
    if identities: assert schema['top_ids']['items']['enum']==list(mapping)


def test_short_top_alias_mapping_distinguishes_cache_identity():
    snapshots=[]
    for identity in ['uuid-a','uuid-b']:
        raw=answer(decision('L1'));raw['response']['initial']['top_ids']=['T1']
        work,calls=workflow([raw],n=1)
        with processing_scope('fast'):
            work.compact_details('fast:detail',{},[{'top_id':identity,'title':'Same title'}],{},0,0)
        snapshots.append((calls[0][0],deepcopy(work.provenance)))
    assert snapshots[0][0]==snapshots[1][0]
    assert snapshots[0][1]!=snapshots[1][1]


def test_slow_keeps_canonical_top_ids_and_full_evidence_budget():
    work,calls=workflow([answer(decision('L1'))],n=1)
    with processing_scope('slow'):
        work.compact_details('detail',{},[{'top_id':'top'}],{},0,0)
    assert calls[0][0]['agenda']==[{'top_id':'top'}]
    schema=calls[0][1]['$defs']['assignment']['properties']
    assert schema['top_ids']['items']['enum']==['top']
    assert 'maxLength' not in schema['reason'] and 'maxItems' not in schema['evidence']
    assert 'model_top_aliases' not in work.provenance


# All eight coverage failures and seven mixed answers from the reported job.
FAILED = [
    (0,79,[(0,12),(13,32),(33,10),(11,14),(15,18),(19,19),(20,20),(21,21)],[]),
    (80,159,[(80,130),(130,159)],[]),
    (320,399,[(320,332),(332,333),(333,399)],[]),
    (400,479,[(400,418),(419,420),(420,420),(421,437),(438,445),(446,450),(451,459),(460,463),(464,479)],[]),
    (720,799,[(720,736),(737,750),(751,776),(778,799)],[]),
    (1040,1119,[(1040,1050),(1050,1069),(1070,1085),(1086,1097),(1098,1119)],[]),
    (1120,1199,[(1120,1128),(1129,1137),(1138,1140),(1141,1164),(1166,1175),(1176,1180),(1181,1188),(1189,1199)],[]),
    (1200,1279,[(1200,1222),(1222,1235),(1236,1251),(1252,1264),(1264,1273),(1273,1279)],[]),
    (240,319,[(240,269),(270,289),(290,319)],[(240,319)]),
    (560,639,[(560,582),(583,604),(605,623),(624,639)],[(560,582),(583,604),(605,623),(624,640)]),
    (960,1039,[(960,1039)],[(960,1039)]),
    (1280,1359,[(1280,1300),(1300,1314),(1314,1325),(1325,1348),(1348,1359)],[(1280,1300),(1300,1314),(1314,1325),(1325,1348),(1348,1359)]),
    (1360,1439,[(1360,1374),(1375,1382),(1383,1392),(1393,1400),(1400,1439)],[(1360,1400),(1400,1439)]),
    (1600,1679,[(1600,1626),(1627,1630),(1631,1658),(1659,1671),(1672,1675),(1676,1679)],[(1600,1626),(1627,1630),(1631,1658),(1659,1671),(1672,1675),(1676,1679)]),
    (1760,1773,[(1760,1773)],[(1760,1773)]),
]


@pytest.mark.parametrize('start,end,spans,requests', FAILED)
def test_actual_old_failures_are_never_silently_converted(start,end,spans,requests):
    raw = {'spans':[dict(start=a,end=b) for a,b in spans],
           'source_ranges':[dict(start=a,end=b) for a,b in requests]}
    work, _ = workflow([raw], n=1774)
    with pytest.raises(AgendaValidationError):
        work.compact_details('detail', {}, [{'top_id':'top'}], {}, start,end)


@pytest.mark.parametrize('kind', ['assignments','source_request'])
def test_mixed_new_variants_and_extra_start_boundaries_fail(kind):
    work, _ = workflow([{'response':{'kind':kind, 'spans':[decision('L80')], 'source_window_ids':['W2']}}])
    with pytest.raises(AgendaValidationError, match='mixed_source_request'):
        work.compact_details('detail', {}, [{'top_id':'top'}], {}, 0,79)
    work, _ = workflow([answer(decision('L1', end=79))])
    with pytest.raises(AgendaValidationError, match='invalid_compact_response'):
        work.compact_details('detail', {}, [{'top_id':'top'}], {}, 0,79)


def test_retrieval_offers_only_missing_original_windows_and_deduplicates():
    work, calls = workflow([{'response':{'kind':'source_request','source_window_ids':['W2']}},
                            answer(decision('L1'))])
    rows = work.compact_details('detail', {}, [{'top_id':'top'}], {}, 0,79)
    assert len(rows) == 80
    assert [w['window_id'] for w in calls[0][0]['source_windows']] == ['W2']
    assert [r['index'] for r in calls[1][0]['requested_originals']] == list(range(80,160))
    assert calls[1][0]['source_windows'] == []
    variants = calls[0][1]['properties']['response']['anyOf']
    assert set(variants[0]['properties']) == {'kind','initial','changes'}
    assert set(variants[1]['properties']) == {'kind','source_window_ids'}


def test_retrieval_limit_is_not_a_context_error_and_repeat_requests_fail():
    request = {'response':{'kind':'source_request','source_window_ids':['W2']}}
    work, _ = workflow([request]); work.retrieval_rounds = 0
    with pytest.raises(AgendaValidationError, match='source_request_limit'):
        work.compact_details('detail', {}, [{'top_id':'top'}], {}, 0,79)
    work, _ = workflow([request, request])
    with pytest.raises(AgendaValidationError, match='invalid_source_request'):
        work.compact_details('detail', {}, [{'top_id':'top'}], {}, 0,79)


@pytest.mark.parametrize('preparation',[False,True])
def test_source_requests_are_bounded_windows_not_the_whole_transcript(preparation):
    request={'response':{'kind':'source_request','source_window_ids':['W2','W3','W4']}}
    result={'response':{'kind':'result','result':{'items':[]}}} if preparation else answer(decision('L1'))
    work,calls=workflow([request,result],n=1774)
    if preparation:
        work.source_call('discover','',{'original':work.rows[:80]},
            {'type':'object','properties':{'items':{'type':'array','items':{'type':'string'}}},'required':['items']},lambda data:None)
    else:
        work.compact_details('detail',{},[{'top_id':'top'}],{},0,79)
    assert len(calls[1][0]['requested_originals'])==240
    options=calls[0][1]['properties']['response']['anyOf'][1]['properties']
    assert set(options)=={'kind','source_window_ids'}
    assert options['source_window_ids']['maxItems']==3
    assert not {'W1','W2','W3','W4'} & {w['window_id'] for w in calls[1][0]['source_windows']}


@pytest.mark.parametrize('response',[
    {'kind':'source_request','source_ranges':[{'start':0,'end':1773}]},
    {'kind':'source_request','source_window_ids':['W1','W2','W3','W4']},
    {'kind':'source_request','source_window_ids':['W1','W1']},
])
def test_actual_full_transcript_request_and_excessive_windows_are_rejected(response):
    work,calls=workflow([{'response':response}],n=1774)
    with pytest.raises(AgendaValidationError):
        work.source_call('states','',{}, {'type':'object','properties':{}},lambda data:None)
    assert len(calls)==1


def test_small_context_subdivides_source_windows_without_model_calls_or_text_loss():
    work,_=workflow([],n=90)
    work.config=replace(work.config,context_tokens=16384)
    for row in work.rows: row['text']='Beratung und Wiederaufnahme. '*10
    result_schema={'type':'object','properties':{'items':{'type':'array','items':{'type':'string'}}}}
    body,schema,offered,limit=work.retrieval_offer('states','',{},result_schema,None,set())
    assert any(len(rows)<80 for rows in offered.values()) and 1<=limit<=3
    assert [r for rows in offered.values() for r in rows]==work.rows
    assert all(r['text']=='Beratung und Wiederaufnahme. '*10 for rows in offered.values() for r in rows)
    largest=sorted(offered.values(),key=lambda rows:len(json.dumps(rows)),reverse=True)[:limit]
    followup=dict(body,requested_originals=[r for rows in largest for r in rows])
    assert work.fits('states','',followup,schema)


def test_fast_incomplete_generation_has_no_hidden_split_repairs(agenda_model):
    agenda_model.overrides['fast:detail'] = IncompleteResponseError('length')
    result = run(['Beratung']*3, processing_mode='fast')
    assert not result.llm.processing_complete
    assert len([b for b,_ in agenda_model.calls if b['phase']=='fast:detail']) == 1


def test_prompt_projection_preserves_sources_and_questions_without_mutating_archive():
    work, _ = workflow([], n=1); work.system='System'
    body = {'original':work.rows, 'note': {'source_ranges': [], 'evidence':[{'line_id':'id-0','quote':'Original 0'}],
            'grounding':{'content_status':'contradicted','questions':['Wurde abgelehnt?'],
                         'source_ids':['id-0'], 'diagnostics':[]}}}
    before = deepcopy(body)
    projected=json.loads(work.messages('detail','',body)[1]['content'])
    assert body == before
    assert projected['original'][0]['text']=='Original 0'
    assert projected['note']['grounding']=={'content_status':'contradicted','questions':['Wurde abgelehnt?']}
    assert projected['note']['evidence'][0]['line_id']=='L1'
    assert 'source_ranges' not in projected['note']


def test_id_based_prompts_do_not_expose_competing_zero_based_indices():
    work,_=workflow([],n=3)
    body={'target_lines':work.rows,'target_start':0,'target_end':2,'source_windows':[]}
    before=deepcopy(body)
    projected=json.loads(work.messages('detail','',body)[1]['content'])
    assert 'target_start' not in projected and 'target_end' not in projected
    assert all('index' not in row for row in projected['target_lines'])
    assert [row['line_id'] for row in projected['target_lines']]==['L1','L2','L3']
    assert [row['text'] for row in projected['target_lines']]==[row['text'] for row in work.rows]
    assert body==before
