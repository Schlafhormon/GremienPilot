"""Source endpoint protocol, including sanitized geometry from job 9965f5c7.

Only ranges/diagnostic codes are retained from the private job, never transcript
content or inferred corrections. Scripted answers do not establish model quality.
"""
from copy import deepcopy
import json
import pytest

from agenda_llm import Workflow, AgendaValidationError
from source_contract import SourceCatalog
from processing_mode import processing_scope
from llm_transport import IncompleteResponseError
from test_agenda_llm import run


def decision(end, **kwargs):
    return dict(end_line_id=end, top_ids=['top'], reason='Beratung',
                evidence=[{'line_id': 'L1'}], uncertain=False, confidence=0.8, **kwargs)


def answer(*spans):
    return {'response': {'kind': 'assignments', 'spans': list(spans)}}


def workflow(responses, n=160):
    work = Workflow.__new__(Workflow)
    work.rows = [dict(line_id=f'id-{i}', index=i, text=f'Original {i}') for i in range(n)]
    work.by_id = {r['line_id']: r for r in work.rows}
    work.catalog = SourceCatalog(work.rows)
    work.retrieval_rounds = 2
    calls = []
    def call(phase, instruction, body, schema, validate):
        calls.append((deepcopy(body), deepcopy(schema)))
        raw = responses.pop(0)
        if callable(raw): raw = raw(body)
        data = work.catalog.prepare(work.catalog.translate(raw, decode=True))
        validate(data)
        return data
    work.call = call
    return work, calls


@pytest.mark.parametrize('start,end', [(0,0), (0,79), (80,159), (1760,1773)])
def test_exact_endpoints_expand_without_changing_source_identities(start, end):
    work, calls = workflow([answer(decision(f'L{end+1}'))], n=end+1)
    rows = work.compact_details('fast:detail', {}, [{'top_id':'top'}], {}, start, end)
    assert [r['line_id'] for r in rows] == [f'id-{i}' for i in range(start, end+1)]
    assert all(r['top_ids'] == ['top'] for r in rows)
    assert len(calls) == 1


def test_endpoint_order_is_source_order_not_lexicographic_and_preserves_joint_and_gap():
    joint = decision('L9'); joint['top_ids'] = ['top','other']
    gap = decision('L10'); gap['top_ids'] = []
    work, _ = workflow([answer(joint, gap)], n=10)
    rows = work.compact_details('detail', {}, [{'top_id':'top'},{'top_id':'other'}], {}, 0,9)
    assert [r['top_ids'] for r in rows] == [['top','other']]*9 + [[]]


@pytest.mark.parametrize('ends', [[], ['L2'], ['L3','L2'], ['L2','L2','L3'], ['L4'], ['missing'], [False]])
def test_unknown_duplicate_reversed_or_missing_final_end_is_not_repaired(ends):
    work, calls = workflow([answer(*(decision(e) for e in ends))], n=3)
    with pytest.raises(AgendaValidationError, match='incomplete_source_coverage'):
        work.compact_details('detail', {}, [{'top_id':'top'}], {}, 0,2)
    assert len(calls) == 1


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
    work, _ = workflow([answer(decision('L80', start=0))])
    with pytest.raises(AgendaValidationError, match='invalid_compact_response'):
        work.compact_details('detail', {}, [{'top_id':'top'}], {}, 0,79)


def test_retrieval_offers_only_missing_original_windows_and_deduplicates():
    work, calls = workflow([{'response':{'kind':'source_request','source_window_ids':['W2']}},
                            answer(decision('L80'))])
    rows = work.compact_details('detail', {}, [{'top_id':'top'}], {}, 0,79)
    assert len(rows) == 80
    assert [w['window_id'] for w in calls[0][0]['source_windows']] == ['W2']
    assert [r['index'] for r in calls[1][0]['requested_originals']] == list(range(80,160))
    assert calls[1][0]['source_windows'] == []
    variants = calls[0][1]['properties']['response']['anyOf']
    assert set(variants[0]['properties']) == {'kind','spans'}
    assert set(variants[1]['properties']) == {'kind','source_window_ids'}


def test_retrieval_limit_is_not_a_context_error_and_repeat_requests_fail():
    request = {'response':{'kind':'source_request','source_window_ids':['W2']}}
    work, _ = workflow([request]); work.retrieval_rounds = 0
    with pytest.raises(AgendaValidationError, match='source_request_limit'):
        work.compact_details('detail', {}, [{'top_id':'top'}], {}, 0,79)
    work, _ = workflow([request, request])
    with pytest.raises(AgendaValidationError, match='invalid_source_request'):
        work.compact_details('detail', {}, [{'top_id':'top'}], {}, 0,79)


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
