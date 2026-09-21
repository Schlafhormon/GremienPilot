import json
from types import SimpleNamespace
import pytest
import agenda_llm
from agenda_context import model_agenda, EvidenceContext, closing_act, section_act
from agenda_detection import segment_known_agenda
from assignment_suggestions import TranscriptUtterance, transition_kind
from transcript_splitter import aligned_segment_line, split_transcript_for_agenda_detection


def utterances(texts):
    return [TranscriptUtterance('M', text) for text in texts]


def test_identity_preserves_original_numbers_sections_duplicates_and_unknowns():
    titles = ['[Öffentlich] 07 Informationen', '[Öffentlich] 08 Schließung',
              '[Nichtöffentlich] 07 Informationen', '02.10 Schule', '02.10 Schule',
              'Ohne Nummer', 'Noch ohne Nummer']
    ids = model_agenda(titles)
    assert [t['top_id'] for t in ids[:3]] == ['public:07', 'public:08', 'nonpublic:07']
    assert ids[3]['number'] == '02.10'
    assert len({t['top_id'] for t in ids}) == len(ids)
    assert [t['number'] for t in ids[-2:]] == [None, None]
    assert not any('agenda:' in t['top_id'] for t in ids)


@pytest.mark.parametrize('text,kind', [
    ('Ich denke, dann kommen wir zum Tagesordnungspunkt 7.', 'call'),
    ('Wenn keine Fragen sind, zum Tagesordnungspunkt 7 habe ich noch einen Hinweis.', 'continuation'),
    ('In der letzten Niederschrift habe ich eine Anmerkung zu Punkt 7.', 'mention'),
    ('Später kommen wir zum Tagesordnungspunkt 7.', 'mention'),
    ('Wir behandeln TOP 7 nicht.', 'mention'),
    ('Wir nehmen TOP 2 wieder auf.', 'call'),
    ('Wir beraten jetzt TOP 9 vorgezogen.', 'call'),
    ('Weitere Anfragen, Informationen zum Tagesordnungspunkt 3?', 'continuation'),
    ('In der nächsten Sitzung weitere Fragen zu TOP 3?', 'mention'),
])
def test_speech_acts(text, kind):
    assert transition_kind(text) == kind


def test_topic_anchor_survives_long_open_discussion_and_separate_section():
    tops = ['[Nichtöffentlich] 01 Tagesordnung', '[Nichtöffentlich] 03 Anfragen und Informationen',
            '[Öffentlich] 05 Gebühren']
    texts = ['Dann kommen wir zum nichtöffentlichen Teil der Sitzung.',
             'Kommen wir zu TOP 3.', 'Anfragen und Informationen.'] + ['Weitere Gebührenauskunft.']*80
    context = EvidenceContext(utterances(texts), tops, model_agenda(tops))
    packet = context.packet(70, 80)
    assert packet['section_anchor']['index'] == 0
    assert packet['topic_anchor']['index'] == 1
    assert packet['topic_anchor']['top_id'] == 'nonpublic:03'
    assert texts[1] in [r['text'] for r in packet['original_evidence']]


def test_revisit_and_unknown_call_do_not_imply_monotonic_order():
    tops = ['1 Haushalt', '2 Schule']
    ctx = EvidenceContext(utterances(['Kommen wir zu TOP 2.', 'Wir nehmen TOP 1 wieder auf.',
                                    'Kommen wir zu TOP 88.']), tops, model_agenda(tops))
    assert ctx.at(1)['topic']['top_id'] == 'unspecified:1'
    assert ctx.at(2)['topic'] is None


def test_unknown_call_invalidates_old_continuation():
    tops=['7 Informationen']
    ctx=EvidenceContext(utterances(['Kommen wir zu TOP 7.', 'Weitere Fragen zu TOP 7?',
        'Kommen wir zu TOP 88.']),tops,model_agenda(tops))
    assert ctx.at(1)['continuation'] is not None
    assert ctx.at(2)['continuation'] is None


def test_review_accepts_current_question_without_making_it_a_new_call(fake_openai_module, monkeypatch):
    monkeypatch.setenv('AGENDA_DETECTION_GAP_REVIEW_MAX_CALLS','0')
    first=fake_reply({'0':'nonpublic:01','1':'nonpublic:03','2':'nonpublic:04'})
    fake_openai_module.responses=[first,fake_reply({'1':'nonpublic:03','2':'nonpublic:04'})]
    result=segment_known_agenda(utterances(['Wir kommen zur Bestätigung der Tagesordnung.',
        'Weitere Anfragen, Informationen zum Tagesordnungspunkt 3?',
        'Ich schließe die nichtöffentliche Sitzung.']),
        ['[Nichtöffentlich] 01 Bestätigung der Tagesordnung','[Nichtöffentlich] 03 Anfragen',
         '[Nichtöffentlich] 04 Schließung'],use_llm=True)
    assert result.assignments==[0,1,2]
    assert 'noncurrent_reference_cannot_change_topic' not in result.llm.failure_reasons
    assert result.llm.failed_calls == 0


def test_protocol_reference_cannot_become_a_continuation_anchor():
    tops = ['[Öffentlich] 02 Niederschrift', '[Öffentlich] 07 Informationen']
    ctx = EvidenceContext(utterances(['Kommen wir zu TOP 2.',
        'Ich habe noch eine Anmerkung zu Punkt 7.', 'Das steht im alten Protokoll.']), tops, model_agenda(tops))
    assert ctx.at(2)['topic']['top_id'] == 'public:02'
    assert ctx.at(2)['section']['section'] == 'public'


def test_continuation_keeps_the_original_call_and_title_in_later_windows():
    tops=['[Öffentlich] 07 Anfragen und Informationen']
    texts=['Kommen wir zu TOP 7.', 'Anfragen und Informationen.'] + ['Sachfrage.']*40
    texts+=['Zum Tagesordnungspunkt 7 habe ich noch einen Hinweis.']+['Weitere Auskunft.']*40
    ctx=EvidenceContext(utterances(texts),tops,model_agenda(tops))
    packet=ctx.packet(75,80)
    assert packet['topic_anchor']['index']==0
    assert packet['continuation_anchor']['index']==42
    assert {0,1,42} <= {r['index'] for r in packet['original_evidence']}


@pytest.mark.parametrize('text,expected', [
    ('Wenn das nicht der Fall ist, dann schließe ich die nichtöffentliche Sitzung.', True),
    ('Falls keine weiteren Fragen vorliegen, schließe ich die öffentliche Sitzung.', True),
    ('Wenn das nicht der Fall ist, werde ich die öffentliche Sitzung schließen.', False),
    ('Gestern sagte er: „Ich schließe die Sitzung.“', False),
    ('Ich schließe die Sitzung nicht.', False),
])
def test_conditional_closing_keeps_polarity_and_tense(text, expected):
    assert closing_act(text) == expected


def test_minutes_call_is_not_a_section_switch():
    assert section_act('Kommen wir zur Niederschrift der öffentlichen Sitzung.') is None
    assert section_act('Kommen wir zum öffentlichen Teil der Sitzung.') == 'public'


def test_repair_child_keeps_distant_original_anchor(fake_openai_module, monkeypatch):
    monkeypatch.setenv('AGENDA_DETECTION_CHUNK_LINES', '2')
    monkeypatch.setenv('AGENDA_DETECTION_GAP_REVIEW_MAX_CALLS', '0')
    fake_openai_module.responses = [fake_reply({'0': 'unspecified:7', '1': 'unspecified:7'}),
        TimeoutError(), fake_reply({'2': 'unspecified:7'}), fake_reply({'3': 'unspecified:7'})]
    result = segment_known_agenda(utterances(['Kommen wir zu TOP 7.', 'Informationen.',
        'Sachfrage zum Haushalt.', 'Antwort auf diese Frage.']), ['7 Informationen'], use_llm=True)
    assert result.assignments == [0]*4
    child = json.loads(fake_openai_module.instances[0].calls[-1]['messages'][1]['content'])
    assert child['evidence_context']['topic_anchor']['index'] == 0
    assert child['evidence_context']['original_evidence'][0]['text'] == 'Kommen wir zu TOP 7.'


def fake_reply(labels, gaps=None):
    return json.dumps({'classification_note': 'Aktueller Sitzungsverlauf', 'assignments': labels,
                       'uncertain_lines': [], 'gaps': gaps or []})


def test_missing_gap_reason_repaired_without_splitting_or_changing_labels(fake_openai_module, monkeypatch):
    monkeypatch.setenv('AGENDA_DETECTION_GAP_REVIEW_MAX_CALLS', '0')
    fake_openai_module.responses = [fake_reply({'0': None, '1': 'unspecified:1'}),
        json.dumps({'gap_reasons': {'0': 'Technikpause'}})]
    result = segment_known_agenda(utterances(['Mikrofontest.', 'Kommen wir zu TOP 1.']), ['1 Haushalt'], use_llm=True)
    assert result.assignments == [None, 0]
    assert result.llm.attempted_calls == 2
    assert len(result.llm.chunks) == 1
    first, second = fake_openai_module.instances[0].calls
    a, b = [json.loads(c['messages'][1]['content']) for c in [first, second]]
    assert a['evidence_context'] == b['evidence_context']
    assert a['target_lines'] == b['target_lines']
    schema = second['response_format']['json_schema']['schema']
    assert schema['required'] == ['gap_reasons']
    assert schema['properties']['gap_reasons']['required'] == ['0']
    assert 'assignments' not in schema['properties']


def test_bad_review_cannot_move_section_call_back_to_public(fake_openai_module, monkeypatch):
    monkeypatch.setenv('AGENDA_DETECTION_GAP_REVIEW_MAX_CALLS', '0')
    fake_openai_module.responses = [fake_reply({'0': 'public:08', '1': 'nonpublic:01'}),
        fake_reply({'0': 'public:08', '1': 'public:08'}),
        fake_reply({'0': 'public:08', '1': 'public:08'})]
    result = segment_known_agenda(utterances(['Ich schließe die öffentliche Sitzung.',
        'Dann kommen wir zum nichtöffentlichen Teil der Sitzung.']),
        ['[Öffentlich] 08 Schließung', '[Nichtöffentlich] 01 Tagesordnung'], use_llm=True)
    assert result.assignments == [0, 1]
    assert result.llm.chunks[-1]['status'] == 'failed'
    assert any(s.uncertain for s in result.segments)
    review = json.loads(fake_openai_module.instances[0].calls[1]['messages'][1]['content'])
    assert 'context_assignments' not in review and review['previous_top'] is None
    schema=fake_openai_module.instances[0].calls[1]['response_format']['json_schema']['schema']
    assert schema['properties']['assignments']['properties']['1']['enum'] == [None, 'nonpublic:01']


def test_closing_identity_requires_original_evidence(fake_openai_module, monkeypatch):
    monkeypatch.setenv('LLM_REPAIR_SPLIT_DEPTH', '0')
    fake_openai_module.content = fake_reply({'0': 'public:08'})
    result = segment_known_agenda(utterances(['Ein Bericht zur neuen Satzung.']),
        ['[Öffentlich] 07 Informationen', '[Öffentlich] 08 Schließung'], use_llm=True)
    assert result.assignments == [None]
    assert 'closing_without_evidence' in result.llm.validation_reasons


def test_minutes_reference_does_not_override_original_call(fake_openai_module, monkeypatch):
    monkeypatch.setenv('LLM_REPAIR_SPLIT_DEPTH','0')
    fake_openai_module.responses=[fake_reply({'0':'public:02','1':'public:07'}),
                                  fake_reply({'0':'public:02','1':'public:02'})]
    result=segment_known_agenda(utterances(['Kommen wir zu TOP 2.', 'Ich habe eine Anmerkung zu Punkt 7.']),
        ['[Öffentlich] 02 Niederschrift','[Öffentlich] 07 Informationen'],use_llm=True)
    assert result.assignments==[0,0]
    assert result.llm.chunks[0]['repair_history'][0]['reason']=='protocol_reference_cannot_change_topic'


def test_fresh_namespace_and_digest_isolate_cache(fake_openai_module, monkeypatch, tmp_path):
    monkeypatch.setenv('LLM_CACHE_DIR', str(tmp_path))
    monkeypatch.setattr(agenda_llm, 'model_fingerprint', lambda c: {'digest': 'one'})
    fake_openai_module.content = fake_reply({'0': 'unspecified:1'})
    def run(namespace):
        return segment_known_agenda(utterances(['Beratung.']), ['1 Haushalt'], use_llm=True, cache_namespace=namespace)
    assert run('a').llm.attempted_calls == 1
    assert run('a').llm.attempted_calls == 0
    assert run('b').llm.attempted_calls == 1
    monkeypatch.setattr(agenda_llm, 'model_fingerprint', lambda c: {'digest': 'two'})
    assert run('a').llm.attempted_calls == 1
    assert len(list(tmp_path.iterdir())) == 3


def test_aligned_times_survive_sentence_split_and_keep_raw_segment_bounds():
    text = 'Diese sehr lange erste Erläuterung gehört noch zum bisherigen Beratungspunkt. Jetzt kommt die nächste Frage.'
    words=[]
    for i, token in enumerate(text.split()):
        words.append({'word':token,'start':100+i,'end':100+i+0.8})
    line=aligned_segment_line({'text':text,'start':70,'end':130,'words':words,'speaker':'S'})
    result=split_transcript_for_agenda_detection([line])
    assert result[0]['start'] == 100
    assert result[1]['start'] == next(w['start'] for w in words if w['word'] == 'Jetzt')
    assert result[1]['timing']['source'] == 'word_alignment'
    assert result[1]['timing']['segments'][0]['start'] == 70
    assert split_transcript_for_agenda_detection(result) == result


def test_timing_survives_job_session_storage_and_api(tmp_path, monkeypatch):
    import persistence, main
    from fastapi.testclient import TestClient
    monkeypatch.setenv('PERSISTENCE_DB_PATH',str(tmp_path/'timing.sqlite3'))
    persistence.init_db()
    line=aligned_segment_line({'text':'Hallo.', 'start':2,'end':8,'words':[{'word':'Hallo.','start':6,'end':7}]})
    with TestClient(main.app) as client:
        response=client.post('/api/sessions',json={'current_step':2,'transcript':[line],'tops':['1 Begrüßung'],'assignments':[0]})
        assert response.status_code == 200
        session=client.get('/api/sessions/'+response.json()['session_id']).json()
        assert session['transcript'][0]['timing'] == line['timing']
        assert session['transcript'][0]['start'] == 6


def test_api_fresh_namespace_uid_mapping_and_exact_cache_replay(fake_openai_module, monkeypatch, tmp_path):
    import main
    from fastapi.testclient import TestClient
    monkeypatch.setenv('LLM_CACHE_DIR', str(tmp_path/'cache'))
    monkeypatch.setattr(agenda_llm, 'model_fingerprint', lambda c: {'digest': 'local-model'})
    fake_openai_module.content = fake_reply({'0': 'public:07'})
    payload = {'tops':['[Öffentlich] 07 Informationen'], 'top_ids':['stable-uid'], 'use_llm':True,
               'transcript':[{'line_id':'line-a','speaker':'S','text':'Beratung.','start':0,'end':1}]}
    client=TestClient(main.app)
    fresh=client.post('/api/agenda-detection',json={**payload,'fresh':True}).json()
    namespace=fresh['llm']['provenance']['cache_namespace']
    assert namespace and fresh['llm']['attempted_calls']==1
    assert fresh['llm']['provenance']['identities'][0]['top_uid']=='stable-uid'
    replay=client.post('/api/agenda-detection',json={**payload,'cache_namespace':namespace}).json()
    assert replay['llm']['attempted_calls']==0
    assert replay['assignments']==fresh['assignments']
    another=client.post('/api/agenda-detection',json={**payload,'fresh':True}).json()
    assert another['llm']['attempted_calls']==1
    assert another['llm']['provenance']['cache_namespace']!=namespace
