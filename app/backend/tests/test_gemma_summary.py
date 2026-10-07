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


def test_headings_removed_without_losing_inline_prose():
    assert gemma.paragraphs('## Zu TOP 2:\n\n**Aus der Beratung**\n\nMüller erkläre, es sei teuer.\n\n'
                            'Beschlüsse und Festlegungen:\nDer Ausschuss beschließt einstimmig.') == [
        'Müller erkläre, es sei teuer.', 'Der Ausschuss beschließt einstimmig.']
    assert gemma.paragraphs('## Zu TOP 1: Der Ausschuss vertagt.') == ['Der Ausschuss vertagt.']


@pytest.mark.parametrize('mode', ['fast', 'slow'])
def test_prose_is_immutable_and_only_adapter_call_uses_training_prompt(monkeypatch, mode):
    cfg = LLMConfig(base_url='http://llama:8080/v1', model='gemma-4-31b', api_key='local',
        provider='llama-cpp', summary_style='gemma4-lora', context_tokens=32768,
        processing_mode=mode, reasoning_effort='none')
    calls = []
    text = 'Müller erkläre, die Kosten betrügen 5 Euro.'
    def complete(client, config, **kwargs):
        calls.append((config, kwargs))
        if config.lora_scale:
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='## Zu TOP 2:\n\n' + text))])
        body = json.loads(kwargs['messages'][1]['content'])
        ids = [r['source_id'] for r in body['source']]
        if body['phase'] == 'protocol_sources':
            answer = {'paragraphs': [{'id': 'P:0', 'section': 'discussion', 'scope': 'current',
                                     'evidence': [{'source_id': ids[0]}]}], 'considered_source_ids': ids}
        else:
            answer = {'checked_claim_ids': [c['claim_id'] for c in body['candidate']],
                      'considered_source_ids': ids, 'issues': []}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(answer)))])
    monkeypatch.setattr(gemma, 'complete', complete)
    monkeypatch.setattr(summary_grounding, 'complete', complete)
    with processing_scope(mode):
        workflow = gemma.GemmaWorkflow(None, cfg, '2. Kosten', '', {})
        claims, issues, rows, count = workflow.run(['Müller: Die Kosten betragen 5 Euro.'])
    assert [c['text'] for c in claims] == [text]
    assert count == 1 and not issues
    assert sum(c.lora_scale == 1 for c, _ in calls) == 1
    assert [json.loads(k['messages'][1]['content'])['phase'] for c, k in calls if not c.lora_scale] == (
        ['protocol_sources'] if mode == 'fast' else ['protocol_sources', 'final_review', 'consolidated_review'])
    assert claims[0]['grounding']['content_status'] == ('unreviewed' if mode == 'fast' else 'supported')
    assert calls[0][1]['messages'][0]['content'].startswith(gemma.PROMPT)
    assert 'response_format' not in calls[0][1]


def test_partial_utterance_retains_speaker():
    cfg = LLMConfig(base_url='http://llama:8080/v1', model='gemma-4-31b', api_key='local',
                    provider='llama-cpp', summary_style='gemma4-lora')
    w = gemma.GemmaWorkflow(None, cfg, 'TOP 1', '', {})
    w.lines = ['Müller: Ein langer Beitrag.']
    assert w.source_text([{'line_index': 0, 'start_char': 15, 'text': 'Beitrag.'}]) == ['Müller: Beitrag.']


def test_protocol_export_understands_sections():
    from export_protocol import parse_summary_sections
    value = parse_summary_sections('Beschlüsse und Festlegungen:\nEinstimmig beschlossen.\n\n'
                                   'Aus der Beratung:\nMüller erläutere den Plan.')
    assert value['decisions'] == ['Einstimmig beschlossen.']
    assert value['discussion'] == ['Müller erläutere den Plan.']
