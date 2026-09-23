"""Run the first real context block and an original-source review locally.

Run inside the Linux backend environment after synthetic tests. Private artifacts
go to a separate SQLite database in the Linux state volume; stdout is content-free.
This is a bounded probe, never an end-to-end certificate or a pipeline checkpoint.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'app'/'backend'))
import durable_jobs as durable
import persistence
from agenda_llm import Workflow, obj, array, TEXT, EVIDENCE
from agenda_context import model_agenda
from agenda_detection import AgendaLLMUsage
from assignment_suggestions import TranscriptUtterance
from llm_config import get_llm_config
from llm_transport import model_fingerprint, request_control


def probe(job_id, output):
    if os.name == 'nt':
        raise RuntimeError('Run inside Linux; never open the live volume from Windows SQLite')
    job = durable.load(job_id)
    pipeline = persistence.load_pipeline_job(job_id)
    with persistence.connect() as db:
        records = dict(db.execute('SELECT step_key,value FROM durable_steps WHERE job_id=?',(job_id,)))
    options = pipeline['result_refs']['options']
    model = options.get('agenda_model') or options.get('model')
    config = get_llm_config(model)
    if urlparse(config.base_url).hostname not in {'ollama','localhost','127.0.0.1','host.docker.internal'}:
        raise RuntimeError('Probe requires a local model service')
    identities = json.loads(records['model-identities'])
    if model_fingerprint(config) != identities.get(config.model):
        raise RuntimeError('Model identity mismatch')
    if config.public_snapshot() != job['payload']['versions']['model']:
        raise RuntimeError('Model configuration mismatch')
    transcript = [TranscriptUtterance(**{k:r.get(k) for k in ('speaker','text','line_id','start','end')})
                  for r in json.loads(records['pipeline:agenda-transcript:v1'])]
    pdf = json.loads(records['pipeline:pdf:page-evidence-v3'])
    agenda = model_agenda(pdf['tops'],[i['id'] for i in pdf['items'] if i['kind']=='agenda'])
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    os.environ['PERSISTENCE_DB_PATH']=str(output/'probe.sqlite3')
    lock=durable.ProcessLock()
    lock.__enter__()
    persistence.init_db()
    with persistence.connect() as db:
        previous=db.execute('select job_id from durable_jobs').fetchone()
    scratch=durable.load(previous[0]) if previous else durable.submit('agenda',{'model':config.model})
    if scratch['payload']['versions'] != durable.version_snapshot(scratch['payload']):
        lock.__exit__()
        raise RuntimeError('Probe code/configuration changed; choose a new output directory')
    with persistence.connect() as db:
        db.execute("UPDATE durable_jobs SET state='queued',owner=NULL,lease_until=NULL")
    runtime=durable.Runtime(durable.claim('bounded-probe',3600),'bounded-probe',threading.Event())
    token=durable.CURRENT.set(runtime)
    usage=AgendaLLMUsage(True,'local_probe')
    began=time.monotonic()
    captured={}
    class BlockDone(Exception): pass
    try:
        with request_control(durable.check,durable.progress):
            work=Workflow(transcript,usage,model,options.get('agenda_system_prompt'),None,'local_probe')
            original=work.call
            def capture(phase,instruction,body,schema,validate):
                captured.update(body=body,phase=phase)
                captured['draft']=original(phase,instruction,body,schema,validate)
                raise BlockDone()
            work.call=capture
            try: work.context('primary',agenda)
            except BlockDone: pass
            if not captured:
                raise RuntimeError('Transcript did not require context condensation')
            rows=captured['body']['sources']
            ids={r['line_id'] for r in rows}
            issue_schema=array(obj({'kind':{'enum':['unsupported','omission','contradiction','unclear']},
                                    'question':TEXT,'evidence':EVIDENCE}))
            instruction=('Prüfe den unbestätigten Kontextentwurf unabhängig gegen ALLE vorliegenden Originalquellen. '
                'Melde konkrete Auslassungen, Widersprüche und unbelegte Aussagen. '
                'Beurteile nur im Quellausschnitt belegbare Probleme. Fehlende lokale Erwähnung ist kein Gegenbeweis. '
                'Prüfe Beschlüsse gegenüber bloßen Vorschlägen, Stimmenzahlen, Verneinungen, '
                'gemeinsame Beratungen und Wiederaufnahmen. reviewed_range bestätigt exakt das vollständig gelesene Quellfenster.')
            checked=[]
            def review_window(window):
                bounds={'start':window[0]['index'],'end':window[-1]['index']}
                schema=obj({'reviewed_range':obj({key:{'type':'integer','enum':[value]} for key,value in bounds.items()}),
                            'issues':issue_schema})
                body={'sources':window,'candidate':captured['draft'],'required_range':bounds}
                if not work.fits('probe:independent_context_review',instruction,body,schema):
                    if len(window)<2: raise RuntimeError('Single source does not fit independent review')
                    middle=len(window)//2
                    return review_window(window[:middle])+review_window(window[middle:])
                expected={r['line_id'] for r in window}
                def check(data):
                    if data.get('reviewed_range') != bounds:
                        raise ValueError('Incomplete original-source review coverage')
                    for issue in data['issues']:
                        work.text(issue['question'])
                        work.evidence(issue['evidence'],required=False)
                answer=original('probe:independent_context_review',instruction,body,schema,check)
                durable.artifact('probe:coverage','completed_original_window',{'range':bounds,
                    'original_ids':sorted(expected),'source_sha256':work.catalog.sha256})
                checked.extend(expected)
                return answer['issues']
            review={'issues':review_window(rows)}
            if set(checked)!=ids or len(checked)!=len(ids):
                raise RuntimeError('Original review coverage gap')
            durable.artifact('probe','review',review)
        with persistence.connect() as db:
            metrics=[json.loads(r[0]) for r in db.execute('SELECT metrics FROM durable_metrics')]
            failures=db.execute("SELECT count(*) FROM durable_artifacts WHERE kind='technical_diagnostic'").fetchone()[0]
        report=dict(job_id=job_id,probe_job_id=scratch['job_id'],model=config.model,
            context_tokens=config.context_tokens,source_count=len(rows),source_start=rows[0]['index'],source_end=rows[-1]['index'],
            context_reference_status=captured['draft']['grounding']['reference_status'],
            context_content_status='unreviewed',independent_source_coverage=len(ids),
            review_issue_counts={kind:sum(i['kind']==kind for i in review['issues']) for kind in ['unsupported','omission','contradiction','unclear']},
            calls=len(metrics),new_calls=usage.attempted_calls,failed_calls=failures,duration_seconds=round(time.monotonic()-began,2),
            open_questions=len(review['issues'])+len(captured['draft']['grounding']['questions']),metrics=metrics,
            end_to_end_verified=False)
        (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        with persistence.connect() as db:
            db.execute("UPDATE durable_jobs SET state='review_required',result=?,owner=NULL,lease_until=NULL",(json.dumps(report),))
        print(json.dumps(report,ensure_ascii=False))
    finally:
        durable.CURRENT.reset(token)
        lock.__exit__()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job_id')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    probe(args.job_id,args.output)
