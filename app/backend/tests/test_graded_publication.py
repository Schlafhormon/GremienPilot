from test_summary_grounding import generate


def test_assignment_uncertainty_survives_summary_generation(summary_model):
    import main
    result=generate()
    session=dict(agenda_proposals=dict(result=dict(llm=dict(line_results=[dict(index=0,line_id='original',
        review_status='unresolved',uncertain=True,grounding=dict(evidence_status='source_range'))]))))
    main.preserve_assignment_questions(result,session,[0,1])
    assert result.llm_usage['processing_complete'] and result.llm_usage['assignment_review_required']
    assert '[UNBESTÄTIGT' in result.summary and 'Wiederaufnahme' in result.summary
    assert result.structured.evidence[0]['grounding']['assignment_origins'][0]['line_id']=='original'


def test_saving_manual_text_retains_open_questions(summary_model):
    import main
    result=generate()
    old=dict(session_id='s',tops=['TOP'],top_ids=['a'],assignments=[0,0],transcript=[
        dict(line_id=str(i),speaker=s,text=t,start=i,end=i+1) for i,(s,t) in enumerate([('A','Sachverhalt.'),('B','Fortsetzung.')])],
        summaries={0:result.summary},summary_reviews={0:dict(structured=result.structured.to_dict(),llm_usage=result.llm_usage)})
    old=main.reconcile_session_summaries(None,old)
    edited=main.reconcile_session_summaries(old,{**old,'summaries':{0:'Beschluss:\nAntrag angenommen.'}})
    assert 'Antrag angenommen.' in edited['summaries'][0]
    assert '[UNBESTÄTIGT' in edited['summaries'][0]
    assert edited['summary_reviews'][0]['retained_evidence']
    assert not edited['summary_reviews'][0]['llm_usage']['processing_complete']
    stripped=main.reconcile_session_summaries(old,{**old,'summaries':{0:'Antrag angenommen.'},'summary_reviews':{0:{}}})
    assert '[UNBESTÄTIGT' in stripped['summaries'][0]
    assert stripped['summary_reviews'][0]['retained_evidence']
