"""Conservative, source-bound order guard; never invent a replacement assignment."""
from copy import deepcopy
from source_contract import reviewed


ORDER_PROMPT = """
Verbindliche feste TOP-Reihenfolge: Es gilt ausschließlich die Position in agenda,
nicht die sichtbare Nummer (öffentlich/nichtöffentlich kann erneut mit 01 beginnen).
Nach Beginn eines späteren TOPs darf kein früherer TOP wieder laufender TOP sein.
Überspringen ist erlaubt, aber nur ein im Original belegter tatsächlicher Beginn
rechtfertigt einen Wechsel, auch beim ersten Zielbeitrag oder nach einer Lücke.
Bloße Erwähnungen, Ankündigungen oder passende Titelwörter belegen keinen Beginn.
boundary=confirmed: evidence belegt den tatsächlichen Beginn an dieser Grenze
(höchstens zwei Zeilen davor/danach). boundary=continuation: derselbe bereits
belegte TOP läuft weiter. boundary=unclear: Grenze nicht eindeutig belegbar;
setze uncertain=true und top_ids=[] für den fraglichen Bereich. Keine gemeinsame
Zuordnung nacheinander behandelter TOPs. Übersprungene TOPs bleiben geplant;
ohne belegte Beratung erhalten sie keine Zeilen und den Status not_evidenced.
"""


def enforce_order(proposals, agenda, sources):
    """Return cleaned copies, concrete review ranges and the furthest confirmed TOP.

    A rejected jump cannot become accepted merely by repeating it in the next
    window. Contradictory backward proposals also reopen the preceding jump.
    Source text meaning remains a model decision, never title-string matching.
    """
    rows = deepcopy(proposals)
    positions = {item['top_id']: i for i, item in enumerate(agenda)}
    source_positions = {row['line_id']: row['index'] for row in sources}
    highwater, highwater_start = -1, 0
    issues = []

    def reject(start, end, reason):
        for row in rows[start:end+1]:
            if row['status'] != 'not_processed':
                row.update(top_ids=[], status='unassigned', uncertain=True,
                           review_status='unresolved', reason=reason)
                if row.get('grounding'):
                    row['grounding'] = reviewed(row['grounding'], questions=[reason])
        if issues and issues[-1]['reason'] == reason and issues[-1]['end_index'] + 1 >= start:
            issues[-1]['end_index'] = max(end, issues[-1]['end_index'])
        else:
            issues.append(dict(start_index=start, end_index=end, reason=reason))

    start = 0
    while start < len(rows):
        value = proposals[start]
        end = start
        # Expansion of a compact decision retains its actual boundary, including
        # at block edges. Do not mistake each line for a fresh transition.
        while end + 1 < len(rows) and all(proposals[end+1].get(k) == value.get(k)
                for k in ('boundary_start', 'top_ids', 'boundary', 'uncertain', 'review_status', 'status')):
            end += 1
        ids = value.get('top_ids', [])
        target = positions.get(ids[0], -1) if len(ids) == 1 else -1
        if value['status'] == 'not_processed':
            pass
        elif target >= 0 and target < highwater:
            reject(highwater_start, end, 'Widersprüchlicher TOP-Rücksprung; vorherigen Übergang und betroffenen Bereich manuell prüfen.')
        elif (target < 0 or value.get('uncertain') or value.get('review_status') in {'unresolved', 'technical_pending'}):
            reject(start, end, 'TOP-Grenze unklar oder widersprüchlich; Bereich manuell zuordnen.')
        else:
            local_evidence = any(abs(source_positions.get(e.get('line_id'), -100) - start) <= 2
                                 for e in value.get('evidence', []))
            continued_span = value.get('boundary_start', start) < start
            confirmed = value.get('boundary') == 'confirmed' and local_evidence and not continued_span
            continuation = ((value.get('boundary') == 'continuation' or continued_span) and target == highwater
                            and start > 0 and rows[start-1]['top_ids'] == ids)
            if not (confirmed or continuation):
                reject(start, end, 'TOP-Übergang ohne eindeutigen lokalen Beginnbeleg; Bereich manuell zuordnen.')
            elif target > highwater:
                highwater, highwater_start = target, start
        start = end + 1
    return rows, issues, highwater
