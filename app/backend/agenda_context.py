"""Identity-bound original sources. No speech-act inference or agenda anchors."""
import hashlib
import json
import math
from agenda_labels import parse_agenda_label


def model_agenda(tops, top_ids=None):
    if top_ids and (len(top_ids) != len(tops) or len(set(top_ids)) != len(tops) or not all(isinstance(t, str) and t for t in top_ids)):
        raise ValueError('invalid_top_ids')
    return [dict(top_id=top_ids[i] if top_ids else f'agenda:{i}', title=title,
                 number=parse_agenda_label(title).original_number,
                 section=parse_agenda_label(title).section)
            for i, title in enumerate(tops)]


def source_rows(transcript):
    digest = hashlib.sha256(json.dumps([(t.speaker, t.text, t.start, t.end) for t in transcript],
                                       ensure_ascii=False).encode()).hexdigest()
    rows = []
    for index, line in enumerate(transcript):
        if ((line.start is None) != (line.end is None) or
            (line.start is not None and (not math.isfinite(line.start) or not math.isfinite(line.end)
                                        or not 0 <= line.start <= line.end))):
            raise ValueError('invalid_source_times')
        rows.append(dict(index=index, line_id=line.line_id or f'transcript:{digest}:{index}',
                         speaker=line.speaker, text=line.text, start=line.start, end=line.end))
    if len({r['line_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate_source_id')
    return rows
