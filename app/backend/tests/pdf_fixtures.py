"""Synthetic public-domain fixtures; model responses below are test doubles."""
from io import BytesIO
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.utils import ImageReader
from PIL import Image, ImageDraw


def pdf_bytes(kinds=('digital',)):
    output = BytesIO()
    canvas = Canvas(output, pagesize=(400, 500))
    for index, kind in enumerate(kinds, 1):
        lines = [f'Einladung Seite {index}', '01 Rat', '02.1 Bau', 'Nichtoeffentlicher Teil', '01 Personal']
        if kind == 'scan':
            image = Image.new('RGB', (800, 1000), 'white')
            draw = ImageDraw.Draw(image)
            for n, line in enumerate(lines):
                draw.text((70, 80 + n * 50), line, fill='black', font_size=24)
            canvas.drawImage(ImageReader(image), 0, 0, 400, 500)
        else:
            for n, line in enumerate(lines):
                canvas.drawString(35, 460 - n * 25, line)
        canvas.showPage()
    canvas.save()
    return output.getvalue()


def agenda(pages=(1,), items=None):
    return {'items': items if items is not None else [item('p1-a')],
            'metadata': dict.fromkeys(['committee', 'date', 'time', 'location', 'title'], ''),
            'metadata_sources': {key: [] for key in ['committee', 'date', 'time', 'location', 'title']},
            'pages': list(pages)}


def item(identity, page=1, number=None, title='Haushalt', section=None, parent_id=None, kind='agenda'):
    return {'id': identity, 'number': number, 'title': title, 'section': section,
            'parent_id': parent_id, 'kind': kind, 'sources': [{'page': page, 'quote': None}]}


def audit(page=1, issues=None):
    return {'page': page, 'complete': not issues, 'issues': issues or []}
