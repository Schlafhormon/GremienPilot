"""Session-scoped processing policy, inherited only within one operation."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal

ProcessingMode = Literal['fast', 'slow']
VERSION = 'processing-modes-v2'
FAST_NOTICE = 'Fast – ohne automatische Inhaltsprüfung. Ungenauere Ergebnisse möglich.'


@dataclass(frozen=True)
class ProcessingPolicy:
    mode: ProcessingMode = 'slow'

    @property
    def fast(self):
        return self.mode == 'fast'

    def attempts(self, slow_attempts):
        return 1 if self.fast else slow_attempts

    def snapshot(self):
        return {'processing_mode': self.mode, 'policy_version': VERSION}


_CURRENT = ContextVar('processing_policy', default=ProcessingPolicy())


def policy():
    return _CURRENT.get()


@contextmanager
def processing_scope(mode=None):
    if mode is None:
        yield policy()
        return
    if mode not in ('fast', 'slow'):
        raise ValueError('processing_mode must be fast or slow')
    token = _CURRENT.set(ProcessingPolicy(mode))
    try:
        yield policy()
    finally:
        _CURRENT.reset(token)


def agenda_complete(usage):
    return bool(usage.get('processing_complete') and (
        usage.get('review_complete') or
        (usage.get('processing_mode') == 'fast' and usage.get('review_status') == 'skipped')))


def pdf_usable(result, mode='slow'):
    if not result or not result.get('processing_complete') or not result.get('tops'):
        return False
    if result.get('processing_mode', 'slow') == 'fast':
        return mode == 'fast' and result.get('review_status') == 'skipped'
    return not result.get('review_required')
