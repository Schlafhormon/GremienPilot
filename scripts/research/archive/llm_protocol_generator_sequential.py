#!/usr/bin/env python3
"""Legacy command redirected to the verified full-source workflow.

Historical heuristic and partial-source inference implementations were removed.
Use --help for the shared JSON input CLI. No old research outputs are modified.
"""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parent.parent / 'model_workflow.py'), run_name='__main__')
