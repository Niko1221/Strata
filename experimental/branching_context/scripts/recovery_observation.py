"""Append small measured subprocess facts, independently of test assertions."""
import json
import os
from pathlib import Path

def record(kind, **values):
    destination = Path(os.environ.get('CTT_RECOVERY_OBSERVATIONS','evidence/runs/recovery-observations.jsonl'))
    destination.parent.mkdir(parents=True,exist_ok=True)
    data = (json.dumps(dict(kind=kind,pid=os.getpid(),**values))+'\n').encode()
    fd = os.open(destination,os.O_CREAT|os.O_WRONLY|os.O_APPEND,0o600)
    try:
        if os.write(fd,data)!=len(data):
            raise OSError('Incomplete evidence append')
        os.fsync(fd)
    finally:
        os.close(fd)
