"""Isolated process crash/CAS worker; never invoked by production API."""
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from context_time_travel.real import BranchingModel
from context_time_travel.store import Store, Conflict
from recovery_observation import record

directory, action, target, point = sys.argv[1:]
def fault(at):
    if at == point:
        record('process-termination',point=at,exit_code=73,power_loss=False,
            native_counters=backend.stats if 'backend' in globals() else None)
        os._exit(73)
if action == 'move':
    store = Store(directory)
    store.fault_hook = fault
    try:
        result = store.move('main',0,target,'attempt')
        print(json.dumps(vars(result)))
    except Conflict:
        sys.exit(74)
elif action == 'race':
    while not (Path(directory)/'go').exists():
        time.sleep(.005)
    store = Store(directory)
    try:
        result = store.move('main',0,target,point)
        print(json.dumps(vars(result)))
    except Conflict:
        sys.exit(74)
elif action == 'feed':
    profile = json.loads(Path(os.environ['CTT_PROFILE']).read_text())
    backend = BranchingModel.load_backend(profile)
    try:
        sut = BranchingModel(directory,backend,profile)
        sut.store.fault_hook = fault
        sut._lease_fault = fault
        tip = sut.feed(target,profile['fixtures']['suffix_a'])
        sut.store.move('main',0,tip,'attempt')
    finally:
        backend.close()
