#!/usr/bin/env python3
"""Prove independent reference checks reject four deliberately wrong outcomes.

This is NOT an all-mutants campaign. It is a smoke check for the reference
oracle; all twelve mandatory real-backend mutants belong in G5 evidence.
"""
from fractions import Fraction
import tempfile
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'reference'))
from context_time_travel import Store, Conflict
from context_time_travel.numerics import assert_logits_close
from test_context_time_travel_toy import Snapshot, generate, attention


def check(description, fn):
    try:
        fn()
    except AssertionError:
        print('KILLED:', description)
        return
    raise RuntimeError('Mutant survived: '+description)


def wrong_branch():
    expected=attention(Snapshot((1,3,9),b'key'))
    wrong=attention(Snapshot((1,3,5,9),b'key'))
    assert wrong==expected, 'Branch leaked a sibling suffix'


def reset_rng():
    s=Snapshot((1,3),b'same-key')
    first,_=generate(s,7)
    second,continuation=generate(first,80)
    reset=Snapshot(first.tokens, first.stream_key, 0)
    _,wrong=generate(reset,80)
    assert continuation==wrong, 'RNG cursor reset at generation chunk boundary'


def perturbed_logits():
    try:
        assert_logits_close([1,3,100],[1,3,4],atol=1e-6,rtol=0)
    except AssertionError:
        raise AssertionError('Numerical difference detected')


def aba_cas():
    with tempfile.TemporaryDirectory() as tmp:
        db=Store(tmp)
        p=db.create_root('toy',(1,3),{})
        a=db.append(p,(5,),{})
        db.fork('main',p)
        db.move('main',0,a,'first')
        db.move('main',1,p,'second')
        try:
            db.move('main',0,a,'stale')
        except Conflict:
            raise AssertionError('ABA stale write refused')


if __name__=='__main__':
    for name,fn in [('sibling suffix leakage', wrong_branch),
                    ('RNG cursor reset', reset_rng),
                    ('logit corruption', perturbed_logits),
                    ('ABA stale reference', aba_cas)]:
        check(name,fn)
    print('REFERENCE MUTANT SMOKE PASS (4 deliberately wrong outcomes rejected)')
