"""Versioned branch-local sampler/controller, independent of backend samplers."""
import codecs
import copy
import hashlib
import hmac
import math
from .store import IntegrityError, InvalidOperation

_RNG_ALGORITHM = 'hmac-sha256-u53-v1'
_SAMPLER_ORDER = ['grammar/support','repetition-penalty','temperature','categorical']
_DRAW_LIMIT = 1 << 128


def _validated_state(state):
    if not isinstance(state, dict):
        raise IntegrityError('Invalid decoder controller state')
    d = copy.deepcopy(state)
    # Idle is the canonical pre-generation state. Forced inputs may extend it
    # without inventing a sampler, grammar or random stream.
    if d.get('status') == 'idle':
        if set(d) != {'status'}:
            raise IntegrityError('Invalid idle decoder controller state')
        return d
    if d.get('schema') != 1:
        raise IntegrityError('Unsupported decoder controller schema')
    if d.get('rng_algorithm') != _RNG_ALGORITHM:
        raise IntegrityError('Unsupported decoder RNG algorithm')
    if d.get('sampler_order') != _SAMPLER_ORDER:
        raise IntegrityError('Unsupported sampler processing order')
    cursor = d.get('next_draw')
    if type(cursor) is not int or not 0 <= cursor < _DRAW_LIMIT:
        raise IntegrityError('Invalid decoder RNG draw cursor')
    return d


def start(policy, stream, history, vocab_size):
    p = dict(allowed=list(range(vocab_size)),temperature=1.0,repetition_penalty=1.0,
             eos_tokens=[],stop_sequences=[],grammar=None,rejection_threshold=1.0)
    if set(policy)-set(p):
        raise InvalidOperation('Unsupported sampler policy field')
    p.update(copy.deepcopy(policy))
    p['allowed'] = sorted(set(p['allowed']))
    if not p['allowed'] or any(type(t) is not int or not 0<=t<vocab_size for t in p['allowed']):
        raise InvalidOperation('Invalid allowed-token support')
    if (not 0<p['rejection_threshold']<=1 or not math.isfinite(p['temperature'])
        or p['temperature']<0 or not math.isfinite(p['repetition_penalty']) or p['repetition_penalty']<=0):
        raise InvalidOperation('Invalid sampler numeric policy')
    counts = {}
    for token in history:
        counts[str(token)] = counts.get(str(token),0)+1
    grammar = p['grammar']
    if grammar is not None and (set(grammar) != {'start','transitions'} or grammar['start'] not in grammar['transitions']):
        raise InvalidOperation('Invalid token-DFA grammar')
    return dict(schema=1,status='active',rng_algorithm=_RNG_ALGORITHM,
        stream_key=hashlib.sha256(stream.encode()).hexdigest(),next_draw=0,policy=p,
        sampler_order=copy.deepcopy(_SAMPLER_ORDER),
        grammar_state=grammar['start'] if grammar else None,seen=counts,stop_tail=[],
        accepted_in_scope=0,output_pending_hex='',output_decoder='utf8-incremental-replace-v1')


def choose(logits, state):
    d = _validated_state(state)
    if d['status'] != 'active':
        raise InvalidOperation('Generation scope is not active')
    p = d['policy']
    allowed = p['allowed']
    if p['grammar']:
        transitions = p['grammar']['transitions'].get(d['grammar_state'],{})
        allowed = [t for t in allowed if str(t) in transitions]
    if not allowed:
        raise IntegrityError('Empty allowed-token support; grammar cannot be bypassed')
    scores = []
    for token in allowed:
        score = logits[token]
        if not math.isfinite(score):
            raise IntegrityError('Non-finite raw logit')
        if str(token) in d['seen']:
            score = score/p['repetition_penalty'] if score>0 else score*p['repetition_penalty']
        scores.append(score)
    if p['temperature'] == 0:
        return allowed[max(range(len(scores)),key=scores.__getitem__)],d
    while True:
        draw = d['next_draw']
        if draw >= _DRAW_LIMIT - 1:
            raise IntegrityError('Decoder RNG draw cursor is exhausted')
        raw = hmac.new(bytes.fromhex(d['stream_key']),b'ctt-u53-v1:'+draw.to_bytes(16,'big'),hashlib.sha256).digest()
        u = (int.from_bytes(raw[:8],'big')>>11)/2**53
        d['next_draw'] += 1
        if u < p['rejection_threshold']:
            u /= p['rejection_threshold']
            break
    high = max(scores)
    weights = [math.exp((score-high)/p['temperature']) for score in scores]
    target = u*sum(weights)
    running = 0.0
    for token,weight in zip(allowed,weights):
        running += weight
        if running > target:
            return token,d
    return allowed[-1],d


def consume(state, token, piece, is_eog):
    d = _validated_state(state)
    if d['status'] != 'active':
        return d
    p = d['policy']
    if p['grammar']:
        transitions = p['grammar']['transitions'].get(d['grammar_state'],{})
        if str(token) not in transitions:
            raise IntegrityError('Token violates active grammar')
        d['grammar_state'] = transitions[str(token)]
    d['seen'][str(token)] = d['seen'].get(str(token),0)+1
    d['accepted_in_scope'] += 1
    max_stop = max(map(len,p['stop_sequences']),default=0)
    d['stop_tail'] = (d['stop_tail']+[token])[-max_stop:] if max_stop else []
    decoder = codecs.getincrementaldecoder('utf8')(errors='replace')
    decoder.setstate((bytes.fromhex(d['output_pending_hex']),0))
    decoder.decode(piece,final=False)
    d['output_pending_hex'] = decoder.getstate()[0].hex()
    if is_eog or token in p['eos_tokens'] or any(d['stop_tail'][-len(s):]==s for s in p['stop_sequences'] if s):
        d['status'] = 'stopped'
    return d
