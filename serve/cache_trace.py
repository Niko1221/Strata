"""Metadata-only cache prefix inspection; no prompt text or token IDs are recorded."""
from . import slot_policy

def snapshot(prompt, held, busy, valid, used, now):
    rows=[]
    for b,h in enumerate(held):
        hi=min(len(prompt),len(h));lo=0
        if h[:hi]==prompt[:hi]:lo=hi
        else:
            while lo<hi:
                middle=(lo+hi+1)//2
                if h[:middle]==prompt[:middle]:lo=middle
                else:hi=middle-1
        state=slot_policy.cache_state(len(h),used[b],now,valid[b])
        rows.append({"slot":b,"held_tokens":len(h),"busy":bool(busy[b]),
                     "valid_hint":bool(valid[b]),"expired":state["expired"],
                     "common_prefix_tokens":lo,"full_prefix_match":bool(h) and lo==len(h) and len(h)<len(prompt)})
    return {"prompt_tokens":len(prompt),"slots":rows}
