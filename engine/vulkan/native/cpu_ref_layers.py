"""Multi-layer CPU reference for qwen35moe; saves layer outputs for comparison."""
import struct, sys
import numpy as np

PATH = "/home/g/Code/llama.cpp/models/Qwen3.8-35B-A3B-Q4_K_M.gguf"
f = open(PATH, "rb"); assert f.read(4) == b"GGUF"
struct.unpack("<I", f.read(4)); nt, nkv = struct.unpack("<QQ", f.read(16))
def rs():
    n = struct.unpack("<Q", f.read(8))[0]; return f.read(n)
def rv(t):
    if t in (0,1,7): f.read(1)
    elif t in (2,3): f.read(2)
    elif t in (4,5,6): f.read(4)
    elif t in (10,11,12): f.read(8)
    elif t==8: rs()
    elif t==9:
        et,n=struct.unpack("<Iq",f.read(12))
        for _ in range(n): rv(et)
for _ in range(nkv): rs(); rv(struct.unpack("<I",f.read(4))[0])
T = {}
for _ in range(nt):
    name=rs().decode(); nd=struct.unpack("<I",f.read(4))[0]
    dims=struct.unpack("<"+"Q"*nd,f.read(8*nd))
    t=struct.unpack("<I",f.read(4))[0]; off=struct.unpack("<Q",f.read(8))[0]
    T[name]=(t,dims,off)
f.close(); DB = 10990752

def h2f(h):
    h=np.uint16(h).astype(np.int32); sign=np.where((h>>15)&1,-1.,1.).astype(np.float32)
    e=(h>>10)&0x1f; m=h&0x3ff
    val=np.where(e==0, m.astype(np.float32)*np.float32(2.**-24),
          np.where(e==31, np.inf, (1.+m.astype(np.float32)/1024.)*np.float32(2.)**(e.astype(np.float32)-15.)))
    return (sign*val).astype(np.float32)
def gsc(j,q):
    if j<4: return q[j]&63,q[j+4]&63
    return (q[j+4]&0xf)|((q[j-4]>>6)<<4),(q[j+4]>>4)|((q[j]>>6)<<4)
def dq4(data,ni,no):
    out=np.zeros((no,ni),np.float32); rs_=ni//256*144
    for r in range(no):
        row=data[r*rs_:(r+1)*rs_]
        for blk in range(ni//256):
            b=row[blk*144:(blk+1)*144]; d=h2f(np.array([int(b[0])|(int(b[1])<<8)]))[0]; mn=h2f(np.array([int(b[2])|(int(b[3])<<8)]))[0]
            sc=b[4:20]; qs=b[16:144]
            for j in range(4):
                s0,m0=gsc(2*j,sc); s1,m1=gsc(2*j+1,sc); base=blk*256+j*64
                out[r,base:base+32]=d*s0*(qs[j*32:(j+1)*32]&0xf)-mn*m0
                out[r,base+32:base+64]=d*s1*(qs[j*32:(j+1)*32]>>4)-mn*m1
    return out
def dq6(data,ni,no):
    out=np.zeros((no,ni),np.float32); rs_=ni//256*210
    for r in range(no):
        row=data[r*rs_:(r+1)*rs_]
        for blk in range(ni//256):
            b=row[blk*210:(blk+1)*210]; ql=b[0:128].astype(np.int32); qh=b[128:192].astype(np.int32)
            sc=b[192:208].astype(np.int8).astype(np.float32); d=h2f(np.array([int(b[208])|(int(b[209])<<8)]))[0]
            for hb in range(2):
                for l in range(32):
                    is_=l//16; q1=(ql[hb*64+l]&0xf)|((qh[hb*32+l]&3)<<4); q2=(ql[hb*64+l+32]&0xf)|(((qh[hb*32+l]>>2)&3)<<4)
                    q3=(ql[hb*64+l]>>4)|(((qh[hb*32+l]>>4)&3)<<4); q4=(ql[hb*64+l+32]>>4)|(((qh[hb*32+l]>>6)&3)<<4)
                    o=blk*256+hb*128+l; so=sc[hb*8:]
                    out[r,o]=d*so[is_]*(q1-32); out[r,o+32]=d*so[is_+2]*(q2-32); out[r,o+64]=d*so[is_+4]*(q3-32); out[r,o+96]=d*so[is_+6]*(q4-32)
    return out
def getw(name):
    t,dims,off=T[name]; be=1 if t in (0,1) else (32 if t==8 else 256)
    bb={0:4,1:2,12:144,14:210,8:272}[t]; nb=(dims[0]//be)*bb
    for d in dims[1:]: nb*=d
    raw=np.fromfile(PATH,dtype=np.uint8,count=nb,offset=DB+off)
    if t in (12,14):
        dq=dq4 if t==12 else dq6
        if len(dims)==2: return dq(raw,dims[0],dims[1])
        per,ne=int(dims[1]),int(dims[2]); return dq(raw,dims[0],per*ne).reshape(ne,per,dims[0])
    if t==0:
        if len(dims)==2: return raw.view(np.float32).reshape((dims[1],dims[0]))
        return raw.view(np.float32).reshape(dims)
    raise Exception((name,t))
def getrow(name,row):
    t,dims,off=T[name]; rs_=(dims[0]//256)*144 if t==12 else dims[0]*4
    raw=np.fromfile(PATH,dtype=np.uint8,count=rs_,offset=DB+off+row*rs_)
    return dq4(raw,dims[0],1)[0] if t==12 else raw.view(np.float32)[0]
def getexpert(name,e):
    t,dims,off=T[name]; per,ni=int(dims[1]),int(dims[0]); rs_=ni//256*(144 if t==12 else 210)
    data=np.fromfile(PATH,dtype=np.uint8,count=per*rs_,offset=DB+off+e*per*rs_)
    return (dq4 if t==12 else dq6)(data,ni,per)
def rms(x,w,eps=1e-6): return x/np.sqrt(np.mean(x*x)+eps)*w
def silu(v): return v/(1+np.exp(-v)); 
def sigm(v): return 1/(1+np.exp(-v))

tok=int(sys.argv[1]) if len(sys.argv)>1 else 3710
NL=int(sys.argv[2]) if len(sys.argv)>2 else 4
N=2048;HK=16;HV=32;DS=128;DC=4;C=8192;H=16;HKV=2;D=256;nrot=64;base=1e7
x=getrow('token_embd.weight',tok).copy()
convS={}; SS={}; KV={}
outs=[]
for L in range(NL):
    h=rms(x,getw(f'blk.{L}.attn_norm.weight'))
    if L%4!=3:  # GDN
        qkv=getw(f'blk.{L}.attn_qkv.weight')@h; z=getw(f'blk.{L}.attn_gate.weight')@h
        alpha=getw(f'blk.{L}.ssm_alpha.weight')@h; beta=getw(f'blk.{L}.ssm_beta.weight')@h
        taps=getw(f'blk.{L}.ssm_conv1d.weight'); st=convS.get(L,np.zeros((DC-1,C),np.float32)); co=np.zeros(C,np.float32)
        for c in range(C):
            acc=sum(st[k,c]*taps[c,k] for k in range(DC-1))+qkv[c]*taps[c,3]
            for k in range(DC-2): st[k,c]=st[k+1,c]
            st[DC-2,c]=qkv[c]; co[c]=silu(np.float32(acc))
        convS[L]=st
        dt=getw(f'blk.{L}.ssm_dt.bias');sa=getw(f'blk.{L}.ssm_a');S=SS.get(L,np.zeros((HV,DS,DS),np.float32));go=np.zeros((HV,DS),np.float32)
        for hh in range(HV):
            kh=hh%HK; qi=co[kh*DS:kh*DS+DS];ki=co[HK*DS+kh*DS:HK*DS+kh*DS+DS];vi=co[2*HK*DS+hh*DS:2*HK*DS+hh*DS+DS]
            q=qi/np.sqrt(np.sum(qi*qi)+1e-6)/np.sqrt(DS);k=ki/np.sqrt(np.sum(ki*ki)+1e-6)
            g=np.exp(np.log1p(np.exp(alpha[hh]+dt[hh]))*sa[hh]);b=sigm(beta[hh]);Si=S[hh]*g
            d=(vi-Si@k)*b; S[hh]=Si+np.outer(d,k); go[hh]=S[hh]@q
        SS[L]=S; nw=getw(f'blk.{L}.ssm_norm.weight');o=np.zeros_like(go)
        for hh in range(HV):
            v=go[hh];o[hh]=v/np.sqrt(np.mean(v*v)+1e-6)*nw*silu(z[hh*DS:(hh+1)*DS])
        y=getw(f'blk.{L}.ssm_out.weight')@o.reshape(-1)
    else:  # attention
        qg=getw(f'blk.{L}.attn_q.weight')@h; kk=getw(f'blk.{L}.attn_k.weight')@h; vv=getw(f'blk.{L}.attn_v.weight')@h
        qw=getw(f'blk.{L}.attn_q_norm.weight');kw=getw(f'blk.{L}.attn_k_norm.weight')
        q=np.zeros((H,D),np.float32);gate=np.zeros((H,D),np.float32);kc,vc=KV.get(L,(np.zeros((HKV,4096,D),np.float32),np.zeros((HKV,4096,D),np.float32)))
        pos=0
        def rope(v,pos):
            half=nrot//2; out=v.copy()
            for i in range(nrot):
                p=i if i<half else i-half; th=pos*base**(-2*p/nrot); cs,sn=np.cos(th),np.sin(th)
                out[i]=v[i]*cs-v[i+half]*sn if i<half else v[i-half]*sn+v[i]*cs
            return out
        for hh in range(H):
            src=qg[hh*2*D:hh*2*D+D]; q[hh]=rope(src/np.sqrt(np.mean(src*src)+1e-6)*qw,pos); gate[hh]=qg[hh*2*D+D:hh*2*D+2*D]
        for hh in range(HKV):
            src=kk[hh*D:hh*D+D]; kc[hh,pos]=rope(src/np.sqrt(np.mean(src*src)+1e-6)*kw,pos); vc[hh,pos]=vv[hh*D:hh*D+D]
        KV[L]=(kc,vc); np.save('/tmp/opencode/ref_qg.npy',qg); np.save('/tmp/opencode/ref_q.npy',q); np.save('/tmp/opencode/ref_gate.npy',gate); np.save('/tmp/opencode/ref_kc.npy',kc); np.save('/tmp/opencode/ref_vc.npy',vc); att=np.zeros((H,D),np.float32)
        for hh in range(H):
            kvh=hh//(H//HKV); sc=np.array([np.dot(q[hh],kc[kvh,p])/np.sqrt(D) for p in range(pos+1)])
            sc=np.exp(sc-sc.max()); sc/=sc.sum(); att[hh]=(sc[:,None]*vc[kvh,:pos+1]).sum(0)*sigm(gate[hh])
        np.save('/tmp/opencode/ref_att.npy',att)
        y=getw(f'blk.{L}.attn_output.weight')@att.reshape(-1)
    x=x+y; outs.append(x.copy())
    h2=rms(x,getw(f'blk.{L}.post_attention_norm.weight'))
    r=getw(f'blk.{L}.ffn_gate_inp.weight')@h2; p=np.exp(r-r.max());p/=p.sum()
    ids=np.argsort(-p)[:8]; w=p[ids];w/=w.sum()
    parts=np.zeros((8,N),np.float32)
    for s,e in enumerate(ids):
        gg=getexpert(f'blk.{L}.ffn_gate_exps.weight',e)@h2; uu=getexpert(f'blk.{L}.ffn_up_exps.weight',e)@h2
        parts[s]=getexpert(f'blk.{L}.ffn_down_exps.weight',e)@(silu(gg)*uu)
    sg=getw(f'blk.{L}.ffn_gate_shexp.weight')@h2; su=getw(f'blk.{L}.ffn_up_shexp.weight')@h2
    sy=getw(f'blk.{L}.ffn_down_shexp.weight')@(silu(sg)*su); gs=sigm(getw(f'blk.{L}.ffn_gate_inp_shexp.weight')@h2)
    x=x+(w[:,None]*parts).sum(0)+gs*sy
    outs[-1]=x.copy()
    print('L',L,'x',x[:3],'abs',np.abs(x).max(),flush=True)
np.save('/tmp/opencode/ref_layers.npy',np.array(outs))
print('saved',len(outs),'layers')
