"""CPU reference for one qwen35moe layer, to compare with the Vulkan dump.
Reads GGUF bytes directly; dequant matches ggml-quants.c / q36ref.cu.
"""
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
pos=f.tell(); f.close()
DB = (pos+31)//32*32
print("DB",DB)

def h2f(h):
    h=np.uint16(h).astype(np.int32)
    sign=np.where((h>>15)&1, -1.0, 1.0).astype(np.float32)
    e=(h>>10)&0x1f; m=h&0x3ff
    val=np.where(e==0, m.astype(np.float32)*np.float32(2.0**-24),
          np.where(e==31, np.where(m==0, np.float32(np.inf), np.float32(np.nan)),
                   (1.0+m.astype(np.float32)/1024.0)*np.float32(2.0)**(e.astype(np.float32)-15.0)))
    return (sign*val).astype(np.float32)

def get_sc_min(j, q):
    if j<4: return q[j]&63, q[j+4]&63
    return (q[j+4]&0xf)|((q[j-4]>>6)<<4), (q[j+4]>>4)|((q[j]>>6)<<4)

def load_raw(name):
    t,dims,off=T[name]
    be = 1 if t in (0,1) else (32 if t==8 else 256)
    bb = {0:4,1:2,12:144,14:210,8:272}[t]
    nb = (dims[0]//be)*bb
    for d in dims[1:]: nb*=d
    data = np.fromfile(PATH, dtype=np.uint8, count=nb, offset=DB+off)
    return t, dims, data

def dequant_q4k(data, n_in, n_out):
    # 3-D expert tensors: rows = n_out * n_experts, reshape below.
    out=np.zeros((n_out,n_in),dtype=np.float32)
    rs_=n_in//256*144
    for r in range(n_out):
        row=data[r*rs_:(r+1)*rs_]
        for blk in range(n_in//256):
            b=row[blk*144:(blk+1)*144]
            d=h2f(np.array([int(b[0])|(int(b[1])<<8)]))[0]; mn=h2f(np.array([int(b[2])|(int(b[3])<<8)]))[0]
            sc=b[4:20]; qs=b[16:144]
            for j in range(4):
                s0,m0=get_sc_min(2*j,sc); s1,m1=get_sc_min(2*j+1,sc)
                base=blk*256+j*64
                out[r,base:base+32]=d*s0*(qs[j*32:(j+1)*32]&0xf)-mn*m0
                out[r,base+32:base+64]=d*s1*(qs[j*32:(j+1)*32]>>4)-mn*m1
    return out

def dequant_q6k(data, n_in, n_out):
    out=np.zeros((n_out,n_in),dtype=np.float32)
    rs_=n_in//256*210
    for r in range(n_out):
        row=data[r*rs_:(r+1)*rs_]
        for blk in range(n_in//256):
            b=row[blk*210:(blk+1)*210]
            ql=b[0:128].astype(np.int32); qh=b[128:192].astype(np.int32)
            sc=b[192:208].astype(np.int8).astype(np.float32)
            d=h2f(np.array([int(b[208])|(int(b[209])<<8)]))[0]
            for hb in range(2):
                for l in range(32):
                    is_=l//16
                    q1=(ql[hb*64+l]&0xf)|((qh[hb*32+l]&3)<<4)
                    q2=(ql[hb*64+l+32]&0xf)|(((qh[hb*32+l]>>2)&3)<<4)
                    q3=(ql[hb*64+l]>>4)|(((qh[hb*32+l]>>4)&3)<<4)
                    q4=(ql[hb*64+l+32]>>4)|(((qh[hb*32+l]>>6)&3)<<4)
                    o=blk*256+hb*128+l
                    sco=sc[hb*8:]
                    out[r,o]=d*sco[is_+0]*(q1-32)
                    out[r,o+32]=d*sco[is_+2]*(q2-32)
                    out[r,o+64]=d*sco[is_+4]*(q3-32)
                    out[r,o+96]=d*sco[is_+6]*(q4-32)
    return out

def dequant_q4k_row(data, row, n_in):
    out=np.zeros(n_in,dtype=np.float32); rs_=n_in//256*144
    r=data[row*rs_:(row+1)*rs_]
    for blk in range(n_in//256):
        b=r[blk*144:(blk+1)*144]
        d=h2f(np.array([int(b[0])|(int(b[1])<<8)]))[0]; mn=h2f(np.array([int(b[2])|(int(b[3])<<8)]))[0]
        sc=b[4:20]; qs=b[16:144]
        for j in range(4):
            s0,m0=get_sc_min(2*j,sc); s1,m1=get_sc_min(2*j+1,sc)
            base=blk*256+j*64
            out[base:base+32]=d*s0*(qs[j*32:(j+1)*32]&0xf)-mn*m0
            out[base+32:base+64]=d*s1*(qs[j*32:(j+1)*32]>>4)-mn*m1
    return out

def getrow(name, row):
    t,dims,off=T[name]; raw=load_raw(name)[2]
    if t==12: return dequant_q4k_row(raw,row,dims[0])
    if t==14: return dequant_q6k(raw,dims[0],1)[0]
    if t==0: return raw.view(np.float32)[row]
    raise Exception((name,t))

def getexpert(name, e):
    t,dims,off=T[name]; per=int(dims[1]); n_in=int(dims[0])
    be=256; bb=144 if t==12 else 210
    rs_=n_in//be*bb; eb=per*rs_
    data=np.fromfile(PATH,dtype=np.uint8,count=eb,offset=DB+off+e*eb)
    return (dequant_q4k if t==12 else dequant_q6k)(data,n_in,per)

def getw(name):
    t,dims,off=T[name]
    raw=load_raw(name)[2]
    if t in (12,14):
        dq = dequant_q4k if t==12 else dequant_q6k
        if len(dims)==2: return dq(raw,dims[0],dims[1])
        if len(dims)==3:
            per = int(dims[1]); ne=int(dims[2])
            arr = dq(raw, dims[0], per*ne)
            return arr.reshape(ne, per, dims[0])
    if t==0:
        if len(dims)==2: return raw.view(np.float32).reshape((dims[1],dims[0]))  # [out,in]
        return raw.view(np.float32).reshape(dims)
    raise Exception((name,t))

def rms(x,w,eps=1e-6): return x/np.sqrt(np.mean(x*x)+eps)*w
def silu(v): return v/(1+np.exp(-v))
def sigmoid(v): return 1/(1+np.exp(-v))

L=0
tok=int(sys.argv[1]) if len(sys.argv)>1 else 3710
N=2048; HK=16; HV=32; DS=128; DC=4; C=8192
# embed (single row only)
x=getrow("token_embd.weight", tok).copy()
# attn norm
h=rms(x,getw(f"blk.{L}.attn_norm.weight"))
np.save('/tmp/opencode/ref_h_norm1.npy',h)
# GDN projections
qkv=getw(f"blk.{L}.attn_qkv.weight")@h
z=getw(f"blk.{L}.attn_gate.weight")@h
alpha=getw(f"blk.{L}.ssm_alpha.weight")@h
beta=getw(f"blk.{L}.ssm_beta.weight")@h
# conv: taps [4,8192] flattened as tap*? -> w[c*4+k]
taps=getw(f"blk.{L}.ssm_conv1d.weight")
state=np.zeros((DC-1,C),dtype=np.float32)
conv_out=np.zeros(C,dtype=np.float32)
for c in range(C):
    acc=sum(state[k,c]*taps[c,k] for k in range(DC-1))+qkv[c]*taps[c,DC-1]
    for k in range(DC-2): state[k,c]=state[k+1,c]
    state[DC-2,c]=qkv[c]
    conv_out[c]=silu(np.float32(acc))
np.save('/tmp/opencode/ref_qkv.npy',qkv); np.save('/tmp/opencode/ref_conv_out.npy',conv_out); np.save('/tmp/opencode/ref_alpha.npy',alpha); np.save('/tmp/opencode/ref_beta.npy',beta)
# gdn step
dt=getw(f"blk.{L}.ssm_dt.bias"); sa=getw(f"blk.{L}.ssm_a")
S=np.zeros((HV,DS,DS),dtype=np.float32)
gdn_out=np.zeros((HV,DS),dtype=np.float32)
for hh in range(HV):
    kh=hh%HK; i=np.arange(DS)
    qi=conv_out[kh*DS:kh*DS+DS]; ki=conv_out[HK*DS+kh*DS:HK*DS+kh*DS+DS]
    vi=conv_out[2*HK*DS+hh*DS:2*HK*DS+hh*DS+DS]
    q=qi/np.sqrt(np.sum(qi*qi)+1e-6)/np.sqrt(DS); k=ki/np.sqrt(np.sum(ki*ki)+1e-6)
    g=np.exp(np.log1p(np.exp(alpha[hh]+dt[hh]))*sa[hh]); b=sigmoid(beta[hh])
    Si=S[hh]*g
    sk=Si@k
    d=(vi-sk)*b
    S[hh]=Si + np.outer(d,k)
    gdn_out[hh]=S[hh]@q
np.save('/tmp/opencode/ref_gdn_out.npy',gdn_out)
# gated norm
nw=getw(f"blk.{L}.ssm_norm.weight")
o=np.zeros_like(gdn_out)
for hh in range(HV):
    v=gdn_out[hh]; o[hh]=v/np.sqrt(np.mean(v*v)+1e-6)*nw*silu(z[hh*DS:(hh+1)*DS])
np.save('/tmp/opencode/ref_gdn_normed.npy',o); np.save('/tmp/opencode/ref_z.npy',z)
y=getw(f"blk.{L}.ssm_out.weight")@o.reshape(-1)
np.save('/tmp/opencode/ref_y_mixer.npy',y)
x=x+y
np.save('/tmp/opencode/ref_x_after_add.npy',x)
# post attn norm
h2=rms(x,getw(f"blk.{L}.post_attention_norm.weight"))
np.save('/tmp/opencode/ref_h_norm2.npy',h2)
# router
r=getw(f"blk.{L}.ffn_gate_inp.weight")@h2
p=np.exp(r-r.max()); p/=p.sum()
ids=np.argsort(-p)[:8]
w=p[ids]; w/=w.sum()
# experts
parts=np.zeros((8,N),dtype=np.float32)
for s,e in enumerate(ids):
    g=getexpert(f"blk.{L}.ffn_gate_exps.weight",e)@h2
    u=getexpert(f"blk.{L}.ffn_up_exps.weight",e)@h2
    parts[s]=getexpert(f"blk.{L}.ffn_down_exps.weight",e)@(silu(g)*u)
# shared
sg=getw(f"blk.{L}.ffn_gate_shexp.weight")@h2
su=getw(f"blk.{L}.ffn_up_shexp.weight")@h2
sy=getw(f"blk.{L}.ffn_down_shexp.weight")@(silu(sg)*su)
gate=sigmoid(getw(f"blk.{L}.ffn_gate_inp_shexp.weight")@h2)
x=x+(w[:,None]*parts).sum(0)+gate*sy
np.save('/tmp/opencode/ref_layer0.npy',x)
print('ids',ids, 'x[:8]',x[:8])
