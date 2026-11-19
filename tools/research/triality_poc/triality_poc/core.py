"""PyTorch reference implementation of the supplied v3 research design.

Not a Prism/TheTom/TurboQuant binary-compatible codec and not a fused CUDA kernel.
Tensor layout: [batch, head, token, channel]. Encoding packs integer codes into
bytes, but reference attention explicitly decodes them. The same operations run
on CPU or CUDA; a requested CUDA device never falls back to CPU.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Sequence
import torch
from torch import Tensor, nn


def resolve_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA explicitly requested, but this PyTorch/environment has no usable CUDA device.')
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('This PoC qualifies only CPU and CUDA.')
    return device


def _conj(x: Tensor) -> Tensor:
    return torch.cat((x[:1], -x[1:]))


def _cd(x: Tensor, y: Tensor) -> Tensor:
    """Cayley–Dickson convention inherited from the supplied v3 reference."""
    if x.numel() == 1:
        return x*y
    h = x.numel()//2
    a,b,cc,d = x[:h],x[h:],y[:h],y[h:]
    return torch.cat((_cd(a,cc)-_cd(_conj(d),b), _cd(d,a)+_cd(b,_conj(cc))))


@lru_cache(maxsize=1)
def _algebra_cached() -> tuple[Tensor, Tensor, Tensor]:
    eye=torch.eye(8,dtype=torch.float64)
    left=torch.stack([torch.stack([_cd(eye[i],eye[j]) for j in range(8)],dim=1) for i in range(8)])
    z=torch.zeros_like(eye)
    gamma=torch.stack([torch.cat((torch.cat((z,l),1),torch.cat((l.T,z),1)),0) for l in left])
    omega=torch.eye(16,dtype=torch.float64)
    for g in gamma:
        omega=omega@g
    plus=torch.nonzero(omega.diag()==1).flatten()
    minus=torch.nonzero(omega.diag()==-1).flatten()
    return gamma,omega,gamma[:,plus[:,None],minus]


def algebra() -> tuple[Tensor,Tensor,Tensor]:
    """Return independent CPU/FP64 copies: positive-Clifford generators, chirality, L."""
    return tuple(x.clone() for x in _algebra_cached())


def spin_views(planes: Sequence[tuple[int,int,float]], *, device: str|torch.device='cpu',
               dtype: torch.dtype=torch.float64) -> tuple[Tensor,Tensor,Tensor]:
    """Same Spin lift; right multiplication, so the last listed factor acts first."""
    gamma,omega,_=algebra()
    rv=torch.eye(8,dtype=torch.float64); S=torch.eye(16,dtype=torch.float64)
    for i,j,theta in planes:
        if not (0<=i<j<8 and math.isfinite(theta)):
            raise ValueError('Finite angle and 0 <= i < j < 8 required.')
        R=torch.eye(8,dtype=rv.dtype)
        R[i,i]=R[j,j]=math.cos(theta)
        R[i,j]=math.sin(theta);R[j,i]=-math.sin(theta)
        rotor=math.cos(theta/2)*torch.eye(16,dtype=S.dtype)+math.sin(theta/2)*(gamma[i]@gamma[j])
        rv=rv@R;S=S@rotor
    plus=torch.nonzero(omega.diag()==1).flatten();minus=torch.nonzero(omega.diag()==-1).flatten()
    return tuple(x.to(device=device,dtype=dtype) for x in (rv,S[plus[:,None],plus],S[minus[:,None],minus]))


def query_basis(*, device: str|torch.device='cpu', dtype: torch.dtype=torch.float64) -> Tensor:
    """[I,J1,...,J7], with C(c)[i,a] = L[i,a,b] c[b], c0=e0."""
    L=_algebra_cached()[2]
    C0=L[:,:,0]
    basis=torch.stack([L[:,:,b]@C0.T for b in range(8)])
    return basis.to(device=device,dtype=dtype)


def unit_control(a: Tensor, theta_max: float=math.pi/8, *, validate: bool=True) -> Tensor:
    """Smooth bounded chart R7 -> S7; nonzero rank-seven derivative at zero.

    Low precision accumulation uses FP32, then returns the input dtype. Direct
    entry validates finite values; QueryAdapter avoids per-token GPU synchronisation.
    """
    if a.shape[-1]!=7 or not math.isfinite(theta_max) or not 0<theta_max<math.pi/2:
        raise ValueError('Last dimension 7 and 0 < theta_max < pi/2 required.')
    if validate and not torch.isfinite(a).all():
        raise ValueError('Non-finite controller input.')
    x=a.float() if a.dtype in (torch.float16,torch.bfloat16) else a
    tau=math.tan(theta_max)
    scale=x.abs().amax(-1,keepdim=True).clamp_min(tau)
    z=x/scale
    u=tau*z/torch.sqrt((tau/scale).square()+z.square().sum(-1,keepdim=True))
    co=torch.cat((torch.ones_like(u[...,:1]),u),-1)
    return (co/torch.sqrt(co.square().sum(-1,keepdim=True))).to(a.dtype)


def apply_control(q: Tensor, co: Tensor, basis: Tensor|None=None) -> Tensor:
    """Apply one controller per head/token to all 8D blocks; tail stays unchanged.

    No per-token 8x8 matrix is constructed. This is a PyTorch contraction, not a
    custom fused kernel. q: [...,D]; co: [...,8].
    """
    if q.shape[:-1]!=co.shape[:-1] or co.shape[-1]!=8 or q.device!=co.device or q.dtype!=co.dtype:
        raise ValueError('q/co leading shapes, dtype and device must agree.')
    d=q.shape[-1];n=d//8*8
    if n==0:
        return q
    b=query_basis(device=q.device,dtype=q.dtype) if basis is None else basis
    blocks=q[...,:n].reshape(*q.shape[:-1],n//8,8)
    # A(c)=sum_b c_b basis_b, applied without materialising A per token.
    out=torch.einsum('cij,...bj,...c->...bi',b,blocks,co)
    out=out.reshape(*q.shape[:-1],n)
    return torch.cat((out,q[...,n:]),-1) if n<d else out


class QueryAdapter(nn.Module):
    """Rank-eight bottleneck, seven bounded controls per Q head. Zero Wout only."""
    def __init__(self, heads:int, dim:int, rank:int=8, theta_max:float=math.pi/8) -> None:
        super().__init__()
        if min(heads,dim,rank)<=0:
            raise ValueError('Positive heads/dim/rank required.')
        self.heads,self.dim,self.theta_max=heads,dim,theta_max
        self.w_in=nn.Parameter(torch.randn(heads,rank,dim)/math.sqrt(dim))
        self.b_in=nn.Parameter(torch.zeros(heads,rank))
        self.w_out=nn.Parameter(torch.zeros(heads,7,rank))
        self.b_out=nn.Parameter(torch.zeros(heads,7))
        self.register_buffer('basis',query_basis(dtype=torch.float32),persistent=True)

    def controls(self,q:Tensor) -> Tensor:
        if q.ndim!=4 or q.shape[1]!=self.heads or q.shape[-1]!=self.dim:
            raise ValueError('Expected q [B,H,T,D] matching adapter.')
        h=torch.tanh(torch.einsum('bhtd,hrd->bhtr',q,self.w_in)+self.b_in[None,:,None,:])
        a=torch.einsum('bhtr,har->bhta',h,self.w_out)+self.b_out[None,:,None,:]
        return unit_control(a,self.theta_max,validate=False)

    def forward(self,q:Tensor) -> Tensor:
        return apply_control(q,self.controls(q),self.basis)


def _hadamard(n:int) -> Tensor:
    h=torch.ones(1,1,dtype=torch.float64)
    while h.shape[0]<n:
        h=torch.cat((torch.cat((h,h),1),torch.cat((h,-h),1)),0)
    return h/math.sqrt(n)


def make_views(width:int, seed:int=0, *, kind:str='triality', dtype:torch.dtype=torch.float32,
               device:str|torch.device='cpu') -> Tensor:
    """Three U=B F matrices; generic control uses three independent SO(8) blocks.

    F is head-wide normalised Hadamard for a power-of-two width; otherwise block
    Hadamard with an identity tail. Model-owned transforms are not represented here.
    """
    if width<=0:
        raise ValueError('width must be positive')
    gen=torch.Generator().manual_seed(seed)
    if kind=='identity':
        return torch.eye(width,dtype=dtype,device=device)[None]
    if width & (width-1)==0:
        F=_hadamard(width)
    else:
        F=torch.block_diag(*([_hadamard(8)]*(width//8)+([torch.eye(width%8,dtype=torch.float64)] if width%8 else [])))
    signs=(torch.randint(0,2,(width,),generator=gen)*2-1).double()
    F=F*signs[None,:]
    if kind=='hadamard':
        return F.to(device=device,dtype=dtype)[None]
    if kind=='triality':
        angles=torch.randn(28,generator=gen,dtype=torch.float64)*.35
        planes=[(i,j,float(angles[n])) for n,(i,j) in enumerate(( (i,j) for i in range(8) for j in range(i+1,8) ))]
        blocks=spin_views(planes)
    elif kind=='generic':
        blocks=[]
        for _ in range(3):
            R,_=torch.linalg.qr(torch.randn(8,8,generator=gen,dtype=torch.float64))
            if torch.linalg.det(R)<0:
                R[:,0]*=-1
            blocks.append(R)
    else:
        raise ValueError('Unknown view family.')
    matrices=[]
    for R in blocks:
        B=torch.block_diag(*([R]*(width//8)+([torch.eye(width%8,dtype=R.dtype)] if width%8 else [])))
        matrices.append(B@F)
    return torch.stack(matrices).to(device=device,dtype=dtype)


def pack_unsigned(x:Tensor,bits:int) -> Tensor:
    """Little-endian bit stream per final dimension; zero padded last byte."""
    if bits not in range(2,9) or x.dtype not in (torch.int64,torch.int32,torch.int16,torch.uint8):
        raise ValueError('Unsigned integer codes and 2..8 bits required.')
    if torch.any(x<0) or torch.any(x>=(1<<bits)):
        raise ValueError('Code outside bit range.')
    offsets=torch.arange(bits,device=x.device)
    bitstream=((x.to(torch.int64).unsqueeze(-1)>>offsets)&1).flatten(-2)
    padding=(-bitstream.shape[-1])%8
    if padding:
        bitstream=torch.nn.functional.pad(bitstream,(0,padding))
    sh=bitstream.shape
    return (bitstream.reshape(*sh[:-1],-1,8)*(1<<torch.arange(8,device=x.device))).sum(-1).to(torch.uint8)


def unpack_unsigned(p:Tensor,bits:int,count:int) -> Tensor:
    if bits not in range(2,9) or p.dtype!=torch.uint8 or count<0 or p.shape[-1]*8<count*bits:
        raise ValueError('Invalid packed representation.')
    stream=((p.long().unsqueeze(-1)>>torch.arange(8,device=p.device))&1).flatten(-2)[...,:count*bits]
    return (stream.reshape(*stream.shape[:-1],count,bits)*(1<<torch.arange(bits,device=p.device))).sum(-1)


def _pack_signs(signs:Tensor) -> Tensor:
    bit=signs.long();pad=(-bit.shape[-1])%8
    if pad:
        bit=torch.nn.functional.pad(bit,(0,pad))
    return (bit.reshape(*bit.shape[:-1],-1,8)*(1<<torch.arange(8,device=bit.device))).sum(-1).to(torch.uint8)


def _unpack_signs(p:Tensor,m:int,dtype:torch.dtype) -> Tensor:
    bit=((p.long().unsqueeze(-1)>>torch.arange(8,device=p.device))&1).flatten(-2)[...,:m]
    return bit.to(dtype)*2-1


@dataclass
class Quantized:
    codes:Tensor
    scales:Tensor
    bits:int
    width:int
    group:int
    ste:Tensor|None=None

    @property
    def storage_bytes(self) -> int:
        return sum(t.numel()*t.element_size() for t in (self.codes,self.scales))


def quantize(x:Tensor,bits:int,group:int=8,*,ste:bool=False) -> Quantized:
    """Symmetric absmax REFERENCE codec (2^bits-1 levels), NOT a TurboQuant reproduction."""
    if bits not in range(2,9) or group<=0 or x.shape[-1]%group:
        raise ValueError('2..8 bits, group positive and dividing head width required.')
    value=x.detach();qmax=(1<<(bits-1))-1
    chunks=value.reshape(*value.shape[:-1],-1,group)
    scales=chunks.abs().amax(-1,keepdim=True)/qmax
    safe=torch.where(scales>0,scales,torch.ones_like(scales))
    signed=torch.round(chunks/safe).clamp(-qmax,qmax).long()
    codes=pack_unsigned((signed+qmax).flatten(-2),bits)
    encoded=Quantized(codes,scales.squeeze(-1),bits,x.shape[-1],group)
    if ste:
        dq=dequantize(encoded)
        encoded.ste=x+(dq-x).detach()
    return encoded


def dequantize(encoded:Quantized|Tensor) -> Tensor:
    if isinstance(encoded,Tensor):
        return encoded
    if encoded.ste is not None:
        return encoded.ste
    qmax=(1<<(encoded.bits-1))-1
    codes=unpack_unsigned(encoded.codes,encoded.bits,encoded.width)-qmax
    shaped=codes.reshape(*codes.shape[:-1],-1,encoded.group).to(encoded.scales.dtype)
    return (shaped*encoded.scales.unsqueeze(-1)).flatten(-2)


@dataclass
class Cache:
    keys:list[Quantized|Tensor]
    value:Quantized|Tensor
    views:Tensor
    weights:Tensor
    beta:float
    G:Tensor|None=None
    signs:Tensor|None=None
    norms:Tensor|None=None

    def reconstructed_key(self) -> Tensor:
        return sum(self.weights[r]*(dequantize(k)@self.views[r]) for r,k in enumerate(self.keys))

    @property
    def payload_bytes(self) -> int:
        def size(x):
            return x.storage_bytes if isinstance(x,Quantized) else x.numel()*x.element_size()
        total=sum(size(x) for x in self.keys)+size(self.value)
        for x in (self.signs,self.norms):
            if x is not None:
                total+=x.numel()*x.element_size()
        return total

    @property
    def profile_bytes(self) -> int:
        return sum(x.numel()*x.element_size() for x in (self.views,self.weights,self.G) if x is not None)


def build_cache(k:Tensor,v:Tensor,views:Tensor,k_bits:int|None,v_bits:int|None,*,
                beta:float=0.,m:int=32,seed:int=123,weights:Tensor|None=None,group:int=8,
                ste:bool=False) -> Cache:
    if k.ndim!=4 or k.shape!=v.shape or k.device!=v.device or k.dtype!=v.dtype:
        raise ValueError('K/V must be matching [B,Hkv,S,D] tensors.')
    if views.ndim!=3 or tuple(views.shape[-2:])!=(k.shape[-1],k.shape[-1]) or views.device!=k.device or views.dtype!=k.dtype:
        raise ValueError('Views must match device, dtype and head dimension.')
    if not math.isfinite(beta) or not 0<=beta<=1:
        raise ValueError('beta must be finite in [0,1].')
    w=torch.full((len(views),),1/len(views),dtype=k.dtype,device=k.device) if weights is None else weights
    if w.shape!=(len(views),) or w.device!=k.device or w.dtype!=k.dtype or not torch.isfinite(w).all() or (w<0).any() or not torch.isclose(w.sum(),w.new_tensor(1.)):
        raise ValueError('Weights must be finite nonnegative, sum one, and match device/dtype.')
    keys=[]
    for U in views:
        rotated=k@U.T
        keys.append(rotated if k_bits is None else quantize(rotated,k_bits,group,ste=ste))
    value=v if v_bits is None else quantize(v,v_bits,group,ste=ste)
    cache=Cache(keys,value,views,w,beta)
    if beta>0:
        if m<=0:
            raise ValueError('Positive QJL dimension required.')
        # Sampling on CPU defines the same profile across CPU/CUDA, then transfer once.
        G=torch.randn(m,k.shape[-1],generator=torch.Generator().manual_seed(seed),dtype=torch.float64).to(k)
        residual=(k-cache.reconstructed_key()).detach()
        cache.G=G
        cache.signs=_pack_signs(residual@G.T>=0)
        cache.norms=torch.linalg.vector_norm(residual,dim=-1)
    return cache


def cache_scores(q:Tensor,cache:Cache,start:int=0,end:int|None=None) -> Tensor:
    """Three raw view scores + optional shared correction, scale applied exactly once."""
    k0=dequantize(cache.keys[0]);hq=q.shape[1];hkv=k0.shape[1]
    if q.ndim!=4 or hq%hkv or q.shape[0]!=k0.shape[0] or q.shape[-1]!=k0.shape[-1]:
        raise ValueError('Invalid query/GQA shape.')
    if q.device!=k0.device or q.dtype!=k0.dtype:
        raise ValueError('Query/cache device or dtype mismatch.')
    stop=k0.shape[-2] if end is None else end
    if not 0<=start<=stop<=k0.shape[-2]:
        raise ValueError('Invalid key range.')
    scores=None
    for r,key in enumerate(cache.keys):
        kk=dequantize(key)[:,:,start:stop].repeat_interleave(hq//hkv,dim=1)
        sc=(q@cache.views[r].T)@kk.transpose(-2,-1)
        scores=cache.weights[r]*sc if scores is None else scores+cache.weights[r]*sc
    if cache.beta>0:
        assert cache.G is not None and cache.signs is not None and cache.norms is not None
        m=cache.G.shape[0]
        signs=_unpack_signs(cache.signs[:,:,start:stop],m,q.dtype).repeat_interleave(hq//hkv,dim=1)
        norms=cache.norms[:,:,start:stop].repeat_interleave(hq//hkv,dim=1).unsqueeze(-2)
        correction=math.sqrt(math.pi/2)/m*((q@cache.G.T)@signs.transpose(-2,-1))*norms
        scores=scores+cache.beta*correction
    return scores/math.sqrt(q.shape[-1])


def _probabilities(scores:Tensor,mask:Tensor|None) -> Tensor:
    if mask is None:
        return scores.softmax(-1)
    if mask.dtype!=torch.bool or mask.device!=scores.device:
        raise ValueError('Boolean mask on the same device required; True means allowed.')
    allowed=mask.any(-1,keepdim=True)
    masked=scores.masked_fill(~mask,-torch.inf)
    safe=torch.where(allowed,masked,torch.zeros_like(masked))
    return torch.where(allowed,safe.softmax(-1),torch.zeros_like(safe))


def attention(q:Tensor,cache:Cache,mask:Tensor|None=None) -> Tensor:
    p=_probabilities(cache_scores(q,cache),mask)
    v=dequantize(cache.value).repeat_interleave(q.shape[1]//dequantize(cache.value).shape[1],dim=1)
    return p@v


def tiled_attention(q:Tensor,cache:Cache,mask:Tensor|None=None,*,q_tile:int=32,k_tile:int=64) -> Tensor:
    """Online softmax reference. Bounded score tiles, NOT FlashAttention backward.

    Autograd parity is tested on tiny shapes, but training retains graph intermediates.
    Large-shape memory-efficiency claims apply only under no_grad/inference_mode.
    """
    if q_tile<=0 or k_tile<=0:
        raise ValueError('Positive tile sizes required.')
    v=dequantize(cache.value).repeat_interleave(q.shape[1]//dequantize(cache.value).shape[1],1)
    S=v.shape[-2];blocks=[]
    for qi in range(0,q.shape[-2],q_tile):
        qq=q[:,:,qi:qi+q_tile];shape=qq.shape[:-1]+(1,)
        max_s=torch.full(shape,-torch.inf,device=q.device,dtype=q.dtype)
        denom=torch.zeros(shape,device=q.device,dtype=q.dtype)
        numer=torch.zeros_like(qq)
        for ki in range(0,S,k_tile):
            stop=min(S,ki+k_tile)
            s=cache_scores(qq,cache,ki,stop)
            if mask is not None:
                s=s.masked_fill(~mask[...,qi:qi+q_tile,ki:stop],-torch.inf)
            local=s.amax(-1,keepdim=True)
            newmax=torch.maximum(max_s,local)
            # Finite substitutes prevent -inf-(-inf) and NaN gradients on all-masked tiles.
            finite_new=torch.where(torch.isfinite(newmax),newmax,torch.zeros_like(newmax))
            old_delta=torch.where(torch.isfinite(max_s),max_s-finite_new,torch.zeros_like(max_s))
            old_scale=torch.where(torch.isfinite(max_s),old_delta.exp(),torch.zeros_like(max_s))
            shifted=torch.where(torch.isfinite(s),s-finite_new,torch.zeros_like(s))
            exp_s=torch.where(torch.isfinite(s),shifted.exp(),torch.zeros_like(s))
            numer=old_scale*numer+exp_s@v[:,:,ki:stop]
            denom=old_scale*denom+exp_s.sum(-1,keepdim=True)
            max_s=newmax
        blocks.append(numer/denom.clamp_min(torch.finfo(q.dtype).tiny))
    return torch.cat(blocks,dim=-2)


def fisher_energy(p:Tensor,e:Tensor) -> Tensor:
    """e^T (diag(p)-pp^T)e, computed by weighted centering for stability."""
    mean=(p*e).sum(-1,keepdim=True)
    return (p*(e-mean).square()).sum(-1)
