import math
import pytest
import torch
from triality_poc import core as c

torch.set_num_threads(1)

def rng(seed=7):
    torch.manual_seed(seed)


def test_clifford_and_chirality():
    g, omega, L = c.algebra()
    eye = torch.eye(16, dtype=torch.float64)
    for i in range(8):
        for j in range(8):
            assert torch.equal(g[i] @ g[j] + g[j] @ g[i], 2*eye if i == j else 0*eye)
    assert torch.equal(omega @ omega, eye)
    assert (omega.diag()==1).sum() == 8
    assert (omega.diag()==-1).sum() == 8
    assert L.shape == (8,8,8)


def test_same_lift_intertwiner_and_sign():
    _,_,L=c.algebra()
    rv,rp,rm=c.spin_views([(0,1,.7),(1,3,-.4),(2,7,.25)])
    for i in range(8):
        target=torch.einsum('i,iab->ab',rv[:,i],L)
        torch.testing.assert_close(rp@L[i]@rm.T,target,atol=2e-14,rtol=2e-14)
    rv,rp,rm=c.spin_views([(0,1,2*math.pi)])
    torch.testing.assert_close(rv,torch.eye(8,dtype=rv.dtype))
    torch.testing.assert_close(rp,-torch.eye(8,dtype=rv.dtype))
    torch.testing.assert_close(rm,-torch.eye(8,dtype=rv.dtype))
    for r in c.spin_views([(0,1,4*math.pi)]):
        torch.testing.assert_close(r,torch.eye(8,dtype=r.dtype))


def test_j_generators_tangent_and_lie_rank():
    J=c.query_basis()[1:]
    I=torch.eye(8,dtype=J.dtype)
    for a in range(7):
        assert torch.equal(J[a].T,-J[a])
        for b in range(7):
            assert torch.equal(J[a]@J[b]+J[b]@J[a],-2*I if a==b else 0*I)
    rng();q=torch.randn(8,dtype=torch.float64);t=torch.einsum('aij,j->ai',J,q)
    torch.testing.assert_close(t@q,torch.zeros(7,dtype=q.dtype),atol=1e-13,rtol=0)
    torch.testing.assert_close(t@t.T,(q@q)*torch.eye(7,dtype=q.dtype))
    basis=torch.stack(list(J)+[J[a]@J[b] for a in range(7) for b in range(a+1,7)])
    assert torch.linalg.matrix_rank(basis.flatten(1)).item()==28


def test_query_chart_identity_norm_bound_grad():
    rng();a=torch.randn(11,7,dtype=torch.float64)*3
    co=c.unit_control(a)
    torch.testing.assert_close(co.square().sum(-1),torch.ones(11,dtype=co.dtype))
    assert torch.acos(co[:,0]).max()<=math.pi/8+1e-14
    q=torch.randn(11,19,dtype=torch.float64)
    z=c.apply_control(q,co)
    torch.testing.assert_close(z.square().sum(-1),q.square().sum(-1))
    assert torch.equal(z[:,-3:],q[:,-3:])
    co0=c.unit_control(torch.zeros(11,7,dtype=q.dtype))
    assert torch.equal(c.apply_control(q,co0),q)
    x=torch.zeros(2,7,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda y:c.apply_control(q[:2],c.unit_control(y)),(x,),atol=1e-6)


def test_controller_identity_and_learning_signal():
    rng();ad=c.QueryAdapter(3,16,rank=8).double()
    q=torch.randn(2,3,5,16,dtype=torch.float64)
    assert torch.equal(ad(q),q)
    (ad(q)*torch.randn_like(q)).sum().backward()
    assert ad.w_out.grad.abs().sum()>0
    assert torch.count_nonzero(ad.w_in.grad)==0


@pytest.mark.parametrize('bits',[2,3,4,5,6,8])
def test_pack_roundtrip(bits):
    rng();x=torch.randint(0,1<<bits,(2,3,37),dtype=torch.int64)
    p=c.pack_unsigned(x,bits)
    assert p.dtype==torch.uint8 and p.shape[-1]==math.ceil(37*bits/8)
    assert torch.equal(c.unpack_unsigned(p,bits,37),x)


@pytest.mark.parametrize('bits',[2,3,4,6])
def test_reference_quantizer(bits):
    rng();x=torch.randn(2,3,11,16,dtype=torch.float64)
    p=c.quantize(x,bits,group=8)
    y=c.dequantize(p)
    assert y.shape==x.shape and y.dtype==x.dtype
    assert torch.isfinite(y).all()
    assert p.storage_bytes == sum(t.numel()*t.element_size() for t in (p.codes,p.scales))
    torch.testing.assert_close(c.dequantize(c.quantize(x*0,bits,8)),x*0)


def test_exact_views_preserve_scores():
    rng();q=torch.randn(2,6,5,16,dtype=torch.float64)
    k=torch.randn(2,2,7,16,dtype=torch.float64);v=torch.randn_like(k)
    cache=c.build_cache(k,v,c.make_views(16,3,dtype=q.dtype),None,None,beta=0)
    s=c.cache_scores(q,cache)
    target=q@k.repeat_interleave(3,dim=1).transpose(-2,-1)/4
    torch.testing.assert_close(s,target,atol=1e-12,rtol=1e-12)
    assert cache.signs is None and cache.G is None


def test_qjl_zero_and_formula():
    rng();q=torch.randn(2,4,3,16,dtype=torch.float64)
    k=torch.randn(2,2,7,16,dtype=q.dtype)
    U=c.make_views(16,5,dtype=q.dtype)
    z=c.build_cache(k,k,U,None,None,beta=1,m=32)
    s=c.cache_scores(q,z)
    t=q@k.repeat_interleave(2,dim=1).transpose(-2,-1)/4
    torch.testing.assert_close(s,t,atol=1e-12,rtol=1e-12)
    p=c.build_cache(k,k,U,2,None,beta=.3,m=32)
    residual=k-p.reconstructed_key()
    signs=torch.where(residual@p.G.T>=0,1.,-1.).to(q.dtype).repeat_interleave(2,dim=1)
    correction=math.sqrt(math.pi/2)/32*(q@p.G.T)@signs.transpose(-2,-1)
    correction=correction*torch.linalg.vector_norm(residual,dim=-1).repeat_interleave(2,dim=1).unsqueeze(-2)
    baseline=q@p.reconstructed_key().repeat_interleave(2,dim=1).transpose(-2,-1)
    torch.testing.assert_close(c.cache_scores(q,p),(baseline+.3*correction)/4)


@pytest.mark.parametrize('beta',[0.,.4])
def test_tiled_attention_mask_and_gradient(beta):
    rng();q=torch.randn(1,4,6,16,dtype=torch.float64,requires_grad=True)
    k=torch.randn(1,2,9,16,dtype=q.dtype);v=torch.randn_like(k)
    cache=c.build_cache(k,v,c.make_views(16,3,dtype=q.dtype),3,3,beta=beta,m=19)
    mask=torch.arange(9)[None,:] <= torch.arange(6)[:,None]+3
    mask[0,:]=False
    y=c.attention(q,cache,mask);z=c.tiled_attention(q,cache,mask,q_tile=2,k_tile=3)
    torch.testing.assert_close(y,z,rtol=1e-12,atol=1e-12)
    assert torch.equal(y[:,:,0],torch.zeros_like(y[:,:,0]))
    w=torch.randn_like(y)
    dy=torch.autograd.grad((y*w).sum(),q,retain_graph=True)[0]
    dz=torch.autograd.grad((z*w).sum(),q)[0]
    assert torch.isfinite(dz).all()
    torch.testing.assert_close(dy,dz,rtol=1e-10,atol=1e-10)


def test_fisher_kl_and_constant_shift():
    rng();s=torch.randn(9,dtype=torch.float64);e=torch.randn_like(s)*1e-4
    p=s.softmax(-1);kl=(p*(s.log_softmax(-1)-(s+e).log_softmax(-1))).sum()
    approx=c.fisher_energy(p,e)/2
    torch.testing.assert_close(kl,approx,atol=1e-11,rtol=1e-3)
    assert c.fisher_energy(p,torch.ones_like(s)*100).abs()<1e-10


def test_invalid_and_cuda_fail_closed():
    with pytest.raises(ValueError):c.pack_unsigned(torch.ones(3,dtype=torch.int64),1)
    with pytest.raises(ValueError):c.make_views(0,3)
    with pytest.raises(ValueError):c.unit_control(torch.ones(7)*float('nan'))
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError,match='CUDA'):c.resolve_device('cuda')


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA not available: no CPU substitution')
def test_cuda_reference_parity():
    rng();cpu=c.QueryAdapter(2,16).double();gpu=c.QueryAdapter(2,16).double().cuda()
    
    with torch.no_grad(): cpu.w_out.normal_(std=.1)
    gpu.load_state_dict(cpu.state_dict());q=torch.randn(2,2,5,16,dtype=torch.float64)
    torch.testing.assert_close(cpu(q),gpu(q.cuda()).cpu(),rtol=1e-11,atol=1e-11)
    U=c.make_views(16,3,dtype=q.dtype)
    k=torch.randn(2,1,9,16,dtype=q.dtype);v=torch.randn_like(k)
    ca=c.build_cache(k,v,U,3,4,beta=.25,m=32)
    cb=c.build_cache(k.cuda(),v.cuda(),U.cuda(),3,4,beta=.25,m=32)
    torch.testing.assert_close(c.attention(q,ca),c.attention(q.cuda(),cb).cpu(),rtol=2e-10,atol=2e-10)


def test_exact_key_roundtrip_and_noncontiguous():
    rng(); k=torch.randn(2,2,8,32,dtype=torch.float64)[...,::2];v=torch.randn_like(k)
    assert not k.is_contiguous()
    cache=c.build_cache(k,v,c.make_views(16,7,dtype=k.dtype),None,None)
    torch.testing.assert_close(cache.reconstructed_key(),k,rtol=1e-13,atol=1e-13)
    q=torch.randn(2,6,5,16,dtype=k.dtype)
    torch.testing.assert_close(c.cache_scores(q,cache),q@k.repeat_interleave(3,1).transpose(-2,-1)/4,rtol=1e-13,atol=1e-13)
