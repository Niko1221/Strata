"""Numerical validation of deductions; independent of real-model claims."""
from __future__ import annotations
import argparse,json,math,platform
from pathlib import Path
import torch
from .core import algebra,query_basis,spin_views,unit_control,apply_control,QueryAdapter,make_views,build_cache,cache_scores,attention,tiled_attention,fisher_energy,resolve_device


def run(device:torch.device) -> dict:
    torch.manual_seed(20261023);torch.set_num_threads(1)
    dtype=torch.float64
    basis=query_basis(device=device,dtype=dtype);J=basis[1:]
    a=torch.randn(512,7,device=device,dtype=dtype)
    co=unit_control(a);A=torch.einsum('ni,ijk->njk',co,basis)
    ident=torch.eye(8,device=device,dtype=dtype)
    q=torch.randn(512,8,device=device,dtype=dtype);y=apply_control(q,co,basis)
    tangent=torch.einsum('aij,nj->nai',J,q)
    lie=torch.stack(list(J)+[J[i]@J[j] for i in range(7) for j in range(i+1,7)])
    metrics={'orthogonality_max':float((A.transpose(-1,-2)@A-ident).abs().max()),
             'determinant_error_max':float((torch.linalg.det(A)-1).abs().max()),
             'norm_error_max':float((y.square().sum(-1)-q.square().sum(-1)).abs().max()),
             'tangent_gram_error_max':float((tangent@tangent.transpose(-1,-2)-q.square().sum(-1)[:,None,None]*torch.eye(7,device=device,dtype=dtype)).abs().max()),
             'lie_algebra_rank':int(torch.linalg.matrix_rank(lie.flatten(1))),
             'maximum_angle_radians':float(torch.acos(co[:,0]).max())}
    z=torch.zeros(2,7,device=device,dtype=dtype,requires_grad=True)
    metrics['zero_gradcheck']=bool(torch.autograd.gradcheck(lambda a:apply_control(q[:2],unit_control(a,validate=False),basis),(z,),atol=1e-6))
    # Conditional QJL moments, with q/e fixed before independently sampled Gaussian G.
    q0=torch.randn(16,device=device,dtype=dtype);e=torch.randn(16,device=device,dtype=dtype)
    m=32;samples=20000;chunks=[]
    for _ in range(samples//250):
        G=torch.randn(250,m,16,device=device,dtype=dtype)
        signs=torch.where(G@e>=0,1.,-1.).to(dtype)
        chunks.append(math.sqrt(math.pi/2)*e.norm()*(signs*(G@q0)).mean(-1))
    Z=torch.cat(chunks);mean=float(q0@e)
    theory=float((math.pi/2*e.square().sum()*q0.square().sum()-(q0@e).square())/m)
    empirical=float(Z.var(unbiased=True));se=math.sqrt(theory/samples)
    metrics['qjl']={'samples':samples,'m':m,'conditional_true_inner_product':mean,
                    'mean_estimate':float(Z.mean()),'mean_error_in_standard_errors':float((Z.mean()-mean)/se),
                    'theory_variance':theory,'empirical_variance':empirical,'variance_ratio':empirical/theory}
    # Attention-Fisher shrinkage with correlated shared-G noise. Calibration/evaluation G are disjoint.
    keys=9;E=torch.randn(keys,16,device=device,dtype=dtype)*.15
    raw=torch.randn(keys,device=device,dtype=dtype);p=raw.softmax(-1)
    missing=E@q0/math.sqrt(16);zs=[]
    for _ in range(40):
        G=torch.randn(250,m,16,device=device,dtype=dtype)
        signs=torch.where(torch.einsum('nmd,kd->nkm',G,E)>=0,1.,-1.).to(dtype)
        val=math.sqrt(math.pi/2)*E.norm(dim=-1)[None,:]*torch.einsum('nkm,nm->nk',signs,G@q0)/m/math.sqrt(16)
        zs.append(val)
    estimates=torch.cat(zs);cal,test=estimates[:5000],estimates[5000:]
    D=float(fisher_energy(p,missing));noise=cal-missing
    N=float(fisher_energy(p,noise).mean());beta=D/(D+N) if D+N else 0.
    arm={}
    for b in (0.,beta,1.):
        error=-(1-b)*missing+b*(test-missing)
        lp=(raw+error).log_softmax(-1)
        kl=(p*(raw.log_softmax(-1)-lp)).sum(-1).mean()
        arm[str(round(b,6))]={'beta':b,'mean_kl':float(kl),'fisher_approx':float(fisher_energy(p,error).mean()/2)}
    metrics['fisher_shrinkage']={'oracle_calibration_beta':beta,'D':D,'N':N,'arms':arm,
                               'limitation':'fixed q/E; oracle residual; not deployable calibration or autoregressive evidence'}
    # Local controllable subspace: regularised Fisher least squares never worsens its own quadratic optimum.
    qv=torch.randn(8,device=device,dtype=dtype);K=torch.randn(12,8,device=device,dtype=dtype)
    Fm=K@torch.einsum('aij,j->ia',J,qv)/math.sqrt(8)
    p0=(K@qv/math.sqrt(8)).softmax(-1)
    err=torch.randn(12,device=device,dtype=dtype)*.01
    H=torch.diag(p0)-p0[:,None]*p0[None,:]
    eta=torch.linalg.solve(Fm.T@H@Fm+1e-6*torch.eye(7,device=device,dtype=dtype),-Fm.T@H@err)
    metrics['fisher_projection']={'before':float(err@H@err),'after':float((err+Fm@eta)@H@(err+Fm@eta)),
                                  'control_rank':int(torch.linalg.matrix_rank(Fm)),'eta_norm':float(eta.norm())}
    # Exact KL identity / Hoeffding bound on a finite logit displacement.
    s=torch.randn(17,device=device,dtype=dtype);delta=torch.randn_like(s)*.1
    pp=s.softmax(-1)
    kl=(pp*(s.log_softmax(-1)-(s+delta).log_softmax(-1))).sum()
    bound=(delta.max()-delta.min()).square()/8
    metrics['logit_kl_bound']={'kl':float(kl),'range_squared_over_8':float(bound),'satisfied':bool(kl<=bound+1e-14)}
    # Residual-network error propagation: observed downstream response vs Jacobian first order.
    mats=[torch.randn(8,8,device=device,dtype=dtype)*.04 for _ in range(6)]
    def suffix(x):
        for W in mats:x=x+torch.tanh(W@x)
        return x
    x=torch.randn(8,device=device,dtype=dtype,requires_grad=True);d=torch.randn_like(x)*1e-5
    jac=torch.autograd.functional.jacobian(suffix,x)
    real=suffix(x+d)-suffix(x);pred=jac@d
    prod=math.prod(1+float(torch.linalg.matrix_norm(W,2)) for W in mats)
    metrics['network_propagation']={'first_order_relative_error':float(((real-pred).norm()/real.norm()).detach()),
                                    'observed_amplification':float((real.norm()/d.norm()).detach()),'product_lipschitz_bound':prod,
                                    'bound_satisfied':bool(real.norm()<=prod*d.norm()+1e-12)}
    # Actual reference payloads on tiny model shape; no false bit-vs-byte claim.
    k=torch.randn(1,2,32,16,device=device,dtype=torch.float32);v=torch.randn_like(k)
    payload={}
    for label,family,bits,vbits in [('teacher_exact','identity',None,None),('single2','hadamard',2,None),
                                  ('single6','hadamard',6,None),('triple2','triality',2,None),('triple2_v3','triality',2,3)]:
        ca=build_cache(k,v,make_views(16,600,kind=family,device=device),bits,vbits)
        payload[label]={'per_layer_per_sequence_payload_bytes':ca.payload_bytes,'shared_profile_bytes':ca.profile_bytes}
    metrics['reference_payloads']=payload
    checks={'orthogonal':metrics['orthogonality_max']<1e-12,'positive_det':metrics['determinant_error_max']<1e-12,
            'norm':metrics['norm_error_max']<1e-11,'tangent':metrics['tangent_gram_error_max']<1e-11,
            'lie_rank28':metrics['lie_algebra_rank']==28,'gradcheck':metrics['zero_gradcheck'],
            'angle_bound':metrics['maximum_angle_radians']<math.pi/8,
            'qjl_mean_mc':abs(metrics['qjl']['mean_error_in_standard_errors'])<5,
            'qjl_variance_mc':.94<metrics['qjl']['variance_ratio']<1.06,
            'fisher_projection':metrics['fisher_projection']['after']<=metrics['fisher_projection']['before']+1e-14,
            'logit_bound':metrics['logit_kl_bound']['satisfied'],'residual_bound':metrics['network_propagation']['bound_satisfied']}
    return {'scope':'new synthetic/analytic research probes; no real LLM inference',
            'environment':{'python':platform.python_version(),'torch':torch.__version__,'device':str(device),
                           'cuda_build':torch.version.cuda,'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None},
            'checks':checks,'passed':all(checks.values()),'metrics':metrics}


def main()->int:
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda');p.add_argument('--out',default='results/validation.json');a=p.parse_args()
    device=resolve_device(a.device);r=run(device);path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(r,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps(r,indent=2))
    return 0 if r['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
