import sys, math
sys.path.insert(0,'.')
import numpy as np
from fastfidelity import paulis as pl
from fastfidelity import StabTableau, SpectrumBackend
from fastfidelity.local import local_candidate
n,N,p=4,1000,0.2
bell=np.zeros(4); bell[0]=bell[3]=1/math.sqrt(2)
psi=np.kron(bell,bell); rho=(1-p)*np.outer(psi,psi)+p*np.eye(16)/16
B=SpectrumBackend(rho); FST=1-p+p/16
def run(seed, verbose=False):
    rng=np.random.default_rng(seed)
    T=StabTableau.random(n,rng); f0=T.fidelity_from_spectrum(B.E)
    keys,signs,us=T.enumerate(); keys=[int(k) for k in keys]
    draws=rng.integers(0,1<<n,size=N); store={}; tot=0
    for u in range(1<<n):
        m=int((draws==u).sum()); k=keys[list(us).index(u)]; s=int(signs[list(us).index(u)])
        if m==0: continue
        if k==0: tot+=s*m; continue
        bits=np.asarray(B.measure(np.full(m,k,dtype=np.int64),rng),dtype=np.int64)
        x,z=pl.split(k,n); sup=x|z
        o=1-2*(pl.popcount(bits & sup)&1); tot+=s*int(o.sum())
        # all nonempty substrings (k=4 covers every weight for n=4)
        sub=sup
        while sub:
            Q=pl.join(x&sub,z&sub,n); oq=1-2*(pl.popcount(bits & sub)&1)
            mm,ss=store.get(Q,(0,0)); store[Q]=(mm+m,ss+int(oq.sum()))
            sub=(sub-1)&sup
    p0=tot/N
    ks=np.array(list(store),dtype=np.int64); cnt=np.array([store[k][0] for k in ks]); sm=np.array([store[k][1] for k in ks])
    cand=local_candidate(ks,cnt,sm,n)
    fc=cand.fidelity_from_spectrum(B.E) if cand is not None else None
    return T,f0,p0,store,cand,fc
for seed in range(1,400):
    T,f0,p0,store,cand,fc=run(seed)
    if cand is not None and abs(fc-FST)<1e-9 and f0<0.1 and p0<0.15:
        print("seed",seed); break
print("start generators:",[ (s and '+' or '-')+pl.to_label(int(g),n) for g,s in zip(T.g,[1 if x>0 else 0 for x in T.s])],"f0=",round(f0,4),"p0_hat=",p0)
rows=[]
for k,(m,s) in store.items():
    mean=s/m; sc=abs(mean)-3/math.sqrt(m); rows.append((sc,pl.to_label(k,n),m,mean,float(B.E[k])))
rows.sort(reverse=True)
print("top stored strings (score=|mean|-3/sqrt(m)):")
for r in rows[:12]: print(f"  {r[1]} m={r[2]:4d} mean={r[3]:+.3f} exact={r[4]:+.3f} score={r[0]:+.3f}")
print("strings stored:",len(store))
print("candidate generators:",[('+' if s>0 else '-')+pl.to_label(int(g),n) for g,s in zip(cand.g,cand.s)],"f(candidate)=",round(fc,4),"F_STAB=",FST)

# --- details for the example ---
T,f0,p0,store,cand,fc=run(1)
from fastfidelity.f2 import Basis
rows=sorted(((abs(s/m)-3/math.sqrt(m),k,m,s/m) for k,(m,s) in store.items()),reverse=True)
gens=[];basis=Basis()
print("\ngreedy selection:")
for sc,k,m,mean in rows:
    if sc<0.2: break
    L=pl.to_label(int(k),n)
    if not all(pl.symp(int(k),g,n)==0 for g in gens): print(f"  {L}: rejected (anticommutes)"); continue
    if not basis.add(int(k)): print(f"  {L}: rejected (product of accepted strings)"); continue
    gens.append(int(k)); print(f"  {L}: accepted with sign {'+' if mean>0 else '-'}")
    if len(gens)==n: break
# store estimate of the candidate fidelity (Eq. store)
rng=np.random.default_rng(99)
ck,cs,cu=cand.enumerate(); ck=[int(k) for k in ck]; cs=[int(x) for x in cs]
need=math.ceil(N/16); det=0.0; Kn=0; R=[]
for k,s in zip(ck,cs):
    if k==0: det+=s; Kn+=1; continue
    if k in store:
        m,sm=store[k]
        if m<need:
            bits=np.asarray(B.measure(np.full(need-m,k,dtype=np.int64),rng),dtype=np.int64)
            o=1-2*(pl.popcount(bits & pl.support(k,n))&1); m+=need-m; sm+=int(o.sum()); store[k]=(m,sm)
        det+=s*sm/m; Kn+=1
    else: R.append((k,s))
NR=math.ceil(N*len(R)/16); idx=rng.integers(0,len(R),size=NR); tot=0
for i,(k,s) in enumerate(R):
    mi=int((idx==i).sum())
    if mi==0: continue
    bits=np.asarray(B.measure(np.full(mi,k,dtype=np.int64),rng),dtype=np.int64)
    tot+=s*int((1-2*(pl.popcount(bits & pl.support(k,n))&1)).sum())
fhat=det/16+len(R)/16*tot/NR
print(f"\ncandidate group: |K|={Kn} strings from the store (incl. identity), |R|={len(R)} sampled with N_R={NR} shots")
print("K:",[('+' if s>0 else '-')+pl.to_label(k,n) for k,s in zip(ck,cs) if k==0 or k in store])
print(f"f_hat(candidate) = {fhat:.3f}  (exact {fc:.4f});  p0_hat = {p0:.3f} (exact {f0:.4f})")
# best neighbor of the start, exactly
best=0
for x in range(1,1<<n):
    for mm in range(4):
        best=max(best,T.neighbor(x,mm).fidelity_from_spectrum(B.E))
print("best neighbor of the start (exact):",round(best,4))
# p_x of the start basis (exact)
px=[T.flip(x).fidelity_from_spectrum(B.E) if x else f0 for x in range(1<<n)]
print("exact p_x of start basis: max",round(max(px),4),"sum",round(sum(px),4))
rng=np.random.default_rng(7); Nf=10000
d=rng.integers(0,16,size=Nf); tot=0
for i,(k,s) in enumerate(zip(ck,cs)):
    m=int((d==i).sum())
    if k==0: tot+=s*m; continue
    bits=np.asarray(B.measure(np.full(m,k,dtype=np.int64),rng),dtype=np.int64)
    tot+=s*int((1-2*(pl.popcount(bits & pl.support(k,n))&1)).sum())
F=tot/Nf; print(f"fresh DFE of the candidate: F_hat={F:.4f}, lower bound={F-math.sqrt(2*math.log(100)/Nf):.4f}")
print("all candidate group elements:",[('+' if s>0 else '-')+pl.to_label(k,n) for k,s in zip(ck,cs)])
