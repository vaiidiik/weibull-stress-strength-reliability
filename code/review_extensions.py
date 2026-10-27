from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from scipy import special, stats
import analysis as a

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/results'


def templates(theta,n=20,mode='typeI',heterogeneous=False):
    alpha,rate,_=theta
    if heterogeneous:
        settings=[(15,.6,.45,'early'),(20,.7,.65,'balanced'),(25,.76,.8,'late')]
    else:
        settings=[(n,.7,.65,'balanced')]*3
    return [dict(n=ni,m=int(ni*mf),T=(-np.log1p(-qt)/rate)**(1/alpha),mode=mode,plan=a.removal_plan(ni,int(ni*mf),pattern)) for ni,mf,qt,pattern in settings]


def batch_generate(theta,blocks,k,r,count,rng):
    alpha,rate,stress=theta
    d=np.zeros(count,dtype=int)
    C=np.zeros(count)
    logs=[]
    weights=[]
    for block in blocks:
        active=np.full(count,block['n'],dtype=int)
        live=np.ones(count,dtype=bool)
        hazard=np.zeros(count)
        terminal=np.zeros(count,dtype=int)
        lx=np.zeros((count,block['m']+1))
        w=np.zeros_like(lx)
        for j in range(block['m']):
            hazard+=rng.exponential(size=count)/np.maximum(active,1)
            x=(hazard/rate)**(1/alpha)
            event=live.copy()
            if block['mode']=='typeI':
                stopped=live&(x>block['T'])
                terminal[stopped]=active[stopped]
                live[stopped]=False
                event&=~stopped
            if not np.any(event):
                continue
            active[event]-=1
            removals=np.full(count,int(block['plan'][j]))
            if block['mode']=='adaptiveII':
                removals[x>block['T']]=0
            if j==block['m']-1:
                removals=active.copy()
            removals=np.minimum(removals,active)
            lx[event,j]=np.log(x[event])
            w[event,j]=1+removals[event]
            C[event]+=lx[event,j]
            d[event]+=1
            active[event]-=removals[event]
        lx[:,-1]=np.log(block['T'])
        w[:,-1]=terminal
        assert np.all(w.sum(axis=1)==block['n'])
        logs.append(lx)
        weights.append(w)
    y=(np.cumsum(rng.exponential(size=(count,r)),axis=1)/(k*stress))**(1/alpha)
    ly=np.log(y)
    C+=ly.sum(axis=1)
    return dict(lx=np.concatenate(logs,axis=1),w=np.concatenate(weights,axis=1),ly=ly,d=d,r=r,q=d+r,C=C,k=k)


def one(batch,i):
    return {key:value[i] if isinstance(value,np.ndarray) else value for key,value in batch.items()}


def batch_fit(batch,restricted=False):
    n=len(batch['d'])
    lo=np.full(n,np.log(.02))
    hi=np.full(n,np.log(30.))
    logw=np.log(np.maximum(batch['w'],1e-300))
    for _ in range(39):
        alpha=np.exp((lo+hi)/2)
        term=logw+alpha[:,None]*batch['lx']
        maximum=term.max(axis=1)
        z=np.exp(term-maximum[:,None])
        total=z.sum(axis=1)
        logA=maximum+np.log(total)
        meanlog=(z*batch['lx']).sum(axis=1)/total
        logB=np.log(batch['k'])+alpha*batch['ly'][:,-1]
        loga=np.log(np.maximum(batch['d'],1))-logA
        logb=np.log(batch['r'])-logB
        score=batch['q']/alpha+batch['C']-batch['d']*meanlog-batch['r']*batch['ly'][:,-1]
        if restricted:
            pooled=loga>logb
            logtotal=np.logaddexp(logA,logB)
            combined=np.exp(logA-logtotal)*meanlog+np.exp(logB-logtotal)*batch['ly'][:,-1]
            score=np.where(pooled,batch['q']/alpha+batch['C']-batch['q']*combined,score)
        lo=np.where(score>0,(lo+hi)/2,lo)
        midpoint=np.log(alpha)
        hi=np.where(score<=0,midpoint,hi)
    alpha=np.exp((lo+hi)/2)
    logA=special.logsumexp(logw+alpha[:,None]*batch['lx'],axis=1)
    logB=np.log(batch['k'])+alpha*batch['ly'][:,-1]
    rate=np.exp(np.log(np.maximum(batch['d'],1))-logA)
    stress=np.exp(np.log(batch['r'])-logB)
    if restricted:
        pooled=rate>stress
        common=np.exp(np.log(batch['q'])-np.logaddexp(logA,logB))
        rate=np.where(pooled,common,rate)
        stress=np.where(pooled,common,stress)
    out=np.column_stack([alpha,rate,stress])
    out[batch['d']==0]=np.nan
    return out


def batch_logratio_se(theta,batch):
    alpha,rate,stress=theta.T
    v=batch['w']*np.exp(alpha[:,None]*batch['lx'])
    Ap=(v*batch['lx']).sum(axis=1)
    App=(v*batch['lx']**2).sum(axis=1)
    B=batch['k']*np.exp(alpha*batch['ly'][:,-1])
    Bp=B*batch['ly'][:,-1]
    Bpp=B*batch['ly'][:,-1]**2
    information=np.zeros((len(theta),3,3))
    information[:,0,0]=batch['q']/alpha**2+rate*App+stress*Bpp
    information[:,0,1]=information[:,1,0]=Ap
    information[:,0,2]=information[:,2,0]=Bp
    information[:,1,1]=batch['d']/rate**2
    information[:,2,2]=batch['r']/stress**2
    covariance=np.linalg.inv(information)
    variance=covariance[:,1,1]/rate**2+covariance[:,2,2]/stress**2-2*covariance[:,1,2]/(rate*stress)
    return np.sqrt(np.maximum(variance,0))


def bootstrap_intervals(theta,blocks,k,r,rng,B=399,restricted=False,s=3,K=5):
    batch=batch_generate(theta,blocks,k,r,B,rng)
    estimates=batch_fit(batch,restricted)
    valid=np.isfinite(estimates).all(axis=1)
    values=a.reliability(estimates[valid,1],estimates[valid,2],s,K)
    result={'Percentile':np.quantile(values,[.025,.975]),'failed':int((~valid).sum()),'B':B}
    if not restricted:
        sub={key:value[valid] if isinstance(value,np.ndarray) else value for key,value in batch.items()}
        se=batch_logratio_se(estimates[valid],sub)
        psi=np.log(theta[2]/theta[1])
        pivot=(np.log(estimates[valid,2]/estimates[valid,1])-psi)/se
        result['pivot_quantiles']=np.quantile(pivot[np.isfinite(pivot)],[.025,.975])
        result['invalid_pivots']=int((~np.isfinite(pivot)).sum())
    return result


def scalar_logratio_se(theta,data):
    covariance=np.linalg.inv(a.observed_information(theta,data))
    gradient=np.array([0.,-1/theta[1],1/theta[2]])
    return float(np.sqrt(gradient@covariance@gradient))


def data_templates(data):
    return [dict(n=b['n'],m=b['m'],T=b['T'],mode=b['mode'],plan=b.get('plan',a.removal_plan(b['n'],b['m']))) for b in data['blocks']]


def order_evidence(data,rng,B=999):
    u=a.fit_mle(data)
    o=a.fit_mle(data,True)
    statistic=max(0.,2*(a.loglik(u,data)-a.loglik(o,data)))
    def objective(logalpha):
        alpha=np.exp(logalpha)
        A,Bexp=a.exposures(alpha,data)
        rate=data['q']/(A+Bexp)
        return -a.loglik((alpha,rate,rate),data)
    from scipy import optimize
    fit=optimize.minimize_scalar(objective,bounds=(np.log(.02),np.log(30)),method='bounded')
    alpha=np.exp(fit.x)
    A,Bexp=a.exposures(alpha,data)
    rate=data['q']/(A+Bexp)
    batch=batch_generate((alpha,rate,rate),data_templates(data),data['k'],data['r'],B,rng)
    ub=batch_fit(batch)
    ob=batch_fit(batch,True)
    def ll(theta):
        aa,rr,ss=theta.T
        exposure=(batch['w']*np.exp(aa[:,None]*batch['lx'])).sum(axis=1)
        stress_exp=batch['k']*np.exp(aa*batch['ly'][:,-1])
        return batch['q']*np.log(aa)+batch['d']*np.log(rr)+batch['r']*np.log(ss)+(aa-1)*batch['C']-rr*exposure-ss*stress_exp
    lr=np.maximum(0,2*(ll(ub)-ll(ob)))
    pvalue=1. if statistic<1e-8 else (1+np.count_nonzero(lr>=statistic))/(B+1)
    grid=a.posterior_grid(data,na=240,np_=320)
    probability=float(grid['weights'][:,grid['p']<=.5].sum())
    return dict(LR=statistic,boundary_bootstrap_p=pvalue,posterior_order_probability=probability,bootstrap_B=B)


def bootstrap_study(outer=500,inner=399):
    records=[]
    start=time.time()
    sid=0
    for regime,rate,stress in [('ordered',.8,1.6),('boundary',1.,1.),('violated',1.6,.8)]:
        for n in [20,40]:
            for mode in ['typeI','adaptiveII']:
                sid+=1
                theta=np.array([1.8,rate,stress])
                r=6 if n==20 else 12
                plan=templates(theta,n,mode)
                rng=np.random.default_rng(a.SEED+900000+sid*1000)
                samples=batch_generate(theta,plan,1,r,outer,rng)
                uf=batch_fit(samples)
                of=batch_fit(samples,True)
                true=float(a.reliability(rate,stress))
                for i in range(outer):
                    data=one(samples,i)
                    se=scalar_logratio_se(uf[i],data)
                    psi=np.log(uf[i,2]/uf[i,1])
                    aci=a.reliability(1,np.exp(psi+np.array([-1,1])*stats.norm.ppf(.975)*se))
                    u=bootstrap_intervals(uf[i],plan,1,r,rng,inner)
                    q=u['pivot_quantiles']
                    student=a.reliability(1,np.exp(psi-q[::-1]*se))
                    o=bootstrap_intervals(of[i],plan,1,r,rng,inner,True)
                    intervals=[('MLE ACI',aci),('MLE Boot-p',u['Percentile']),('MLE Boot-t (log ratio)',student),('Ordered MLE Boot-p',o['Percentile'])]
                    for label,interval in intervals:
                        records.append(dict(scenario=sid,regime=regime,n=n,protocol=mode,replicate=i+1,method=label,true_R=true,lower=interval[0],upper=interval[1],covered=float(interval[0]<=true<=interval[1]),width=interval[1]-interval[0],failed_inner=o['failed'] if label.startswith('Ordered') else u['failed']))
                pd.DataFrame(records).to_csv(OUT/'bootstrap_extended_raw.csv',index=False)
                print('Bootstrap setting',sid,'/12 complete; seconds',round(time.time()-start),flush=True)
    raw=pd.DataFrame(records)
    rows=[]
    for keys,g in raw.groupby(['scenario','regime','n','protocol','method'],sort=False):
        cp=g.covered.mean()
        rows.append(dict(zip(['scenario','regime','n','protocol','method'],keys),outer=outer,inner=inner,coverage=cp,coverage_mcse=np.sqrt(cp*(1-cp)/len(g)),average_width=g.width.mean(),failed_inner_max=g.failed_inner.max()))
    pd.DataFrame(rows).to_csv(OUT/'bootstrap_extended_summary.csv',index=False)


def design_study(repetitions=500):
    theta=np.array([1.8,.8,1.6])
    settings=[]
    for mode in ['typeI','adaptiveII']:
        for n in [20,40]:
            for r in [6,12]:
                settings.append((f'uniform_n{n}_r{r}_{mode}',templates(theta,n,mode),r))
        settings.append((f'heterogeneous_{mode}',templates(theta,mode=mode,heterogeneous=True),6))
        for pattern in ['early','late']:
            plan=templates(theta,20,mode)
            for block in plan:
                block['plan']=a.removal_plan(block['n'],block['m'],pattern)
            settings.append((f'{pattern}_{mode}',plan,6))
    rows=[]
    for sid,(label,plan,r) in enumerate(settings,1):
        rng=np.random.default_rng(a.SEED+700000+sid*1000)
        samples=batch_generate(theta,plan,1,r,repetitions,rng)
        uf=batch_fit(samples)
        of=batch_fit(samples,True)
        for i in range(repetitions):
            data=one(samples,i)
            for ordered,fit in [(False,uf[i]),(True,of[i])]:
                grid=a.posterior_grid(data,ordered,na=80,np_=110,s=2,K=4)
                truth=float(a.reliability(theta[1],theta[2],2,4))
                for method,estimate in [('MLE',float(a.reliability(fit[1],fit[2],2,4))),('Bayes',grid['means'][-1])]:
                    rows.append(dict(setting=label,replicate=i+1,method=('Ordered ' if ordered else '')+method,s=2,K=4,r=r,strength_total=sum(b['n'] for b in plan),target_total=sum(b['m'] for b in plan),true_R=truth,estimate=estimate,hpd_covered=float(grid['hpd'][0]<=truth<=grid['hpd'][1]) if method=='Bayes' else np.nan,hpd_width=np.diff(grid['hpd'])[0] if method=='Bayes' else np.nan))
        print('Design extension',sid,'/',len(settings),flush=True)
    raw=pd.DataFrame(rows)
    raw.to_csv(OUT/'design_extensions_raw.csv',index=False)
    summary=[]
    for keys,g in raw.groupby(['setting','method'],sort=False):
        err=g.estimate-g.true_R
        cp=g.hpd_covered.mean()
        summary.append(dict(setting=keys[0],method=keys[1],s=2,K=4,repetitions=repetitions,bias=err.mean(),absolute_bias=abs(err.mean()),mse=np.mean(err**2),bias_mcse=err.std(ddof=1)/np.sqrt(repetitions),hpd_coverage=cp,hpd_coverage_mcse=np.sqrt(cp*(1-cp)/repetitions) if np.isfinite(cp) else np.nan,hpd_width=g.hpd_width.mean()))
    pd.DataFrame(summary).to_csv(OUT/'design_extensions_summary.csv',index=False)


def prior_study(repetitions=500):
    alternative=dict(a0=.5,b0=.125,c1=1.,c2=1.,a1=.5,b1=.25)
    rows=[]
    for sid,(regime,rate,stress) in enumerate([('ordered',.8,1.6),('boundary',1.,1.),('violated',1.6,.8)],1):
        theta=(1.8,rate,stress)
        rng=np.random.default_rng(a.SEED+600000+sid*1000)
        samples=batch_generate(theta,templates(theta),1,6,repetitions,rng)
        for i in range(repetitions):
            data=one(samples,i)
            for label,prior in [('Primary',a.PRIOR),('Broader proper',alternative)]:
                for ordered in [False,True]:
                    grid=a.posterior_grid(data,ordered,prior,na=80,np_=110)
                    truth=float(a.reliability(rate,stress))
                    rows.append(dict(regime=regime,prior=label,ordered=ordered,replicate=i+1,true_R=truth,estimate=grid['means'][-1],covered=float(grid['hpd'][0]<=truth<=grid['hpd'][1]),width=np.diff(grid['hpd'])[0],edge_mass=grid['edge_mass']))
        print('Prior extension',regime,flush=True)
    raw=pd.DataFrame(rows)
    raw.to_csv(OUT/'prior_extensions_raw.csv',index=False)
    summary=[]
    for keys,g in raw.groupby(['regime','prior','ordered'],sort=False):
        err=g.estimate-g.true_R
        cp=g.covered.mean()
        summary.append(dict(regime=keys[0],prior=keys[1],ordered=keys[2],repetitions=repetitions,bias=err.mean(),mse=np.mean(err**2),hpd_coverage=cp,coverage_mcse=np.sqrt(cp*(1-cp)/repetitions),average_width=g.width.mean(),maximum_edge_mass=g.edge_mass.max()))
    pd.DataFrame(summary).to_csv(OUT/'prior_extensions_summary.csv',index=False)


def nonparametric_reliability(x,y,s=3,K=5):
    count=(np.asarray(x)[:,None]>np.asarray(y)[None,:]).sum(axis=0)
    return float(stats.hypergeom.sf(s-1,len(x),count,K).mean())


def nonparametric_comparison(samples,B=2000):
    rows=[]
    for sid,(label,x,y) in enumerate(samples,1):
        rng=np.random.default_rng(a.SEED+500000+sid)
        value=nonparametric_reliability(x,y)
        draws=[nonparametric_reliability(rng.choice(x,len(x),replace=True),rng.choice(y,len(y),replace=True)) for _ in range(B)]
        lo,hi=np.quantile(draws,[.025,.975])
        rows.append(dict(dataset=label,strength_n=len(x),stress_n=len(y),estimate=value,lower=lo,upper=hi,bootstrap_B=B,information='Complete observations; hypergeometric U-statistic'))
    pd.DataFrame(rows).to_csv(OUT/'nonparametric_reliability.csv',index=False)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('task',choices=['check','bootstrap','design','prior'])
    args=parser.parse_args()
    if args.task=='bootstrap':
        bootstrap_study()
    elif args.task=='design':
        design_study()
    elif args.task=='prior':
        prior_study()
    else:
        rng=np.random.default_rng(443)
        errors=[]
        start=time.perf_counter()
        for mode in ['typeI','adaptiveII']:
            plan=templates((1.8,1.6,.8),heterogeneous=True,mode=mode)
            batch=batch_generate((1.8,1.6,.8),plan,3,8,300,rng)
            for ordered in [False,True]:
                estimates=batch_fit(batch,ordered)
                for i in range(50):
                    data=one(batch,i)
                    reference=a.fit_mle(data,ordered)
                    error=np.max(np.abs(reference-estimates[i]))
                    assert error<2e-5,error
                    errors.append(error)
            assert np.isfinite(batch_logratio_se(batch_fit(batch),batch)).all()
        plan=templates((1.8,.8,1.6))
        b=batch_generate((1.8,.8,1.6),plan,1,6,1000,rng)
        start=time.perf_counter()
        batch_fit(b)
        print('1000 batch fits, seconds',time.perf_counter()-start)
        (OUT/'batch_validation.json').write_text(json.dumps({'independent_fit_comparisons':len(errors),'maximum_parameter_error':max(errors)},indent=2)+'\n')
        print('Batch optimizer and heterogeneous censoring checks passed',flush=True)
