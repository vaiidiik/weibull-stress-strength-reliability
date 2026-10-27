from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import integrate,stats
import analysis as a
import review_extensions as e

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/results'


def run_checks():
    rows=[]
    for s,K in [(3,5),(2,4),(1,3),(4,6)]:
        for rho in [.5,1.,2.]:
            exact=float(a.reliability(1.,rho,s,K))
            integral=integrate.quad(lambda h:stats.binom.sf(s-1,K,np.exp(-h))*rho*np.exp(-rho*h),0,np.inf,epsabs=1e-11)[0]
            assert abs(exact-integral)<1e-9
            data=a.simulate_data((1.8,.8,1.6),rng=np.random.default_rng(144))
            for ordered in [False,True]:
                grid=a.posterior_grid(data,ordered,s=s,K=K)
                assert abs(grid['hpd_mass']-.95)<1e-8
                if ordered:
                    assert grid['hpd'][0]>=a.reliability(1.,1.,s,K)
            rows.append(dict(s=s,K=K,rho=rho,formula=exact,integral=integral,error=abs(exact-integral)))
    pd.DataFrame(rows).to_csv(OUT/'general_system_checks.csv',index=False)
    generator=[]
    for mode in ['typeI','adaptiveII']:
        theta=(1.8,.8,1.6)
        rng=np.random.default_rng(a.SEED+888)
        batch=e.batch_generate(theta,e.templates(theta,20,mode),1,6,30000,rng)
        complete=np.array([a.simulate_data(theta,n=20,k=1,r=6,mode=mode,rng=rng)['d'] for _ in range(3000)])
        se=np.sqrt(batch['d'].var()/len(batch['d'])+complete.var()/len(complete))
        standardized=(batch['d'].mean()-complete.mean())/se if se else 0.
        assert abs(standardized)<5
        generator.append(dict(protocol=mode,fast_n=30000,complete_generation_n=3000,fast_mean_failures=batch['d'].mean(),complete_mean_failures=complete.mean(),difference_in_MCSE=standardized))
    pd.DataFrame(generator).to_csv(OUT/'generator_validation.csv',index=False)
    diagnostics=[]
    for case,(aa,bb) in enumerate([(.8,1.6),(1.,1.),(1.6,.8)],1):
        data=a.simulate_data((1.8,aa,bb),rng=np.random.default_rng(a.SEED+88000+case))
        for ordered in [False,True]:
            chains,_=a.mcmc(data,ordered,chains=4,steps=10000,burn=2000,s=2,K=4)
            frame=a.chain_diagnostics(chains,s=2,K=4)
            grid=a.posterior_grid(data,ordered,na=240,np_=320,s=2,K=4)
            frame['scenario']=case
            frame['ordered']=ordered
            frame['quadrature_mean']=grid['means']
            frame['difference']=frame['mean']-frame.quadrature_mean
            assert (abs(frame.difference)<5*frame.MCSE+.002).all()
            diagnostics.append(frame)
            print('MCMC comparison',case,ordered,flush=True)
    pd.concat(diagnostics,ignore_index=True).to_csv(OUT/'mcmc_scenario_checks.csv',index=False)


if __name__=='__main__':
    run_checks()
