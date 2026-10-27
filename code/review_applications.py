from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
import analysis as a
import review_extensions as e

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw'
PROCESSED=ROOT/'data/processed'
OUT=ROOT/'data/results'


def load_cases():
    jute=pd.read_csv(RAW/'jute_fibre.csv')
    table=pd.read_csv(PROCESSED/'real_censored_strength.csv')
    blocks=[]
    for _,g in table.groupby('block'):
        failures=g.failure.to_numpy()
        removals=g.withdrawn_at_failure.to_numpy(int)
        q=int(g.terminal_survivors.iloc[0])
        T=float(g.terminal_time.iloc[0])
        blocks.append(dict(x=failures,R=removals,z=np.r_[failures,T] if q else failures,w=np.r_[1+removals,q] if q else 1+removals,terminal=T,terminal_count=q,n=10,m=7,T=.5,mode='typeI',plan=a.removal_plan(10,7)))
    records=pd.read_csv(PROCESSED/'real_upper_k_records.csv').value.to_numpy()
    cases=[('Jute',jute.gauge_15mm.to_numpy(),jute.gauge_20mm.to_numpy(),a.make_data(blocks,records,2))]
    carbon=pd.read_csv(RAW/'carbon_fibre.csv').carbon_strength_GPa.to_numpy()
    rng=np.random.default_rng(a.SEED+70)
    ix=rng.permutation(100)
    x,y=carbon[ix[:50]]/3,carbon[ix[50:]]/3
    blocks=[a.censor_block(x[p],int(.7*len(p)),1.,a.removal_plan(len(p),int(.7*len(p))),rng) for p in np.array_split(np.arange(50),3)]
    records=pd.read_csv(PROCESSED/'carbon_records.csv').scaled_value.to_numpy()
    cases.append(('Carbon random-split boundary check',x,y,a.make_data(blocks,records,4)))
    primary=pd.read_csv(RAW/'primary_carbon_gauges.csv')
    x=primary.query('gauge_mm==20').strength_GPa.to_numpy()/5
    y=primary.query('gauge_mm==80').strength_GPa.to_numpy()/5
    rng=np.random.default_rng(a.SEED+201)
    partition=np.array_split(rng.permutation(len(x)),3)
    stress_order=rng.permutation(len(y))
    blocks=[a.censor_block(x[p],int(.7*len(p)),1.,a.removal_plan(len(p),int(.7*len(p))),rng) for p in partition]
    records,positions=a.extract_records(y[stress_order],2,6)
    pd.DataFrame({'record':range(1,len(records)+1),'scaled_value':records,'acquisition_position':positions,'source_position':stress_order[positions-1]+1}).to_csv(PROCESSED/'primary_records.csv',index=False)
    rows=[]
    for i,(part,block) in enumerate(zip(partition,blocks),1):
        for position in part:
            rows.append(dict(role='Strength',block=i,source_position=position+1,scaled_value=x[position]))
    for i,position in enumerate(stress_order,1):
        rows.append(dict(role='Comparison',block=0,source_position=position+1,acquisition_position=i,scaled_value=y[position]))
    pd.DataFrame(rows).to_csv(PROCESSED/'primary_allocation.csv',index=False)
    rows=[]
    for i,block in enumerate(blocks,1):
        for j,(value,removed) in enumerate(zip(block['x'],block['R']),1):
            rows.append(dict(block=i,failure=j,value=value,withdrawn=int(removed),terminal=block['terminal'],terminal_survivors=block['terminal_count']))
    pd.DataFrame(rows).to_csv(PROCESSED/'primary_censored_strength.csv',index=False)
    cases.append(('Primary carbon 20 mm versus 80 mm',x,y,a.make_data(blocks,records,2)))
    return cases



def quadrature_checks(data,label):
    rows=[]
    for ordered in [False,True]:
        grid=a.posterior_grid(data,ordered,na=240,np_=320)
        fine=a.posterior_grid(data,ordered,na=400,np_=520)
        row=dict(dataset=label,ordered=ordered,maximum_mean_difference=float(np.max(abs(grid['means']-fine['means']))),maximum_equal_tail_difference=float(np.max(abs(grid['ci']-fine['ci']))),maximum_hpd_difference=float(np.max(abs(grid['hpd']-fine['hpd']))),edge_mass=grid['edge_mass'],hpd_mass=grid['hpd_mass'])
        assert row['maximum_mean_difference']<1e-4
        assert row['maximum_equal_tail_difference']<.002
        assert row['maximum_hpd_difference']<.002
        rows.append(row)
    return rows


def run_applications():
    cases=load_cases()
    primary=cases[-1]
    fit,shape=a.goodness_of_fit(primary[1],primary[2],np.random.default_rng(a.SEED+200),B=1000)
    fit['sample']=['20 mm','80 mm']
    fit.to_csv(OUT/'primary_goodness_of_fit.csv',index=False)
    (OUT/'primary_common_shape_test.json').write_text(json.dumps(shape,indent=2)+'\n')
    evidence=[]
    intervals=[]
    primary_rows=[]
    refinement=[]
    for i,(label,x,y,data) in enumerate(cases,1):
        refinement.extend(quadrature_checks(data,label))
        rng=np.random.default_rng(a.SEED+300000+i)
        result=e.order_evidence(data,rng)
        evidence.append(dict(dataset=label,**result))
        u=a.fit_mle(data)
        o=a.fit_mle(data,True)
        psi=np.log(u[2]/u[1])
        se=e.scalar_logratio_se(u,data)
        boot=e.bootstrap_intervals(u,e.data_templates(data),data['k'],data['r'],rng,B=3999)
        ordered=e.bootstrap_intervals(o,e.data_templates(data),data['k'],data['r'],rng,B=3999,restricted=True)
        aci=a.asymptotic_intervals(u,data)[1]
        student=a.reliability(1,np.exp(psi-boot['pivot_quantiles'][::-1]*se))
        for method,ci in [('MLE ACI',aci),('MLE Boot-p',boot['Percentile']),('MLE Boot-t (log ratio)',student),('Ordered MLE Boot-p',ordered['Percentile'])]:
            intervals.append(dict(dataset=label,method=method,estimate=float(a.reliability(*(o if method.startswith('Ordered') else u)[1:])),lower=ci[0],upper=ci[1],bootstrap_B=0 if method=='MLE ACI' else 3999))
        if i==3:
            for method,fit,ci in [('MLE',u,aci),('Ordered MLE',o,ordered['Percentile'])]:
                primary_rows.append(dict(method=method,alpha=fit[0],a=fit[1],b=fit[2],R=float(a.reliability(*fit[1:])),lower=ci[0],upper=ci[1]))
            for ordered_flag in [False,True]:
                grid=a.posterior_grid(data,ordered_flag,na=240,np_=320)
                primary_rows.append(dict(method='Ordered Bayes' if ordered_flag else 'Bayes',alpha=grid['means'][0],a=grid['means'][1],b=grid['means'][2],R=grid['means'][3],lower=grid['ci'][0],upper=grid['ci'][1],hpd_lower=grid['hpd'][0],hpd_upper=grid['hpd'][1]))
        print('Application',label,result,flush=True)
    pd.DataFrame(evidence).to_csv(OUT/'order_evidence.csv',index=False)
    pd.DataFrame(intervals).to_csv(OUT/'application_bootstrap_intervals.csv',index=False)
    pd.DataFrame(primary_rows).to_csv(OUT/'primary_estimates.csv',index=False)
    pd.DataFrame(refinement).to_csv(OUT/'application_quadrature_refinement.csv',index=False)
    pd.DataFrame([row for row in refinement if row['dataset']==primary[0]]).to_csv(OUT/'primary_quadrature_refinement.csv',index=False)
    e.nonparametric_comparison([(label,x,y) for label,x,y,data in cases])
    comparison=pd.read_csv(OUT/'nonparametric_reliability.csv')
    complete=[]
    for file in ['common_shape_test.json','carbon_common_shape_test.json','primary_common_shape_test.json']:
        fit=json.loads((OUT/file).read_text())
        complete.append(float(a.reliability(fit['lambda_X'],fit['lambda_Y'])))
    comparison['complete_weibull_estimate']=complete
    comparison.to_csv(OUT/'nonparametric_reliability.csv',index=False)


if __name__=='__main__':
    run_applications()
