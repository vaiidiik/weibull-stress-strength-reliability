from pathlib import Path
import nbformat

root=Path(__file__).resolve().parents[1]
cells=[]

def md(text):
    cells.append(nbformat.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbformat.v4.new_code_cell(text.strip()))

md((root/'code/notes/derivations.md').read_text().split('## 1.')[0]+'''\nRun all cells in order. The notebook is stored in code; datasets and results are in the project data folder. The notebook retains computed outputs and exports results, figures and datasets. The saved outputs are the computed project results. Set RUN_SIMULATION=True in the implementation cell to repeat all 14,400 simulated datasets. The default loads the saved Monte Carlo tables and repeats the other calculations.\n\n| Requirement | Where it is fulfilled |\n|---|---|\n| Unrestricted and restricted MLEs | Sections 5-6; implemented profile fits and real-data/Monte Carlo tables |\n| Joint Bayesian priors | Sections 7-8; dependent gamma-Dirichlet prior with a_0=2 and c_1+c_2=4 |\n| Unrestricted and ordered Bayesian estimates | Support-specific quadrature, posterior means, LINEX, credible intervals and checked sampling |\n| Complete simulation study | 48 scenarios, 300 replications each, all four estimators, parameter/reliability errors, interval coverage and MCSEs |\n| Suitable-data validation | Published 15 mm/20 mm jute groups, calibrated fit checks, retained observations, inference, sensitivity and predictive checks |''')
theory=(root/'code/notes/derivations.md').read_text()
sections=theory.split('\n## ')[2:]
for section in sections[:8]:
    md('## '+section)
md('## 9. Executable implementation\nThe following cell contains the complete implementation, including the censoring protocols, record likelihood, profile MLEs, dependent prior, posterior integration, bootstrap and numerical diagnostics.')
code('%matplotlib inline\n'+(root/'code/analysis.py').read_text()+'\nRUN_SIMULATION=False')
md('## 10. Distribution and reliability figures')
code('''
z=np.linspace(.02,3.5,500)
fig,axes=plt.subplots(2,2,figsize=(11,7))
for ash in [.6,1.,1.8,3.]:
    rate=1.
    axes[0,0].plot(z,ash*rate*z**(ash-1)*np.exp(-rate*z**ash),label=f'alpha={ash}')
    axes[0,1].plot(z,1-np.exp(-rate*z**ash),label=f'alpha={ash}')
    axes[1,0].plot(z,np.exp(-rate*z**ash),label=f'alpha={ash}')
    axes[1,1].plot(z,ash*rate*z**(ash-1),label=f'alpha={ash}')
for ax,title in zip(axes.flat,['Weibull density','Weibull distribution','Weibull survival','Weibull hazard']):
    ax.set(title=title,xlabel='Positive measurement',ylabel=title.split()[-1]);ax.legend()
fig.tight_layout();save_figure('01_weibull_functions')
rho=np.linspace(.01,5,300)
fig,axes=plt.subplots(1,2,figsize=(11,4))
for s in [1,2,3,4,5]:
    axes[0].plot(rho,reliability(np.ones_like(rho),rho,s,5),label=f'{s}-out-of-5')
axes[0].axvline(1,color='black',linestyle=':');axes[0].set(xlabel='Stress/strength rate ratio rho',ylabel='System reliability',title='Shared-stress system configurations');axes[0].legend()
correct=reliability(np.ones_like(rho),rho)
independent=stats.binom.sf(2,5,rho/(1+rho))
axes[1].plot(rho,correct,label='One shared stress');axes[1].plot(rho,independent,label='Independent stress for each component',linestyle='--')
axes[1].set(xlabel='Stress/strength rate ratio rho',ylabel='3-out-of-5 reliability',title='Effect of common-stress dependence');axes[1].legend()
fig.tight_layout();save_figure('02_reliability')
''')
md('## 11. Numerical verification\nThe theoretical reliability is checked against direct integration and independent simulation of complete systems. The record transformation, restricted optimization, information matrix and posterior integration are also checked independently.')
code('''
rng=np.random.default_rng(SEED+1)
checks=[]
for ash,aa,bb in [(.8,.8,1.6),(1.8,1.,1.),(2.5,1.6,.8)]:
    exact=float(reliability(aa,bb))
    integral=integrate.quad(lambda t: stats.binom.sf(2,5,np.exp(-aa*t))*bb*np.exp(-bb*t),0,np.inf,epsabs=1e-11)[0]
    strengths=(rng.exponential(size=(200000,5))/aa)**(1/ash)
    stress=(rng.exponential(size=200000)/bb)**(1/ash)
    direct=np.mean((strengths>stress[:,None]).sum(axis=1)>=3)
    se=np.sqrt(exact*(1-exact)/len(stress))
    assert abs(exact-integral)<1e-9
    assert abs(direct-exact)<5*se
    checks.append(dict(check='Reliability',alpha=ash,a=aa,b=bb,formula=exact,integral=integral,system_MC=direct,MCSE=se))
verification_data=simulate_data((1.8,1.6,.8),rng=rng)
u=fit_mle(verification_data);o=fit_mle(verification_data,True)
independent_fit=optimize.minimize(lambda zz:-loglik(np.exp(zz),verification_data),np.log(u)+.1,method='BFGS',options={'gtol':1e-7})
assert np.max(np.abs(np.exp(independent_fit.x)-u))<1e-4
constraint={'type':'ineq','fun':lambda zz:zz[2]-zz[1]}
independent_ordered=optimize.minimize(lambda zz:-loglik(np.exp(zz),verification_data),np.log(o),method='SLSQP',constraints=[constraint],options={'ftol':1e-10,'maxiter':1000})
assert abs(loglik(np.exp(independent_ordered.x),verification_data)-loglik(o,verification_data))<1e-6
numerical_hessian=np.zeros((3,3));eps=1e-4
for i in range(3):
    for j in range(3):
        ei=np.eye(3)[i]*eps;ej=np.eye(3)[j]*eps
        numerical_hessian[i,j]=-(loglik(u+ei+ej,verification_data)-loglik(u+ei-ej,verification_data)-loglik(u-ei+ej,verification_data)+loglik(u-ei-ej,verification_data))/(4*eps**2)
hessian_error=np.max(np.abs(numerical_hessian-observed_information(u,verification_data)))
assert hessian_error<1e-3
hazard_end=rng.gamma(8,1,size=100000)
record_gamma_check=dict(empirical_mean=hazard_end.mean(),theoretical_mean=8.,empirical_variance=hazard_end.var(),theoretical_variance=8.)
record_paths=np.cumsum(rng.exponential(1/3,size=(100000,8)),axis=1)
assert abs((3*record_paths[:,-1]).mean()-8)<.04
reliability_checks=pd.DataFrame(checks)
reliability_checks.to_csv(RESULTS/'reliability_verification.csv',index=False)
display(reliability_checks)
display(pd.DataFrame([dict(check='Observed information finite-difference error',value=hessian_error),dict(check='Ordered optimizer log-likelihood difference',value=abs(loglik(np.exp(independent_ordered.x),verification_data)-loglik(o,verification_data)))]))
''')
md('## 12. Monte Carlo study\n'+sections[8].split('\n',1)[1])
code('''
if RUN_SIMULATION:
    design,simulation_raw,simulation=run_simulation(300)
else:
    design=pd.read_csv(RESULTS/'simulation_design.csv')
    simulation_raw=pd.read_csv(RESULTS/'simulation_raw.csv')
    simulation=pd.read_csv(RESULTS/'simulation_summary.csv')
display(design)
Rsummary=simulation[simulation.parameter=='R_3_5']
display(Rsummary[['scenario','regime','censoring','n_per_block','record_k','method','truth','bias','mse','bias_mcse','mse_mcse','absolute_bias','coverage','coverage_mcse','average_length','hpd_coverage','hpd_average_length']])
refinement=[]
for row in design.itertuples():
    rr=np.random.default_rng(SEED+row.scenario*10000)
    dd=simulate_data((row.alpha,row.lambda_X,row.lambda_Y),n=row.n_per_block,k=row.record_k,r=row.record_count,mode=row.censoring,rng=rr)
    for ordered in [False,True]:
        coarse=posterior_grid(dd,ordered,na=80,np_=110)
        fine=posterior_grid(dd,ordered,na=200,np_=280)
        refinement.append(dict(scenario=row.scenario,restricted=ordered,max_mean_difference=np.max(np.abs(coarse['means']-fine['means'])),R_mean_difference=abs(coarse['means'][-1]-fine['means'][-1]),max_interval_endpoint_difference=np.max(np.abs(coarse['ci']-fine['ci'])),edge_mass=coarse['edge_mass']))
refinement=pd.DataFrame(refinement)
refinement.to_csv(RESULTS/'quadrature_refinement.csv',index=False)
assert refinement.R_mean_difference.max()<.002
assert refinement.edge_mass.max()<1e-6
assert simulation_raw.grid_edge_mass.max()<1e-5
assert len(simulation_raw)==48*300*4
assert np.isfinite(simulation_raw[PARAMETERS].to_numpy()).all()
display(refinement.describe().loc[['mean','max']])
''')
code('''
colors={'MLE':'#0f566c','Restricted MLE':'#da9132','Bayes':'#3c9b7d','Restricted Bayes':'#9266b0'}
fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True)
for column,regime in enumerate(['ordered','boundary','violated']):
    sub=Rsummary[Rsummary.regime==regime]
    for method in colors:
        ss=sub[sub.method==method].groupby('n_per_block')[['mse','bias']].mean()
        axes[0,column].plot(ss.index,ss.mse,'o-',label=method,color=colors[method])
        axes[1,column].plot(ss.index,ss.bias,'o-',label=method,color=colors[method])
    axes[0,column].set(title=regime.capitalize()+' order',ylabel='Mean scenario MSE')
    axes[1,column].set(xlabel='Starting units per block',ylabel='Mean scenario signed bias');axes[1,column].axhline(0,color='black',linewidth=.6)
axes[0,0].legend(fontsize=8)
fig.suptitle('Reliability estimation across 48 scenarios; each point averages eight scenarios',y=1.02)
fig.tight_layout();save_figure('03_simulation_bias_mse')
fig,axes=plt.subplots(1,3,figsize=(13,4))
for ax,regime in zip(axes,['ordered','boundary','violated']):
    sub=Rsummary[Rsummary.regime==regime]
    for method in ['MLE','Bayes','Restricted Bayes']:
        ss=sub[sub.method==method]
        ax.errorbar(ss.scenario,ss.coverage,yerr=1.96*ss.coverage_mcse,fmt='o',ms=3,label=method,color=colors[method],alpha=.8)
    ax.axhline(.95,color='black',linestyle='--');ax.set(title=regime.capitalize(),xlabel='Scenario',ylabel='Coverage with Monte Carlo error bars',ylim=(-.03,1.03))
axes[0].legend(fontsize=8);fig.tight_layout();save_figure('04_simulation_coverage')
fig,axes=plt.subplots(1,2,figsize=(11,4))
for method,metric,length,label,style in [('Bayes','coverage','average_length','Unrestricted equal-tail','-'),('Bayes','hpd_coverage','hpd_average_length','Unrestricted HPD','--'),('Restricted Bayes','coverage','average_length','Ordered equal-tail','-'),('Restricted Bayes','hpd_coverage','hpd_average_length','Ordered HPD','--')]:
    sub=Rsummary[Rsummary.method==method].groupby('regime')[[metric,length]].mean().reindex(['ordered','boundary','violated'])
    color=colors[method]
    axes[0].plot(range(3),sub[metric],'o',linestyle=style,color=color,label=label)
    axes[1].plot(range(3),sub[length],'o',linestyle=style,color=color,label=label)
for ax in axes:
    ax.set(xticks=range(3),xticklabels=['Ordered','Boundary','Violated'],xlabel='Rate regime')
    ax.legend(fontsize=8)
axes[0].set(ylabel='Coverage probability',title='Bayesian interval coverage',ylim=(-.03,1.03));axes[0].axhline(.95,color='black',linestyle=':')
axes[1].set(ylabel='Average interval width',title='Bayesian interval width')
fig.tight_layout();save_figure('13_hpd_comparison')
fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,mode in zip(axes,['typeI','adaptiveII']):
    sub=simulation_raw.merge(design[['scenario','censoring','n_per_block']],on='scenario')
    sub=sub[(sub.method=='MLE')&(sub.censoring==mode)]
    ax.hist(sub.observed_failures,bins=15,alpha=.75,color='#0f566c');ax.set(title=mode+' retained failures',xlabel='Observed strength failures across three blocks',ylabel='Replications')
fig.tight_layout();save_figure('05_observed_information')
''')
code('''
selected=design[(design.alpha==1.8)&(design.n_per_block==20)&(design.record_k==3)]
boot_rows=[]
for row in selected.itertuples():
    rr=np.random.default_rng(SEED+500000+row.scenario)
    covered=[];length=[];failed=0
    for outer in range(100):
        dd=simulate_data((row.alpha,row.lambda_X,row.lambda_Y),n=20,k=3,r=6,mode=row.censoring,rng=rr)
        estimate=fit_mle(dd,True)
        draws,nfailed=bootstrap_reliability(estimate,dd,rr,B=199,restricted=True)
        ci=np.quantile(draws,[.025,.975])
        covered.append(ci[0]-1e-10<=row.truth_R<=ci[1]+1e-10)
        length.append(ci[1]-ci[0]);failed+=nfailed
    cp=np.mean(covered)
    boot_rows.append(dict(scenario=row.scenario,regime=row.regime,censoring=row.censoring,outer_replications=100,inner_resamples=199,coverage=cp,coverage_mcse=np.sqrt(cp*(1-cp)/100),average_length=np.mean(length),failed_inner_fits=failed))
    print(f'Restricted bootstrap coverage: scenario {row.scenario} complete',flush=True)
bootstrap_coverage=pd.DataFrame(boot_rows)
bootstrap_coverage.to_csv(RESULTS/'restricted_bootstrap_coverage.csv',index=False)
display(bootstrap_coverage)
''')
md('## 13. Published-data validation\n'+sections[9].split('\n',1)[1])
code('''
measurements=pd.read_csv(RAW/'jute_fibre.csv')
x=measurements.gauge_15mm.to_numpy();y=measurements.gauge_20mm.to_numpy()
assert len(x)==len(y)==30 and np.all(x>0) and np.all(y>0)
gof,shape_test=goodness_of_fit(x,y,np.random.default_rng(SEED+20),B=1000)
gof.to_csv(RESULTS/'goodness_of_fit.csv',index=False)
(RESULTS/'common_shape_test.json').write_text(json.dumps(shape_test,indent=2))
display(gof);display(pd.DataFrame([shape_test]))
fig,axes=plt.subplots(2,3,figsize=(13,7))
for row,(label,sample) in enumerate([('15 mm strength',x),('20 mm comparison stress',y)]):
    ash,rate,ll=iid_fit(sample);zz=np.linspace(.005,1.1,400);zsort=np.sort(sample);prob=(np.arange(1,31)-.5)/30
    axes[row,0].hist(sample,bins=8,density=True,alpha=.5,color='#3c9b7d');axes[row,0].plot(zz,ash*rate*zz**(ash-1)*np.exp(-rate*zz**ash),color='#0f566c');axes[row,0].set(title=label+' density',xlabel='Scaled breaking strength',ylabel='Density')
    axes[row,1].step(zsort,np.arange(1,31)/30,where='post',label='Empirical');axes[row,1].plot(zz,1-np.exp(-rate*zz**ash),label='Fitted Weibull');axes[row,1].set(title=label+' CDF',xlabel='Scaled breaking strength',ylabel='Cumulative probability');axes[row,1].legend()
    theoretical=(-np.log1p(-prob)/rate)**(1/ash)
    axes[row,2].scatter(theoretical,zsort,color='#0f566c');axes[row,2].plot([0,1],[0,1],linestyle='--',color='black');axes[row,2].set(title=label+' Q-Q',xlabel='Fitted Weibull quantile',ylabel='Empirical quantile')
fig.tight_layout();save_figure('06_real_data_fit')
''')
code('''
rng=np.random.default_rng(SEED)
partition=np.array_split(rng.permutation(np.arange(30)),3)
blocks=[];block_rows=[]
for i,indices in enumerate(partition,1):
    block=censor_block(x[indices],7,.5,removal_plan(10,7),rng,'typeI');blocks.append(block)
    for j,(failure,removed) in enumerate(zip(block['x'],block['R']),1):
        block_rows.append(dict(block=i,failure_number=j,failure=failure,withdrawn_at_failure=removed,terminal_time=block['terminal'],terminal_survivors=block['terminal_count']))
    if len(block['x'])==0:
        block_rows.append(dict(block=i,failure_number=0,failure=np.nan,withdrawn_at_failure=0,terminal_time=block['terminal'],terminal_survivors=block['terminal_count']))
block_table=pd.DataFrame(block_rows)
block_table.to_csv(PROCESSED/'real_censored_strength.csv',index=False)
pd.DataFrame({'block':np.repeat([1,2,3],10),'source_position':np.concatenate(partition)+1,'measurement':x[np.concatenate(partition)]}).to_csv(PROCESSED/'real_block_allocation.csv',index=False)
records,positions=extract_records(y,2,6)
record_table=pd.DataFrame({'record_number':np.arange(1,7),'k':2,'value':records,'source_position':positions})
record_table.to_csv(PROCESSED/'real_upper_k_records.csv',index=False)
real_data=make_data(blocks,records,2)
u=fit_mle(real_data);o=fit_mle(real_data,True)
gu=posterior_grid(real_data,na=240,np_=320);go=posterior_grid(real_data,True,na=240,np_=320)
parameter_aci,Raci=asymptotic_intervals(u,real_data)
posterior_u=posterior_draws(gu,np.random.default_rng(SEED+50),40000)
posterior_o=posterior_draws(go,np.random.default_rng(SEED+51),40000)
boot_u,fail_u=bootstrap_reliability(u,real_data,np.random.default_rng(SEED+52),B=500)
boot_o,fail_o=bootstrap_reliability(o,real_data,np.random.default_rng(SEED+53),B=500,restricted=True)
intervals={'MLE':Raci,'Restricted MLE':np.quantile(boot_o,[.025,.975]),'Bayes':gu['ci'],'Restricted Bayes':go['ci']}
fits={'MLE':np.r_[u,reliability(*u[1:])],'Restricted MLE':np.r_[o,reliability(*o[1:])],'Bayes':gu['means'],'Restricted Bayes':go['means']}
real_rows=[]
for method,est in fits.items():
    real_rows.append(dict(method=method,alpha=est[0],lambda_X=est[1],lambda_Y=est[2],R_3_5=est[3],lower=intervals[method][0],upper=intervals[method][1],interval_type='Log-ratio asymptotic CI' if method=='MLE' else 'Parametric bootstrap percentile CI' if method=='Restricted MLE' else 'Posterior equal-tail credible interval'))
real_estimates=pd.DataFrame(real_rows)
real_estimates.to_csv(RESULTS/'real_estimates.csv',index=False)
pd.DataFrame([dict(dataset='Jute',model=label,lower=g['hpd'][0],upper=g['hpd'][1],posterior_mass=g['hpd_mass']) for label,g in [('Unrestricted',gu),('Ordered',go)]]).to_csv(RESULTS/'jute_hpd_intervals.csv',index=False)
pd.DataFrame({'unrestricted_bootstrap_R':boot_u,'restricted_bootstrap_R':boot_o}).to_csv(RESULTS/'real_bootstrap.csv',index=False)
param_rows=[]
for j,name in enumerate(PARAMETERS[:3]):
    for label,draws in [('Bayes',posterior_u),('Restricted Bayes',posterior_o)]:
        ci=np.quantile(draws[:,j],[.025,.975]);param_rows.append(dict(method=label,parameter=name,estimate=fits[label][j],lower=ci[0],upper=ci[1],interval_type='Posterior equal-tail'))
    param_rows.append(dict(method='MLE',parameter=name,estimate=u[j],lower=parameter_aci[j,0],upper=parameter_aci[j,1],interval_type='Positive log-normal asymptotic CI'))
parameter_estimates=pd.DataFrame(param_rows)
parameter_estimates.to_csv(RESULTS/'real_parameter_intervals.csv',index=False)
display(block_table);display(record_table);display(real_estimates);display(parameter_estimates)
display(pd.DataFrame([dict(model='Unrestricted',LINEX_positive=gu['linex'][1.5],LINEX_negative=gu['linex'][-1.5],HPD_lower=gu['hpd'][0],HPD_upper=gu['hpd'][1],bootstrap_lower=np.quantile(boot_u,.025),bootstrap_upper=np.quantile(boot_u,.975),failed_bootstrap_fits=fail_u),dict(model='Restricted',LINEX_positive=go['linex'][1.5],LINEX_negative=go['linex'][-1.5],HPD_lower=go['hpd'][0],HPD_upper=go['hpd'][1],bootstrap_lower=np.quantile(boot_o,.025),bootstrap_upper=np.quantile(boot_o,.975),failed_bootstrap_fits=fail_o)]))
fig,axes=plt.subplots(1,2,figsize=(11,4))
for i,block in enumerate(blocks,1):
    axes[0].scatter(block['x'],np.full(len(block['x']),i),s=30,label=f'Block {i}')
    axes[0].scatter(block['x'],np.full(len(block['x']),i),s=30+45*block['R'],facecolors='none',edgecolors='black')
axes[0].axvline(.5,color='black',linestyle='--',label='Threshold T');axes[0].set(xlabel='Scaled strength; ring size indicates withdrawal count',ylabel='Block',title='Retained failures and censoring');axes[0].legend(fontsize=8)
axes[1].plot(np.arange(1,len(y)+1),y,'o',alpha=.35,label='Complete comparison sample');axes[1].step(positions,records,where='post',color='#0f566c',label='First six upper 2-records');axes[1].scatter(positions,records,color='#0f566c');axes[1].axvline(positions[-1],linestyle=':',color='black',label='Record experiment stops');axes[1].set(xlabel='Printed table position',ylabel='Scaled comparison measurement',title='Auditable upper k-record extraction');axes[1].legend(fontsize=8)
fig.tight_layout();save_figure('07_real_sampling')
fig,axes=plt.subplots(1,2,figsize=(11,4))
for label,draws in [('Bayes',posterior_u),('Restricted Bayes',posterior_o)]:
    axes[0].hist(draws[:,-1],bins=50,density=True,alpha=.45,label=label)
axes[0].axvline(.5,linestyle='--',color='black');axes[0].set(xlabel='3-out-of-5 reliability',ylabel='Posterior density',title='Posterior reliability');axes[0].legend()
for j,row in enumerate(real_estimates.itertuples()):
    axes[1].plot([row.lower,row.upper],[j,j],color=colors[row.method],linewidth=3);axes[1].scatter(row.R_3_5,j,color=colors[row.method],s=50)
axes[1].set(yticks=np.arange(4),yticklabels=real_estimates.method,xlabel='Reliability with 95% interval',title='Point and interval estimates');axes[1].axvline(.5,linestyle='--',color='black')
fig.tight_layout();save_figure('08_real_reliability_intervals')
''')
md('## 13.1. Additional validation on published carbon-fibre strengths\nThe calibrated jute checks reveal limited Weibull adequacy in one group. A second common-unit dataset provides an independent compatibility check. The 100 carbon-fibre strengths in GPa are transcribed from the supplied Journal of Mathematics article, section 5.1.1, pages 12 and 14, attributed there to Nichols and Padgett. A seeded random split produces disjoint strength and comparison groups of 50 each. Both remain laboratory breaking-strength populations. The analysis unit is 3 GPa, so measurements are divided by three before inference and the same fixed primary prior is used. No flood-discharge observations are mixed with these data. The first five upper 4-records contain no tied retained values; no jitter is applied to rounded measurements.')
code("""
carbon=pd.read_csv(RAW/'carbon_fibre.csv')
assert len(carbon)==100
crng=np.random.default_rng(SEED+70)
ci=crng.permutation(100)
cx=carbon.carbon_strength_GPa.to_numpy()[ci[:50]]/3
cy=carbon.carbon_strength_GPa.to_numpy()[ci[50:]]/3
carbon_gof,carbon_shape=goodness_of_fit(cx,cy,np.random.default_rng(SEED+71),B=1000)
carbon_gof['sample']=['Carbon strength half','Carbon comparison half']
carbon_gof.to_csv(RESULTS/'carbon_goodness_of_fit.csv',index=False)
(RESULTS/'carbon_common_shape_test.json').write_text(json.dumps(carbon_shape,indent=2))
cblocks=[];carbon_alloc=[]
for number,part in enumerate(np.array_split(np.arange(50),3),1):
    n=len(part);m=int(.7*n)
    cblocks.append(censor_block(cx[part],m,1.,removal_plan(n,m),crng))
    for local in part:
        carbon_alloc.append(dict(role='Strength',block=number,source_position=ci[local]+1,scaled_measurement=cx[local]))
for local in range(50):
    carbon_alloc.append(dict(role='Comparison stress',block=0,source_position=ci[50+local]+1,scaled_measurement=cy[local]))
pd.DataFrame(carbon_alloc).to_csv(PROCESSED/'carbon_allocation.csv',index=False)
crecords,cpositions=extract_records(cy,4,5)
pd.DataFrame({'record_number':range(1,6),'k':4,'scaled_value':crecords,'source_position':ci[50+cpositions-1]+1,'acquisition_position':cpositions}).to_csv(PROCESSED/'carbon_records.csv',index=False)
carbon_data=make_data(cblocks,crecords,4)
cu=fit_mle(carbon_data);co=fit_mle(carbon_data,True)
cgu=posterior_grid(carbon_data,na=240,np_=320);cgo=posterior_grid(carbon_data,True,na=240,np_=320)
_,caci=asymptotic_intervals(cu,carbon_data)
cboot,cfailed=bootstrap_reliability(co,carbon_data,np.random.default_rng(SEED+72),B=500,restricted=True)
carbon_rows=[]
for label,estimate,interval in [('MLE',np.r_[cu,reliability(*cu[1:])],caci),('Restricted MLE',np.r_[co,reliability(*co[1:])],np.quantile(cboot,[.025,.975])),('Bayes',cgu['means'],cgu['ci']),('Restricted Bayes',cgo['means'],cgo['ci'])]:
    carbon_rows.append(dict(method=label,alpha=estimate[0],lambda_X=estimate[1],lambda_Y=estimate[2],R_3_5=estimate[3],lower=interval[0],upper=interval[1]))
carbon_estimates=pd.DataFrame(carbon_rows)
carbon_estimates.to_csv(RESULTS/'carbon_estimates.csv',index=False)
pd.DataFrame([dict(dataset='Carbon',model=label,lower=g['hpd'][0],upper=g['hpd'][1],posterior_mass=g['hpd_mass']) for label,g in [('Unrestricted',cgu),('Ordered',cgo)]]).to_csv(RESULTS/'carbon_hpd_intervals.csv',index=False)
display(carbon_gof);display(pd.DataFrame([carbon_shape]));display(carbon_estimates)
fig,axes=plt.subplots(2,3,figsize=(13,7))
for row,(label,sample) in enumerate([('Carbon strength half',cx),('Carbon comparison half',cy)]):
    ash,rate,ll=iid_fit(sample);zz=np.linspace(.001,2.2,400);ss=np.sort(sample);prob=(np.arange(1,len(sample)+1)-.5)/len(sample)
    axes[row,0].hist(sample,bins=9,density=True,alpha=.5);axes[row,0].plot(zz,ash*rate*zz**(ash-1)*np.exp(-rate*zz**ash));axes[row,0].set(title=label+' density',xlabel='Strength / 3 GPa',ylabel='Density')
    axes[row,1].step(ss,np.arange(1,len(sample)+1)/len(sample),where='post');axes[row,1].plot(zz,1-np.exp(-rate*zz**ash));axes[row,1].set(title=label+' CDF',xlabel='Strength / 3 GPa',ylabel='Probability')
    tt=(-np.log1p(-prob)/rate)**(1/ash);axes[row,2].scatter(tt,ss);axes[row,2].plot([0,2],[0,2],'--',color='black');axes[row,2].set(title=label+' Q-Q',xlabel='Weibull quantile',ylabel='Empirical quantile')
fig.tight_layout();save_figure('12_carbon_data_fit')
""")
md('## 14. Independent posterior sampler and convergence diagnostics\nThe quadrature computation is checked against Metropolis-Hastings in the collapsed transformed coordinates. The trace plots below show actual retained samples after warm-up; the conditional total-rate draw uses its exact Gamma distribution.')
code('''
chain_u,accept_u=mcmc(real_data)
chain_o,accept_o=mcmc(real_data,True)
diag_u=chain_diagnostics(chain_u);diag_o=chain_diagnostics(chain_o)
diag_u['model']='Unrestricted';diag_o['model']='Restricted'
diagnostics=pd.concat([diag_u,diag_o],ignore_index=True)
diagnostics['quadrature_mean']=np.r_[gu['means'],go['means']]
diagnostics['difference']=diagnostics['mean']-diagnostics.quadrature_mean
diagnostics.to_csv(RESULTS/'posterior_diagnostics.csv',index=False)
np.savez_compressed(RESULTS/'posterior_chains.npz',unrestricted=chain_u,restricted=chain_o)
display(diagnostics);display(pd.DataFrame({'chain':range(1,5),'unrestricted_acceptance':accept_u,'restricted_acceptance':accept_o}))
assert diagnostics.split_Rhat.max()<1.02
assert diagnostics.ESS.min()>400
assert np.all(np.abs(diagnostics.difference)<5*diagnostics.MCSE+.002)
fig,axes=plt.subplots(4,2,figsize=(12,10))
for j,name in enumerate(PARAMETERS):
    for c in range(4):
        axes[j,0].plot(chain_u[c,::10,j],alpha=.55,linewidth=.6)
        axes[j,1].plot(chain_o[c,::10,j],alpha=.55,linewidth=.6)
    axes[j,0].set(ylabel=name,title='Unrestricted' if j==0 else None);axes[j,1].set(ylabel=name,title='Restricted' if j==0 else None)
axes[-1,0].set_xlabel('Retained iteration / 10');axes[-1,1].set_xlabel('Retained iteration / 10')
fig.tight_layout();save_figure('09_posterior_traces')
fig,axes=plt.subplots(1,3,figsize=(12,4))
for j,name in enumerate(PARAMETERS[:3]):
    axes[j].hist(posterior_u[:,j],bins=50,density=True,alpha=.5,label='Unrestricted');axes[j].hist(posterior_o[:,j],bins=50,density=True,alpha=.5,label='Restricted');axes[j].set(xlabel=name,ylabel='Posterior density',title=name)
axes[0].legend();fig.tight_layout();save_figure('10_parameter_posteriors')
''')
md('## 15. Sensitivity analyses and posterior predictive checks\nHyperparameters, hybrid scheme, record order/count, and the imposed acquisition order are varied explicitly. All settings and exclusions are recorded. Posterior predictive checks regenerate the actual censored/record experiment; they do not compare dependent records to an iid empirical distribution.')
code('''
sensitivity=[]
priors={'Primary':PRIOR,'Broader total rate':dict(a0=1.,b0=.25,c1=2.,c2=2.,a1=1.,b1=.5),'More concentrated rates':dict(a0=6.,b0=1.5,c1=4.,c2=4.,a1=4.,b1=2.)}
for label,pr in priors.items():
    for ordered in [False,True]:
        gg=posterior_grid(real_data,ordered,pr,na=240,np_=320)
        sensitivity.append(dict(analysis='Prior',setting=label,restricted=ordered,R=gg['means'][-1],lower=gg['ci'][0],upper=gg['ci'][1]))
for mode in ['typeI','adaptiveII']:
    rr=np.random.default_rng(SEED)
    rr.permutation(np.arange(30))
    bb=[censor_block(x[ind],7,.5,removal_plan(10,7),rr,mode) for ind in partition]
    dd=make_data(bb,records,2)
    for ordered in [False,True]:
        gg=posterior_grid(dd,ordered,na=240,np_=320)
        sensitivity.append(dict(analysis='Hybrid scheme',setting=mode,restricted=ordered,R=gg['means'][-1],lower=gg['ci'][0],upper=gg['ci'][1]))
for kk,rc in [(1,4),(2,4),(2,8),(3,6)]:
    yy,ii=extract_records(y,kk,rc)
    dd=make_data(blocks,yy,kk)
    gg=posterior_grid(dd,na=240,np_=320)
    sensitivity.append(dict(analysis='Record design',setting=f'k={kk}, r={rc}',restricted=False,R=gg['means'][-1],lower=gg['ci'][0],upper=gg['ci'][1]))
sensitivity=pd.DataFrame(sensitivity)
sensitivity.to_csv(RESULTS/'sensitivity.csv',index=False);display(sensitivity)
order_results=[];order_failures=0;rr=np.random.default_rng(SEED+60)
for rep in range(200):
    try:
        yy,ii=extract_records(rr.permutation(y),2,6)
        gg=posterior_grid(make_data(blocks,yy,2),na=110,np_=160)
        order_results.append(dict(permutation=rep+1,R=gg['means'][-1],record_stop_position=ii[-1],largest_record=yy[-1]))
    except ValueError:
        order_failures+=1
order_results=pd.DataFrame(order_results)
order_results.to_csv(RESULTS/'acquisition_order_sensitivity.csv',index=False)
display(pd.DataFrame([dict(attempted_orders=200,orders_reaching_six_records=len(order_results),orders_not_reaching_target=order_failures,mean_R=order_results.R.mean(),minimum_R=order_results.R.min(),maximum_R=order_results.R.max())]))
rr=np.random.default_rng(SEED+61);predictive=[]
for i in range(2000):
    ash,aa,bb,_=posterior_u[rr.integers(len(posterior_u))]
    bbk=[]
    for old in blocks:
        raw=(rr.exponential(size=old['n'])/aa)**(1/ash)
        bbk.append(censor_block(raw,old['m'],old['T'],removal_plan(old['n'],old['m']),rr,old['mode']))
    yy=(np.cumsum(rr.exponential(scale=1/(2*bb),size=6)))**(1/ash)
    predictive.append(dict(failure_count=sum(len(bk['x']) for bk in bbk),largest_record=yy[-1]))
predictive=pd.DataFrame(predictive)
predictive.to_csv(RESULTS/'posterior_predictive.csv',index=False)
ppcheck=pd.DataFrame([dict(statistic='Observed failure count',observed=real_data['d'],predictive_lower=np.quantile(predictive.failure_count,.025),predictive_upper=np.quantile(predictive.failure_count,.975),posterior_predictive_upper_tail=np.mean(predictive.failure_count>=real_data['d'])),dict(statistic='Largest record',observed=records[-1],predictive_lower=np.quantile(predictive.largest_record,.025),predictive_upper=np.quantile(predictive.largest_record,.975),posterior_predictive_upper_tail=np.mean(predictive.largest_record>=records[-1]))])
ppcheck.to_csv(RESULTS/'predictive_checks.csv',index=False);display(ppcheck)
fig,axes=plt.subplots(1,3,figsize=(13,4))
axes[0].hist(predictive.failure_count,bins=np.arange(0,23)-.5,color='#0f566c',alpha=.75);axes[0].axvline(real_data['d'],color='#da9132',linewidth=2);axes[0].set(title='Predictive retained failures',xlabel='Failure count',ylabel='Predictive replications')
axes[1].hist(predictive.largest_record,bins=40,color='#0f566c',alpha=.75);axes[1].axvline(records[-1],color='#da9132',linewidth=2);axes[1].set(title='Predictive largest record',xlabel='Scaled measurement',ylabel='Predictive replications')
axes[2].hist(order_results.R,bins=25,color='#3c9b7d',alpha=.75);axes[2].axvline(gu['means'][-1],color='#da9132',linewidth=2);axes[2].set(title='Acquisition-order sensitivity',xlabel='Posterior mean reliability',ylabel='Successful permutations')
fig.tight_layout();save_figure('11_predictive_order_sensitivity')
''')
md('## 16. Results interpretation and requirement audit')
code(r"""
agg=Rsummary.groupby(['regime','method'])[['bias','absolute_bias','mse','coverage','average_length','hpd_coverage','hpd_average_length']].mean().reset_index()
agg.to_csv(RESULTS/'reliability_regime_summary.csv',index=False);display(agg)
real_refinement=[]
for ordered,base in [(False,gu),(True,go)]:
    gg=posterior_grid(real_data,ordered,na=400,np_=520)
    real_refinement.append(dict(restricted=ordered,maximum_mean_difference=np.max(np.abs(base['means']-gg['means'])),maximum_interval_difference=np.max(np.abs(base['ci']-gg['ci'])),maximum_hpd_difference=np.max(np.abs(base['hpd']-gg['hpd'])),hpd_mass=base['hpd_mass'],edge_mass=base['edge_mass']))
real_refinement=pd.DataFrame(real_refinement)
real_refinement.to_csv(RESULTS/'real_quadrature_refinement.csv',index=False);display(real_refinement)
assert real_refinement.maximum_mean_difference.max()<1e-4
assert real_refinement.maximum_interval_difference.max()<.002
assert real_refinement.maximum_hpd_difference.max()<.002
assert np.max(np.abs(real_refinement.hpd_mass-.95))<1e-8
findings=f'''The complete 15 mm and 20 mm samples give calibrated Weibull K-S p-values {gof.bootstrap_p.iloc[0]:.3f} and {gof.bootstrap_p.iloc[1]:.3f}. The common-shape likelihood-ratio statistic is {shape_test['LR']:.3f}, with bootstrap p-value {shape_test['bootstrap_p']:.3f}. The 15 mm calibrated test rejects Weibull adequacy at the 5% level, so the jute estimates are reported with a model-fit limitation. The additional carbon-fibre halves have calibrated K-S p-values {carbon_gof.bootstrap_p.iloc[0]:.3f} and {carbon_gof.bootstrap_p.iloc[1]:.3f}; their common-shape bootstrap p-value is {carbon_shape['bootstrap_p']:.3f}. These are compatible with the fitted Weibull model at this sample size, without proving it true. Their unrestricted and restricted posterior reliability means are {cgu['means'][-1]:.4f} and {cgo['means'][-1]:.4f}, respectively.

The retrospective Type-I experiment retains {real_data['d']} strength failures and {real_data['r']} upper 2-records. The unrestricted MLE reliability is {fits['MLE'][-1]:.4f}; its log-ratio 95% confidence interval is [{Raci[0]:.4f}, {Raci[1]:.4f}]. The unrestricted posterior mean is {gu['means'][-1]:.4f}, with a 95% equal-tail credible interval [{gu['ci'][0]:.4f}, {gu['ci'][1]:.4f}]. The restricted MLE is {fits['Restricted MLE'][-1]:.4f}; the restricted posterior mean is {go['means'][-1]:.4f}, with credible interval [{go['ci'][0]:.4f}, {go['ci'][1]:.4f}]. The restricted posterior excludes reliability below one half by construction, so this shift reflects the assumed rate order as well as the measurements.

Across 14,400 simulated datasets, all four estimators were compared against fixed truths and all raw outputs were retained. Scenario-level tables include Monte Carlo uncertainty. Both equal-tail and numerical HPD interval coverage and average width are recorded for the Bayesian estimators. Under a false order restriction, the true reliability is below one half while restricted estimates cannot be; the resulting bias and low coverage are a structural consequence rather than an implementation error. At an equality boundary, a continuous truncated posterior has all interval mass above one half, so an equal-tail restricted credible interval need not include the boundary truth. This is why boundary coverage is reported explicitly.

The largest sampler split R-hat is {diagnostics.split_Rhat.max():.4f} and the minimum effective sample size is {diagnostics.ESS.min():.0f}. Sampler means are checked against quadrature with their Monte Carlo standard errors. The numerical HPD intervals have posterior probability 0.95 under the interpolated reliability density, with endpoints checked on a finer grid. The maximum real-data quadrature refinement difference in a mean is {real_refinement.maximum_mean_difference.max():.2g}. The model remains conditional on common shape, homogeneous block populations, noninformative withdrawals, shared stress and the laboratory interpretation of the comparison group. Acquisition-order sensitivity and prior sensitivity should be retained when presenting the estimates.

All five assigned requirements are implemented. The SAGE reference was available at abstract level only; full-text review of that unsupplied article remains access-limited. All five supplied PDFs were examined and their uses or exclusions are recorded in the literature review.'''
(NOTES/'results_discussion.md').write_text('# Results and discussion\n\nVaidik Maheshwari (250150045) and Anant Gupta (250150004)\n\n'+findings)
display(Markdown(findings))
audit=pd.DataFrame([dict(requirement='Unrestricted and restricted MLEs',evidence='Profile derivation, independent optimization check, real-data and 48-scenario tables',status='Implemented'),dict(requirement='Joint priors',evidence='Dependent gamma-Dirichlet family, Jacobian, covariance, truncated ordered prior',status='Implemented'),dict(requirement='Unrestricted and restricted Bayes estimates',evidence='Quadrature, squared-error and LINEX, credible intervals, independent sampler',status='Implemented'),dict(requirement='Complete simulation',evidence='48 x 300 datasets; point metrics for four targets; ACI, equal-tail and HPD coverage/width; MCSE; bootstrap subset',status='Implemented'),dict(requirement='Suitable-data validation',evidence='Published jute data; calibrated K-S/LR; fit, sampling, posterior and predictive figures',status='Implemented')])
audit.to_csv(RESULTS/'requirements_audit.csv',index=False);display(audit)
""")
md('## 17. Literature review and bibliography\n'+(root/'code/notes/literature_review.md').read_text().split('\n',3)[-1])
nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.14'},'title':'DA2141 - Vaidik Maheshwari and Anant Gupta'})
nbformat.write(nb,root/'code/Vaidik_Maheshwari_Anant_Gupta_DA2141.ipynb')
print(len(cells),'cells written')
