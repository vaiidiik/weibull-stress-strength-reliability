from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import analysis as a

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'data/results'
TABLES=ROOT/'latex/tables'
GRAPHS=ROOT/'graphs'
index=json.loads((RESULTS/'report_graph_index.json').read_text())
plt.rcParams.update({'font.size':11,'axes.grid':True,'grid.alpha':.2})


def table(file,caption,label,headers,rows,alignment):
    lines=[r'\begin{table}[htbp]',r'\centering',r'\caption{'+caption+r'}\label{'+label+'}',r'\small',r'\begin{tabular}{'+alignment+'}',r'\toprule',' & '.join(headers)+r' \\',r'\midrule']
    lines.extend(' & '.join(str(x) for x in row)+r' \\' for row in rows)
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    (TABLES/file).write_text('\n'.join(lines)+'\n')


def save(group,fig):
    fig.tight_layout()
    fig.savefig(GRAPHS/(group+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(GRAPHS/(group+'.pdf'),bbox_inches='tight')
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    parts=[]
    for i,axis in enumerate(fig.axes,1):
        box=axis.get_tightbbox(renderer).expanded(1.01,1.01).transformed(fig.dpi_scale_trans.inverted())
        states=[ax.get_visible() for ax in fig.axes]
        for ax in fig.axes:
            ax.set_visible(ax is axis)
        file=GRAPHS/f'{group}_panel_{i:02}.png'
        fig.savefig(file,dpi=240,bbox_inches=box,facecolor='white')
        for ax,state in zip(fig.axes,states):
            ax.set_visible(state)
        parts.append(dict(path=str(file.relative_to(ROOT)),title=axis.get_title(),panel=i))
    index[group]=parts
    plt.close(fig)


names={'Jute':'Jute','Carbon random-split boundary check':'Split check','Primary carbon 20 mm versus 80 mm':'20 vs 80 mm'}
methods={'MLE ACI':'ACI','MLE Boot-p':'Boot-p','MLE Boot-t (log ratio)':'Boot-t','Ordered MLE Boot-p':'O-Boot-p'}
app=pd.read_csv(RESULTS/'application_bootstrap_intervals.csv')
rows=[[names[r.dataset],methods[r.method],f'{r.estimate:.4f}',f'{r.lower:.4f}',f'{r.upper:.4f}',str(r.bootstrap_B)] for r in app.itertuples()]
table('review_application_bootstrap.tex','Additional application intervals. Boot-t studentizes the unrestricted log-rate ratio and transforms endpoints to reliability. O-Boot-p is ordered MLE percentile bootstrap.','tab:application-bootstrap',['Data','Interval','Estimate','Lower','Upper','$B$'],rows,'llrrrr')
order=pd.read_csv(RESULTS/'order_evidence.csv')
rows=[[names[r.dataset],f'{r.LR:.3f}',f'{r.boundary_bootstrap_p:.3f}',f'{r.posterior_order_probability:.3f}'] for r in order.itertuples()]
table('review_order.tex','Evidence about the rate order from retained data. The likelihood diagnostic uses 999 boundary-bootstrap samples. Posterior probability is calculated under the unrestricted primary prior.','tab:order-evidence',['Data','LR','Boundary $p$',r'$P(a\le b\mid D)$'],rows,'lrrr')
primary=pd.read_csv(RESULTS/'primary_estimates.csv')
rows=[[r.method,f'{r.alpha:.3f}',f'{r.a:.3f}',f'{r.b:.3f}',f'{r.R:.4f}',f'[{r.lower:.4f}, {r.upper:.4f}]'] for r in primary.itertuples()]
table('review_primary_estimates.tex','Retained-data estimates for the distinct 20 mm and 80 mm primary carbon groups. Units are 5 GPa. Bayesian intervals are equal-tail; the ordered MLE interval is percentile bootstrap.','tab:primary-estimates',['Estimator',r'$\alpha$','$a$','$b$','$R$','95 percent interval'],rows,'lrrrrl')
gof=pd.read_csv(RESULTS/'primary_goodness_of_fit.csv')
rows=[[r.sample,str(r.n),f'{r.alpha:.3f}',f'{r.rate:.3f}',f'{r.KS:.4f}',f'{r.bootstrap_p:.3f}'] for r in gof.itertuples()]
table('review_primary_gof.tex','Separate Weibull fits for all 153 valid source rows in each primary gauge group, using the fixed 5 GPa unit. K-S p-values use 1000 refitted bootstrap samples.','tab:primary-gof',['Gauge','$n$',r'$\alpha$','Rate','K-S','$p$'],rows,'lrrrrr')
nonparam=pd.read_csv(RESULTS/'nonparametric_reliability.csv')
rows=[[names[r.dataset],f'{r.estimate:.4f}',f'{r.complete_weibull_estimate:.4f}',f'[{r.lower:.4f}, {r.upper:.4f}]',str(r.strength_n),str(r.stress_n)] for r in nonparam.itertuples()]
table('review_nonparametric.tex','Complete-data nonparametric shared-stress estimates with 2000 two-sample bootstrap resamples. These use all measurements, unlike the retained-data likelihood.','tab:nonparametric',['Data','U-statistic','Weibull','NP 95 percent interval','$n_X$','$n_Y$'],rows,'lrrlrr')
boot=pd.read_csv(RESULTS/'bootstrap_extended_summary.csv')
lines=[r'\begin{longtable}{rllllrrr}',r'\caption{Bootstrap comparison in 12 settings. Every setting uses 500 outer datasets and 399 inner resamples. I and A denote Type I and adaptive Type II. The shape is 1.8 and record order is one.}\label{tab:bootstrap-expanded}\\',r'\toprule',r'ID & Regime & $n$ & Rule & Interval & CP & MCSE & Width \\',r'\midrule',r'\endfirsthead',r'\toprule',r'ID & Regime & $n$ & Rule & Interval & CP & MCSE & Width \\',r'\midrule',r'\endhead']
for row in boot.itertuples():
    lines.append(' & '.join([str(row.scenario),row.regime.capitalize(),str(row.n),'I' if row.protocol=='typeI' else 'A',methods[row.method],f'{row.coverage:.3f}',f'{row.coverage_mcse:.3f}',f'{row.average_width:.3f}'])+r' \\')
lines.extend([r'\bottomrule',r'\end{longtable}'])
(TABLES/'review_bootstrap.tex').write_text('\n'.join(lines)+'\n')
design=pd.read_csv(RESULTS/'design_extensions_summary.csv')
rows=[]
for i,(label,g) in enumerate(design.groupby('setting',sort=False),1):
    values=g.set_index('method')
    protocol='A' if label.endswith('adaptiveII') else 'I'
    if label.startswith('uniform'):
        parts=label.split('_')
        desc=parts[1]
        r=parts[2][1:]
    else:
        desc=label.split('_')[0]
        r='6'
    rows.append([str(i),desc,protocol,r]+[f'{values.loc[m,"mse"]:.4f}' for m in ['MLE','Ordered MLE','Bayes','Ordered Bayes']]+[f'{values.loc[m,"hpd_coverage"]:.3f}' for m in ['Bayes','Ordered Bayes']])
table('review_design.tex','Additional 2-out-of-4 designs, with 500 replications each. Columns U and O denote unrestricted and ordered estimators. CP columns refer to numerical HPD intervals.','tab:design-extensions',['ID','Design','Rule','$r$','U MLE','O MLE','U Bayes','O Bayes','U CP','O CP'],rows,'rlllrrrrrr')
prior=pd.read_csv(RESULTS/'prior_extensions_summary.csv')
rows=[[r.regime.capitalize(),r.prior,'Yes' if r.ordered else 'No',f'{r.bias:.4f}',f'{r.mse:.4f}',f'{r.hpd_coverage:.3f}',f'{r.average_width:.3f}'] for r in prior.itertuples()]
table('review_prior.tex','Prior comparison in 500 paired datasets per regime, with 20 specimens per block and six records. Intervals are numerical HPD intervals.','tab:prior-extensions',['Regime','Prior','Ordered','Bias','MSE','CP','Width'],rows,'lllrrrr')
checks=pd.read_csv(RESULTS/'mcmc_scenario_checks.csv')
checks['standardized']=abs(checks.difference)/checks.MCSE
rows=[]
for (sid,ordered),g in checks.groupby(['scenario','ordered']):
    rows.append([str(sid),'Yes' if ordered else 'No',f'{g.split_Rhat.max():.4f}',f'{g.ESS.min():.0f}',f'{g.standardized.max():.2f}'])
table('review_mcmc.tex','Independent 2-out-of-4 checks in ordered, boundary and violated scenarios (IDs 1, 2 and 3). Four chains retain 8000 iterations each. The last column is the largest absolute mean difference divided by its sampler MCSE.','tab:mcmc-extra',['ID','Ordered','Max R-hat','Min ESS','Max MCSE units'],rows,'rlrrr')
summary=pd.read_csv(RESULTS/'simulation_summary.csv')
aci=summary[(summary.parameter=='R_3_5')&(summary.method=='MLE')]
minimum=aci.loc[aci.coverage.idxmin()]
(TABLES/'review_findings.tex').write_text(f"In the enlarged main study, the smallest unrestricted ACI coverage is {minimum.coverage:.3f} in scenario {int(minimum.scenario)}, with {int(minimum.n_per_block)} specimens per block and MCSE {minimum.coverage_mcse:.4f}. A difference of a few coverage points must still be compared with its Monte Carlo uncertainty.\n")
fig,axes=plt.subplots(1,2,figsize=(11,4))
for method,label,color in [('MLE ACI','ACI','#222222'),('MLE Boot-p','Boot-p','#5887a3'),('MLE Boot-t (log ratio)','Boot-t','#3c9b7d'),('Ordered MLE Boot-p','Ordered Boot-p','#9a6a9e')]:
    g=boot[boot.method==method].groupby('regime')[['coverage','average_width']].mean().reindex(['ordered','boundary','violated'])
    axes[0].plot(range(3),g.coverage,'o-',label=label,color=color)
    axes[1].plot(range(3),g.average_width,'o-',label=label,color=color)
for ax in axes:
    ax.set(xticks=range(3),xticklabels=['Ordered','Boundary','Violated'],xlabel='Rate regime')
    ax.legend(fontsize=9)
axes[0].set(ylabel='Coverage',title='Bootstrap-study coverage',ylim=(-.03,1.03))
axes[0].axhline(.95,color='black',linestyle=':')
axes[1].set(ylabel='Average width',title='Bootstrap-study interval width')
save('14_bootstrap_comparison',fig)
fig,axes=plt.subplots(1,2,figsize=(11,4))
for method,style in [('MLE','-'),('Bayes','--')]:
    for n,color in [(20,'#0f566c'),(40,'#b57835')]:
        vals=[design[(design.setting==f'uniform_n{n}_r{r}_typeI')&(design.method==method)].mse.iloc[0] for r in [6,12]]
        axes[0].plot([6,12],vals,marker='o',linestyle=style,color=color,label=f'{method}, n={n}')
axes[0].set(xlabel='Stress record count',ylabel='MSE for R(2,4)',title='Crossed size / record settings',xticks=[6,12])
axes[0].legend(fontsize=9)
for i,(mode,label) in enumerate([('typeI','Type I'),('adaptiveII','Adaptive II')]):
    settings=[f'uniform_n20_r6_{mode}',f'heterogeneous_{mode}',f'early_{mode}',f'late_{mode}']
    vals=[design[(design.setting==key)&(design.method=='Bayes')].mse.iloc[0] for key in settings]
    axes[1].bar(np.arange(4)+(i-.5)*.35,vals,width=.35,label=label)
axes[1].set(xticks=range(4),xticklabels=['Uniform','Mixed blocks','Early','Late'],ylabel='Unrestricted Bayes MSE',title='Censoring design comparison')
axes[1].tick_params(axis='x',labelsize=9)
axes[1].legend(fontsize=9)
save('15_design_extensions',fig)
f=pd.read_csv(ROOT/'data/raw/primary_carbon_gauges.csv')
fig,axes=plt.subplots(2,3,figsize=(12,7))
for i,gauge in enumerate([20,80]):
    values=f[f.gauge_mm==gauge].strength_GPa.to_numpy()/5
    alpha,rate,_=a.iid_fit(values)
    x=np.linspace(.001,1.3,400)
    values=np.sort(values)
    p=(np.arange(1,len(values)+1)-.5)/len(values)
    axes[i,0].hist(values,bins=12,density=True,alpha=.5)
    axes[i,0].plot(x,alpha*rate*x**(alpha-1)*np.exp(-rate*x**alpha))
    axes[i,0].set(xlabel='Failure strength / 5 GPa',ylabel='Density',title=f'{gauge} mm density')
    axes[i,1].step(values,np.arange(1,len(values)+1)/len(values),where='post',label='Empirical')
    axes[i,1].plot(x,1-np.exp(-rate*x**alpha),label='Fitted Weibull')
    axes[i,1].set(xlabel='Failure strength / 5 GPa',ylabel='Probability',title=f'{gauge} mm CDF')
    axes[i,1].legend(fontsize=8)
    q=(-np.log1p(-p)/rate)**(1/alpha)
    axes[i,2].scatter(q,values,s=10)
    axes[i,2].plot([0,1.3],[0,1.3],'--',color='black')
    axes[i,2].set(xlabel='Fitted quantile',ylabel='Observed quantile',title=f'{gauge} mm Q-Q')
save('16_primary_carbon_fits',fig)
fig,ax=plt.subplots(figsize=(6,3.7))
for i,r in enumerate(primary.itertuples()):
    ax.plot([r.lower,r.upper],[i,i],color='#0f566c')
    ax.scatter(r.R,i,color='#0f566c')
ax.set(yticks=range(4),yticklabels=primary.method,xlabel='Reliability with 95 percent interval',title='Distinct primary gauge groups',xlim=(.45,1))
ax.axvline(.5,color='gray',linestyle=':')
save('17_primary_reliability',fig)
fig,ax=plt.subplots(figsize=(6.2,3.8))
for i,r in enumerate(nonparam.itertuples()):
    ax.plot([r.lower,r.upper],[i,i],color='black')
    ax.scatter(r.estimate,i,color='black')
ax.set(yticks=range(3),yticklabels=[names[x] for x in nonparam.dataset],xlabel='Complete-data reliability',title='Nonparametric shared-stress comparison',xlim=(0,1))
ax.axvline(.5,color='gray',linestyle=':')
save('18_nonparametric_comparison',fig)
(RESULTS/'report_graph_index.json').write_text(json.dumps(index,indent=2)+'\n')
sources=set()


def collect_source(path):
    path=path.resolve()
    if path in sources:
        return
    sources.add(path)
    for name in re.findall(r'\\input\{([^}]+)\}',path.read_text()):
        collect_source(ROOT/'latex'/name)


collect_source(ROOT/'latex/main.tex')
report_text='\n'.join(path.read_text() for path in sources)
report_panels=sorted(set(re.findall(r'\{(\d\d_[^{}]+_panel_\d\d\.png)\}',report_text)))
all_files=[p['path'] for group in index.values() for p in group]
report_files=['graphs/'+name for name in report_panels]
assert set(report_files).issubset(all_files)
coverage=dict(figure_groups=len(index),individual_panels=len(all_files),report_panels=len(report_files),files=all_files,report_files=report_files,supplementary_files=sorted(set(all_files)-set(report_files)),all_report_panels_exist=all((ROOT/file).exists() for file in report_files))
(RESULTS/'report_graph_coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
for folder in ['sections','tables']:
    for path in (ROOT/'latex'/folder).glob('*.tex'):
        if path.resolve() not in sources:
            path.unlink()
print('Added review tables and graphs; retained only manuscript source files')
