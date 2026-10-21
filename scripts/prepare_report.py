from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import analysis

OUT=ROOT/'manuscript'
TABLES=OUT/'tables'
FIGURES=OUT/'figures'
TABLES.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.size':13,'axes.labelsize':13,'axes.titlesize':13,'legend.fontsize':10,'figure.dpi':130,'savefig.dpi':240,'axes.grid':False})


def table(name,caption,label,headers,rows,alignment):
    text=['\\begin{table}[htbp]','\\centering','\\caption{'+caption+'}\\label{'+label+'}','\\small','\\begin{tabular}{'+alignment+'}','\\toprule',' & '.join(headers)+r' \\','\\midrule']
    text.extend(' & '.join(str(v) for v in row)+r' \\' for row in rows)
    text.extend(['\\bottomrule','\\end{tabular}','\\end{table}',''])
    (TABLES/name).write_text('\n'.join(text))


def short(method):
    return {'Restricted MLE':'Ordered MLE','Restricted Bayes':'Ordered Bayes'}.get(method,method)


def save(name):
    plt.tight_layout()
    plt.savefig(FIGURES/name,bbox_inches='tight',facecolor='white')
    plt.close()


summary=pd.read_csv(ROOT/'results/simulation_summary.csv')
system=summary.query("parameter=='R_3_5'")
point=system.groupby(['regime','method'])[['absolute_bias','mse']].mean()
rows=[]
methods=['MLE','Restricted MLE','Bayes','Restricted Bayes']
for regime in ['ordered','boundary','violated']:
    for method in methods:
        row=point.loc[(regime,method)]
        rows.append([regime.capitalize(),short(method),f'{row.absolute_bias:.4f}',f'{row.mse:.4f}'])
table('simulation_points.tex','Absolute bias and MSE of reliability, averaged over 16 scenarios in each rate regime.','tab:points',['Rate regime','Estimator','Absolute bias','MSE'],rows,'llrr')
rows=[]
intervals=[]
for regime in ['ordered','boundary','violated']:
    for method,label,coverage,width in [('MLE','MLE ACI','coverage','average_length'),('Bayes','Bayes equal-tail','coverage','average_length'),('Bayes','Bayes HPD','hpd_coverage','hpd_average_length'),('Restricted Bayes','Ordered Bayes equal-tail','coverage','average_length'),('Restricted Bayes','Ordered Bayes HPD','hpd_coverage','hpd_average_length')]:
        sub=system[(system.regime==regime)&(system.method==method)]
        cp=float(sub[coverage].mean())
        aw=float(sub[width].mean())
        rows.append([regime.capitalize(),label,f'{cp:.3f}',f'{aw:.3f}'])
        intervals.append(dict(regime=regime,interval=label,coverage=cp,average_width=aw))
table('simulation_intervals.tex','Coverage probability and average width of 95 percent reliability intervals, averaged over 16 scenarios in each regime. HPD denotes the numerical approximation described in Section~\\ref{sec:credible}.','tab:intervals',['Rate regime','Interval','CP','AW'],rows,'llrr')
pd.DataFrame(intervals).to_csv(ROOT/'results/interval_comparison.csv',index=False)
boundary=system[(system.regime=='boundary')&(system.method=='Restricted Bayes')]
(TABLES/'hpd_findings.tex').write_text(f"At the equality boundary, ordered equal-tail intervals have coverage {boundary.coverage.mean():.3f}, whereas ordered numerical HPD intervals have coverage {boundary.hpd_coverage.mean():.3f}. Their mean widths are {boundary.average_length.mean():.3f} and {boundary.hpd_average_length.mean():.3f}, respectively. The HPD calculation allows the lower endpoint to equal one half. This explains why its boundary coverage differs from the equal-tail result; it does not establish nominal coverage in every scenario.\n")
boot=pd.read_csv(ROOT/'results/restricted_bootstrap_coverage.csv')
rows=[]
for row in boot.itertuples():
    rows.append([row.regime.capitalize(),'Type I' if row.censoring=='typeI' else 'Adaptive II',f'{row.coverage:.3f}',f'{row.coverage_mcse:.3f}',f'{row.average_length:.3f}'])
table('bootstrap.tex','Ordered MLE bootstrap intervals in six selected scenarios: 100 outer datasets and 199 resamples per dataset. MCSE refers to coverage.','tab:bootstrap',['Rate regime','Protocol','CP','MCSE','AW'],rows,'llrrr')
jute=pd.read_csv(ROOT/'results/goodness_of_fit.csv')
carbon=pd.read_csv(ROOT/'results/carbon_goodness_of_fit.csv')
rows=[]
for dataset,frame in [('Jute',jute),('Carbon',carbon)]:
    for i,row in enumerate(frame.itertuples()):
        name=dataset+(' strength' if i==0 else ' comparison')
        rows.append([name,str(row.n),f'{row.alpha:.3f}',f'{row.rate:.3f}',f'{row.KS:.3f}',f'{row.bootstrap_p:.3f}'])
table('goodness_of_fit.tex','Complete-data Weibull fits and bootstrap-calibrated K-S tests. Carbon rates use the fixed 3 GPa measurement unit.','tab:gof',['Sample','$n$','$\\widehat\\alpha$','Rate','K-S','$p$-value'],rows,'lrrrrr')
import json
jshape=json.loads((ROOT/'results/common_shape_test.json').read_text())
cshape=json.loads((ROOT/'results/carbon_common_shape_test.json').read_text())
(TABLES/'shape_findings.tex').write_text(f"For the jute groups, the common-shape bootstrap p-value is {jshape['bootstrap_p']:.3f}. The calibrated K-S p-values are {jute.bootstrap_p.iloc[0]:.3f} and {jute.bootstrap_p.iloc[1]:.3f}. Thus the first group rejects Weibull adequacy at 5 percent, even though the common-shape test does not reject. The jute estimates must be read with this fit limitation. The carbon common-shape bootstrap p-value is {cshape['bootstrap_p']:.3f}. Table~\\ref{{tab:gof}} reports its separate fit checks.\n")
rows=[]
for dataset,file in [('Jute','real_estimates.csv'),('Carbon','carbon_estimates.csv')]:
    frame=pd.read_csv(ROOT/'results'/file)
    for row in frame.itertuples():
        rows.append([dataset,short(row.method),f'{row.alpha:.3f}',f'{row.lambda_X:.3f}',f'{row.lambda_Y:.3f}',f'{row.R_3_5:.3f}',f'[{row.lower:.3f}, {row.upper:.3f}]'])
table('real_estimates.tex','Parameter and reliability estimates with 95 percent reliability intervals. Interval types are explained in the text.','tab:real-estimates',['Data','Estimator','$\\alpha$','$a$','$b$','$R$','95 percent interval'],rows,'llrrrrl')
rows=[]
for file in ['jute_hpd_intervals.csv','carbon_hpd_intervals.csv']:
    frame=pd.read_csv(ROOT/'results'/file)
    for row in frame.itertuples():
        rows.append([row.dataset,row.model,f'[{row.lower:.4f}, {row.upper:.4f}]',f'{row.upper-row.lower:.4f}',f'{row.posterior_mass:.3f}'])
table('real_hpd.tex','Numerical 95 percent HPD intervals for reliability in the data examples. The probability column checks mass under the interpolated posterior density.','tab:real-hpd',['Data','Posterior','HPD interval','Width','Probability'],rows,'lllrr')
rho=np.linspace(.01,5,400)
fig,ax=plt.subplots(figsize=(6.2,4))
for s in range(1,6):
    ax.plot(rho,analysis.reliability(np.ones_like(rho),rho,s,5),label=f'{s}-out-of-5',linewidth=2 if s==3 else 1.2)
ax.set(xlabel='Stress / strength rate ratio',ylabel='Reliability',ylim=(0,1))
ax.axvline(1,color='gray',linestyle=':',linewidth=1)
ax.legend()
save('reliability.png')
fig,ax=plt.subplots(figsize=(6.5,4))
x=np.arange(3)
for i,method in enumerate(methods):
    values=[point.loc[(regime,method),'mse'] for regime in ['ordered','boundary','violated']]
    ax.bar(x+(i-1.5)*.18,values,width=.18,label=short(method),color=['#333333','#888888','#385b75','#9eb4c3'][i])
ax.set(xticks=x,xticklabels=['Ordered','Boundary','Violated'],xlabel='Rate regime',ylabel='Mean scenario MSE')
ax.legend()
save('simulation_mse.png')
for dataset,file,columns,unit in [('jute','jute_fibre.csv',['gauge_15mm','gauge_20mm'],1),('carbon','carbon_fibre.csv',['carbon_strength_GPa'],3)]:
    frame=pd.read_csv(ROOT/'data'/file)
    if dataset=='jute':
        samples=[frame[c].to_numpy() for c in columns]
    else:
        values=frame[columns[0]].to_numpy()
        allocation=np.random.default_rng(analysis.SEED+70).permutation(100)
        samples=[values[allocation[:50]]/unit,values[allocation[50:]]/unit]
    fig,axes=plt.subplots(1,2,figsize=(8,3.6))
    for ax,sample,title in zip(axes,samples,['Strength','Comparison stress']):
        alpha,rate,_=analysis.iid_fit(sample)
        probs=(np.arange(1,len(sample)+1)-.5)/len(sample)
        quantile=(-np.log1p(-probs)/rate)**(1/alpha)
        ax.scatter(quantile,np.sort(sample),s=20,color='black')
        end=max(quantile.max(),sample.max())*1.03
        ax.plot([0,end],[0,end],color='gray',linestyle='--')
        ax.set(title=title,xlabel='Fitted Weibull quantile',ylabel='Observed quantile')
    save(dataset+'_qq.png')
real=pd.read_csv(ROOT/'results/real_estimates.csv')
fig,ax=plt.subplots(figsize=(6.4,3.7))
for i,row in enumerate(real.itertuples()):
    ax.plot([row.lower,row.upper],[i,i],color='black',linewidth=1.3)
    ax.scatter(row.R_3_5,i,color='black',s=28)
ax.set(yticks=range(4),yticklabels=[short(x) for x in real.method],xlabel='Reliability',xlim=(.3,1))
ax.axvline(.5,color='gray',linestyle='--')
save('jute_intervals.png')
print('Manuscript tables and figures prepared')
