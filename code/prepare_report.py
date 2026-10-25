from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import analysis

OUT=ROOT/'latex'
TABLES=OUT/'tables'
FIGURES=ROOT/'graphs'
TABLES.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


def table(name,caption,label,headers,rows,alignment):
    text=['\\begin{table}[htbp]','\\centering','\\caption{'+caption+'}\\label{'+label+'}','\\small','\\begin{tabular}{'+alignment+'}','\\toprule',' & '.join(headers)+r' \\','\\midrule']
    text.extend(' & '.join(str(v) for v in row)+r' \\' for row in rows)
    text.extend(['\\bottomrule','\\end{tabular}','\\end{table}',''])
    (TABLES/name).write_text('\n'.join(text))


def short(method):
    return {'Restricted MLE':'Ordered MLE','Restricted Bayes':'Ordered Bayes'}.get(method,method)


summary=pd.read_csv(ROOT/'data/results/simulation_summary.csv')
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
pd.DataFrame(intervals).to_csv(ROOT/'data/results/interval_comparison.csv',index=False)
boundary=system[(system.regime=='boundary')&(system.method=='Restricted Bayes')]
(TABLES/'hpd_findings.tex').write_text(f"At the equality boundary, ordered equal-tail intervals have coverage {boundary.coverage.mean():.3f}, whereas ordered numerical HPD intervals have coverage {boundary.hpd_coverage.mean():.3f}. Their mean widths are {boundary.average_length.mean():.3f} and {boundary.hpd_average_length.mean():.3f}, respectively. The HPD calculation allows the lower endpoint to equal one half. This explains why its boundary coverage differs from the equal-tail result; it does not establish nominal coverage in every scenario.\n")
boot=pd.read_csv(ROOT/'data/results/restricted_bootstrap_coverage.csv')
rows=[]
for row in boot.itertuples():
    rows.append([row.regime.capitalize(),'Type I' if row.censoring=='typeI' else 'Adaptive II',f'{row.coverage:.3f}',f'{row.coverage_mcse:.3f}',f'{row.average_length:.3f}'])
table('bootstrap.tex','Ordered MLE bootstrap intervals in six selected scenarios: 100 outer datasets and 199 resamples per dataset. MCSE refers to coverage.','tab:bootstrap',['Rate regime','Protocol','CP','MCSE','AW'],rows,'llrrr')
jute=pd.read_csv(ROOT/'data/results/goodness_of_fit.csv')
carbon=pd.read_csv(ROOT/'data/results/carbon_goodness_of_fit.csv')
rows=[]
for dataset,frame in [('Jute',jute),('Carbon',carbon)]:
    for i,row in enumerate(frame.itertuples()):
        name=dataset+(' strength' if i==0 else ' comparison')
        rows.append([name,str(row.n),f'{row.alpha:.3f}',f'{row.rate:.3f}',f'{row.KS:.3f}',f'{row.bootstrap_p:.3f}'])
table('goodness_of_fit.tex','Complete-data Weibull fits and bootstrap-calibrated K-S tests. Carbon rates use the fixed 3 GPa measurement unit.','tab:gof',['Sample','$n$','$\\widehat\\alpha$','Rate','K-S','$p$-value'],rows,'lrrrrr')
import json
jshape=json.loads((ROOT/'data/results/common_shape_test.json').read_text())
cshape=json.loads((ROOT/'data/results/carbon_common_shape_test.json').read_text())
(TABLES/'shape_findings.tex').write_text(f"For the jute groups, the common-shape bootstrap p-value is {jshape['bootstrap_p']:.3f}. The calibrated K-S p-values are {jute.bootstrap_p.iloc[0]:.3f} and {jute.bootstrap_p.iloc[1]:.3f}. Thus the first group rejects Weibull adequacy at 5 percent, even though the common-shape test does not reject. The jute estimates must be read with this fit limitation. The carbon common-shape bootstrap p-value is {cshape['bootstrap_p']:.3f}. Table~\\ref{{tab:gof}} reports its separate fit checks.\n")
rows=[]
for dataset,file in [('Jute','real_estimates.csv'),('Carbon','carbon_estimates.csv')]:
    frame=pd.read_csv(ROOT/'data/results'/file)
    for row in frame.itertuples():
        rows.append([dataset,short(row.method),f'{row.alpha:.3f}',f'{row.lambda_X:.3f}',f'{row.lambda_Y:.3f}',f'{row.R_3_5:.3f}',f'[{row.lower:.3f}, {row.upper:.3f}]'])
table('real_estimates.tex','Parameter and reliability estimates with 95 percent reliability intervals. Interval types are explained in the text.','tab:real-estimates',['Data','Estimator','$\\alpha$','$a$','$b$','$R$','95 percent interval'],rows,'llrrrrl')
rows=[]
for file in ['jute_hpd_intervals.csv','carbon_hpd_intervals.csv']:
    frame=pd.read_csv(ROOT/'data/results'/file)
    for row in frame.itertuples():
        rows.append([row.dataset,row.model,f'[{row.lower:.4f}, {row.upper:.4f}]',f'{row.upper-row.lower:.4f}',f'{row.posterior_mass:.3f}'])
table('real_hpd.tex','Numerical 95 percent HPD intervals for reliability in the data examples. The probability column checks mass under the interpolated posterior density.','tab:real-hpd',['Data','Posterior','HPD interval','Width','Probability'],rows,'lllrr')
for parameter,label,name in [('alpha',r'$\alpha$','shape'),('lambda_X',r'$a$','strength_rate'),('lambda_Y',r'$b$','stress_rate')]:
    frame=summary[summary.parameter==parameter].groupby(['regime','method'])[['absolute_bias','mse','rmse']].mean()
    rows=[]
    for regime in ['ordered','boundary','violated']:
        for method in methods:
            row=frame.loc[(regime,method)]
            rows.append([regime.capitalize(),short(method),f'{row.absolute_bias:.4f}',f'{row.mse:.4f}',f'{row.rmse:.4f}'])
    table('parameter_'+name+'.tex','Point-estimation errors for '+label+', averaged over the 16 scenarios in each rate regime.','tab:'+name,['Regime','Estimator','Absolute bias','MSE','RMSE'],rows,'llrrr')
raw=pd.read_csv(ROOT/'data/results/simulation_raw.csv')
design=pd.read_csv(ROOT/'data/results/simulation_design.csv')
failure=raw[raw.method=='MLE'].merge(design[['scenario','censoring','n_per_block']],on='scenario')
rows=[]
for (protocol,n),group in failure.groupby(['censoring','n_per_block']):
    rows.append(['Type I' if protocol=='typeI' else 'Adaptive II',str(n),str(3*int(.7*n)),f'{group.observed_failures.mean():.2f}',f'{group.observed_failures.std():.2f}',f'{group.time_stopped_blocks.mean():.2f}'])
table('observed_failures.tex','Retained strength information across the simulation. Means pool all rate and shape settings at the indicated size and protocol. The target is the sum across three blocks.','tab:failures',['Protocol','$n$','Target','Mean failures','SD','Time-stopped blocks'],rows,'llrrrr')
rows=[]
parameter_names={'alpha':r'$\alpha$','lambda_X':r'$a$','lambda_Y':r'$b$','R_3_5':r'$R$'}
for row in pd.read_csv(ROOT/'data/results/real_parameter_intervals.csv').itertuples():
    rows.append([parameter_names[row.parameter],short(row.method),f'{row.estimate:.4f}',f'[{row.lower:.4f}, {row.upper:.4f}]'])
table('parameter_intervals.tex','Jute parameter estimates with positive asymptotic intervals for MLE and equal-tail intervals for the Bayesian rows. These intervals refer to the retained-data likelihood.','tab:parameter-intervals',['Parameter','Estimator','Estimate','95 percent interval'],rows,'llrl')
rows=[]
for row in pd.read_csv(ROOT/'data/results/posterior_diagnostics.csv').itertuples():
    rows.append(['Ordered' if row.model=='Restricted' else row.model,parameter_names[row.parameter],f'{row.split_Rhat:.4f}',f'{row.ESS:.0f}',f'{row.mean:.4f}',f'{row.MCSE:.4f}',f'{row.difference:.4f}'])
table('diagnostics.tex','Checks from four retained chains for each model. Difference means sampler mean minus quadrature mean. R-hat is the classical split statistic.','tab:diagnostics',['Model','Target','R-hat','ESS','Mean','MCSE','Difference'],rows,'llrrrrr')
rows=[]
for row in pd.read_csv(ROOT/'data/results/sensitivity.csv').itertuples():
    setting={'typeI':'Type I','adaptiveII':'Adaptive II'}.get(row.setting,row.setting)
    rows.append([row.analysis,setting,'Yes' if row.restricted else 'No',f'{row.R:.4f}',f'[{row.lower:.4f}, {row.upper:.4f}]'])
table('sensitivity.tex','Jute sensitivity calculations. All intervals in this table are posterior equal-tail intervals; each row changes the stated input.','tab:sensitivity',['Change','Setting','Ordered','$R$','95 percent interval'],rows,'lllrl')
rows=[]
for row in pd.read_csv(ROOT/'data/results/predictive_checks.csv').itertuples():
    rows.append([row.statistic,f'{row.observed:.4f}',f'[{row.predictive_lower:.4f}, {row.predictive_upper:.4f}]',f'{row.posterior_predictive_upper_tail:.3f}'])
table('predictive.tex','Posterior predictive summaries from 2000 regenerated censored/record experiments. The last column is a descriptive upper-tail probability.','tab:predictive',['Statistic','Observed','95 percent predictive range','Upper tail'],rows,'lrll')
jute_raw=pd.read_csv(ROOT/'data/raw/jute_fibre.csv')
carbon_raw=pd.read_csv(ROOT/'data/raw/carbon_fibre.csv').carbon_strength_GPa.to_numpy()
rows=[]
for sample,values in [('Jute 15 mm',jute_raw.gauge_15mm.to_numpy()),('Jute 20 mm',jute_raw.gauge_20mm.to_numpy()),('Carbon (GPa)',carbon_raw)]:
    rows.append([sample,str(len(values)),f'{np.mean(values):.4f}',f'{np.std(values,ddof=1):.4f}',f'{np.median(values):.4f}',f'{np.min(values):.4f}',f'{np.max(values):.4f}'])
table('descriptive.tex','Descriptive statistics for all supplied observations used in the applications, before retrospective censoring. Jute values use the source scaling; carbon values in this table are in GPa.','tab:descriptive',['Sample','$n$','Mean','SD','Median','Minimum','Maximum'],rows,'lrrrrrr')
rows=[]
for row in pd.read_csv(ROOT/'data/processed/real_censored_strength.csv').itertuples():
    rows.append([str(row.block),str(row.failure_number),f'{row.failure:.5f}',str(row.withdrawn_at_failure),str(row.terminal_survivors)])
table('jute_retained.tex','All observed jute failures, actual withdrawals and terminal survivor counts. Terminal survivors are counted once per block, not once per displayed row; all blocks stop at time 0.5.','tab:jute-retained',['Block','Failure no.','Failure','Withdrawn','Terminal survivors'],rows,'rrrrr')
records=pd.read_csv(ROOT/'data/processed/real_upper_k_records.csv')
rows=[[str(row.record_number),str(row.k),f'{row.value:.5f}',str(row.source_position)] for row in records.itertuples()]
table('jute_records.tex','The six upper 2-records retained from the jute comparison group, with positions in printed acquisition order.','tab:jute-records',['Record','$k$','Value','Acquisition position'],rows,'rrrr')
rows=[]
for row in pd.read_csv(ROOT/'data/processed/carbon_records.csv').itertuples():
    rows.append([str(row.record_number),f'{row.scaled_value:.5f}',str(row.source_position),str(row.acquisition_position)])
table('carbon_records.tex','The five upper 4-records from the carbon comparison half. Source position refers to the original table; acquisition position refers to the seeded comparison ordering.','tab:carbon-records',['Record','Scaled value','Source position','Acquisition position'],rows,'rrrr')
crng=np.random.default_rng(analysis.SEED+70)
allocation=crng.permutation(100)
cx=carbon_raw[allocation[:50]]/3
cblocks=[]
crows=[]
for number,part in enumerate(np.array_split(np.arange(50),3),1):
    n=len(part)
    block=analysis.censor_block(cx[part],int(.7*n),1.,analysis.removal_plan(n,int(.7*n)),crng)
    cblocks.append(block)
    for j,(value,withdrawn) in enumerate(zip(block['x'],block['R']),1):
        crows.append(dict(block=number,failure_number=j,failure=value,withdrawn_at_failure=int(withdrawn),terminal_time=block['terminal'],terminal_survivors=block['terminal_count']))
pd.DataFrame(crows).to_csv(ROOT/'data/processed/carbon_censored_strength.csv',index=False)
crecords=pd.read_csv(ROOT/'data/processed/carbon_records.csv').scaled_value.to_numpy()
cdata=analysis.make_data(cblocks,crecords,4)
expected=pd.read_csv(ROOT/'data/results/carbon_estimates.csv').query("method=='MLE'")[['alpha','lambda_X','lambda_Y']].iloc[0].to_numpy()
assert np.max(np.abs(analysis.fit_mle(cdata)-expected))<1e-7
rows=[]
for i,block in enumerate(cblocks,1):
    rows.append([str(i),str(block['n']),str(block['m']),str(len(block['x'])),str(int(block['R'].sum())),str(block['terminal_count'])])
table('carbon_sampling.tex','Retrospective carbon strength blocks under Type-I progressive hybrid censoring. The time limit is one in units of 3 GPa. Each row conserves all starting specimens.','tab:carbon-sampling',['Block','$n_i$','$m_i$','Failures','Withdrawn','Terminal'],rows,'rrrrrr')
rows=[[str(i+1),f'{row.gauge_15mm:.5f}',f'{row.gauge_20mm:.5f}'] for i,row in enumerate(jute_raw.itertuples())]
table('jute_complete.tex','All 30 measurements in each jute group, in the source table order and with the source scaling already applied.','tab:jute-complete',['Position','15 mm','20 mm'],rows,'rrr')
rows=[]
for i in range(0,len(carbon_raw),8):
    values=[f'{v:.2f}' for v in carbon_raw[i:i+8]]
    rows.append([f'{i+1}--{min(i+8,len(carbon_raw))}']+values+['']*(8-len(values)))
table('carbon_complete.tex','All 100 carbon-fibre observations in original printed order, in GPa. Each row contains consecutive source positions.','tab:carbon-complete',['Positions']+[str(i) for i in range(1,9)],rows,'lrrrrrrrr')
lines=[r'\begin{longtable}{rrrllrr}',r'\caption{Complete 48-scenario design. Every row has three strength blocks and 300 independent replications. O, B and V mean ordered, boundary and violated.}\label{tab:design}\\',r'\toprule',r'ID & $\alpha$ & $n$ & Protocol & Regime & $k$ & $r$ \\',r'\midrule',r'\endfirsthead',r'\toprule',r'ID & $\alpha$ & $n$ & Protocol & Regime & $k$ & $r$ \\',r'\midrule',r'\endhead']
for row in design.itertuples():
    lines.append(' & '.join([str(row.scenario),f'{row.alpha:.1f}',str(row.n_per_block),'Type I' if row.censoring=='typeI' else 'Adaptive II',{'ordered':'O','boundary':'B','violated':'V'}[row.regime],str(row.record_k),str(row.record_count)])+r' \\')
lines.extend([r'\bottomrule',r'\end{longtable}'])
(TABLES/'scenario_design.tex').write_text('\n'.join(lines)+'\n')
print('All report tables prepared')
verification=pd.read_csv(ROOT/'data/results/reliability_verification.csv')
refine=pd.read_csv(ROOT/'data/results/quadrature_refinement.csv')
real_refine=pd.read_csv(ROOT/'data/results/real_quadrature_refinement.csv')

def scientific(value):
    if value==0:
        return '$0$'
    mantissa,exponent=f'{value:.2e}'.split('e')
    return '$'+mantissa+r'\times10^{'+str(int(exponent))+'}$'

rows=[['Reliability integral versus product',scientific(np.max(np.abs(verification.formula-verification.integral))),'Same theoretical target'],['Direct-system simulation versus product',f'{np.max(np.abs(verification.system_MC-verification.formula)):.5f}','Within the stated Monte Carlo checks'],['Simulation reliability means, finer grid',scientific(refine.R_mean_difference.max()),'One dataset per scenario and support'],['Application means, finer grid',scientific(real_refine.maximum_mean_difference.max()),'Both posterior supports'],['Application equal-tail endpoints',scientific(real_refine.maximum_interval_difference.max()),'Both posterior supports'],['Application HPD endpoints',scientific(real_refine.maximum_hpd_difference.max()),'Both posterior supports'],['HPD posterior probability versus 0.95',scientific(np.max(np.abs(raw.hpd_mass.dropna()-.95))),'All saved Bayesian simulation fits']]
table('numerical_checks.tex','Numerical comparisons used to check the calculation. Probability-mass checks refer to the interpolated density used for numerical HPD intervals.','tab:numerical-checks',['Comparison','Largest difference','Scope'],rows,r'p{.43\linewidth}rp{.30\linewidth}')
order=pd.read_csv(ROOT/'data/results/acquisition_order_sensitivity.csv')
(TABLES/'order_findings.tex').write_text(f"Of 200 attempted permutations, {len(order)} reach the prescribed six upper 2-records and {200-len(order)} do not. Among successful orders, posterior mean reliability ranges from {order.R.min():.4f} to {order.R.max():.4f}, with mean {order.R.mean():.4f} and standard deviation {order.R.std():.4f}. The original printed-order mean is 0.6656.\n")
loss=ROOT/'data/results/bayesian_loss_comparison.csv'
if loss.exists():
    rows=[]
    for row in pd.read_csv(loss).itertuples():
        rows.append([row.model,f'{row.squared_error:.4f}',f'{row.linex_positive:.4f}',f'{row.linex_negative:.4f}'])
    table('loss_comparison.tex','Jute Bayesian reliability estimates under squared error and two LINEX loss choices. The posterior distribution is unchanged across loss choices in a row.','tab:loss',['Posterior','Squared error','LINEX $c=1.5$','LINEX $c=-1.5$'],rows,'lrrr')
frame=summary.groupby(['parameter','regime','method'])[['absolute_bias','mse']].mean()
point_lines=[]
for regime in ['ordered','boundary','violated']:
    u=point.loc[(regime,'MLE')]
    b=point.loc[(regime,'Bayes')]
    o=point.loc[(regime,'Restricted Bayes')]
    point_lines.append(f"For the {regime} rate regime, mean scenario-specific reliability MSE is {u.mse:.4f} for unrestricted MLE, {b.mse:.4f} for unrestricted Bayes and {o.mse:.4f} for ordered Bayes.")
(TABLES/'simulation_findings.tex').write_text('\n\n'.join(point_lines)+'\n')
print('Extended tables and data summaries saved')
