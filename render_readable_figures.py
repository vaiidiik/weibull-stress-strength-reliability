from pathlib import Path
import json
import nbformat
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import analysis

root=Path.cwd()
out=root/'reports/readable_figures'
out.mkdir(exist_ok=True)
namespace=dict(vars(analysis))
namespace['plt']=plt
namespace['ROOT']=root
manifest={}


def save_figure(name):
    fig=plt.gcf()
    for axis in fig.axes:
        for getter,setter in [(axis.get_xlabel,axis.set_xlabel),(axis.get_ylabel,axis.set_ylabel),(axis.get_title,axis.set_title)]:
            label=getter()
            for old,new in [('lambda_X','Strength rate a'),('lambda_Y','Stress rate b'),('R_3_5','Reliability R'),('alpha','α')]:
                label=label.replace(old,new)
            setter(label)
    for index,axis in enumerate(fig.axes):
        if name=='03_simulation_bias_mse':
            regime=['Ordered','Boundary','Violated'][index%3]
            axis.set_title(regime+' rate order')
            axis.set_xlabel('Starting units per block')
            axis.tick_params(labelbottom=True)
        if name=='09_posterior_traces':
            model='Unrestricted' if index%2==0 else 'Restricted'
            axis.set_title(model+' posterior: '+axis.get_ylabel())
            axis.set_xlabel('Retained iteration / 10')
            for chain,line in enumerate(axis.lines,1):
                line.set_label('Chain '+str(chain))
        if name=='05_observed_information':
            axis.set_title(['Type I retained failures','Adaptive II retained failures'][index])
        if name=='12_carbon_data_fit' and index%3 in [0,1]:
            axis.lines[0].set_label('Fitted Weibull' if index%3==0 else 'Empirical CDF')
            if index%3==1:
                axis.lines[1].set_label('Fitted Weibull CDF')
        if name=='11_predictive_order_sensitivity':
            axis.lines[0].set_label('Observed value' if index<2 else 'Original acquisition order')
        handles,labels=axis.get_legend_handles_labels()
        if handles:
            labels=[label.replace('alpha=','α = ') for label in labels]
            axis.legend(handles,labels,fontsize=9)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    panels=[]
    for index,axis in enumerate(fig.axes,1):
        box=axis.get_tightbbox(renderer).expanded(1.01,1.01)
        box=box.transformed(fig.dpi_scale_trans.inverted())
        visibility=[other.get_visible() for other in fig.axes]
        for other in fig.axes:
            other.set_visible(other is axis)
        target=out/f'{name}_panel_{index:02}.png'
        fig.savefig(target,dpi=240,bbox_inches=box,facecolor='white')
        for other,visible in zip(fig.axes,visibility):
            other.set_visible(visible)
        panels.append({'path':str(target.relative_to(root)),'title':axis.get_title(),'panel':index})
    manifest[name]=panels
    plt.close(fig)


namespace['save_figure']=save_figure
nb=nbformat.read(root/'Vaidik_Maheshwari_Anant_Gupta_DA2141.ipynb',as_version=4)
namespace['design']=pd.read_csv(root/'results/simulation_design.csv')
namespace['simulation_raw']=pd.read_csv(root/'results/simulation_raw.csv')
namespace['simulation']=pd.read_csv(root/'results/simulation_summary.csv')
namespace['Rsummary']=namespace['simulation'].query("parameter=='R_3_5'")
measurements=pd.read_csv(root/'data/jute_fibre.csv')
namespace['x']=measurements.gauge_15mm.to_numpy()
namespace['y']=measurements.gauge_20mm.to_numpy()
blocktable=pd.read_csv(root/'data/real_censored_strength.csv')
blocks=[]
for _,group in blocktable.groupby('block'):
    failures=group[group.failure_number>0].failure.to_numpy()
    R=group[group.failure_number>0].withdrawn_at_failure.to_numpy(int)
    terminal=float(group.terminal_time.iloc[0])
    count=int(group.terminal_survivors.iloc[0])
    blocks.append(dict(x=failures,R=R,terminal=terminal,terminal_count=count,z=np.r_[failures,terminal] if count else failures,w=np.r_[1+R,count] if count else 1+R,n=10,m=7,T=.5,mode='typeI'))
records=pd.read_csv(root/'data/real_upper_k_records.csv')
namespace['blocks']=blocks
namespace['records']=records.value.to_numpy()
namespace['positions']=records.source_position.to_numpy(int)
namespace['real_data']=analysis.make_data(blocks,namespace['records'],2)
namespace['real_estimates']=pd.read_csv(root/'results/real_estimates.csv')
for ordered,label,seed in [(False,'posterior_u',analysis.SEED+50),(True,'posterior_o',analysis.SEED+51)]:
    grid=analysis.posterior_grid(namespace['real_data'],ordered,na=240,np_=320)
    namespace[label]=analysis.posterior_draws(grid,np.random.default_rng(seed),40000)
    namespace['go' if ordered else 'gu']=grid
chains=np.load(root/'results/posterior_chains.npz')
namespace['chain_u']=chains['unrestricted']
namespace['chain_o']=chains['restricted']
carbon=pd.read_csv(root/'data/carbon_fibre.csv').carbon_strength_GPa.to_numpy()
indices=np.random.default_rng(analysis.SEED+70).permutation(100)
namespace['cx']=carbon[indices[:50]]/3
namespace['cy']=carbon[indices[50:]]/3
namespace['predictive']=pd.read_csv(root/'results/posterior_predictive.csv')
namespace['order_results']=pd.read_csv(root/'results/acquisition_order_sensitivity.csv')
for index in [12,17,20,21,23,25,27]:
    source=nb.cells[index].source
    if index not in [12,17]:
        source=source[source.index('fig,axes='):]
    exec(compile(source,f'figure_cell_{index}','exec'),namespace)
(root/'reports/readable_figures/manifest.json').write_text(json.dumps(manifest,indent=2))
print('Readable panels saved:',sum(len(v) for v in manifest.values()))
