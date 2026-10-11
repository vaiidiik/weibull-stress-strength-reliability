from pathlib import Path
import json
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize, special, stats, integrate
from numpy.polynomial.legendre import leggauss
from IPython.display import display, Markdown

ROOT = Path.cwd()
for directory in ['data', 'results', 'figures', 'reports']:
    (ROOT / directory).mkdir(exist_ok=True)
plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 180, 'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.2})
SEED = 250150045
PRIOR = dict(a0=2.0, b0=0.5, c1=2.0, c2=2.0, a1=2.0, b1=1.0)
PARAMETERS = ['alpha', 'lambda_X', 'lambda_Y', 'R_3_5']


def save_figure(name):
    plt.savefig(ROOT / 'figures' / (name + '.png'), bbox_inches='tight')
    plt.savefig(ROOT / 'figures' / (name + '.pdf'), bbox_inches='tight')
    plt.show()
    plt.close()


def reliability(a, b, s=3, K=5):
    rho = np.asarray(b) / np.asarray(a)
    return -np.expm1(sum(-np.log1p(rho / j) for j in range(s, K + 1)))


def removal_plan(n, m, pattern='balanced'):
    plan = np.zeros(m, dtype=int)
    if pattern == 'late':
        plan[-1] = n - m
    else:
        for i in range(n - m):
            plan[i % max(m - 1, 1)] += 1
    return plan


def censor_block(values, m, T, plan, rng, mode='typeI'):
    active = np.arange(len(values))
    failures, removed = [], []
    terminal, terminal_count = 0.0, 0
    case = 'target'
    for j in range(m):
        selected = active[np.argmin(values[active])]
        failure = float(values[selected])
        if mode == 'typeI' and failure > T:
            terminal, terminal_count, case = float(T), len(active), 'time'
            break
        active = active[active != selected]
        failures.append(failure)
        if j == m - 1:
            r = len(active)
        elif mode == 'adaptiveII' and failure > T:
            r = 0
        else:
            r = min(int(plan[j]), len(active))
        if r:
            withdrawn = rng.choice(active, r, replace=False)
            active = active[~np.isin(active, withdrawn)]
        removed.append(r)
        terminal = failure
    x = np.asarray(failures)
    R = np.asarray(removed, dtype=int)
    exposure_x = np.r_[x, terminal] if terminal_count else x.copy()
    exposure_w = np.r_[1 + R, terminal_count] if terminal_count else 1 + R
    assert len(x) + int(R.sum()) + terminal_count == len(values)
    return dict(x=x, R=R, terminal=terminal, terminal_count=terminal_count, z=exposure_x, w=exposure_w, n=len(values), m=m, T=T, mode=mode, case=case)


def make_data(blocks, records, k):
    x = np.concatenate([b['x'] for b in blocks])
    z = np.concatenate([b['z'] for b in blocks])
    w = np.concatenate([b['w'] for b in blocks])
    y = np.asarray(records)
    if len(x) == 0 or len(y) < 2:
        raise ValueError('A finite joint MLE requires strength failures and at least two distinct records.')
    if np.any(z <= 0) or np.any(y <= 0) or np.any(np.diff(y) <= 0):
        raise ValueError('Positive observations and strictly increasing records are required.')
    return dict(blocks=blocks, x=x, z=z, w=w, y=y, k=k, d=len(x), r=len(y), C=np.log(x).sum() + np.log(y).sum(), lx=np.log(z), ly=np.log(y), q=len(x) + len(y))


def exposures(alpha, data):
    vals = data['w'] * np.exp(alpha * data['lx'])
    A = vals.sum()
    B = data['k'] * np.exp(alpha * data['ly'][-1])
    return A, B


def loglik(theta, data):
    alpha, a, b = theta
    if min(theta) <= 0:
        return -np.inf
    A, B = exposures(alpha, data)
    return data['q'] * np.log(alpha) + data['d'] * np.log(a) + data['r'] * np.log(b) + (alpha - 1) * data['C'] - a * A - b * B


def conditional_rates(alpha, data, restricted=False):
    A, B = exposures(alpha, data)
    a, b = data['d'] / A, data['r'] / B
    if restricted and a > b:
        a = b = data['q'] / (A + B)
    return a, b


def fit_mle(data, restricted=False):
    def objective(logalpha):
        alpha = np.exp(logalpha)
        a, b = conditional_rates(alpha, data, restricted)
        return -loglik((alpha, a, b), data)
    fit = optimize.minimize_scalar(objective, bounds=(np.log(0.02), np.log(30)), method='bounded', options={'xatol': 1e-10})
    if not fit.success:
        raise RuntimeError(fit.message)
    alpha = np.exp(fit.x)
    if alpha < .021 or alpha > 29:
        raise RuntimeError('Shape optimizer reached its search boundary.')
    a, b = conditional_rates(alpha, data, restricted)
    return np.array([alpha, a, b])


def observed_information(theta, data):
    alpha, a, b = theta
    v = data['w'] * np.exp(alpha * data['lx'])
    Ap, App = np.sum(v * data['lx']), np.sum(v * data['lx'] ** 2)
    B = data['k'] * np.exp(alpha * data['ly'][-1])
    Bp, Bpp = B * data['ly'][-1], B * data['ly'][-1] ** 2
    return np.array([[data['q'] / alpha ** 2 + a * App + b * Bpp, Ap, Bp], [Ap, data['d'] / a ** 2, 0], [Bp, 0, data['r'] / b ** 2]])


def asymptotic_intervals(theta, data):
    covariance = np.linalg.inv(observed_information(theta, data))
    if np.any(np.linalg.eigvalsh(covariance) <= 0):
        raise RuntimeError('Observed information is not positive definite.')
    parameter_ci = np.exp(np.log(theta)[:, None] + np.array([-1, 1]) * stats.norm.ppf(.975) * np.sqrt(np.diag(covariance))[:, None] / theta[:, None])
    grad = np.array([0., -1 / theta[1], 1 / theta[2]])
    se = np.sqrt(grad @ covariance @ grad)
    rho_ci = np.exp(np.log(theta[2] / theta[1]) + np.array([-1, 1]) * stats.norm.ppf(.975) * se)
    return parameter_ci, reliability(np.ones(2), rho_ci)


def weighted_quantile(values, weights, probs=(.025, .975)):
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    total = w.sum()
    cw = (np.cumsum(w) - 0.5 * w) / total
    return np.interp(probs, cw, v)


def shortest_interval(values, weights, mass=.95):
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    cum = np.r_[0, np.cumsum(w) / np.sum(w)]
    ends = np.searchsorted(cum, cum[:-1] + mass)
    good = ends <= len(v)
    starts = np.arange(len(v))[good]
    ends = np.minimum(ends[good] - 1, len(v) - 1)
    j = np.argmin(v[ends] - v[starts])
    return np.array([v[starts[j]], v[ends[j]]])


_QUADRATURE = {}


def posterior_grid(data, restricted=False, prior=None, na=110, np_=160):
    pr = PRIOR if prior is None else prior
    key = (na, np_, restricted)
    if key not in _QUADRATURE:
        ea, wa = leggauss(na)
        ep, wp = leggauss(np_)
        lo, hi = np.log(.02), np.log(30.)
        eta = lo + (ea + 1) * (hi - lo) / 2
        alpha = np.exp(eta)
        pmax = .5 if restricted else 1.
        p = (ep + 1) * pmax / 2
        logwa = np.log(wa * (hi - lo) / 2) + eta
        logwp = np.log(wp * pmax / 2)
        _QUADRATURE[key] = alpha, p, logwa, logwp
    alpha, p, logwa, logwp = _QUADRATURE[key]
    A = np.exp(alpha[:, None] * data['lx']) @ data['w']
    B = data['k'] * np.exp(alpha * data['ly'][-1])
    H = pr['b0'] + A[:, None] * p + B[:, None] * (1 - p)
    Q = pr['a0'] + data['q']
    logw = ((data['q'] + pr['a1'] - 1) * np.log(alpha) + (alpha - 1) * data['C'] - pr['b1'] * alpha + logwa)[:, None]
    logw = logw + (data['d'] + pr['c1'] - 1) * np.log(p) + (data['r'] + pr['c2'] - 1) * np.log1p(-p) - Q * np.log(H) + logwp
    logw -= special.logsumexp(logw)
    weights = np.exp(logw)
    R = reliability(p, 1 - p)
    means = np.array([np.sum(weights * alpha[:, None]), np.sum(weights * p * Q / H), np.sum(weights * (1 - p) * Q / H), np.sum(weights * R)])
    pweights = weights.sum(axis=0)
    reliability_ci = weighted_quantile(R, pweights)
    hpd = shortest_interval(R, pweights)
    linex = {c: -special.logsumexp(np.log(pweights + 1e-300) - c * R) / c for c in [1.5, -1.5]}
    edge_mass = weights[:2].sum() + weights[-2:].sum()
    return dict(alpha=alpha, p=p, H=H, Q=Q, weights=weights, means=means, ci=reliability_ci, hpd=hpd, linex=linex, edge_mass=edge_mass, prior=pr, restricted=restricted)


def posterior_draws(grid, rng, size=30000):
    selected = rng.choice(grid['weights'].size, size=size, p=grid['weights'].ravel())
    ia, ip = np.unravel_index(selected, grid['weights'].shape)
    t = rng.gamma(grid['Q'], 1 / grid['H'][ia, ip])
    p = grid['p'][ip]
    return np.column_stack([grid['alpha'][ia], p * t, (1 - p) * t, reliability(p, 1 - p)])


def simulate_data(theta, n=20, g=3, m_fraction=.7, T_quantile=.65, k=2, r=8, mode='typeI', pattern='balanced', rng=None):
    rng = np.random.default_rng(SEED) if rng is None else rng
    alpha, a, b = theta
    m = int(n * m_fraction)
    T = (-np.log1p(-T_quantile) / a) ** (1 / alpha)
    blocks = []
    for _ in range(g):
        values = (rng.exponential(size=n) / a) ** (1 / alpha)
        blocks.append(censor_block(values, m, T, removal_plan(n, m, pattern), rng, mode))
    records = (np.cumsum(rng.exponential(scale=1 / (k * b), size=r))) ** (1 / alpha)
    return make_data(blocks, records, k)


def bootstrap_reliability(theta, template, rng, B=500, restricted=False):
    values = []
    failure_count = 0
    for _ in range(B):
        alpha, a, b = theta
        blocks = []
        for block in template['blocks']:
            raw = (rng.exponential(size=block['n']) / a) ** (1 / alpha)
            blocks.append(censor_block(raw, block['m'], block['T'], removal_plan(block['n'], block['m'], block.get('pattern', 'balanced')), rng, block['mode']))
        y = (np.cumsum(rng.exponential(scale=1 / (template['k'] * b), size=template['r']))) ** (1 / alpha)
        try:
            estimate = fit_mle(make_data(blocks, y, template['k']), restricted)
            values.append(reliability(*estimate[1:]))
        except (ValueError, RuntimeError):
            failure_count += 1
    if len(values) < .95 * B:
        raise RuntimeError('Too many bootstrap fits failed.')
    return np.asarray(values), failure_count


def run_simulation(repetitions=300):
    records = []
    scenarios = []
    sid = 0
    for regime, a, b in [('ordered', .8, 1.6), ('boundary', 1., 1.), ('violated', 1.6, .8)]:
        for alpha in [.8, 1.8]:
            for n in [20, 40]:
                for mode in ['typeI', 'adaptiveII']:
                    for k in [1, 3]:
                        sid += 1
                        truth = np.array([alpha, a, b, reliability(a, b)])
                        r = 6 if n == 20 else 12
                        scenarios.append(dict(scenario=sid, regime=regime, alpha=alpha, lambda_X=a, lambda_Y=b, n_per_block=n, blocks=3, m_per_block=int(.7*n), T_quantile=.65, record_k=k, record_count=r, censoring=mode, truth_R=truth[-1], repetitions=repetitions))
                        rng = np.random.default_rng(SEED + sid * 10000)
                        for rep in range(repetitions):
                            data = simulate_data(truth[:3], n=n, k=k, r=r, mode=mode, rng=rng)
                            unrestricted = fit_mle(data)
                            restricted = fit_mle(data, True)
                            grid_u = posterior_grid(data, na=80, np_=110)
                            grid_o = posterior_grid(data, True, na=80, np_=110)
                            _, aci = asymptotic_intervals(unrestricted, data)
                            fits = [('MLE', np.r_[unrestricted, reliability(*unrestricted[1:])], aci), ('Restricted MLE', np.r_[restricted, reliability(*restricted[1:])], None), ('Bayes', grid_u['means'], grid_u['ci']), ('Restricted Bayes', grid_o['means'], grid_o['ci'])]
                            for method, est, ci in fits:
                                row = dict(scenario=sid, replicate=rep+1, method=method, observed_failures=data['d'], time_stopped_blocks=sum(x['case']=='time' for x in data['blocks']), pooled=bool(np.isclose(restricted[1], restricted[2])), alpha=est[0], lambda_X=est[1], lambda_Y=est[2], R_3_5=est[3], lower=np.nan if ci is None else ci[0], upper=np.nan if ci is None else ci[1], covered=np.nan if ci is None else float(ci[0] <= truth[-1] <= ci[1]), grid_edge_mass=max(grid_u['edge_mass'],grid_o['edge_mass']))
                                records.append(row)
                        if sid % 4 == 0:
                            print(f'Completed {sid}/48 scenarios', flush=True)
    raw = pd.DataFrame(records)
    design = pd.DataFrame(scenarios)
    raw.to_csv(ROOT/'results/simulation_raw.csv', index=False)
    design.to_csv(ROOT/'results/simulation_design.csv', index=False)
    summary = []
    for (scenario, method), group in raw.groupby(['scenario','method'],sort=False):
        truth = design.loc[design.scenario==scenario, ['alpha','lambda_X','lambda_Y','truth_R']].iloc[0].to_numpy(float)
        for j, parameter in enumerate(PARAMETERS):
            err = group[parameter].to_numpy() - truth[j]
            row = dict(scenario=scenario,method=method,parameter=parameter,truth=truth[j],mean=group[parameter].mean(),bias=err.mean(),absolute_bias=abs(err.mean()),mse=np.mean(err**2),rmse=np.sqrt(np.mean(err**2)),bias_mcse=err.std(ddof=1)/np.sqrt(len(err)),mse_mcse=(err**2).std(ddof=1)/np.sqrt(len(err)))
            if parameter=='R_3_5' and method!='Restricted MLE':
                cp = group.covered.mean()
                row.update(coverage=cp, coverage_mcse=np.sqrt(cp*(1-cp)/len(group)),average_length=(group.upper-group.lower).mean())
            summary.append(row)
    summary = pd.DataFrame(summary).merge(design, on='scenario')
    summary.to_csv(ROOT/'results/simulation_summary.csv',index=False)
    return design, raw, summary


def ks_stat(sample, alpha, rate):
    z = np.sort(sample)
    f = -np.expm1(-rate * z ** alpha)
    n = len(z)
    return max(np.max(np.arange(1,n+1)/n-f), np.max(f-np.arange(n)/n))


def iid_fit(sample):
    def objective(logalpha):
        alpha = np.exp(logalpha)
        rate = len(sample)/np.sum(sample**alpha)
        return -(len(sample)*np.log(alpha*rate)+(alpha-1)*np.log(sample).sum()-rate*np.sum(sample**alpha))
    opt = optimize.minimize_scalar(objective,bounds=(np.log(.02),np.log(30)),method='bounded')
    alpha = np.exp(opt.x)
    return alpha, len(sample)/np.sum(sample**alpha), -opt.fun


def goodness_of_fit(x, y, rng, B=1000):
    rows=[]
    for label,sample in [('15 mm',x),('20 mm',y)]:
        alpha,rate,ll = iid_fit(sample)
        observed=ks_stat(sample,alpha,rate)
        bs=[]
        for _ in range(B):
            simulated=(rng.exponential(size=len(sample))/rate)**(1/alpha)
            ash,rsh,_=iid_fit(simulated)
            bs.append(ks_stat(simulated,ash,rsh))
        rows.append(dict(sample=label,n=len(sample),alpha=alpha,rate=rate,KS=observed,bootstrap_p=(1+np.sum(np.array(bs)>=observed))/(B+1),bootstrap_B=B,AIC=4-2*ll))
    n1,n2=len(x),len(y)
    def pooled(logalpha):
        alpha=np.exp(logalpha)
        a=n1/np.sum(x**alpha)
        b=n2/np.sum(y**alpha)
        return -(n1*np.log(alpha*a)+n2*np.log(alpha*b)+(alpha-1)*(np.log(x).sum()+np.log(y).sum())-n1-n2)
    opt=optimize.minimize_scalar(pooled,bounds=(np.log(.02),np.log(30)),method='bounded')
    common=np.exp(opt.x)
    LR=2*(sum(iid_fit(z)[2] for z in [x,y])+opt.fun)
    lrstats=[]
    a=n1/np.sum(x**common)
    b=n2/np.sum(y**common)
    for _ in range(B):
        xx=(rng.exponential(size=n1)/a)**(1/common)
        yy=(rng.exponential(size=n2)/b)**(1/common)
        def null_obj(z):
            ash=np.exp(z)
            aa=n1/np.sum(xx**ash)
            bb=n2/np.sum(yy**ash)
            return -(n1*np.log(ash*aa)+n2*np.log(ash*bb)+(ash-1)*(np.log(xx).sum()+np.log(yy).sum())-n1-n2)
        null=optimize.minimize_scalar(null_obj,bounds=(np.log(.02),np.log(30)),method='bounded')
        lrstats.append(2*(iid_fit(xx)[2]+iid_fit(yy)[2]+null.fun))
    test=dict(LR=LR,df=1,asymptotic_p=stats.chi2.sf(LR,1),bootstrap_p=(1+np.sum(np.array(lrstats)>=LR))/(B+1),common_shape=common,lambda_X=a,lambda_Y=b,B=B)
    return pd.DataFrame(rows),test


def extract_records(stream,k,r):
    top=sorted(stream[:k])
    records=[top[0]]
    positions=[k]
    for pos,value in enumerate(stream[k:],k+1):
        if value>top[0]:
            top=sorted(top[1:]+[value])
            records.append(top[0])
            positions.append(pos)
        if len(records)==r:
            break
    if len(records)<r:
        raise ValueError('The prescribed record count was not reached.')
    return np.asarray(records),np.asarray(positions)


def collapsed_logposterior(z,data,restricted=False,prior=None):
    pr=PRIOR if prior is None else prior
    alpha=np.exp(z[0])
    if alpha<.005 or alpha>100:
        return -np.inf
    pmax=.5 if restricted else 1.
    v=special.expit(z[1])
    p=pmax*v
    A,B=exposures(alpha,data)
    H=pr['b0']+p*A+(1-p)*B
    value=(data['q']+pr['a1']-1)*np.log(alpha)+(alpha-1)*data['C']-pr['b1']*alpha+(data['d']+pr['c1']-1)*np.log(p)+(data['r']+pr['c2']-1)*np.log1p(-p)-(data['q']+pr['a0'])*np.log(H)
    return value+z[0]+np.log(pmax)+special.log_expit(z[1])+special.log_expit(-z[1])


def mcmc(data,restricted=False,chains=4,steps=15000,burn=4000):
    outputs=[]
    acceptance=[]
    mode=optimize.minimize(lambda z:-collapsed_logposterior(z,data,restricted),[np.log(1.5),-.5],method='BFGS')
    base=mode.x
    eps=1e-4
    hess=np.empty((2,2))
    f=lambda z:-collapsed_logposterior(z,data,restricted)
    for i in range(2):
        for j in range(2):
            ei=np.eye(2)[i]*eps
            ej=np.eye(2)[j]*eps
            hess[i,j]=(f(base+ei+ej)-f(base+ei-ej)-f(base-ei+ej)+f(base-ei-ej))/(4*eps**2)
    cov=np.linalg.inv(hess)
    L=np.linalg.cholesky(cov)*1.5
    for chain in range(chains):
        rng=np.random.default_rng(SEED+8000+100*restricted+chain)
        current=base+rng.normal(size=2)*.4
        lp=collapsed_logposterior(current,data,restricted)
        draws=[]
        accepts=0
        for step in range(steps):
            proposal=current+L@rng.normal(size=2)
            lp_new=collapsed_logposterior(proposal,data,restricted)
            if np.log(rng.random())<lp_new-lp:
                current,lp=proposal,lp_new
                if step>=burn:
                    accepts+=1
            if step>=burn:
                alpha=np.exp(current[0])
                p=(.5 if restricted else 1.)*special.expit(current[1])
                A,B=exposures(alpha,data)
                t=rng.gamma(PRIOR['a0']+data['q'],1/(PRIOR['b0']+p*A+(1-p)*B))
                draws.append([alpha,p*t,(1-p)*t,reliability(p,1-p)])
        outputs.append(np.asarray(draws))
        acceptance.append(accepts/(steps-burn))
    return np.array(outputs),np.array(acceptance)


def chain_diagnostics(chains):
    m,n,p=chains.shape
    half=n//2
    split=np.concatenate([chains[:,:half],chains[:,-half:]],axis=0)
    mm,nn,_=split.shape
    W=split.var(axis=1,ddof=1).mean(axis=0)
    B=nn*split.mean(axis=1).var(axis=0,ddof=1)
    varplus=(nn-1)/nn*W+B/nn
    rhat=np.sqrt(varplus/W)
    ess=[]
    for j in range(p):
        xx=split[:,:,j]-split[:,:,j].mean(axis=1,keepdims=True)
        size=2**int(np.ceil(np.log2(2*nn)))
        fft=np.fft.rfft(xx,n=size,axis=1)
        autocov=np.fft.irfft(fft*np.conj(fft),n=size,axis=1)[:,:nn]/nn
        rho=1-(W[j]-autocov.mean(axis=0))/varplus[j]
        pairs=[]
        for lag in range(0,nn-1,2):
            pair=rho[lag]+rho[lag+1]
            if pair<0:
                break
            pairs.append(pair if not pairs else min(pair,pairs[-1]))
        tau=max(-1+2*sum(pairs),1.)
        ess.append(mm*nn/tau)
    return pd.DataFrame({'parameter':PARAMETERS,'split_Rhat':rhat,'ESS':ess,'mean':chains.mean(axis=(0,1)),'MCSE':chains.std(axis=(0,1),ddof=1)/np.sqrt(ess)})
