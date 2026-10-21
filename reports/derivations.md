# Classical and Bayesian estimation of multicomponent stress-strength reliability

## DA2141 project 3

**Vaidik Maheshwari (250150045)** and **Anant Gupta (250150004)**

**Assigned topic:** Classical and Bayesian estimation of multicomponent stress-strength reliability under block progressive hybrid censoring and k-record for Weibull distribution.

This project derives and implements a common-shape Weibull model, unrestricted and order-restricted maximum likelihood estimates, a dependent gamma-Dirichlet joint prior, unrestricted and order-restricted Bayesian estimates, interval estimation, a 48-scenario Monte Carlo experiment, and a reproducible application to published jute-fibre measurements. All estimates and figures are computed from the stated likelihood. The assigned spreadsheet's row 3 supplies the five mandatory requirements.

## 1. Model, notation and assumptions

There are $K$ independent component strengths $X_1,\ldots,X_K$ and a single stress $Y$ shared by the components. The system works if at least s components exceed Y. All strengths are identically distributed and independent of Y. Strength observations used for estimation come from independent blocks drawn from this same population; a block is a testing facility, not an extra component in the system. We use $K=5$ and $s=3$ for the main experiment. The record order $k$ is different from the number of components $K$.

For a positive variable Z, the rate parameterization of the Weibull distribution is

$$f(z;\alpha,\lambda)=\alpha\lambda z^{\alpha-1}\exp(-\lambda z^\alpha),\qquad F(z)=1-\exp(-\lambda z^\alpha).$$

Thus

$$S(z)=\exp(-\lambda z^\alpha),\quad h(z)=\alpha\lambda z^{\alpha-1},\quad F^{-1}(u)=\left(\frac{-\log(1-u)}{\lambda}\right)^{1/\alpha}.$$

The conventional Weibull scale is $\eta=\lambda^{-1/\alpha}$. Calling $\lambda$ a rate avoids reversing the stochastic ordering. We assume $X\sim\operatorname{Weibull}(\alpha,a)$ and $Y\sim\operatorname{Weibull}(\alpha,b)$, with $\alpha,a,b>0$. Their common shape yields a tractable reliability formula and is checked on the complete application data. Under the increasing transformation $z\mapsto z^\alpha$, strengths become exponential with rate $a$ and stress becomes exponential with rate $b$.

The natural engineering restriction is $a\le b$: strength has larger survival at every positive threshold. Equivalently, $\eta_X\ge\eta_Y$. This restriction is an assumption about populations, not a conclusion automatically established by the data. Simulation deliberately includes $a>b$ to quantify the effect of imposing a false restriction.

## 2. Reliability derivation

Conditional on $Y=y$, the number of components surviving is binomial with success probability $S_X(y)$. The conditional binomial probability must be integrated over the common stress:

$$R_{s,K}=\int_0^\infty\sum_{j=s}^{K}\binom{K}{j}S_X(y)^j[1-S_X(y)]^{K-j}f_Y(y)\,dy.$$

Set $\rho=b/a$ and $u=\exp(-ay^\alpha)$. The transformed stress density satisfies

$$f_Y(y)\,dy=-\rho u^{\rho-1}\,du.$$

$$R_{s,K}=\rho\sum_{j=s}^{K}\binom{K}{j}\int_0^1u^{\rho+j-1}(1-u)^{K-j}\,du=\rho\sum_{j=s}^{K}\binom{K}{j}B(\rho+j,K-j+1).$$

An equivalent and more stable expression follows from exponential order statistics. At least s strengths exceed stress exactly when $Y^\alpha$ is below the $(K-s+1)$-th smallest transformed strength. That order statistic is the sum of independent exponentials with rates $Ka,(K-1)a,\ldots,sa$. Its Laplace transform at $b$ is the product of $ja/(ja+b)$. Therefore

$$R_{s,K}=1-\prod_{j=s}^{K}\frac{j}{j+\rho}.$$

Useful limiting checks are

$$R_{1,1}=\frac{b}{a+b},\qquad R_{K,K}=\frac{b}{Ka+b},$$

$$\lim_{\rho\to0}R_{s,K}=0,\qquad\lim_{\rho\to\infty}R_{s,K}=1.$$ Reliability decreases as $s$ increases. For fixed $a$ and $b$, reliability is independent of the common shape $\alpha$. The shape still affects rate estimates through the likelihood. For $s=3$, $K=5$ and $a=b$,

$$R_{3,5}=1-\frac34\frac45\frac56=\frac12.$$

The common stress makes component success indicators dependent after $Y$ is integrated out:

$$\operatorname{Cov}(I[X_i>Y],I[X_j>Y])=\frac{b}{b+2a}-\left(\frac{b}{b+a}\right)^2>0.$$

Consequently replacing $S_X(y)$ by $P(X>Y)$ in the conditional binomial sum computes a different model. This project uses the shared-stress integral rather than the unconditional binomial expression in equations (2.5)-(2.7) of Newer (2025).

For stable numerical evaluation,

$$R=-\operatorname{expm1}\left(-\sum_{j=s}^{K}\log(1+\rho/j)\right).$$

## 3. Strength sampling: block progressive hybrid censoring

In block $i$, $n_i$ specimens start together. A target $m_i$, threshold $T_i$ and planned withdrawal vector $(R_{i1},\ldots,R_{im_i})$ are specified before testing, with $\sum_jR_{ij}=n_i-m_i$. At observed failure $x_{ij}$, the prescribed number of survivors is selected uniformly for withdrawal. Withdrawals are noninformative and block assignment is random.

The principal Type-I scheme stops at $\tau_i=\min(x_{im_i},T_i)$. If the target is reached before $T_i$, all remaining survivors are withdrawn at the last failure. Otherwise $d_i<m_i$ failures are observed. Using the realized withdrawal counts $r_{ij}$, the number of terminal survivors is

$$q_i=n_i-d_i-\sum_{j=1}^{d_i}r_{ij}.$$ The block likelihood, up to design factors independent of the model parameters, is

$$L_i\propto\prod_{j=1}^{d_i}f_X(x_{ij})S_X(x_{ij})^{r_{ij}}S_X(\tau_i)^{q_i}.$$

Here $q_i=0$ when the target is reached because final survivors are already included in $r_{i,d_i}$. There is no double-counting of final withdrawals. A block with zero failures still contributes its terminal survival factor.

The adaptive Type-II alternative matches the main reference's design. It guarantees $m_i$ failures: intermediate withdrawals at failures after $T_i$ are canceled, and all survivors are withdrawn at the final target failure. Its duration may exceed $T_i$. Its likelihood uses the same expression with $d_i=m_i$, $q_i=0$ and the actual adapted counts $r_{ij}$. The notebook separately implements both protocols and records specimen counts and stopping outcomes.

Multiplying independent block likelihoods gives

$$L_X\propto\alpha^d a^d\exp\left[(\alpha-1)\sum_{i,j}\log x_{ij}-a A(\alpha)\right],$$

$$d=\sum_i d_i,\qquad A(\alpha)=\sum_{i,j}(1+r_{ij})x_{ij}^{\alpha}+\sum_i q_i\tau_i^{\alpha}.$$

Scheduled removals after termination contribute nothing. The specimen-conservation check is $n_i=d_i+\sum_jr_{ij}+q_i$. Exposure is the sum of transformed failure times and transformed right-censoring times for every starting unit.

## 4. Stress sampling: upper k-record values

Starting with the first $k$ independent stress observations, record the $k$-th largest, which is their minimum. Subsequently a record transition occurs whenever a new observation exceeds the current k-th largest; after updating the largest k observations, retain their new minimum. Collect exactly $r$ successive records $y_1<\cdots<y_r$ and stop at the final prescribed record. Original table positions in the application are an artificial acquisition sequence, not historical timestamps.

On the cumulative-hazard scale $H_Y(y)=-\log S_Y(y)$, the successive k-record increments are independent exponential variables with rate $k$. One justification is that the k surviving exceedances above the current threshold have independent exponential residual hazards; their next minimum has rate k. The joint density on the hazard scale is $k^r\exp[-kH_Y(y_r)]$ over the ordered region. Transforming back using $H_Y^{\prime}(y)=h_Y(y)$ gives

$$L_Y=k^r\exp[-kH_Y(y_r)]\prod_{j=1}^{r}h_Y(y_j)=k^r S_Y(y_r)^k\prod_{j=1}^{r}\frac{f_Y(y_j)}{S_Y(y_j)}.$$

For Weibull observations,

$$L_Y\propto\alpha^r b^r\exp\left[(\alpha-1)\sum_{j=1}^{r}\log y_j-b B(\alpha)\right],\qquad B(\alpha)=k y_r^{\alpha}.$$

The records are dependent and are never treated as an independent Weibull sample. Simulation uses

$$y_j=\left(\frac{\sum_{l=1}^{j}E_l}{kb}\right)^{1/\alpha},\qquad E_l\overset{\mathrm{iid}}{\sim}\operatorname{Exp}(1).$$ Record waiting times are not included in this marginal likelihood. The notebook retains acquisition positions for audit only. Including waiting times requires a different joint likelihood; the inter-record expression in the main reference is not copied into this marginal analysis.

## 5. Joint likelihood and maximum likelihood estimates

Define $q=d+r$ and

$$C=\sum_{i,j}\log x_{ij}+\sum_{j=1}^{r}\log y_j.$$ The parameter-dependent log-likelihood is

$$\ell(\alpha,a,b)=q\log\alpha+d\log a+r\log b+(\alpha-1)C-aA(\alpha)-bB(\alpha).$$

The score equations are

$$\ell_a=d/a-A(\alpha),\quad\ell_b=r/b-B(\alpha),\quad\ell_\alpha=q/\alpha+C-aA'(\alpha)-bB'(\alpha).$$

For fixed $\alpha$, the unique positive rate estimates are

$$\widehat a(\alpha)=\frac{d}{A(\alpha)},\qquad\widehat b(\alpha)=\frac{r}{B(\alpha)}.$$ The unrestricted shape estimate maximizes

$$\ell_p(\alpha)=q\log\alpha-d\log A(\alpha)-r\log B(\alpha)+(\alpha-1)C+\text{constant}.$$

The profile score is

$$\ell_p^{\prime}(\alpha)=\frac{q}{\alpha}+C-d\frac{A^{\prime}(\alpha)}{A(\alpha)}-r\frac{B^{\prime}(\alpha)}{B(\alpha)}.$$

Write $A(\alpha)=\sum_lw_l\exp(\alpha\log z_l)$. The second derivative of $\log A$ is the weighted variance of $\log z_l$, with weights proportional to $w_lz_l^\alpha$. Since $\log B$ is linear in $\alpha$,

$$\ell_p''(\alpha)=-q/\alpha^2-d\operatorname{Var}_{w_lz_l^\alpha/A}(\log z_l)<0.$$

Distinct records ensure a finite interior maximum: the score diverges positively near zero and is negative eventually because the largest record is larger than earlier records. This proof assumes $d>0$ and $r\ge2$. If $d=0$, the unrestricted strength-rate supremum can occur at $a=0$; the software reports this instead of presenting a finite interior MLE. All delivered experiments meet the stated conditions.

For fixed $\alpha$ under $a\le b$, either the unrestricted pair satisfies the order or the optimum lies on the boundary. On the boundary $a=b=\lambda$, differentiation yields $\lambda=q/(A+B)$. Therefore

$$ (\widetilde a,\widetilde b)=\begin{cases}(d/A,r/B),&d/A\le r/B,\\(q/(A+B),q/(A+B)),&d/A>r/B.\end{cases}$$

Reoptimize $\alpha$ with this pair at every trial shape. Simply sorting the unrestricted estimates, or pooling at the unrestricted shape without reoptimization, is not the restricted MLE. On the pooled branch, the second derivative is

$$\ell_{p,O}^{\prime\prime}(\alpha)=-\frac{q}{\alpha^2}-q\frac{d^2}{d\alpha^2}\log[A(\alpha)+B(\alpha)]<0.$$

The score agrees across a pooling transition. This gives a unique profile maximum under the same conditions. Reliability estimates follow by invariance:

$$\widehat R=R(\widehat a,\widehat b),\qquad\widetilde R=R(\widetilde a,\widetilde b).$$

## 6. Observed information and intervals

Differentiate exposures directly:

$$A'=\sum_lw_lz_l^\alpha\log z_l,\quad A''=\sum_lw_lz_l^\alpha(\log z_l)^2,\quad B'=B\log y_r,\quad B''=B(\log y_r)^2.$$

The negative Hessian in the parameter order $(\alpha,a,b)$ is

$$I=\begin{pmatrix}q/\alpha^2+aA''+bB''&A'&B'\\A'&d/a^2&0\\B'&0&r/b^2\end{pmatrix}.$$

At an unrestricted MLE, the covariance approximation is

$$V=I(\widehat\alpha,\widehat a,\widehat b)^{-1}.$$

For parameter $\theta_j$, a positive 95% confidence interval is

$$\left[\exp\left(\log\widehat\theta_j-1.96\frac{\sqrt{V_{jj}}}{\widehat\theta_j}\right),\quad\exp\left(\log\widehat\theta_j+1.96\frac{\sqrt{V_{jj}}}{\widehat\theta_j}\right)\right].$$

For the log-rate ratio, define

$$\psi=\log\rho=\log b-\log a,\qquad g=\left(0,-\frac1a,\frac1b\right)^{\mathsf T}.$$

Its estimated standard error and ratio endpoints are

$$\operatorname{SE}(\widehat\psi)=\sqrt{g^{\mathsf T}Vg},$$

$$\rho_{\mathrm L}=\exp[\widehat\psi-1.96\operatorname{SE}(\widehat\psi)],$$

$$\rho_{\mathrm U}=\exp[\widehat\psi+1.96\operatorname{SE}(\widehat\psi)].$$

Transform through the increasing reliability function:

$$[R(1,\rho_{\mathrm L}),\ R(1,\rho_{\mathrm U})].$$

This construction keeps parameter endpoints positive and reliability endpoints between zero and one.


A normal approximation at a pooled restricted MLE is nonregular. The application instead reports percentile parametric bootstrap intervals, regenerating complete strength blocks, withdrawals, hybrid stopping and stress k-records under the fitted model and refitting each replicate. Bootstrap intervals at an order boundary can have point mass at $R=1/2$; they are descriptive finite-sample approximations, not a claim of exact boundary coverage. Their simulation coverage is assessed separately. Bayesian intervals are posterior credible intervals and have a different interpretation from confidence intervals.

## 7. Dependent joint prior and its derivation

Use $t=a+b$ and $p=a/(a+b)$. Independently assign

$$t\sim\operatorname{Gamma}(a_0,b_0),\quad p\sim\operatorname{Beta}(c_1,c_2),\quad\alpha\sim\operatorname{Gamma}(a_1,b_1),$$

where Gamma is shape-rate. The absolute Jacobian from $(t,p)$ to $(a,b)$ is $t$. The rate-pair density is

$$\pi(a,b)=\frac{b_0^{a_0}\Gamma(c_1+c_2)}{\Gamma(a_0)\Gamma(c_1)\Gamma(c_2)}e^{-b_0(a+b)}(a+b)^{a_0-c_1-c_2}a^{c_1-1}b^{c_2-1}.$$

This is the two-dimensional gamma-Dirichlet family used in Sultana et al. It gives independent Gamma rates only if $a_0=c_1+c_2$. The primary choice is

$$(a_0,b_0,c_1,c_2,a_1,b_1)=(2,0.5,2,2,2,1).$$

Since $a_0\ne c_1+c_2$, the rates are dependent. For $c=c_1+c_2$,

$$E(a)=\frac{a_0c_1}{b_0c},\quad E(b)=\frac{a_0c_2}{b_0c},\quad\operatorname{Cov}(a,b)=\frac{a_0c_1c_2(c-a_0)}{b_0^2c^2(c+1)}.$$

Here prior means of both rates are 2, prior mean shape is 2, and covariance is 0.8. These are fixed choices rather than hyperparameters fitted to each simulation's true parameters. They are proper, moderately broad priors, not literally noninformative. Prior sensitivity is reported and the application does not recycle the same data into a prior presented as external information.

For the order-restricted model condition this prior on $p\le1/2$:

$$\pi_O(\alpha,a,b)=\pi(\alpha,a,b)\frac{I(a\le b)}{P(p\le1/2)},\qquad P(p\le1/2)=I_{1/2}(c_1,c_2).$$

With symmetric $c_1=c_2=2$ the normalizing probability is 1/2. This is a truncated dependent joint prior; sorting draws from a posterior with labelled strength and stress roles would target a different posterior.

## 8. Posterior, Bayesian estimators and computation

In the original coordinates the posterior kernel is

$$\pi(\alpha,a,b\mid D)\propto\alpha^{q+a_1-1}e^{(\alpha-1)C-b_1\alpha}a^{d+c_1-1}b^{r+c_2-1}(a+b)^{a_0-c_1-c_2}e^{-a(A+b_0)-b(B+b_0)}.$$

In $(\alpha,t,p)$, including the Jacobian,

$$\pi(\alpha,t,p\mid D)\propto\alpha^{q+a_1-1}e^{(\alpha-1)C-b_1\alpha}p^{d+c_1-1}(1-p)^{r+c_2-1}t^{q+a_0-1}e^{-tH},$$

$$H=b_0+pA(\alpha)+(1-p)B(\alpha).$$

Hence $t\mid\alpha,p,D\sim\operatorname{Gamma}(q+a_0,H)$. Integrating t exactly gives

$$\pi(\alpha,p\mid D)\propto\alpha^{q+a_1-1}e^{(\alpha-1)C-b_1\alpha}p^{d+c_1-1}(1-p)^{r+c_2-1}H^{-(q+a_0)}.$$

The unrestricted support is $\alpha>0$, $0<p<1$; the restricted support is $\alpha>0$, $0<p\le1/2$. Two-dimensional Gauss-Legendre quadrature integrates $\log\alpha$ from $\log(0.02)$ to $\log(30)$ and p on its model support. The $\exp(\log\alpha)$ Jacobian is included, and weights are normalized with log-sum-exp. Tail weights and refinement comparisons check truncation and discretization. Simulation uses 80 by 110 nodes, whereas the application uses 240 by 320 and a denser refinement. These numerical bounds are integration settings; edge-mass checks assess whether they materially truncate the posterior.

The conditional moments are

$$E(t\mid\alpha,p,D)=\frac{q+a_0}{H},$$

$$E(t^2\mid\alpha,p,D)=\frac{(q+a_0)(q+a_0+1)}{H^2}.$$

Consequently, the posterior rate means are

$$E(a\mid D)=E\left[\frac{p(q+a_0)}{H}\,\middle|\,D\right],$$

$$E(b\mid D)=E\left[\frac{(1-p)(q+a_0)}{H}\,\middle|\,D\right].$$

The posterior mean shape follows directly from the normalized quadrature weights.


Under squared error loss, the Bayes reliability estimate is the posterior mean of R itself:

$$\widehat R_B=E[R(p,1-p)\mid D].$$

It is generally different from evaluating R at posterior mean rates. Under LINEX loss

$$L(\delta,R)=\exp[c(\delta-R)]-c(\delta-R)-1,$$

Differentiating posterior expected loss gives

$$\widehat R_{B,c}=-\frac1c\log E[e^{-cR}\mid D],\qquad c\ne0.$$

The notebook reports $c=+1.5$ and $c=-1.5$, equal-tail 95% credible intervals, and numerical HPD intervals for $R$. For HPD calculation, transform the quadrature marginal density of $p$ to the reliability scale using the absolute derivative $|dR/dp|$. A shape-preserving cubic interpolant approximates this density. Its integrated CDF is inverted and the shortest interval of probability 0.95 is found, checking both support endpoints as candidates. For the unimodal marginals in this study this approximates an HPD interval. The order boundary $R=1/2$ can therefore be an endpoint. Probability mass and grid-refinement checks are recorded. HPD intervals and equal-tail intervals are not interchangeable.

For an independent computational check, four Metropolis-Hastings chains sample the collapsed density in $(\log\alpha,\operatorname{logit}v)$, where $p=v$ for the unrestricted model and $p=v/2$ for the restricted model. The transformed target includes the log-shape Jacobian and

$$\log\left(\frac{dp}{dz}\right)=\log p_{\max}+\log v+\log(1-v).$$

Symmetric Gaussian random-walk proposals use a mode-based covariance fixed before sampling. Given each accepted state, t is drawn from its exact conditional Gamma distribution. Each chain uses 15,000 steps with 4,000 discarded warm-up steps; no adaptation occurs in the retained samples. Trace plots, density plots, split R-hat, effective sample sizes, and Monte Carlo standard errors accompany comparisons against quadrature. These diagnostics are classical split variance R-hat and an initial monotone autocorrelation ESS, not rank-normalized diagnostics.

## 9. Simulation design and estimands

The factorial design comprises 3 order regimes x 2 shapes x 2 block sizes x 2 hybrid protocols x 2 record orders = 48 scenarios. Each has 300 independent replications, for 14,400 independently simulated datasets. There are three blocks, $n=20$ or $n=40$ per block, $m=\lfloor0.7n\rfloor$, a balanced withdrawal plan, and T at the true strength distribution's 65th percentile. The threshold is fixed before generating each experiment. Shapes are 0.8 and 1.8; $k$ is 1 or 3; the prescribed record count is 6 for $n=20$ and 12 for $n=40$. This size contrast increases both strength and stress information; it does not isolate the effect of strength sample size alone.

Rate regimes are ordered $(a,b)=(0.8,1.6)$, boundary $(a,b)=(1,1)$, and violated $(a,b)=(1.6,0.8)$. Parameters are fixed, not sampled from priors. All four estimators use exactly the same simulated observations in a replication. Reported targets are $\alpha$, $a$, $b$ and $R_{3,5}$. Restricted estimators in the violated regime are intentionally misspecified and are judged against the true unrestricted target.

For estimates $\widehat\theta_l$ and truth $\theta$, let $e_l=\widehat\theta_l-\theta$. Report

$$\operatorname{Bias}=\frac1M\sum_{l=1}^{M}e_l,\qquad\operatorname{MSE}=\frac1M\sum_{l=1}^{M}e_l^2,$$

$$\operatorname{RMSE}=\sqrt{\operatorname{MSE}},\qquad\operatorname{MCSE}(\operatorname{Bias})=\frac{\operatorname{SD}(e_l)}{\sqrt{M}},$$

$$\operatorname{MCSE}(\operatorname{MSE})=\frac{\operatorname{SD}(e_l^2)}{\sqrt{M}}.$$

For asymptotic, Bayesian equal-tail and numerical HPD intervals, report coverage $\widehat{\operatorname{CP}}$, average length and

$$\operatorname{MCSE}(\widehat{\operatorname{CP}})=\sqrt{\frac{\widehat{\operatorname{CP}}(1-\widehat{\operatorname{CP}})}{M}}.$$ The main coverage table includes unrestricted log-ratio asymptotic intervals and unrestricted/restricted Bayesian equal-tail and HPD intervals. Restricted MLE bootstrap coverage is separately measured in six selected scenarios with 100 outer replications and 199 inner resamples; its greater Monte Carlo uncertainty is displayed. The primary simulation's restricted MLE has no invented regular normal interval.

All raw replications, scenario settings, summary tables, numerical-refinement results and figures are exported. Seeds are fixed, and computational failures and specimen conservation are audited. Conclusions refer to the measured results rather than asserting universal Bayesian superiority.

## 10. Real-data application and scope

The source is Table 10, printed page 3453 (PDF page 19), of Sultana et al. (2026), attributing jute-fibre breaking-strength data to Xia et al. (2009). The provided values have already been divided by 1000. We transcribe all 30 observations at each of the 15 mm and 20 mm gauge lengths in table order. We do not silently repair the printed 70.32383 in the 10 mm group; that group is excluded because its anomalous value needs verification against the original experiment.

The 15 mm population is the strength population and the 20 mm population is the comparison stress population. This is a same-unit laboratory stress-strength comparison of breaking-strength populations. The latter observations are not measured external operating loads, so the reliability is not a certified in-service engineering probability. The order assumption is assessed as a sensitivity case.

Complete data are first checked using separate Weibull fits, histograms and densities, empirical versus fitted CDFs, Q-Q plots, and bootstrap-calibrated K-S goodness-of-fit tests with 1,000 resamples and refitting in every resample. A common-shape likelihood-ratio test uses one degree of freedom for two populations and a 1,000-replicate parametric bootstrap calibration. Failure to reject is model compatibility, not proof of the Weibull model or equal shapes.

A reproducible retrospective design then splits the 15 mm observations randomly into three disjoint blocks of ten, with target seven failures, balanced withdrawals, and $T=0.5$ in the scaled measurement unit. The principal application uses Type-I progressive hybrid censoring; the adaptive Type-II version is also fitted. From the 20 mm values in printed table order we retain the first six upper $k=2$ records, stopping at original position nine. Remaining stress observations are used only in the complete-data fit check. All full and retained datasets and realized withdrawals are saved. The table order is an imposed acquisition order; a random-order sensitivity study quantifies its effect.

The application reports unrestricted/restricted likelihood and posterior estimates, parameter intervals, reliability intervals, 500-replicate parametric bootstraps, prior sensitivity, hybrid-protocol sensitivity, record-count/order sensitivity, posterior diagnostics, and posterior predictive checks. A finite-record sequence contains much less information than the complete stress sample; its uncertainty should be retained in the interpretation.


## 11. Additional empirical validation

Because the calibrated K-S test for the 15 mm jute sample detects a Weibull-fit limitation at 5%, the notebook also analyzes the 100 carbon-fibre strengths printed in Newer and Alanazi (2026), section 5.1.1, PDF pages 12 and 14. Their attributed source is Nichols and Padgett (2006), A Bootstrap Control Chart for Weibull Percentiles, Quality and Reliability Engineering International 22(2), 141-151, DOI 10.1002/qre.691. These measurements are transcribed in GPa and randomly split into disjoint groups of 50 using seed 250150115. Measurements are divided by a fixed 3 GPa unit before inference. Both groups represent breaking strengths; the comparison is a laboratory reference stress rather than operational loads. Separate bootstrap-calibrated Weibull fits and a common-shape test support compatibility. The strength half is partitioned into blocks of 17,17,16, tested under Type-I progressive hybrid censoring with $m=\lfloor0.7n\rfloor$, $T=1$ and balanced withdrawals. The comparison half supplies the first five upper 4-records, which have no ties despite source measurement rounding. All four estimators and their reliability intervals are computed. No hydrological data are combined with material strengths.
