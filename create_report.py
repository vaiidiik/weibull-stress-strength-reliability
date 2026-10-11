from pathlib import Path
from io import BytesIO
import json
import re
import html
import hashlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.mathtext import MathTextParser
from matplotlib.font_manager import FontProperties
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether, Flowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

root=Path.cwd()
assets=root/'reports/typeset_assets'
assets.mkdir(exist_ok=True)
fonts=Path('/System/Library/Fonts/Supplemental')
for name,file in [('ProjectSans','Arial.ttf'),('ProjectBold','Arial Bold.ttf'),('ProjectItalic','Arial Italic.ttf'),('ProjectBoldItalic','Arial Bold Italic.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(fonts/file)))
pdfmetrics.registerFontFamily('ProjectSans',normal='ProjectSans',bold='ProjectBold',italic='ProjectItalic',boldItalic='ProjectBoldItalic')
regular='ProjectSans'
bold='ProjectBold'
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyProject',fontName=regular,fontSize=11.5,leading=17,spaceAfter=11,textColor=colors.HexColor('#253746'),splitLongWords=True))
styles.add(ParagraphStyle(name='TitleProject',fontName=bold,fontSize=25,leading=31,spaceAfter=22,textColor=colors.HexColor('#0f566c')))
styles.add(ParagraphStyle(name='HeadingProject',fontName=bold,fontSize=17,leading=23,spaceAfter=15,textColor=colors.HexColor('#0f566c'),keepWithNext=True))
styles.add(ParagraphStyle(name='SmallProject',fontName=regular,fontSize=10.5,leading=14,spaceAfter=9))
styles.add(ParagraphStyle(name='CaptionProject',fontName=regular,fontSize=10.5,leading=14,spaceAfter=15,textColor=colors.HexColor('#465a65')))
story=[]
math_parser=MathTextParser('path')
math_fonts={}
figure_manifest=json.loads((root/'reports/readable_figures/manifest.json').read_text())


def normalize_math(value):
    value=value.replace(r'\middle|',r'\mid').replace(r'\overset{\mathrm{iid}}{\sim}',r'\sim').replace(r'\text{constant}',r'\mathrm{constant}')
    value=re.sub(r'\\(le|ge|ne)(?![A-Za-z])',lambda m:{'le':r'\leq','ge':r'\geq','ne':r'\neq'}[m[1]],value)
    value=re.sub(r'\\frac([0-9])([0-9A-Za-z])',r'\\frac{\1}{\2}',value)
    value=re.sub(r'\\mathrm\s+([A-Za-z])',r'\\mathrm{\1}',value)
    return value.replace(r'\mathsf T',r'\mathsf{T}')


class MathEquation(Flowable):
    def __init__(self,value,size=14):
        super().__init__()
        self.value=normalize_math(value.strip().rstrip('.,'))
        self.parsed=math_parser.parse('$'+self.value+'$',dpi=72,prop=FontProperties(size=size))
        if self.parsed.width>500 and size>12:
            size=max(12,size*500/self.parsed.width)
            self.parsed=math_parser.parse('$'+self.value+'$',dpi=72,prop=FontProperties(size=size))
        if self.parsed.width>505:
            raise ValueError('Equation needs an explicit line break: '+self.value)
        self.width=float(self.parsed.width)
        self.height=float(self.parsed.height)+18
        self.hAlign='CENTER'
        self.spaceBefore=3
        self.spaceAfter=10
        for font,fontsize,num,x,y in self.parsed.glyphs:
            file=font.fname
            if file not in math_fonts:
                name='MathFont'+str(len(math_fonts))
                pdfmetrics.registerFont(TTFont(name,file))
                math_fonts[file]=name

    def draw(self):
        canvas=self.canv
        canvas.setFillColor(colors.HexColor('#162b39'))
        baseline=float(self.parsed.depth)+8
        for font,fontsize,num,x,y in self.parsed.glyphs:
            canvas.setFont(math_fonts[font.fname],fontsize)
            canvas.drawString(float(x),float(y)+baseline,chr(num))
        for x,y,w,h in self.parsed.rects:
            canvas.rect(float(x),float(y)+baseline,float(w),float(h),stroke=0,fill=1)


def inline_math(value,size=11.5):
    value=normalize_math(value)
    parsed=math_parser.parse('$'+value+'$',dpi=72,prop=FontProperties(size=size))
    key=hashlib.sha256((value+str(size)).encode()).hexdigest()[:20]
    file=assets/f'inline_{key}.png'
    if not file.exists():
        buffer=BytesIO()
        matplotlib.mathtext.math_to_image('$'+value+'$',buffer,prop=FontProperties(size=size),dpi=300,format='png',color='#253746')
        file.write_bytes(buffer.getvalue())
    with __import__('PIL.Image',fromlist=['Image']).open(file) as im:
        w,h=im.size
    width=w*72/300
    height=h*72/300
    return f'<img src="{file}" width="{width:.3f}" height="{height:.3f}" valign="{-float(parsed.depth):.3f}"/>'


def p(text,style='BodyProject'):
    text=re.sub(r'\$([^$]+)\$',lambda m:inline_math(m[1],styles[style].fontSize),text)
    for old,new in [('a<=b','a ≤ b'),('scale>=stress','scale ≥ stress'),('R>=0.5','R ≥ 0.5')]:
        text=text.replace(old,new)
    text=re.sub(r'\b(alpha|lambda|rho)\b',lambda m:{'alpha':'α','lambda':'λ','rho':'ρ'}[m[0]],text)
    text=re.sub(r'\b([abc])([012])\b',r'\1<sub>\2</sub>',text)
    text=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',text)
    story.append(Paragraph(text,styles[style]))


def section(title):
    if story and not isinstance(story[-1],PageBreak):
        story.append(PageBreak())
    p(title,'HeadingProject')


def equation(value):
    value=normalize_math(value.strip().rstrip('.,'))
    if r'\begin{pmatrix}' in value:
        cells=[r'q/\alpha^2+aA^{\prime\prime}+bB^{\prime\prime}',r'A^\prime',r'B^\prime',r'A^\prime',r'd/a^2',r'0',r'B^\prime',r'0',r'r/b^2']
        grid=Table([[MathEquation(cells[i*3+j],size=13) for j in range(3)] for i in range(3)],colWidths=[260,90,90],hAlign='CENTER')
        grid.setStyle(TableStyle([('BOX',(0,0),(-1,-1),.7,colors.HexColor('#0f566c')),('VALIGN',(0,0),(-1,-1),'MIDDLE')]))
        p('Observed-information matrix in the parameter order $(α,a,b)$:')
        story.append(grid);story.append(Spacer(1,12))
        return
    if r'\begin{cases}' in value:
        equation(r'(\widetilde a,\widetilde b)=(d/A,\ r/B)\quad\mathrm{when}\quad d/A\le r/B')
        equation(r'\widetilde a=\widetilde b=q/(A+B)\quad\mathrm{when}\quad d/A>r/B')
        return
    replacements={
        r'f(z;\alpha,\lambda)=\alpha\lambda z^{\alpha-1}\exp(-\lambda z^\alpha),\qquad F(z)=1-\exp(-\lambda z^\alpha).':[r'f(z;\alpha,\lambda)=\alpha\lambda z^{\alpha-1}e^{-\lambda z^\alpha}',r'F(z)=1-e^{-\lambda z^\alpha}'],
    }
    if value+'.' in replacements:
        for part in replacements[value+'.']:
            equation(part)
        return
    if value.startswith(r'S(z)='):
        for part in [r'S(z)=e^{-\lambda z^\alpha},\qquad h(z)=\alpha\lambda z^{\alpha-1}',r'F^{-1}(u)=\left[\frac{-\log(1-u)}{\lambda}\right]^{1/\alpha}']:
            story.append(MathEquation(part))
        return
    if value.startswith(r'R_{s,K}=\rho\sum') and r'\,du=' in value:
        story.append(MathEquation(r'R_{s,K}=\rho\sum_{j=s}^{K}\binom{K}{j}\int_0^1u^{\rho+j-1}(1-u)^{K-j}\,du'))
        story.append(MathEquation(r'R_{s,K}=\rho\sum_{j=s}^{K}\binom{K}{j}B(\rho+j,K-j+1)'))
        return
    if value.startswith(r'\ell_a='):
        story.append(MathEquation(r'\ell_a=d/a-A(\alpha),\qquad\ell_b=r/b-B(\alpha)'))
        equation(r'\ell_\alpha=q/\alpha+C-aA^{\prime}(\alpha)-bB^{\prime}(\alpha)');return
    if value.startswith(r'A\prime=') or value.startswith("A'="):
        equation(r'A^{\prime}=\sum_lw_lz_l^\alpha\log z_l')
        equation(r'A^{\prime\prime}=\sum_lw_lz_l^\alpha(\log z_l)^2')
        equation(r'B^{\prime}=B\log y_r,\qquad B^{\prime\prime}=B(\log y_r)^2');return
    if value.startswith(r'\left[\exp\left(\log\widehat\theta_j'):
        equation(r'\theta_{j,\mathrm L}=\exp\left[\log\widehat\theta_j-1.96\frac{\sqrt{V_{jj}}}{\widehat\theta_j}\right]')
        equation(r'\theta_{j,\mathrm U}=\exp\left[\log\widehat\theta_j+1.96\frac{\sqrt{V_{jj}}}{\widehat\theta_j}\right]');return
    if value.startswith(r'\pi(a,b)='):
        equation(r'C_0=\frac{b_0^{a_0}\Gamma(c_1+c_2)}{\Gamma(a_0)\Gamma(c_1)\Gamma(c_2)}')
        story.append(MathEquation(r'\pi(a,b)=C_0e^{-b_0(a+b)}(a+b)^{a_0-c_1-c_2}a^{c_1-1}b^{c_2-1}'));return
    if value.startswith(r'\pi(\alpha,a,b\mid D)'):
        story.append(MathEquation(r'\pi(\alpha,a,b\mid D)\propto\alpha^{q+a_1-1}e^{(\alpha-1)C-b_1\alpha}'))
        equation(r'\quad\times a^{d+c_1-1}b^{r+c_2-1}(a+b)^{a_0-c_1-c_2}')
        equation(r'\quad\times e^{-a(A+b_0)-b(B+b_0)}');return
    if value.startswith(r'\pi(\alpha,t,p\mid D)'):
        story.append(MathEquation(r'\pi(\alpha,t,p\mid D)\propto\alpha^{q+a_1-1}e^{(\alpha-1)C-b_1\alpha}'))
        equation(r'\quad\times p^{d+c_1-1}(1-p)^{r+c_2-1}t^{q+a_0-1}e^{-tH}');return
    if value.startswith(r'\pi(\alpha,p\mid D)') and 'H^' in value:
        equation(r'\pi(\alpha,p\mid D)\propto\alpha^{q+a_1-1}e^{(\alpha-1)C-b_1\alpha}')
        equation(r'\quad\times p^{d+c_1-1}(1-p)^{r+c_2-1}H^{-(q+a_0)}');return
    if value.startswith(r'L_Y\propto'):
        story.append(MathEquation(value.split(r',\qquad')[0]));equation(r'B(\alpha)=ky_r^\alpha');return
    story.append(MathEquation(value))


def markdown_block(text):
    pieces=re.split(r'(\$\$[\s\S]*?\$\$)',text)
    for piece in pieces:
        if piece.startswith('$$'):
            equation(piece[2:-2])
        else:
            for paragraph in piece.strip().split('\n\n'):
                if paragraph.strip():
                    p(paragraph.replace('\n',' '))


def figure(name,caption,width=490):
    panels=figure_manifest[name]
    heading=None
    if story and isinstance(story[-1],Paragraph) and story[-1].style.name=='HeadingProject':
        heading=story.pop()
    if story and not isinstance(story[-1],PageBreak):
        story.append(PageBreak())
    for i in range(0,len(panels),2):
        if i:
            story.append(PageBreak())
        if i==0 and heading is not None:
            story.append(heading)
            p(caption,'CaptionProject')
        else:
            p(caption,'HeadingProject')
        for panel in panels[i:i+2]:
            reader=ImageReader(str(root/panel['path']));w,h=reader.getSize()
            displayed_width=min(495,290*w/h)
            image=Image(str(root/panel['path']),width=displayed_width,height=displayed_width*h/w)
            story.append(image)
            p(panel['title'] or f'Panel {panel["panel"]}','CaptionProject')



def table(frame,widths=None):
    labels={'method':'Estimator','regime':'Rate order','bias':'Bias','mse':'MSE','coverage':'Coverage','sample':'Sample','n':'Size','alpha':'Shape α','rate':'Rate','KS':'K-S','bootstrap_p':'Bootstrap p','lambda_X':'Strength rate a','lambda_Y':'Stress rate b','R_3_5':'Reliability','lower':'95% lower','upper':'95% upper','model':'Model','parameter':'Parameter','split_Rhat':'Split R-hat','ESS':'Effective size','MCSE':'MC error','difference':'Mean difference','censoring':'Protocol','coverage_mcse':'Coverage MC error','average_length':'Mean length','analysis':'Analysis','setting':'Setting','restricted':'Ordered'}
    def cell(value):
        if pd.isna(value):
            value='n/a'
        elif isinstance(value,(float,np.floating)):
            value=f'{value:.4f}' if abs(value)<100 else f'{value:.0f}'
        value={'lambda_X':'a','lambda_Y':'b','alpha':'α','R_3_5':'R','typeI':'Type I','adaptiveII':'Adaptive II'}.get(str(value),value)
        return Paragraph(html.escape(str(value)),styles['SmallProject'])
    rows=[[cell(labels.get(c,c)) for c in frame.columns]]+[[cell(v) for v in row] for row in frame.itertuples(index=False,name=None)]
    if widths:
        widths=[v*505/sum(widths) for v in widths]
    t=Table(rows,colWidths=widths,repeatRows=1,hAlign='CENTER')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e4f0f3')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#0f566c')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f8f8')]),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
    story.append(t);story.append(Spacer(1,12))

real=pd.read_csv(root/'results/real_estimates.csv')
sim=pd.read_csv(root/'results/reliability_regime_summary.csv')
gof=pd.read_csv(root/'results/goodness_of_fit.csv')
shape=json.loads((root/'results/common_shape_test.json').read_text())
carbon=pd.read_csv(root/'results/carbon_estimates.csv')
cgof=pd.read_csv(root/'results/carbon_goodness_of_fit.csv')
cshape=json.loads((root/'results/carbon_common_shape_test.json').read_text())
diag=pd.read_csv(root/'results/posterior_diagnostics.csv')

p('Classical and Bayesian estimation of multicomponent stress-strength reliability','TitleProject')
p('Weibull populations under block progressive hybrid censoring and upper k-record sampling','HeadingProject')
p('<b>Vaidik Maheshwari</b> | 250150045<br/><b>Anant Gupta</b> | 250150004')
p('DA2141 | Assigned project 3 | 11 October 2026')
story.append(Spacer(1,20))
p('This report accompanies the fully executed Jupyter notebook. It presents the distribution, principal derivations, unrestricted and ordered inference, 48-scenario simulation, two published-data applications, uncertainty assessments and source-use decisions. The notebook and derivations report contain the complete mathematical development and executable calculation.')
p('The main reliability target is the probability that at least three of five independent component strengths exceed one common random stress. Strength measurements are progressively censored in independent blocks; stress observations are upper k-records stopped at a prescribed record count.')
table(pd.DataFrame({'Assigned requirement':['MLEs and restricted MLEs','Joint Bayesian prior','Unrestricted and restricted Bayes','Complete simulation study','Suitable-data validation'],'Evidence':['Profile optimization and boundary pooling','Dependent gamma-Dirichlet prior','Quadrature, posterior sampling and intervals','48 scenarios x 300 replications','Jute and carbon-fibre data with calibrated checks']}),[180,310])
p('The reported probabilities concern same-unit laboratory comparisons of breaking-strength populations. Neither application supplies measured in-service operating loads. The jute fit limitation is reported explicitly; the additional carbon dataset supports the Weibull validation.', 'CaptionProject')


content=(root/'reports/derivations.md').read_text().split('\n## ')[2:]
for index,chapter in enumerate(content[:8],1):
    title,body=chapter.split('\n',1)
    section(title)
    markdown_block(body)
    if index==1:
        figure('01_weibull_functions','Weibull density, distribution, survival and hazard')
    if index==2:
        figure('02_reliability','Shared-stress system reliability')

section('9. Simulation design and measured performance')
markdown_block(content[8].split('\n',1)[1])
table(sim[['regime','method','bias','mse','coverage']],[72,128,90,90,110])
p('This table averages scenario-specific metrics within each regime. Restricted MLE confidence coverage is not included here because the main simulation does not assign it a regular boundary-normal interval. Its bootstrap assessment follows separately. Scenario-level values and their MCSEs are available in simulation_summary.csv.')

section('10. Simulation figures and restricted bootstrap coverage')
figure('03_simulation_bias_mse','Measured scenario-averaged reliability bias and MSE. Each plotted size value averages eight scenarios.',480)
figure('05_observed_information','Retained strength-failure counts under both hybrid protocols')
figure('04_simulation_coverage','Scenario-level coverage with 1.96 times Monte Carlo standard-error bars.',480)
table(pd.read_csv(root/'results/restricted_bootstrap_coverage.csv')[['regime','censoring','coverage','coverage_mcse','average_length']],[75,105,105,105,100])
p('Restricted bootstrap coverage uses six selected scenarios, 100 outer datasets each and 199 inner resamples per dataset. Its Monte Carlo precision is lower than the primary simulation. False-order reliability truths below one half cannot be covered by intervals restricted above one half. Continuous ordered Bayesian equal-tail intervals also exclude the equality endpoint, explaining their boundary coverage behavior. No universal advantage is inferred from narrower intervals alone.','CaptionProject')

section('11. Jute-fibre data provenance and fit checks')
p('Data are transcribed from Sultana et al., Table 10, printed page 3453 (PDF page 19), attributed to Xia et al. (2009). Each of the 15 mm and 20 mm gauge groups has 30 breaking-strength measurements already divided by 1000. The 15 mm group represents strength and the 20 mm group a same-unit laboratory comparison stress. The printed 70.32383 anomaly in the 10 mm group is visually confirmed; that group is excluded rather than silently repaired.')
table(gof[['sample','n','alpha','rate','KS','bootstrap_p']],[60,45,95,95,95,100])
p(f'The two-population common-shape likelihood-ratio statistic is {shape["LR"]:.4f}, with one degree of freedom and bootstrap p-value {shape["bootstrap_p"]:.4f}. Each goodness-of-fit calibration and the common-shape calibration uses 1,000 resamples with refitting. The 15 mm K-S result rejects Weibull adequacy at 5%, so its reliability estimates carry an explicit model-fit limitation. A high uncalibrated fitted-parameter K-S p-value would not resolve this issue.')
figure('06_real_data_fit','Complete jute measurements: fitted density, empirical/fitted CDF and Q-Q diagnostics.',460)

section('12. Jute retained observations and reliability estimates')
p('A seeded allocation splits the strength group into three blocks of ten. Each uses target seven, T=0.5 and balanced withdrawals. The comparison group supplies the first six upper 2-records in printed table order, stopping at position nine. The printed order is an imposed acquisition order, not a historical record sequence. Complete measurements are used for fit checks; the censored/record likelihood uses only the retained experiment.')
figure('07_real_sampling','Strength failures with withdrawals and auditable upper-record extraction.',470)
table(real[['method','alpha','lambda_X','lambda_Y','R_3_5','lower','upper']],[116,58,62,62,65,64,63])
p('MLE intervals use the unrestricted log-ratio approximation; restricted MLE intervals are parametric bootstrap percentile intervals. Bayesian rows use equal-tail credible intervals. Parameter intervals, LINEX values, shortest intervals and both bootstrap samples are retained in the executed notebook and exported result tables.')
figure('08_real_reliability_intervals','Actual posterior reliability densities and point/interval comparisons.',470)

section('13. Additional carbon-fibre validation')
p('The supplied Newer and Alanazi article lists 100 carbon-fibre strengths in GPa on PDF pages 12 and 14, attributing them to Nichols and Padgett (2006). A seeded random split creates disjoint groups of fifty for strength and comparison stress. Both groups measure breaking strengths; no river discharge data are used. A fixed unit of 3 GPa scales measurements before inference. The same fixed prior is used on the scaled rates.')
table(cgof[['sample','n','alpha','rate','KS','bootstrap_p']],[135,35,75,75,75,95])
p(f'Both calibrated K-S tests are compatible with Weibull at 5%. The common-shape bootstrap p-value is {cshape["bootstrap_p"]:.4f}. This provides a suitable supplementary empirical validation, while retaining the jute fit limitation. Compatibility is not proof of the distribution.')
p('Strength blocks have 17,17 and 16 specimens, target floor(0.7n), T=1 in scaled units and balanced withdrawals. The comparison half supplies the first five upper 4-records. Retained record values are distinct; no jitter is applied to the rounded source data. The allocation, records, calibrated tests and all four estimates are saved.')
table(carbon,[116,58,62,62,65,64,63])
figure('12_carbon_data_fit','Additional carbon-fibre Weibull diagnostics in the fixed 3 GPa unit.',440)

section('14. Posterior trace and density figures')
figure('09_posterior_traces','Four actual retained chains for each parameter and reliability, shown every tenth iteration for readability.',410)
figure('10_parameter_posteriors','Marginal parameter distributions from conditional-Gamma posterior draws on the quadrature grid.',480)

section('15. Sensitivity, predictive checks and interpretation')
table(pd.read_csv(root/'results/sensitivity.csv')[['analysis','setting','restricted','R','lower','upper']],[88,144,58,65,65,70])
figure('11_predictive_order_sensitivity','Posterior prediction under the actual observation protocol and variability across successful random acquisition orders.',480)
p('Prior sensitivity uses fixed alternatives; hybrid sensitivity compares min-time Type-I and adaptive Type-II protocols; record sensitivity varies k and the retained count. Acquisition-order permutations that fail to reach the prescribed count are explicitly counted. The histogram conditions on successful permutations and is a sensitivity description, not an extra iid sampling experiment.')
p('Predictive checks compare the retained failure count and largest stress record with regenerated experiments. Posterior predictive tail probabilities are descriptive checks rather than uniformly distributed significance-test p-values.')

section('16. Literature review and source-use decisions')
source_rows=[('Newer (2025), Scientific Reports 15, 43820. DOI 10.1038/s41598-025-30570-9','Supplied 26-page full PDF. IER model with block adaptive Type-II censoring and upper k-records. Used for the combined sampling framework and performance objectives. Weibull derivations are developed separately. Its unconditional binomial expression is replaced by the appropriate conditional common-stress integral.'),('Sultana et al. (2026), Communications in Statistics - Theory and Methods 55(11), 3436-3459. DOI 10.1080/03610926.2025.2579720','Supplied 25-page full PDF. Common-shape Weibull block Type-I censoring, ordered rates and dependent gamma-Dirichlet priors. Used for parameterization, profiling, pooling and joint-prior construction. Table 10 supplies the jute data. Progressive withdrawals and upper records extend its design.'),('Newer and Alanazi (2026), Journal of Mathematics, 5561353. DOI 10.1155/jom/5561353','Supplied 17-page full PDF. Generalized exponential ranked-set/record model and stochastic system size. Used for its conditional reliability integral, posterior reliability expectations and bootstrap methodology. Section 5.1.1 supplies carbon strengths. Its lower-record ranked-set likelihood, truncated-binomial component count and empirical Bayes procedures are outside this assignment.')]
for title,body in source_rows:
    p(html.escape(title),'SmallProject');p(body)
p('All five supplied articles were extracted page by page; relevant mathematical and application sections were reviewed. Original PDFs and extracted text remain in the project folder. The full source-use register is also available as literature_review.md.')

section('17. Other supplied articles and access limitation')
for title,body in [('Hasaballah et al. (2024), AIP Advances 14, 045111. DOI 10.1063/5.0201467','Supplied 13-page PDF. Generalized inverted exponential unified hybrid inference. Used as methodological background for survival/hazard displays, Bayesian loss and interval reporting. Its distribution, unified stopping scheme and COVID-19 data are not substituted for this Weibull model.'),('Yeganeh et al. (2025), Engineering Applications of Artificial Intelligence 141, 109652. DOI 10.1016/j.engappai.2024.109652','Supplied 24-page PDF. Normal-ratio attention control charts and Ethereum backtesting. Examined and excluded from the assigned reliability calculations. No trading data or attention/control-chart procedure is used.'),('Kumari et al. (2025), Proceedings of the Institution of Mechanical Engineers, Part O: Journal of Risk and Reliability 239(3), 596-608. DOI 10.1177/1748006X241247866','Assigned SAGE reference. Publisher abstract accessed, but full text was not supplied and the full-PDF request was unavailable. Used only for abstract-level context on block progressive Type-II censoring and hierarchical inference. Full-text review of this reference remains access-limited; no inaccessible derivation is claimed to have been reproduced.')]:
    p(html.escape(title),'SmallProject');p(body)
p('Xia et al. (2009), Study on the breaking strength of jute fibres using modified Weibull distribution, Composites Part A 40(1), 54-59. Jute measurements are taken from the supplied Sultana et al. reproduction, not a directly obtained original experimental file. Nichols and Padgett (2006), A Bootstrap Control Chart for Weibull Percentiles, Quality and Reliability Engineering International 22(2), 141-151, DOI 10.1002/qre.691. Carbon data are taken from the supplied Newer and Alanazi reproduction. Both transcription sources are recorded explicitly.')

section('18. Findings, limitations and deliverable map')
for paragraph in (root/'reports/results_discussion.md').read_text().split('\n\n')[2:]:
    p(html.escape(paragraph))
p('Main artifact: Vaidik_Maheshwari_Anant_Gupta_DA2141.ipynb. A readable HTML export retains outputs. reports/derivations.md provides the complete equations and assumptions; reports/literature_review.md records source use and exclusions. data contains transcriptions and observation designs; results contains actual computed tables, raw simulations and sampler chains; figures contains PNG and PDF charts. analysis.py and build_notebook.py support reproducibility. Python code contains no comments.')


def footer(canvas,doc):
    canvas.setFont(regular,9)
    canvas.setFillColor(colors.HexColor('#465a65'))
    canvas.drawString(40,23,'DA2141 | Vaidik Maheshwari and Anant Gupta')
    canvas.drawRightString(555,23,str(doc.page))


doc=SimpleDocTemplate(str(root/'reports/Project_Report.pdf'),pagesize=(595,842),rightMargin=40,leftMargin=40,topMargin=42,bottomMargin=45,title='DA2141 Weibull stress-strength reliability',author='Vaidik Maheshwari and Anant Gupta')
for current,following in zip(story,story[1:]):
    if isinstance(current,Paragraph) and isinstance(following,MathEquation):
        current.keepWithNext=True
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print('Readable project report created')
