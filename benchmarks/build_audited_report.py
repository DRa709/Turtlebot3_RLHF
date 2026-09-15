"""Rebuild audited aggregate results; no training, new labels, or simulator runs."""
from pathlib import Path
import argparse
import html
import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
if __package__:
    from .audited_results import load_audited_data, validate_data
else:
    from audited_results import load_audited_data, validate_data

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA, FIG = HERE / 'data/audited', HERE / 'figures'
ORDER = ['DQN','DoubleDQN','DuelingDoubleDQN','RainbowDQN','DiscreteSAC','SDSAC']
LABELS = ['DQN','Double DQN','Dueling Double','Rainbow','Discrete SAC','SD-SAC']
COLORS = ['#53758D','#1779AA','#B77937','#178471','#7959A1','#B65275']
ARMS = ['original_reward','feedback_200']
ARM_LABELS = ['Original reward','Preference reward (200)']
ARM_COLORS = ['#207AA4','#C3652B']
CHECKPOINTS = [0,1499,1500,1750,2000,5000,10000]
TICKLABELS = ['0','1,499','1,500','1,750','2,000','5,000','10,000']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
    'axes.titlesize':8.7,'axes.labelsize':8,'xtick.labelsize':7.4,'ytick.labelsize':7.4,
    'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#A2ACA9',
    'text.color':'#243B35','axes.labelcolor':'#243B35','xtick.color':'#3B4C48',
    'ytick.color':'#3B4C48','grid.color':'#E1E7E4','grid.linewidth':.55,
    'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})

def read(name): return pd.read_csv(DATA / name)
def save(fig, name):
    for ext in ('pdf','svg','png'):
        fig.savefig(FIG / f'{name}.{ext}', dpi=300, bbox_inches='tight', pad_inches=.08)
        if ext == 'svg':
            path = FIG / f'{name}.{ext}'
            path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines()) + '\n', encoding='utf-8')
    plt.close(fig)
def grid(ax):
    ax.grid(axis='y', zorder=0)
    ax.set_axisbelow(True)

def benchmark():
    curve, seed = read('evaluation_curves.csv'), read('seed_results.csv')
    f, ax = plt.subplots(1,2,figsize=(7,2.35),gridspec_kw={'width_ratios':[1.35,1]})
    f.subplots_adjust(left=.085,right=.985,bottom=.20,top=.77,wspace=.43)
    for a,label,col in zip(ORDER,LABELS,COLORS):
        g=curve[curve.algorithm==a].sort_values('checkpoint_step')
        ax[0].plot(g.checkpoint_step/1000,100*g.success,color=col,label=label,lw=1.2)
    ax[0].set(xlabel='Training actions (thousands)',ylabel='Goal success (%)',ylim=(0,104),xlim=(25,500))
    ax[0].set_xticks([100,200,300,400,500]);grid(ax[0])
    ax[0].set_title('(a) Scheduled evaluation means',loc='left')
    for i,(a,col) in enumerate(zip(ORDER,COLORS)):
        vals=seed[seed.algorithm==a].sort_values('learning_seed').final_success.to_numpy()*100
        ax[1].scatter(vals,i+np.linspace(-.18,.18,5),s=20,color=col,edgecolors='white',linewidths=.3,zorder=3)
        ax[1].plot([vals.mean()]*2,[i-.31,i+.31],color='#162E25',lw=1.5,zorder=4)
    ax[1].set(yticks=range(6),yticklabels=LABELS,xlim=(70,102),xlabel='Final goal success (%)')
    ax[1].invert_yaxis();ax[1].grid(axis='x');ax[1].set_axisbelow(True)
    ax[1].set_title('(b) Five learners per method',loc='left')
    f.legend(*ax[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,
             bbox_to_anchor=(.51,1.03),columnspacing=1.9,handlelength=2)
    save(f,'benchmark')

def series(c,arm,metric):
    g=c[(c.suite=='E2')&(c.policy_mode=='all')]
    b=g[g.study_arm=='frozen_baseline']
    return pd.concat([b,g[g.study_arm==arm]]).sort_values('added_steps')[metric].to_numpy()

def retention():
    c, seed=read('checkpoint_summary.csv'),read('seed_summary.csv')
    f, axes=plt.subplots(1,2,figsize=(7,2.35))
    f.subplots_adjust(left=.085,right=.985,bottom=.26,top=.79,wspace=.27)
    xx=np.arange(7)
    for ax,metric,title in zip(axes,['goal_pct','safety_pct'],['(a) Goal reached','(b) Safety stop']):
        for arm,label,col,mark in zip(ARMS,ARM_LABELS,ARM_COLORS,['o','s']):
            y=series(c,arm,metric)
            ax.plot(xx,y,marker=mark,ms=3.6,lw=1.4,label=label,color=col)
            rates=[]
            for s in [101,202]: rates.append(series(seed[seed.learning_seed==s],arm,metric))
            ax.fill_between(xx,np.min(rates,axis=0),np.max(rates,axis=0),color=col,alpha=.13,lw=0)
        ax.set(xticks=xx,xticklabels=TICKLABELS,ylim=(0,105),ylabel='Episodes (%)',title=title)
        ax.tick_params(axis='x',labelrotation=32);grid(ax)
    f.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.51,1.02))
    f.text(.52,.025,'Additional training actions (equally spaced checkpoint categories)',ha='center',fontsize=8)
    save(f,'retention')

def diagnostics():
    c, p=read('checkpoint_summary.csv'),read('policy_diagnostics.csv')
    f,axes=plt.subplots(1,2,figsize=(7,2.3))
    f.subplots_adjust(left=.09,right=.985,bottom=.27,top=.77,wspace=.31)
    for arm,label,col,mark in zip(ARMS,ARM_LABELS,ARM_COLORS,['o','s']):
        axes[0].plot(range(7),series(c,arm,'mean_min_clearance_m'),marker=mark,ms=3.6,color=col,label=label,lw=1.4)
        for seed,style in [(101,'-'),(202,'--')]:
            g=p[(p.study_arm==arm)&(p.learning_seed==seed)].sort_values('added_step')
            axes[1].plot(g.added_step/1000,g.kl_baseline_to_current,color=col,linestyle=style,lw=1.2)
    axes[0].set(xticks=range(7),xticklabels=TICKLABELS,ylabel='Mean minimum clearance (m)',ylim=(0,.45),title='(a) Clearance on E2')
    axes[0].tick_params(axis='x',labelrotation=32)
    axes[0].set_xlabel('Additional actions (checkpoint categories)')
    axes[1].set(xlim=(0,10),ylim=(0,3),xlabel='Additional actions (thousands)',ylabel='Baseline-to-current KL (nats)',title='(b) Policy change on fixed observations')
    axes[1].text(.03,.94,'Solid: seed 101\nDashed: seed 202',transform=axes[1].transAxes,va='top',fontsize=7.5)
    for ax in axes:grid(ax)
    f.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.51,1.02))
    save(f,'diagnostics')

def comparison_budget():
    m=read('comparison_budget_metrics.csv').sort_values('comparisons')
    x=m.comparisons.to_numpy()
    f,axes=plt.subplots(1,2,figsize=(7,2.5))
    f.subplots_adjust(left=.09,right=.98,bottom=.25,top=.77,wspace=.32)
    axes[0].plot(x,m.strict_accuracy_pct,'o-',color='#178471',ms=4.5,lw=1.6)
    axes[0].set(xticks=x,ylim=(0,103),xlim=(8,213),ylabel='Strict prediction accuracy (%)',title='(a) Ranking accuracy: 22 strict pairs')
    for xx,yy,cc in zip(x,m.strict_accuracy_pct,m.strict_correct):
        axes[0].annotate(f'{yy:.1f}%\n{int(cc)}/22',(xx,yy),xytext=(0,6),textcoords='offset points',ha='center',fontsize=7)
    axes[1].plot(x,m.pair_cross_entropy,'s-',color='#207AA4',ms=4,lw=1.5,label='Pair mean')
    axes[1].plot(x,m.scenario_cross_entropy,'^--',color='#C3652B',ms=4,lw=1.2,label='Scenario mean')
    axes[1].axhline(np.log(2),color='#6A7773',ls=':',lw=1.1,label='Constant p = 0.5')
    axes[1].set(xticks=x,xlim=(8,213),ylim=(.57,1.33),ylabel='Cross-entropy (nats)',title='(b) Prediction loss: all 37 pairs')
    axes[1].annotate('Lowest pair loss\n0.698 at 150',(150,m.loc[m.comparisons==150,'pair_cross_entropy'].iloc[0]),
                     xytext=(103,1.13),arrowprops={'arrowstyle':'-','color':'#485A53','lw':.7},fontsize=7.3)
    f.legend(*axes[1].get_legend_handles_labels(),loc='upper center',ncol=3,fontsize=7.5,
             frameon=False,handlelength=2,bbox_to_anchor=(.52,1.015))
    for ax in axes:
        ax.set_xlabel('Total collected human comparisons');grid(ax)
    f.text(.52,.025,'Common reused validation set; one reward-model seed. These are prediction metrics.',ha='center',fontsize=7.5)
    save(f,'comparison_budget')


def outcomes():
    summary = read('summary.csv').set_index('algorithm').loc[ORDER]
    figure, axis = plt.subplots(figsize=(7, 3.0))
    figure.subplots_adjust(left=.20, right=.97, bottom=.19, top=.82)
    left = np.zeros(6)
    for column, label, color in [
        ('final_success', 'Goal reached', '#178471'),
        ('safety', 'Safety stop', '#C3652B'),
        ('collision', 'Physical contact', '#B65275'),
        ('timeout', 'Timeout', '#8B98A5'),
    ]:
        values = summary[column].to_numpy() * 100
        axis.barh(LABELS, values, left=left, label=label, color=color, height=.63)
        if column == 'final_success':
            for index, value in enumerate(values):
                axis.text(value / 2, index, f'{value:.0f}/100', va='center',
                          ha='center', color='white', fontsize=8)
        left += values
    axis.invert_yaxis()
    axis.set(xlim=(0, 100), xlabel='Final evaluation episodes (%)')
    figure.legend(*axis.get_legend_handles_labels(), loc='upper center',
                  ncol=4, frameon=False, fontsize=7.5)
    figure.text(.59, .02, '100 trials per method; zero recorded contacts in this panel.',
                ha='center', fontsize=7.5)
    save(figure, 'outcomes')


def markdown_table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def report_tables(tables):
    summary = tables['summary'].set_index('algorithm')
    benchmark_rows = []
    for name, label in zip(ORDER, LABELS):
        row = summary.loc[name]
        benchmark_rows.append([label, f'{int(row.final_goals)}/100',
                               f'{100 * row.final_success:.0f}', f'{row.final_sd_pp:.2f}',
                               str(round(100 * row.safety)), str(round(100 * row.collision)),
                               str(round(100 * row.timeout))])
    c = tables['checkpoint_summary']
    c = c[(c.suite == 'E2') & (c.policy_mode == 'all')]
    continuation_rows = []
    for arm, step, label in [('frozen_baseline', 0, 'Frozen baseline'),
                              ('original_reward', 10000, 'Original reward, +10k actions'),
                              ('feedback_200', 10000, 'Preference reward (200), +10k actions')]:
        row = c[(c.study_arm == arm) & (c.added_steps == step)].iloc[0]
        continuation_rows.append([label, f'{int(row.goal)}/{int(row.n)}',
                                  f'{row.goal_pct:g}', int(row.safety), int(row.collision),
                                  int(row.timeout)])
    budget_rows = []
    for row in tables['comparison_budget_metrics'].sort_values('comparisons').itertuples():
        budget_rows.append([row.comparisons, row.fitting_pairs,
                            f'{int(row.strict_correct)}/{row.strict_n}',
                            f'{row.strict_accuracy_pct:.2f}', f'{row.pair_cross_entropy:.3f}'])
    return {
        'AUDITED_BENCHMARK': (['Method', 'Goals', 'Success (%)', 'Learner SD (pp)',
                               'Safety stops', 'Contacts', 'Timeouts'], benchmark_rows),
        'AUDITED_CONTINUATION': (['Condition', 'Goals', 'Success (%)', 'Safety stops',
                                  'Contacts', 'Timeouts'], continuation_rows),
        'AUDITED_COMPARISONS': (['Collected comparisons', 'Fitting pairs', 'Correct / strict pairs',
                                 'Accuracy (%)', 'Pair cross-entropy'], budget_rows),
    }


def render_html(tables):
    rendered = report_tables(tables)
    sections = []
    descriptions = [
        ('AUDITED_BENCHMARK', 'Benchmark: 97 goals in 100 trials',
         'Final primary E1 evaluation: five learners with 20 trials each. Discrete SAC, Double DQN, and Rainbow tie at 97%. SD is across five learner rates.',
         [('benchmark', 'Recorded checkpoint means and all five final learner rates. Cases change between checkpoints.'),
          ('outcomes', 'Exclusive terminal outcomes. Safety stops are distinct from recorded physical contacts.')]),
        ('AUDITED_CONTINUATION', 'Continuation: goal retention failed',
         'E2 differs from E1: 20 scenarios, two learners, and two action-selection modes, totaling 80 trials. Both conditions use actor-only continuation. Neither retained baseline performance.',
         [('retention', 'Seed-range shading is not a confidence interval. Uneven training-action counts appear as equally spaced checkpoint categories.'),
          ('diagnostics', 'Clearance falls in both conditions. KL describes policy change, not its cause. Zero contacts does not establish improved obstacle avoidance.')]),
        ('AUDITED_COMPARISONS', 'Comparison budgets: prediction, not navigation success',
         'The same 22 strict validation pairs determine accuracy; cross-entropy uses all 37 pairs including 15 ties. Validation was reused for selection and only one reward-model seed is available.',
         [('comparison_budget', 'Strict accuracy rises from 50.00% to 86.36%; cross-entropy is lowest at 150. Navigation was evaluated only at 200 comparisons. Improvements across navigation budgets remain untested.')]),
    ]
    for key, title, description, figures in descriptions:
        headers, rows = rendered[key]
        table = pd.DataFrame(rows, columns=headers).to_html(index=False, border=0)
        images = ''.join(f'<figure><img src="figures/{name}.png" alt="{html.escape(caption)}">'
                         f'<figcaption>{html.escape(caption)}</figcaption></figure>' for name, caption in figures)
        sections.append(f'<section><h2>{title}</h2><p>{description}</p><div class="table">{table}</div>{images}</section>')
    return '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Audited TurtleBot3 navigation results</title><style>
body{font:16px/1.6 system-ui,sans-serif;margin:0;background:#f3f6f5;color:#233a34}
main{max-width:1060px;margin:auto;padding:36px 24px}h1{font-size:34px;line-height:1.2}
h2{line-height:1.25}section{background:white;padding:28px;margin:28px 0;border:1px solid #d8e2de;border-radius:12px}
a{color:#086b62}.lead{font-size:18px}img{max-width:100%;height:auto}figure{margin:24px 0}
figcaption{font-size:14px;color:#4c625a}.table{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:9px 11px;border-bottom:1px solid #dae4df;text-align:left}th{background:#edf5f1}
footer{font-size:14px;color:#4c625a}@media(max-width:600px){main{padding:16px 12px}section{padding:16px}h1{font-size:27px}}
</style></head><body><main><h1>Audited TurtleBot3 navigation results</h1>
<p class="lead">Goal reaching, safety stops, physical contacts, and reward prediction are reported separately.</p>
<p><a href="../README.md">Project documentation</a> · <a href="data/audited/README.md">Data and provenance</a> ·
<a href="../docs/RESULTS_ALIGNMENT.md">Corrections and remaining limits</a></p>
''' + ''.join(sections) + '''<footer>Source: aggregate tables audited for manuscript revision 6, 15 September 2026.
These summaries reproduce the figures; they do not verify historical deployed code or establish physical-robot performance.
Experiment-specific code is available upon request. OpenAI Codex assisted with documentation and rendering from existing records.</footer>
</main></body></html>\n'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Validate data and generated text without rewriting files')
    args = parser.parse_args()
    tables = load_audited_data(DATA)
    readme_path = ROOT / 'README.md'
    readme = readme_path.read_text(encoding='utf-8')
    updated = readme
    for key, (headers, rows) in report_tables(tables).items():
        pattern = rf'<!-- BEGIN {key} -->.*?<!-- END {key} -->'
        replacement = f'<!-- BEGIN {key} -->\n{markdown_table(headers, rows)}\n<!-- END {key} -->'
        updated, count = re.subn(pattern, replacement, updated, flags=re.S)
        if count != 1:
            raise ValueError(f'Expected one generated table block: {key}')
    dashboard = render_html(tables)
    if args.check:
        if updated != readme or dashboard != (HERE / 'index.html').read_text(encoding='utf-8'):
            raise ValueError('Generated documentation differs from the audited tables; rebuild it')
        for name in ['benchmark', 'outcomes', 'retention', 'diagnostics', 'comparison_budget']:
            if not (FIG / f'{name}.png').exists():
                raise ValueError(f'Missing reported figure: {name}')
    else:
        FIG.mkdir(parents=True, exist_ok=True)
        benchmark(); outcomes(); retention(); diagnostics(); comparison_budget()
        readme_path.write_text(updated, encoding='utf-8')
        (HERE / 'index.html').write_text(dashboard, encoding='utf-8')
    print(json.dumps(validate_data(tables), indent=2))


if __name__ == '__main__':
    main()
