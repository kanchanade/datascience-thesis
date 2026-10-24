"""Make the three Chapter 4 figures from the saved result files.

Reads (strict cleaning level, the main analysis):
    experiments/results/runs_strict.csv       cross-project runs
    experiments/results/within_strict.csv     within-project runs
    experiments/results/rq3_summary_strict.csv  SHAP agreement (RQ3)
Writes:
    experiments/figures/Fig4_1_within_vs_cross.png
    experiments/figures/Fig4_2_change_vs_unweighted.png
    experiments/figures/Fig4_3_shap_agreement.png

Run from the repository root:   python src/make_figures.py
Needs: pandas, numpy, matplotlib.
"""
from pathlib import Path
import sys
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "experiments" / "results"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "experiments" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

rs = pd.read_csv(RES / "runs_strict.csv")
w = pd.read_csv(RES / "within_strict.csv")
r3 = pd.read_csv(RES / "rq3_summary_strict.csv")
CL=['Logistic Regression','Random Forest','XGBoost','MLP']; SHORT={'Logistic Regression':'Logistic\nRegression','Random Forest':'Random\nForest','XGBoost':'XGBoost','MLP':'MLP'}
S1,S2,S3='#2a78d6','#eb6834','#1baf7a'; INK='#0b0b0b'; INK2='#52514e'; GRID='#e4e3df'; BG='#fcfcfb'
plt.rcParams.update({'font.size':9,'axes.edgecolor':'#bdbcb6','axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'text.color':INK,'figure.facecolor':BG,'axes.facecolor':BG,'savefig.facecolor':BG,'axes.spines.top':False,'axes.spines.right':False})
# Fig 1 within vs cross
fig,axs=plt.subplots(1,2,figsize=(8.2,3.9),sharey=False)
for ax,met in zip(axs,['MCC','AUC-PR']):
    x=np.arange(4); bw=0.36
    for off,lab,col in [(-bw/2-0.01,'Within-project (10-fold CV)',S1),(bw/2+0.01,'Cross-project (leave-one-project-out)',S2)]:
        means=[];dots=[]
        for c in CL:
            if lab.startswith('Within'): v=w[w.Classifier==c].groupby('Target_Dataset')[met].mean()
            else: v=rs[(rs.Method=='unweighted')&(rs.Classifier==c)].groupby('Target_Dataset')[met].mean()
            means.append(v.mean()); dots.append(v.values)
        ax.bar(x+off,means,bw,color=col,label=lab,zorder=2,edgecolor='none')
        for xi,d,m in zip(x+off,dots,means):
            ax.scatter(np.full(len(d),xi)+np.linspace(-0.07,0.07,len(d)),d,s=11,color=INK,alpha=.55,zorder=3,linewidths=0)
            ax.text(xi,m+0.012 if False else 0.005,f"{m:.2f}",ha='center',va='bottom',fontsize=8,color='white',zorder=4,fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([SHORT[c] for c in CL]); ax.set_title(met,loc='left',fontsize=10,color=INK,fontweight='bold')
    ax.yaxis.grid(True,color=GRID,zorder=0); ax.set_axisbelow(True); ax.set_ylim(0,0.72 if met=='AUC-PR' else 0.5); ax.tick_params(length=0)
h,l=axs[0].get_legend_handles_labels()
fig.legend(h,l,loc='lower center',ncol=2,frameon=False,fontsize=8.5,bbox_to_anchor=(0.5,-0.01))
fig.text(0.01,0.99,'Bars: mean over the five projects and ten seeds. Dots: the five per-project means.',fontsize=8,color=INK2,va='top')
fig.tight_layout(rect=(0,0.07,1,0.95)); fig.savefig(OUT/'Fig4_1_within_vs_cross.png',dpi=300); plt.close(fig)
# Fig 2 delta MCC vs unweighted per classifier
PRIM={'Logistic Regression':'sfrt_soft','MLP':'sfrt_soft','Random Forest':'sfrt_hard_k10','XGBoost':'sfrt_hard_k10'}
fig,axs=plt.subplots(2,2,figsize=(8.2,5.4),sharex=True)
for ax,c in zip(axs.ravel(),CL):
    meths=['mutualinfo_k10','variance_k10',PRIM[c]]; labs=['Mutual information (k=10)','Variance (k=10)','SFRT: soft reweighting' if PRIM[c]=='sfrt_soft' else 'SFRT: hard top-10']
    cols=[S1,S1,S2]
    base=rs[(rs.Method=='unweighted')&(rs.Classifier==c)].groupby(['Seed','Target_Dataset']).MCC.mean()
    for i,(m,lab,col) in enumerate(zip(meths,labs,cols)):
        v=rs[(rs.Method==m)&(rs.Classifier==c)].groupby(['Seed','Target_Dataset']).MCC.mean()-base
        per=v.groupby('Target_Dataset').mean(); y=len(meths)-1-i
        ax.barh(y,per.mean(),0.5,color=col,zorder=2,alpha=.9)
        ax.scatter(per.values,np.full(len(per),y)+np.linspace(-.14,.14,len(per)),s=12,color=INK,alpha=.6,zorder=3,linewidths=0)
        ax.text(0.228,y,f"{per.mean():+.3f}",va='center',ha='right',fontsize=8.5,color=INK,fontweight='bold')
    ax.axvline(0,color=INK2,lw=.8,zorder=1); ax.set_yticks(range(len(meths))); ax.set_yticklabels(labs[::-1]); ax.set_title(c,loc='left',fontsize=10,fontweight='bold')
    ax.xaxis.grid(True,color=GRID,zorder=0); ax.set_axisbelow(True); ax.tick_params(length=0); ax.set_xlim(-0.08,0.23); ax.set_xticks([-0.05,0,0.05,0.1,0.15])
for ax in axs[1]: ax.set_xlabel('Change in MCC versus unweighted')
from matplotlib.patches import Patch
fig.legend([Patch(color=S1),Patch(color=S2)],['Baseline feature selection','SFRT (primary variant)'],loc='lower center',ncol=2,frameon=False,fontsize=8.5,bbox_to_anchor=(0.5,-0.005))
fig.text(0.01,0.995,'Bars: mean over targets and seeds. Dots: the five targets (each averaged over ten seeds). Strict cleaning level.',fontsize=8,color=INK2,va='top')
fig.tight_layout(rect=(0,0.05,1,0.96)); fig.savefig(OUT/'Fig4_2_change_vs_unweighted.png',dpi=300); plt.close(fig)
# Fig 3 SHAP agreement
def lab(r):
    c=r.Comparison
    if c.startswith('NOISE'): return r.Classifier
    if c.startswith('RQ3(b)'): return r.Classifier
    return r.Classifier.replace('Logistic Regression','LR').replace('Random Forest','RF')
fig,axs=plt.subplots(1,2,figsize=(8.4,4.9),sharey=True)
order=[('NOISE FLOOR','Same model, same project, different seed (noise floor)',S1),('RQ3(b)','Same model, different projects (RQ3b)',S2),('RQ3(a)','Different models, same project (RQ3a)',S3)]
for ax,space,title in zip(axs,['features','groups'],['21 individual metrics','10 groups of correlated metrics']):
    d=r3[r3.Space==space]; y=0; yt=[];yl=[]
    for key,name,col in order:
        sub=d[d.Comparison.str.startswith(key)]
        for _,r in sub.iterrows():
            ax.scatter(r.Spearman,-y,s=46,color=col,zorder=3,linewidths=0); ax.text(r.Spearman+0.025,-y,f"{r.Spearman:.2f}",va='center',fontsize=7.5,color=INK)
            yt.append(-y); yl.append(lab(r)); y+=1
        y+=0.6
    ax.axvline(0,color=INK2,lw=.8,ls=(0,(3,3)),zorder=1); ax.set_xlim(-0.05,1.08); ax.xaxis.grid(True,color=GRID,zorder=0); ax.set_axisbelow(True)
    ax.set_title(title,loc='left',fontsize=10,fontweight='bold'); ax.set_xlabel('Spearman rank correlation of SHAP importances'); ax.tick_params(length=0)
    ax.set_yticks(yt); ax.set_yticklabels(yl,fontsize=8)
from matplotlib.lines import Line2D
fig.legend([Line2D([],[],marker='o',ls='',color=c,markersize=7) for _,_,c in order],[n for _,n,_ in order],loc='lower center',ncol=1,frameon=False,fontsize=8.5,bbox_to_anchor=(0.5,-0.005))
fig.text(0.01,0.995,'Dashed line: expected agreement of two random rankings (about 0). Mean over projects, seeds and pairs.',fontsize=8,color=INK2,va='top')
fig.tight_layout(rect=(0,0.1,1,0.965)); fig.savefig(OUT/'Fig4_3_shap_agreement.png',dpi=300); plt.close(fig)

print('Figures written to', OUT)
