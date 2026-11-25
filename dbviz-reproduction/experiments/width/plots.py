import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from .training import job_dir


NAMES=("Airplane","Automobile","Bird","Cat","Deer","Dog","Frog","Horse","Ship","Truck")
PALETTE=["#ffcb00","#ff8000","#2ca02c","#00e5ec","#a65628","#ec00ec","#ff003d","#7900fa","#0045ff","#3bff00"]
FIG_NAMES={6:"Test error",7:"Decision regions across widths",8:"Correctly labelled triplets at k = 10",
           9:"One mislabeled automobile per triplet",10:"Fragmentation",11:"Region similarity",12:"Class-region margins"}


def render(root,cfg,figure,widths,partial):
    tag=cfg["profile"].upper()
    suffix="-subset" if partial else ""
    if partial:
        suffix += "-" + "_".join(map(str,widths))
    path=root/"figures"/f"fig{figure:02d}{suffix}.png"
    label=f"arXiv v1 Fig. {figure}    |    {FIG_NAMES[figure]}"
    note=f"{tag}  •  {cfg['epochs']} epochs per model"
    if partial:
        note += "  •  selected widths only"
    def read(k,noise):
        if figure==6:
            p=job_dir(root,k,noise,1)/"metric.json"
        else:
            p=root/"measures"/f"fig{figure:02d}-k{k:02d}-n{noise:02d}.json"
        r=json.loads(p.read_text())
        if figure!=6 and not r["complete"]:
            raise ValueError(f"Incomplete {p}")
        return r
    plotdata=[]
    if figure in [7,8,9]:
        mat=[]
        row_titles=[]
        col_titles=[]
        if figure==7:
            col_titles=[f"k = {k}" for k in widths]
            for panel, name in [(0,"Three different classes"),(1,"Three automobiles")]:
                for noise in [0,20]:
                    mat.append([(k,noise,panel) for k in widths])
                    row_titles.append(f"{name}\n{noise}% noise")
        elif figure==8:
            col_titles=["Airplane","Bird","Horse"]
            for noise in [0,20]:
                mat.append([(10,noise,j) for j in range(3)])
                row_titles.append(f"k = 10\n{noise}% noise")
        else:
            col_titles=["→ Bird","→ Horse","→ Ship"]
            for k in widths:
                mat.append([(k,20,j) for j in range(3)])
                row_titles.append(f"k = {k}\n20% noise")
        fig,axes=plt.subplots(len(mat),len(col_titles),squeeze=False,
            figsize=(max(6.8,len(col_titles)*2.2+1.1),len(mat)*1.8+1.6))
        fig.subplots_adjust(left=.12,right=.995,top=.81,bottom=.10,wspace=.08,hspace=.18)
        rgb=np.array([matplotlib.colors.to_rgb(c) for c in PALETTE])
        for row,panels in enumerate(mat):
            for col,(k,noise,pindex) in enumerate(panels):
                ax=axes[row,col]
                item=read(k,noise)["panels"][pindex]
                with np.load(root/"measures"/item["file"]) as f:
                    vals=f["label"]
                    l,r,b,t=f["extent"]
                    dx=(r-l)/(vals.shape[1]-1)
                    dy=(t-b)/(vals.shape[0]-1)
                    # Confidence shading follows the paper's description (temperature 5).
                    shaded=1-(1-rgb[vals])*f["confidence"][...,None]
                    ax.imshow(shaded,extent=(l-dx/2,r+dx/2,b-dy/2,t+dy/2),origin="lower",aspect="auto",
                              interpolation="nearest")
                    for idx,(x,y) in enumerate(f["xy"]):
                        marker=("o","^","X")[idx]
                        ax.scatter(x,y,s=17,marker=marker,c="#171717",linewidths=0)
                ax.set_xticks([])
                ax.set_yticks([])
                if row==0:
                    ax.set_title(col_titles[col],fontsize=10)
                if col==0:
                    ax.set_ylabel(row_titles[row],fontsize=9)
                plotdata.append({"row":row+1,"column":col+1,"k":k,"noise_percent":noise,**item})
        fig.legend([Patch(color=c) for c in PALETTE],NAMES,loc="lower center",ncols=5,
                   fontsize=7,frameon=False,handlelength=1,columnspacing=1.4)
    else:
        chart = {
            6:[("test_error","Test error")],
            10:[("random_train_mean","Random training triplets"),("same_class_correct_mean","Same class, correctly labelled")],
            11:[("test_agreement_mean","Agreement on test planes")],
            12:[("all_median","All images"),("correct_median","Correctly labelled"),("mislabeled_median","Mislabeled")],
        }[figure]
        nrows=2 if figure==10 else 1
        fig,axs=plt.subplots(nrows,1,squeeze=False,figsize=(7,5.5 if nrows==2 else 4.5))
        fig.subplots_adjust(top=.80,bottom=.13,hspace=.35,left=.13,right=.97)
        for j,(field,series) in enumerate(chart):
            ax=axs[j if nrows==2 else 0,0]
            for noise in (0,20):
                if figure==12 and noise==0 and field!="all_median":
                    continue
                ys=[read(k,noise)[field] for k in widths]
                legend=f"{noise}% noise" if figure!=12 else f"{series}, {noise}% noise"
                style="o-" if noise==0 else "x--"
                color=(["#227a91","#337e42","#ba6621"][j] if figure==12 else ["#227a91","#bb562c"][noise//20])
                ax.plot(widths,ys,style,color=color,markersize=4,lw=1.4,label=legend)
                plotdata.extend({"series":field,"k":k,"noise_percent":noise,"value":y} for k,y in zip(widths,ys))
            if figure==10:
                ax.set_title(series,fontsize=10)
            ax.set_ylabel({6:"Test error",10:"Mean component count",11:"Agreement",12:"Median margin"}[figure])
            ax.grid(axis="y",alpha=.18)
            handles,labels=ax.get_legend_handles_labels()
            unique=dict(zip(labels,handles))
            ax.legend(unique.values(),unique.keys(),frameon=False,fontsize=8)
        axs[-1,0].set_xlabel("ResNet width k")
    fig.suptitle(label+"\n"+note,fontsize=12,fontweight="semibold",y=.98)
    temp=path.with_suffix(".new.png")
    fig.savefig(temp,dpi=145,bbox_inches="tight")
    temp.replace(path)
    plt.close(fig)
    from .training import atomic_json
    atomic_json(path.with_suffix(".json"),{"figure":figure,"profile":cfg["profile"],"epochs":cfg["epochs"],
                  "widths":widths,"partial_width_axis":partial,"measurements":plotdata})
    print(f"Figure written: {path}",flush=True)
