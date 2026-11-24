"""Select and render a disclosed introduction example from real 224px predictions."""
from pathlib import Path
import argparse
import hashlib
import json
import os

os.environ.setdefault('MPLCONFIGDIR','/tmp/vit224-report-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import torch

BASE=Path(__file__).resolve().parent
OUT=BASE
FIG=OUT/'figures'
NAMES={'lulu':'噜噜','nailong':'奶龙','naiwa':'奶蛙','niulai':'牛来'}
VISUAL_PREFERRED={'lulu_0023','lulu_0045','nailong_0052','nailong_0133',
                  'naiwa_0019','naiwa_0056','niulai_0031'}
for f in [str(BASE/'fonts/Times New Roman.ttf'),str(BASE/'fonts/simsun.ttc')]:
    fontManager.addfont(f)
FONT=FontProperties(family=['Times New Roman','SimSun'])
plt.rcParams.update({'svg.fonttype':'path','pdf.fonttype':42,
                     'font.family':['Times New Roman','SimSun'],'axes.unicode_minus':False})


def select_case(results,predictions,widths):
    """Rank genuine rise/fall triplets; keep the full search table for review."""
    candidates=[]
    for seed, rr in results.groupby('seed'):
        if len(rr)!=len(widths):continue
        rr=rr.set_index('width').loc[widths]
        errors=rr.test_error.to_numpy()
        pred=predictions[predictions.seed==seed]
        correct=pred.pivot(index='sample_id',columns='width',values='correct')[widths]
        classes=pred.drop_duplicates('sample_id').set_index('sample_id').true_class
        for a in range(1,len(widths)-2):
            for b in range(a+1,len(widths)-1):
                for c in range(b+1,len(widths)):
                    rise=float(errors[b]-errors[a]);fall=float(errors[b]-errors[c])
                    pattern=correct.iloc[:,a]&~correct.iloc[:,b]&correct.iloc[:,c]
                    pool=list(correct.index[pattern])
                    coverage=int(classes.loc[pool].nunique())
                    early=float(errors[0]-errors[a])
                    # Overall error shape dominates; diversity and near-fit status break ties.
                    score=1000*min(rise,fall)+250*(rise+fall)+20*early+2*coverage+.5*min(6,len(pool))-20*float(rr.iloc[b].train_observed_error)
                    candidates.append(dict(seed=int(seed),a=a,b=b,c=c,width_early=widths[a],
                        width_peak=widths[b],width_late=widths[c],rise=rise,fall=fall,
                        initial_improvement=early,flip_images=len(pool),flip_classes=coverage,
                        peak_train_error=float(rr.iloc[b].train_observed_error),score=score))
    ranked=pd.DataFrame(candidates).sort_values(['score','flip_images'],ascending=False)
    ranked.to_csv(OUT/'illustration_candidates.csv',index=False,encoding='utf-8-sig')
    if ranked.empty:raise RuntimeError('No complete seed available for reporting')
    return ranked.iloc[0].to_dict()


def choose_images(predictions,case,widths):
    seed=int(case['seed']);a,b,c=[int(case[k]) for k in ['width_early','width_peak','width_late']]
    pred=predictions[predictions.seed==seed]
    correct=pred.pivot(index='sample_id',columns='width',values='correct')[widths]
    classes=pred.drop_duplicates('sample_id').set_index('sample_id').true_class
    rows=[]
    ia,ib,ic=[widths.index(d) for d in [a,b,c]]
    for sid,rr in correct.iterrows():
        ca,cb,cc=[int(rr[d]) for d in [a,b,c]]
        exact=ca==1 and cb==0 and cc==1
        stability=float(rr.iloc[ic:].mean())+float(rr.iloc[max(0,ia-1):ia+1].mean())
        visual=.75 if sid in VISUAL_PREFERRED else 0.
        rows.append(dict(sample_id=sid,true_class=classes.loc[sid],exact_flip=exact,
                         visual_preference=visual,
                         score=20*exact+3*ca+3*(1-cb)+3*cc+stability+visual))
    rank=pd.DataFrame(rows).sort_values(['score','sample_id'],ascending=[False,True])
    chosen=[]
    for cls in NAMES:
        chosen.append(rank[rank.true_class==cls].iloc[0].sample_id)
    for row in rank.itertuples():
        if row.sample_id not in chosen and sum(classes[sid]==row.true_class for sid in chosen)<2:
            chosen.append(row.sample_id)
        if len(chosen)==6:break
    chosen.sort(key=lambda sid:(list(NAMES).index(classes[sid]),sid))
    rank['selected']=rank.sample_id.isin(chosen)
    rank.to_csv(OUT/'image_selection.csv',index=False,encoding='utf-8-sig')
    return chosen


def choose_columns(widths,case,count):
    # Display the selected local rise/recovery window; full 12-width board remains separate.
    right=min(len(widths)-1,max(int(case['c']),count-1))
    anchors={0,right,int(case['a']),int(case['b']),int(case['c'])}
    peak=int(case['b'])
    for i in [peak-1,peak+1,peak-2,peak+2]:
        if len(anchors)>=count:break
        if 0<=i<=right:anchors.add(i)
    while len(anchors)<count:
        unused=[i for i in range(right+1) if i not in anchors]
        anchors.add(max(unused,key=lambda i:(min(abs(i-j) for j in anchors),-abs(i-peak))))
    return [widths[i] for i in sorted(anchors)]


def render_board(predictions,metrics,ids,widths,case,data,samples,stem):
    fig=plt.figure(figsize=(18,10.125),facecolor='#f7f8fc')
    ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,3600),ylim=(2025,0));ax.axis('off')
    def text(x,y,s,size=34,color='#18263b',ha='center'):
        ax.text(x,y,s,fontsize=size*.36,ha=ha,va='center',color=color,fontproperties=FONT,zorder=10)
    def rect(x,y,w,h,fill,radius=18):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={radius}',
                                  facecolor=fill,edgecolor='none'))
    seed=int(case['seed']);peak=int(case['width_peak'])
    pred=predictions[predictions.seed==seed]
    text(96,112,'更宽的ResNet，会认错吗？',90,ha='left')
    text(98,224,f'224×224输入 · ResNet-18 · 约10%训练标签噪声 · seed={seed}',37,color='#607087',ha='left')
    text(3504,112,f'6张图片 × {len(widths)}个模型',37,color='#607087',ha='right')
    for x,fill,label in [(2700,'#e1f0e9','符合真值'),(3110,'#fae5e3','不符真值')]:
        rect(x,207,36,36,fill,9);text(x+52,224,label,31,color='#607087',ha='left')
    first_x,gap=444,14;cw=(3504-first_x-(len(widths)-1)*gap)/len(widths)
    text(252,326,'测试输入 / 真值',36)
    text(252,379,'同一行始终是同一张图',24,color='#607087')
    for j,d in enumerate(widths):
        x=first_x+j*(cw+gap);cx=x+cw/2
        if d==peak:rect(x-5,278,cw+10,1432,'#fff0db',20)
        text(cx,326,f'b = {d}',44 if len(widths)<12 else 38)
        text(cx,379,f'{int(metrics.loc[d,"parameters"])/1000:.1f}k参数',27 if len(widths)<12 else 23,color='#607087')
    for i,sid in enumerate(ids):
        rr=pred[pred.sample_id==sid].set_index('width');row=rr.iloc[0];y=445+i*207
        index=int(samples.index[samples.sample_id==sid][0])
        rgb=np.clip((data['x'][index]*data['std'][0]+data['mean'][0]).transpose(1,2,0),0,1)
        ax.imshow(rgb,extent=(178,326,y+148,y),interpolation='bilinear',zorder=8)
        text(252,y+174,'真值：'+NAMES[row.true_class],34)
        for j,d in enumerate(widths):
            p=rr.loc[d];good=bool(p.correct);x=first_x+j*(cw+gap);cx=x+cw/2
            bg,fg=('#e1f0e9','#245947') if good else ('#fae5e3','#a3303f')
            rect(x,y+4,cw,180,bg)
            text(cx,y+76,NAMES[p.predicted_class],62 if len(widths)<12 else 53,fg)
            text(cx,y+140,f'置信度 {p.confidence:.0%}',28 if len(widths)<12 else 23,fg)
    ax.plot([96,3504],[1728,1728],color='#cbd3df',linewidth=.6)
    text(252,1795,'这6张中判对',31)
    counts=[]
    for j,d in enumerate(widths):
        rr=pred[(pred.sample_id.isin(ids))&(pred.width==d)];count=int(rr.correct.sum());counts.append(count)
        text(first_x+j*(cw+gap)+cw/2,1795,f'{count} / 6',45)
    text(96,1889,'模型容量增大 →   b为首阶段通道数；四阶段通道为b、2b、4b、8b。',32,color='#607087',ha='left')
    text(96,1960,'介绍示例：种子、图片及展示容量点按结果选取；底部数量仅对应这6张图。全部12个容量点另附。',29,color='#607087',ha='left')
    for ext in ['png','pdf','svg']:
        fig.savefig(FIG/f'{stem}.{ext}',dpi=200,facecolor=fig.get_facecolor())
    plt.close(fig)
    return dict(widths=widths,correct_out_of_six=counts,file=f'figures/{stem}.png')


