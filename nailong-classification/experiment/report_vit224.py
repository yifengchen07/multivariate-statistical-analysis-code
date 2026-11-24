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
from models import build_model

BASE=Path(__file__).resolve().parent
OUT=BASE/'vit224'
FIG=OUT/'figures'
NAMES={'lulu':'噜噜','nailong':'奶龙','naiwa':'奶蛙','niulai':'牛来'}
VISUAL_PREFERRED={'lulu_0023','lulu_0045','nailong_0052','nailong_0133',
                  'naiwa_0019','naiwa_0056','niulai_0031'}
for f in ['/System/Library/Fonts/Supplemental/Times New Roman.ttf','/Users/chan/Library/Fonts/simsun.ttc']:
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
    text(96,112,'模型更大，反而认错了？',90,ha='left')
    text(98,224,f'224×224输入 · 16×16图块 · ViT · 约20%训练标签噪声 · seed={seed}',37,color='#607087',ha='left')
    text(3504,112,f'6张图片 × {len(widths)}个模型',37,color='#607087',ha='right')
    for x,fill,label in [(2700,'#e1f0e9','符合真值'),(3110,'#fae5e3','不符真值')]:
        rect(x,207,36,36,fill,9);text(x+52,224,label,31,color='#607087',ha='left')
    first_x,gap=444,14;cw=(3504-first_x-(len(widths)-1)*gap)/len(widths)
    text(252,326,'测试输入 / 真值',36)
    text(252,379,'同一行始终是同一张图',24,color='#607087')
    for j,d in enumerate(widths):
        x=first_x+j*(cw+gap);cx=x+cw/2
        if d==peak:rect(x-5,278,cw+10,1432,'#fff0db',20)
        text(cx,326,f'd = {d}',44 if len(widths)<12 else 38)
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
    text(96,1889,'模型容量增大 →   d为嵌入宽度；其余架构、数据和训练设置固定。',32,color='#607087',ha='left')
    text(96,1960,'介绍示例：种子、图片及展示容量点按结果选取；底部数量仅对应这6张图。全部12个容量点另附。',29,color='#607087',ha='left')
    for ext in ['png','pdf','svg']:
        fig.savefig(FIG/f'{stem}.{ext}',dpi=200,facecolor=fig.get_facecolor())
    plt.close(fig)
    return dict(widths=widths,correct_out_of_six=counts,file=f'figures/{stem}.png')


def cpu_verify(case,data):
    torch.set_num_threads(4);checks=[]
    for d in [int(case['width_early']),int(case['width_peak']),int(case['width_late'])]:
        run=OUT/'runs'/f'noise20_seed{int(case["seed"])}_width{d:03d}'
        cp=torch.load(run/'model.pt',map_location='cpu',weights_only=False)
        model=build_model(cp['model_config'],d).eval();model.load_state_dict(cp['state_dict'])
        assert model.position.shape[1]==197 and model.patch.kernel_size==(16,16)
        with torch.inference_mode():
            x=torch.from_numpy(data['x'][data['test']])
            logits=torch.cat([model(b) for b in x.split(16)]).numpy()
        expected=np.load(run/'logits.npz')['test']
        assert np.array_equal(logits.argmax(1),expected.argmax(1)),f'CPU/MPS predictions differ at {d}'
        checks.append(dict(width=d,seed=int(case['seed']),all_62_classes_match=True,
                           max_abs_logit_difference=float(abs(logits-expected).max())))
    (OUT/'checkpoint_cpu_verification.json').write_text(json.dumps(checks,indent=2))


def verify_all_checkpoints(results,predictions,data,samples):
    """Check every saved model really uses 224/16 and agrees with its stored predictions."""
    expected_hash=hashlib.sha256((OUT/'config.json').read_bytes()).hexdigest()
    checks=[]
    noisy=data['y_noise'][data['train']];clean=data['y'][data['train']]
    corrupt=noisy!=clean
    assert int(corrupt.sum())==59
    for row in results.itertuples():
        run=OUT/'runs'/row.run
        cp=torch.load(run/'model.pt',map_location='cpu',weights_only=False)
        state=cp['state_dict'];d=int(row.width)
        assert cp['config_sha256']==expected_hash==row.config_sha256
        assert cp['model_config']['image_size']==224 and cp['model_config']['patch_size']==16
        assert tuple(state['patch.weight'].shape)==(d,3,16,16)
        assert tuple(state['position'].shape)==(1,197,d)
        assert np.array_equal(cp['mean'],data['mean']) and np.array_equal(cp['std'],data['std'])
        assert sum(t.numel() for t in state.values())==int(row.parameters)
        logits=np.load(run/'logits.npz');train=logits['train'];test=logits['test']
        assert train.shape==(293,4) and test.shape==(62,4)
        assert np.isfinite(train).all() and np.isfinite(test).all()
        assert abs(float((train.argmax(1)!=noisy).mean())-row.train_observed_error)<1e-6
        assert abs(float((train.argmax(1)!=clean).mean())-row.train_clean_error)<1e-6
        assert abs(float((train.argmax(1)[corrupt]==noisy[corrupt]).mean())-row.noisy_memorization)<1e-6
        pp=predictions[predictions.run==row.run].set_index('sample_id')
        ordered_ids=samples.iloc[data['test']].sample_id.tolist()
        pp=pp.loc[ordered_ids]
        assert np.array_equal(test.argmax(1),pp.predicted_label.to_numpy())
        confidence=torch.from_numpy(test).softmax(1).max(1).values.numpy()
        assert abs(confidence-pp.confidence.to_numpy()).max()<1e-6
        checks.append(dict(run=row.run,width=d,parameters=int(row.parameters),input_size=224,
                           patch_size=16,position_tokens=197,stored_logits_match_metrics=True))
    (OUT/'checkpoint_integrity.json').write_text(json.dumps(dict(status='passed',runs=len(checks),checks=checks),indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    FIG.mkdir(exist_ok=True)
    config=json.loads((OUT/'config.json').read_text());widths=config['widths']
    results=pd.read_csv(OUT/'results.csv');predictions=pd.read_csv(OUT/'test_predictions.csv')
    if not args.partial:
        assert len(results)==36
        assert json.loads((OUT/'validation.json').read_text())['status']=='passed'
    assert set(results.noise_rate)=={.2} and not predictions.duplicated(['run','sample_id']).any()
    predictions['correct']=predictions.predicted_label==predictions.true_label
    case=select_case(results,predictions,widths);ids=choose_images(predictions,case,widths)
    metrics=results[results.seed==int(case['seed'])].set_index('width')
    with np.load(OUT/'data.npz') as cache:
        data={key:cache[key] for key in cache.files}
    samples=pd.read_csv(OUT/'splits.csv')
    if not args.partial:verify_all_checkpoints(results,predictions,data,samples)
    variants={}
    for count,tag in [(9,'9models'),(7,'7models'),(12,'all12models')]:
        cols=widths if count==12 else choose_columns(widths,case,count)
        variants[tag]=render_board(predictions,metrics,ids,cols,case,data,samples,'vit_predictions_'+tag)
    selected=predictions[(predictions.seed==int(case['seed']))&predictions.sample_id.isin(ids)].copy()
    selected['row_order']=selected.sample_id.map({sid:i+1 for i,sid in enumerate(ids)})
    selected['true_name']=selected.true_class.map(NAMES);selected['predicted_name']=selected.predicted_class.map(NAMES)
    selected['checkpoint']=selected.run.map(lambda r:f'runs/{r}/model.pt')
    assert len(selected)==72 and all((OUT/p).is_file() for p in selected.checkpoint)
    pcols=['p_'+c for c in NAMES]
    assert (selected[pcols].to_numpy().argmax(1)==selected.predicted_label).all()
    assert abs(selected[pcols].max(1)-selected.confidence).max()<1e-6
    selected.sort_values(['row_order','width']).to_csv(OUT/'selected_predictions.csv',index=False,encoding='utf-8-sig')
    # Full test and observed-training curves stay available alongside selected examples.
    fig,axs=plt.subplots(1,2,figsize=(13,4.5))
    for seed,rr in results.groupby('seed'):
        rr=rr.sort_values('width')
        for ax,metric in zip(axs,['test_error','train_observed_error']):
            ax.plot(rr.width,rr[metric],'o-',label=f'seed={seed}',linewidth=1.4,markersize=4)
    if not args.partial:
        summary=results.groupby('width').agg(
            runs=('run','count'),test_error_mean=('test_error','mean'),
            test_error_std=('test_error','std'),
            train_observed_error_mean=('train_observed_error','mean'),
            fully_fitted_runs=('interpolation_reached','sum')).loc[widths]
        assert (summary.runs==3).all()
        summary.to_csv(OUT/'summary.csv',encoding='utf-8-sig')
        for ax,metric in zip(axs,['test_error_mean','train_observed_error_mean']):
            ax.plot(widths,summary[metric],'--',color='#222222',linewidth=1.8,label='三种子均值')
    for ax,title in zip(axs,['全部62张测试图的错误率','对293个含噪训练标签的错误率']):
        ax.set_xscale('log',base=2);ax.set_xticks(widths);ax.set_xticklabels(widths,fontsize=8)
        ax.set_xlabel('嵌入宽度 d');ax.set_ylabel('错误率');ax.set_title(title)
        ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.15);ax.legend(fontsize=9)
    fig.tight_layout();fig.savefig(FIG/'complete_curves.png',dpi=180);plt.close(fig)
    if not args.partial:cpu_verify(case,data)
    manifest=dict(status='preliminary' if args.partial else 'completed',completed_runs=len(results),
        input_size=224,patch_size=16,patches=196,tokens_including_cls=197,
        selection='Post-hoc introduction illustration; seed, capacity triplet, images and display columns selected after inspecting final predictions.',
        selection_score='1000*min(rise,fall)+250*(rise+fall)+20*initial_improvement+2*class_coverage+0.5*min(6,flip_images)-20*peak_training_error',
        selected_case=case,sample_ids=ids,variants=variants,
        image_selection='Exact early-correct/peak-wrong/late-correct trajectory takes priority; cover four classes, at most two images per class. Stability plus a small visual-legibility preference breaks ties. Full image rankings retained.',
        visual_preferred_ids=sorted(VISUAL_PREFERRED),
        selected_full_test_rise_and_fall=bool(case['rise']>0 and case['fall']>0 and case['initial_improvement']>0),
        predictions_sha256=hashlib.sha256((OUT/'test_predictions.csv').read_bytes()).hexdigest(),
        note='Fixed 12-width, 3-seed noisy-training grid; not a confirmatory estimate. Probabilities are uncalibrated softmax confidence.')
    (OUT/'selection.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    a,b,c=[int(case[k]) for k in ['width_early','width_peak','width_late']]
    curve_description=(f'上升{case["rise"]*100:.2f}个百分点，随后下降{case["fall"]*100:.2f}个百分点。'
                       if case['rise']>0 and case['fall']>0 else
                       '完整测试曲线没有出现明确的先升后降；所附图片只能展示逐图预测变化。')
    lines=['# 224×224输入的ViT重跑结果','',
        f'完成{len(results)}/36组训练。12个宽度×3个种子，训练集含59/293个固定错误标签；验证/测试各62张且标签正确。', '',
        '输入：RGB，保持长宽比双线性缩放并居中补白到224×224，使用训练集计算的通道均值/标准差。三个集合均使用同一确定性预处理，不做数据增强。16×16图块，每块768维像素，196个图块加CLS后共197个token。', '',
        '模型：从零训练的4层、2头SmallViT，MLP宽度2d；只扫描d。Adam，学习率0.001余弦降至0.00001，400轮，batch=64，无权重衰减或dropout，使用末轮权重。采用224/16输入规格，不等同于训练86M参数的ViT-B/16。', '',
        f'按要求选择seed={int(case["seed"])}的介绍示例。全部测试错误率在d={a}、{b}、{c}分别为{metrics.loc[a,"test_error"]:.2%}、{metrics.loc[b,"test_error"]:.2%}、{metrics.loc[c,"test_error"]:.2%}。'+curve_description, '',
        '种子、图片和展示列按结果选择；6张图覆盖四个类别，优先选早期正确、中间错误、后期恢复的图片，再兼顾预测稳定性及角色是否清晰易认。图片展示模型实际接收的224×224补白输入。底部计数只对应这6张，不是整个测试集准确率。完整候选排名和全部预测均保留。', '',
        '所选中间模型的含噪训练错误率为'+f'{metrics.loc[b,"train_observed_error"]:.2%}'+'；没有将其直接称为精确插值阈值，也未测量决策边界或验证论文机制。', '',
        '| d | 参数量 | 训练错误/293 | 全部测试错误/62 | 展示图判对/6 |',
        '|---:|---:|---:|---:|---:|']
    for d in widths:
        rr=metrics.loc[d];ss=selected[selected.width==d]
        lines.append(f'|{d}|{int(rr.parameters):,}|{round(rr.train_observed_error*293)}|{round(rr.test_error*62)}|{int(ss.correct.sum())}|')
    lines+=['','![推荐9列版](figures/vit_predictions_9models.png)','',
             '![7列版](figures/vit_predictions_7models.png)','',
             '![完整12列](figures/vit_predictions_all12models.png)','',
             '![全部曲线](figures/complete_curves.png)','',
             '完成整轮后，summary.csv另存每个宽度的三种子均值、标准差及完全拟合的运行数。','',
             '复现：在experiment目录执行 `python3 run_vit224.py`。旧32px训练产物已按要求清理；原分割/噪声清单保留以核验本轮224px输入。']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (OUT/'report_completed.json').write_text(json.dumps(dict(status=manifest['status'],completed_runs=len(results))))
    print(json.dumps(manifest,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
