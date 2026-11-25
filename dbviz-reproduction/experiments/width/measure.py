"""Measurements for arXiv v1 Figures 7–12. All results derive from model predictions."""
import json
import os
import random

import numpy as np
import torch

from .training import atomic_json, load_model


def plane(model, data, source, ids, grid, margin, with_confidence=False):
    """Gram–Schmidt plane, using dbViz's ranges and flattening order."""
    images = data.norm(source[torch.tensor(ids, device=data.device)])
    base = images[0]
    x = images[1]-base
    nx = torch.linalg.vector_norm(x)
    x = x / nx
    b = images[2]-base
    bx = torch.sum(b*x)
    y = b - bx*x
    by = torch.linalg.vector_norm(y)
    y = y / by
    a, b2, c = nx.item(), bx.item(), by.item()
    if min(a,c) < 1e-6 or not np.isfinite([a,b2,c]).all():
        raise ValueError(f"Degenerate CIFAR triplet {ids}")
    low, high = min(0,b2), max(a,b2)
    delta = high-low
    extent = (low-margin*delta,high+margin*delta,-margin*c,(1+margin)*c)
    gx = torch.linspace(*extent[:2], grid, device=data.device)
    gy = torch.linspace(*extent[2:], grid, device=data.device)
    labels, conf = [], []
    for i in range(0, grid*grid, 128):
        ii=torch.arange(i,min(i+128,grid*grid),device=data.device)
        batch = base + gx[ii//grid,None,None,None]*x + gy[ii%grid,None,None,None]*y
        logits = model(batch)
        labels.append(logits.argmax(1).to(torch.uint8))
        if with_confidence:
            conf.append((logits/5.0).softmax(1).amax(1))
    result = {"label":torch.cat(labels).reshape(grid,grid).T.cpu().numpy(),
              "extent":np.asarray(extent),"xy":np.array([[0.,0.],[a,0.],[b2,c]]),
              "ids":np.asarray(ids)}
    if conf:
        result["confidence"] = torch.cat(conf).reshape(grid,grid).T.cpu().numpy()
    return result


def fixed_panels(figure, data):
    if figure == 7:
        return [(672,766,781),(75,32,304)]
    if figure == 8:
        return [(989,417,93),(41,1024,5707),(994,412,52)]
    bad = np.flatnonzero((data.true==1)&(data.corrupt!=1))
    good = np.flatnonzero((data.true==1)&(data.corrupt==1))
    # The plotting shell provides candidate ordinals; retain the first matching
    # candidate in each of the three published corrupted classes.
    candidates = bad[[11,12,15,16,18,29,38,45,76,89]]
    return [(int(good[34]),int(good[12]),int(candidates[data.corrupt[candidates]==cls][0]))
            for cls in (2,7,8)]


def components(labels):
    """8-neighbour components per label, equivalent to skimage connectivity=2."""
    try:
        from scipy.ndimage import label
    except ImportError:
        from collections import deque
        a = labels.copy()
        rows, cols = a.shape
        total = 0
        for r,c in np.ndindex(rows,cols):
            if a[r,c]==255:
                continue
            kind=a[r,c]
            total += 1
            a[r,c]=255
            todo=deque([(r,c)])
            while todo:
                y,x=todo.pop()
                for yy in range(max(0,y-1),min(rows,y+2)):
                    for xx in range(max(0,x-1),min(cols,x+2)):
                        if a[yy,xx]==kind:
                            a[yy,xx]=255
                            todo.append((yy,xx))
        return total
    return int(sum(label(labels==cls, structure=np.ones((3,3),dtype=bool))[1]
                   for cls in np.unique(labels)))


def distinct_classes(targets, number):
    rng=random.Random(1+number)
    ids, used=[],set()
    while len(ids)<3:
        j=rng.randrange(len(targets))
        if int(targets[j]) not in used:
            ids.append(j)
            used.add(int(targets[j]))
    return ids


def deltas():
    values = [0.]
    for index in range(199):
        if index<=10: step = .01
        elif index<20: step = .1
        elif index<=50: step = .5
        elif index<=75: step = 1.
        elif index<=100: step = 10.
        else: step = 20.
        values.append(values[-1]+step)
    return np.asarray(values)


def margin_for_sample(model, data, pool, index, cfg):
    rng = np.random.default_rng(2305+index)
    i = int(rng.choice(pool))
    base = data.norm(data.train[i:i+1])
    trials = rng.integers(0,len(data.true),size=cfg["rays"]+10)
    distances = deltas()
    steps = torch.as_tensor(distances,device=data.device,dtype=torch.float32)
    margins = []
    for j in trials:
        if j == i:
            continue
        diff = data.norm(data.train[int(j):int(j)+1])-base
        magnitude = torch.linalg.vector_norm(diff)
        if magnitude.item()==0:
            continue
        unit=diff/magnitude
        # First adjacent prediction change. Probe the published nonuniform grid
        # in order; stop as soon as the first bracket is observed.
        previous = None
        found = False
        for pos in range(0,len(distances),40):
            batch=base+steps[pos:pos+40,None,None,None]*unit
            pred=model(batch).argmax(1).cpu().numpy()
            if previous is not None and previous!=pred[0]:
                margins.append(float(distances[pos-1]))
                found=True
                break
            changes=np.flatnonzero(pred[:-1]!=pred[1:])
            if len(changes):
                margins.append(float(distances[pos+int(changes[0])]))
                found=True
                break
            previous=pred[-1]
        if found and len(margins)>=cfg["rays"]:
            break
    if len(margins)>=max(1,cfg["rays"]//2):
        return float(np.mean(margins))
    return None


def evaluate_cell(root, data, cfg, figure, k, noise):
    stem = f"fig{figure:02d}-k{k:02d}-n{noise:02d}"
    path=root/"measures"/(stem+".json")
    if path.exists():
        result=json.loads(path.read_text())
        if result.get("complete"):
            return
    else:
        result={"figure":figure,"width":k,"noise_percent":noise,"complete":False}
    model=load_model(root,data,k,noise,1)
    if figure in (7,8,9):
        records=[]
        for i,idx in enumerate(fixed_panels(figure,data)):
            panel=root/"measures"/f"{stem}-p{i+1}.npz"
            if not panel.exists():
                grid=plane(model,data,data.train,list(idx),cfg["panel_grid"],.5,True)
                tmp=panel.with_suffix(".new.npz")
                np.savez_compressed(tmp,**grid)
                os.replace(tmp,panel)
            records.append({"file":panel.name,"ids":list(idx),
                            "true_labels":data.true[list(idx)].tolist(),
                            "training_labels":(data.corrupt if noise else data.true)[list(idx)].tolist()})
        result["panels"]=records
    elif figure in (10,11,12):
        groups={10:("random_train","same_class_correct"),11:("test_agreement",),
                12:(("all","correct","mislabeled") if noise else ("all",))}[figure]
        all_ids=np.arange(len(data.true))
        corr=np.flatnonzero(data.true==data.corrupt)
        same=[corr[data.true[corr]==c] for c in range(10)]
        other=load_model(root,data,k,noise,25) if figure==11 else None
        for group in groups:
            records=result.setdefault(group,[])
            N=cfg["margin_points"] if figure==12 else cfg["planes"]
            pool = {"all":all_ids,"correct":corr,"mislabeled":np.flatnonzero(data.true!=data.corrupt)}.get(group)
            for n in range(len(records),N):
                if figure==10:
                    rng=np.random.default_rng(75007+n)
                    group_ids=all_ids if group=="random_train" else same[n%10]
                    idx=rng.choice(group_ids,3,replace=False).tolist()
                    pred=plane(model,data,data.train,idx,cfg["score_grid"],.1)["label"]
                    value=components(pred)
                elif figure==11:
                    idx=distinct_classes(data.test_y,n)
                    p=plane(model,data,data.test,idx,cfg["score_grid"],.1)["label"]
                    q=plane(other,data,data.test,idx,cfg["score_grid"],.1)["label"]
                    value=int((p==q).sum())
                else:
                    value=margin_for_sample(model,data,pool,n,cfg)
                records.append(value)
                if (n+1)%10==0 or n+1==N:
                    atomic_json(path,result)
                if (n+1)%100==0:
                    print(f"{stem} {group}: {n+1}/{N}",flush=True)
            kept=[v for v in records if v is not None]
            if figure==12:
                result[group+"_usable_images"]=len(kept)
                result[group+"_median"]=float(np.median(kept)) if kept else None
            else:
                mean=float(np.mean(kept))
                result[group+"_mean"]=mean/(cfg["score_grid"]**2) if figure==11 else mean
    result["complete"]=True
    atomic_json(path,result)
    print("Measured",stem,flush=True)
