"""Train and evaluate the width/noise experiments for arXiv v1 figures 6, 7, 10 and 12."""
import argparse
import fcntl
import json
from pathlib import Path

PAPER_WIDTHS = list(range(1,21)) + [30,40,50,60,64]
PANEL_WIDTHS = {7:[1,4,7,10,20,64]}
MEASURED_WIDTHS = [1, 2, 4, 6, 7, 8, 10, 12, 20, 40, 64]
FIGURES = [6, 7, 10, 12]
PROFILES = {
    "paper":dict(epochs=4000,train_size=50000,test_size=10000,panel_grid=500,planes=1000,score_grid=50,margin_points=5000,rays=10),
    "measured":dict(epochs=500,train_size=50000,test_size=10000,panel_grid=500,planes=1000,score_grid=50,margin_points=5000,rays=10),
    "pilot":dict(epochs=25,train_size=50000,test_size=10000,panel_grid=100,planes=20,score_grid=32,margin_points=30,rays=3),
    "smoke":dict(epochs=1,train_size=16,test_size=32,panel_grid=10,planes=3,score_grid=6,margin_points=2,rays=2),
}


def ints(text):
    return list(dict.fromkeys(int(i.strip()) for i in text.split(",")))


def selection(profile,figures,selected):
    curves = {"paper": PAPER_WIDTHS, "measured": MEASURED_WIDTHS}.get(profile, [1,10,64])
    result={}
    jobs=set()
    for f in sorted(figures):
        ideal=PANEL_WIDTHS.get(f,curves)
        w=[k for k in ideal if selected is None or k in selected]
        if not w:
            continue
        result[f]=(w,w!=ideal or (profile!="paper" and f not in PANEL_WIDTHS))
        for k in w:
            for n in (0,20):
                jobs.add((k,n,1))
    return result,sorted(jobs)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=["plan","train","evaluate","plot","all"],default="plan")
    p.add_argument("--profile",choices=PROFILES,default="measured")
    p.add_argument("--data",type=Path,default=Path("datasets"),help="directory containing cifar-10-batches-py")
    p.add_argument("--out",type=Path,default=Path("runs/width"))
    p.add_argument("--figures",type=ints,default=FIGURES,help="e.g. 7 or 6,7,10,12")
    p.add_argument("--widths",type=ints,help="optional subset of the reference grid")
    p.add_argument("--device",choices=["cpu","cuda","mps"],default="cuda")
    p.add_argument("--max-epochs-this-run",type=int,help="stop after this many additional model epochs across jobs")
    a=p.parse_args()
    if not a.figures or set(a.figures)-set(FIGURES):
        p.error("--figures must be a subset of 6,7,10,12")
    if a.widths is not None and (not a.widths or set(a.widths)-set(PAPER_WIDTHS)):
        p.error("--widths must be a subset of 1..20,30,40,50,60,64")
    if a.max_epochs_this_run is not None and a.max_epochs_this_run<1:
        p.error("--max-epochs-this-run must be positive")
    needed,jobs=selection(a.profile,a.figures,a.widths)
    if not jobs:
        p.error("width selection has no overlap with selected figures")
    cfg={"schema":1,"profile":a.profile,**PROFILES[a.profile],"optimizer":"Adam","lr":.0001,
         "batch":128,"data_seed":1,"reference_commit":"aeea84c42a32efafebbb29c3d944204aa74963cf"}
    root=a.out.resolve()/a.profile
    def model_folder(job):
        return root / "models" / ("k%02d-n%02d-s%d"%job)
    complete=sum((model_folder(j)/"metric.json").exists() and (model_folder(j)/"model.pt").exists() for j in jobs)
    print(f"Profile {a.profile}; experiment: {root}\nSelected models: {len(jobs)}; completed: {complete}; "
          f"target: {cfg['epochs']} epochs each ({len(jobs)*cfg['epochs']:,} model epochs from scratch).")
    for f,(widths,sub) in needed.items():
        print(f"Figure {f}: widths {','.join(map(str,widths))}"+(" [subset]" if sub else ""))
    if a.stage=="plan":
        return
    import torch
    from .training import Data, atomic_json, train
    from .measure import evaluate_cell
    from .plots import render
    if a.stage!="plot" and not (a.data/"cifar-10-batches-py").is_dir():
        p.error(f"Missing extracted CIFAR-10: {a.data}/cifar-10-batches-py")
    if a.stage!="plot" and a.device=="cuda" and not torch.cuda.is_available():
        p.error("CUDA is unavailable; select a CUDA PyTorch image or explicitly use --device cpu")
    torch.set_num_threads(2 if a.device=="cpu" else 4)
    root.mkdir(parents=True,exist_ok=True)
    with (root/".lock").open("a") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            p.error("Another process is using this output profile")
        for name in ("measures","figures"):
            (root/name).mkdir(exist_ok=True)
        conf=root/"config.json"
        if conf.exists() and json.loads(conf.read_text())!=cfg:
            p.error("Existing config differs; choose a new --out")
        atomic_json(conf,cfg)
        data=Data(a.data,a.device) if a.stage!="plot" else None
        if a.stage in ("train","all"):
            remaining=a.max_epochs_this_run
            for job in jobs:
                used,done=train(root,data,cfg,job,remaining)
                if remaining is not None:
                    remaining-=used
                if not done:
                    print("Epoch budget consumed; checkpoint saved. Repeat the command to continue.")
                    return
        if a.stage in ("all","evaluate"):
            absent=[j for j in jobs if not (model_folder(j)/"model.pt").exists() or not (model_folder(j)/"metric.json").exists()]
            if absent:
                p.error(f"Missing final checkpoints ({len(absent)}); run --stage train or all")
            with torch.inference_mode():
                for f,(widths,_) in needed.items():
                    if f==6:
                        continue
                    for k in widths:
                        for n in (0,20):
                            evaluate_cell(root,data,cfg,f,k,n)
        if a.stage in ("all","plot"):
            for f,(widths,partial) in needed.items():
                render(root,cfg,f,widths,partial)


if __name__=="__main__":
    main()
