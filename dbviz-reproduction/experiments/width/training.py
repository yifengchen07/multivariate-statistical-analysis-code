"""Width-dependent CIFAR-10 training and data; dbViz commit aeea84c."""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import CIFAR10

from .resnet18k import make_resnet18k


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".new")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def save_weights(path, obj):
    tmp = path.with_suffix(".new.pt")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def noisy_labels(true):
    """Faithful to double-descent/data_noisy.py: reset RandomState(42) per class."""
    labels = true.copy()
    candidates = []
    for cls in range(10):
        rng = np.random.RandomState(42)
        group = np.flatnonzero(true == cls)
        candidates.extend(rng.choice(group, len(group) // 5, replace=False))
    for idx in candidates:
        other = np.delete(np.arange(10), true[idx])
        labels[idx] = rng.choice(other)
    return labels


class Data:
    def __init__(self, root, device):
        a, b = (CIFAR10(root, train=flag, download=False) for flag in (True, False))
        self.true = np.asarray(a.targets)
        self.corrupt = noisy_labels(self.true)
        self.test_y = np.asarray(b.targets)
        self.train = torch.as_tensor(a.data, device=device)
        self.test = torch.as_tensor(b.data, device=device)
        self.device = device
        self.mean = torch.tensor([.4914, .4822, .4465], device=device).view(1,3,1,1)
        self.std = torch.tensor([.2023, .1994, .2010], device=device).view(1,3,1,1)
        self.padded = torch.nn.functional.pad(self.train, (0,0,4,4,4,4))
        self.pixels = torch.arange(32, device=device)

    def norm(self, x):
        return (x.permute(0,3,1,2).float().div_(255) - self.mean) / self.std

    def augment(self, ids, offsets, flips):
        p = self.pixels
        ys = offsets[0,:,None] + p
        xs = offsets[1,:,None] + torch.where(flips[:,None], 31-p, p)
        return self.norm(self.padded[ids[:,None,None], ys[:,:,None], xs[:,None,:]])


def job_dir(root, k, noise, seed):
    return root / "models" / f"k{k:02d}-n{noise:02d}-s{seed}"


def train(root, data, cfg, job, remaining):
    k,noise,seed = job
    folder = job_dir(root, *job)
    folder.mkdir(parents=True, exist_ok=True)
    final = folder / "model.pt"
    resume = folder / "resume.pt"
    metrics_file = folder / "metric.json"
    if final.exists() and metrics_file.exists():
        # Final save can precede cleanup if the process is interrupted.
        if resume.exists():
            resume.unlink()
        return 0, True
    epochs = cfg["epochs"]
    torch.manual_seed(seed)
    model = make_resnet18k(k).to(device=data.device, memory_format=torch.channels_last)
    opt = torch.optim.Adam(model.parameters(), lr=0.0001)
    done = 0
    if resume.exists():
        ck = torch.load(resume, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["weights"])
        opt.load_state_dict(ck["optimizer"])
        done = ck["epoch"]
    if remaining == 0 and done < epochs:
        return 0, False
    labels = torch.as_tensor(data.true if noise == 0 else data.corrupt, device=data.device)
    n = cfg["train_size"]
    batch = min(128, n)
    indices = torch.randperm(len(labels), generator=torch.Generator().manual_seed(29))[:n]
    if n == len(labels):
        indices = torch.arange(n)
    target_epoch = epochs if remaining is None else min(epochs, done + remaining)
    start = time.monotonic()
    for epoch in range(done+1, target_epoch+1):
        # Shared epoch-specific streams make model initialization independent of shuffle/crop.
        gen = torch.Generator().manual_seed(1000000 + epoch)
        order = indices[torch.randperm(n, generator=gen)].to(data.device)
        offsets = torch.randint(9,(2,n),generator=gen).to(data.device)
        flips = (torch.rand(n, generator=gen) < .5).to(data.device)
        model.train()
        for i in range(0,n,batch):
            ids = order[i:i+batch]
            inp = data.augment(ids,offsets[:,i:i+batch],flips[i:i+batch])
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(inp), labels[ids])
            loss.backward()
            opt.step()
        if epoch == done+1 or epoch % 10 == 0 or epoch == target_epoch:
            loss_value = loss.item()
            if not np.isfinite(loss_value):
                raise RuntimeError(f"non-finite loss for {job} at epoch {epoch}")
            save_weights(resume, {"epoch":epoch,"weights":model.state_dict(),"optimizer":opt.state_dict()})
            sec = (time.monotonic()-start)/(epoch-done)
            print(f"{folder.name}: epoch {epoch}/{epochs}, last-batch loss {loss_value:.4f}, {sec:.2f} s/epoch",flush=True)
    if target_epoch < epochs:
        return target_epoch - done, False
    count = cfg["test_size"]
    with torch.inference_mode():
        model.eval()
        test_y = torch.as_tensor(data.test_y[:count], device=data.device)
        hits = torch.zeros((),dtype=torch.long, device=data.device)
        for i in range(0,count,128):
            x = data.norm(data.test[i:min(i+128,count)])
            hits += (model(x).argmax(1)==test_y[i:i+128]).sum()
        correct = int(hits.item())
    save_weights(final, {key:t.detach().cpu() for key,t in model.state_dict().items()})
    atomic_json(metrics_file, {"width":k,"noise_percent":noise,"init_seed":seed,
                              "epochs":epochs,"test_samples":count,"correct":correct,
                              "test_error":1-correct/count,"device":str(data.device),
                              "torch":str(torch.__version__),"cuda":torch.version.cuda})
    if resume.exists():
        resume.unlink()
    return target_epoch - done, True


def load_model(root, data, k, noise, seed):
    path = job_dir(root,k,noise,seed) / "model.pt"
    model = make_resnet18k(k).to(device=data.device,memory_format=torch.channels_last)
    model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
    return model.eval()
