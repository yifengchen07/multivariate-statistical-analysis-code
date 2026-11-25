"""Train the CIFAR-10 models needed for figures 1-3, resuming at epoch boundaries."""
import argparse
import fcntl
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F
from torchvision.datasets import CIFAR10

from .models import checkpoint, create

# Rates for resnet, vgg, vit and mixer follow train_optimizers.sh (SGD).
# Other rates are explicit reproduction choices in the paper's search grid.
RATES = {"resnet": .05, "vgg": .005, "vit": .02, "mixer": .02, "fc": .05,
         "dense": .05, "wrn10": .05}
# Optimizer batch = 128 in every official run. Heavy nets use accumulated gradients.
MICRO = {"dense": 32, "wrn10": 32,
         "vit": 32, "mixer": 32, "vgg": 64}
IMPORTED_RESNET = {0: "5e0df4392497752d870123162efff6fb112b75f89a4caf96356291b9d6195156",
                   1: "c43aef6fa42c804129db230e6e57bb747f4ed9950178a3690d25eb7db9c790e0"}


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def initial_signature(model):
    result = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        result.update(key.encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


@dataclass(frozen=True)
class Job:
    name: str
    seed: int

    def dest(self, work):
        return checkpoint(work, self.name, self.seed)


def jobs(figures):
    selected = []
    for fig, names, seeds in [
        (1, ("fc", "vit", "wrn10"), range(2)),
        (2, ("vit", "dense", "resnet"), range(1)),
        (3, ("fc", "mixer", "vit", "vgg", "dense", "resnet", "wrn10"), range(3)),
    ]:
        if fig in figures:
            selected.extend(Job(name, seed) for name in names for seed in seeds)
    return list(dict.fromkeys(selected))


def save(path, obj, json_text=False):
    temp = path.with_suffix(path.suffix + ".partial")
    if json_text:
        temp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf8")
    else:
        torch.save(obj, temp)
    os.replace(temp, path)


def inputs(data):
    train = CIFAR10(str(data), train=True, download=False)
    test = CIFAR10(str(data), train=False, download=False)
    raw = torch.from_numpy(train.data)
    return (F.pad(raw, (0, 0, 4, 4, 4, 4)), torch.tensor(train.targets),
            torch.from_numpy(test.data), torch.tensor(test.targets))


def augment(padded, epoch):
    n = len(padded)
    generator = torch.Generator().manual_seed(1_000_000 + epoch)
    order = torch.randperm(n, generator=generator)
    offsets = torch.randint(9, (2, n), generator=generator)
    flips = torch.rand(n, generator=generator) < .5
    pixel = torch.arange(32)
    for start in range(0, n, 128):
        cut = slice(start, start + 128)
        ids = order[cut]
        ys = offsets[0, cut, None] + pixel
        xs = offsets[1, cut, None] + torch.where(flips[cut, None], 31 - pixel, pixel)
        yield ids, padded[ids[:, None, None], ys[:, :, None], xs[:, None, :], :]


def normalized(raw, device, mean, std):
    x = raw.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).div_(255)
    return (x - mean) / std


def forward(net, name, x):
    if name == "fc":
        x = x.contiguous()
    return net(x)


@torch.inference_mode()
def accuracy(model, name, raw, y, batch, device, mean, std):
    model.eval()
    correct = torch.zeros((), device=device, dtype=torch.int64)
    for i in range(0, len(y), batch):
        logits = forward(model, name, normalized(raw[i:i+batch], device, mean, std))
        correct += (logits.argmax(1) == y[i:i+batch].to(device)).sum()
    return correct.item() / len(y)


def train_epoch(job, model, optimizer, arrays, epoch, micro, device, mean, std):
    padded, labels, _, _ = arrays
    model.train()
    sums = torch.zeros(2, device=device, dtype=torch.float64 if device.type != "mps" else torch.float32)
    for ids, cropped in augment(padded, epoch):
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(ids), micro):
            cut = slice(start, start + micro)
            x = normalized(cropped[cut], device, mean, std)
            y = labels[ids[cut]].to(device)
            logits = forward(model, job.name, x)
            loss = F.cross_entropy(logits, y)
            (loss * (len(y) / len(ids))).backward()
            sums[0] += loss.detach() * len(y)
            sums[1] += (logits.detach().argmax(1) == y).sum()
        optimizer.step()
    loss, accuracy_value = (sums / len(padded)).tolist()
    if not torch.isfinite(torch.tensor(loss)):
        raise ValueError(f"Non-finite loss on {job} epoch {epoch+1}")
    return loss, accuracy_value


def lr_at(rate, epochs, epoch):
    return rate * .1 ** sum(epoch >= int(epochs / d) for d in (2.667, 1.6, 1.142))


def run_job(job, args, arrays):
    path = job.dest(args.work)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run_locked(job, args, arrays, path)


def run_locked(job, args, arrays, path):
    dev = torch.device(args.device)
    contract = {"recipe": 1, "arch": job.name, "seed": job.seed, "epochs": args.epochs, "lr": RATES[job.name],
                "optimizer_batch": 128, "micro": args.microbatch or MICRO.get(job.name, 128),
                "teacher": "", "teacher_sha256": None,
                "control": False, "cifar_samples": [len(arrays[0]), len(arrays[2])]}
    micro = contract["micro"]
    record = path.with_suffix(".json")
    resume = path.with_suffix(".resume.pt")
    if path.is_file() and record.is_file():
        meta = json.loads(record.read_text())
        if meta["config"] != contract or meta["sha256"] != digest(path):
            raise RuntimeError(f"Completed file/config mismatch: {path}. Use a separate --work for a changed recipe.")
        print(f"Reusing {path}", flush=True)
        return
    if path.is_file() and not resume.is_file() and not record.is_file():
        imported = (job.name == "resnet" and args.epochs == 100
                    and len(arrays[0]) == 50000 and digest(path) == IMPORTED_RESNET.get(job.seed))
        if not imported:
            raise RuntimeError(f"Unverified existing checkpoint: {path}")
        print(f"Reusing verified imported epoch-100 weights {path}", flush=True)
        return
    torch.manual_seed(job.seed)
    model = create(job.name)
    init_hash = initial_signature(model)
    model = model.to(dev, memory_format=torch.channels_last) if job.name == "resnet" else model.to(dev)
    optimizer = torch.optim.SGD(model.parameters(), lr=RATES[job.name], momentum=.9, weight_decay=.0005,
                                foreach=False)
    history = []
    if resume.is_file():
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if state["config"] != contract or state["init_hash"] != init_hash:
            raise RuntimeError(f"Resume configuration/initialization mismatch: {resume}")
        model.load_state_dict(state["weights"])
        optimizer.load_state_dict(state["optimizer"])
        history = state["history"]
        del state
    mean = torch.tensor([.4914, .4822, .4465], device=dev).view(1,3,1,1)
    std = torch.tensor([.2023, .1994, .2010], device=dev).view(1,3,1,1)
    print(f"Training {path} from epoch {len(history)+1}/{args.epochs} microbatch={micro}", flush=True)
    for epoch in range(len(history), args.epochs):
        start = time.monotonic()
        # Fix stochastic dropout separately from the CPU data plan, including after a restart.
        torch.manual_seed(10_000_000 + 1000*job.seed + epoch)
        lr = lr_at(RATES[job.name], args.epochs, epoch)
        optimizer.param_groups[0]["lr"] = lr
        loss, training_acc = train_epoch(job, model, optimizer, arrays, epoch, micro, dev, mean, std)
        test_acc = accuracy(model, job.name, arrays[2], arrays[3], micro, dev, mean, std)
        row = dict(epoch=epoch+1, lr=lr, loss=loss, train_accuracy=training_acc,
                   test_accuracy=test_acc, seconds=time.monotonic()-start)
        history.append(row)
        save(resume, dict(config=contract, init_hash=init_hash, weights=model.state_dict(),
                          optimizer=optimizer.state_dict(), history=history))
        print(path.parent.name, path.stem, json.dumps(row), flush=True)
    save(path, {k: v.detach().cpu() for k, v in model.state_dict().items()})
    save(record, {"config": contract, "init_sha256": init_hash, "sha256": digest(path),
                  "torch": str(torch.__version__), "device": str(dev), "history": history}, True)
    resume.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--figures", type=int, choices=(1, 2, 3), nargs="+", required=True)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--microbatch", type=int, default=0, help="Override only in a separate working directory")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or not 0 <= args.microbatch <= 128:
        parser.error("epochs must be positive; microbatch must be 0..128")
    todo = jobs(args.figures)
    if args.plan:
        for job in todo:
            p = job.dest(args.work)
            print("on disk" if p.exists() else "pending", p)
        print(f"{len(todo)} model files; {sum(not j.dest(args.work).is_file() for j in todo)} missing.")
        return
    torch.set_num_threads(4)
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    arrays = inputs(args.data)
    for job in todo:
        run_job(job, args, arrays)


if __name__ == "__main__":
    main()
