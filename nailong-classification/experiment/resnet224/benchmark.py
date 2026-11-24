"""Measure actual 224px training batches on the selected device."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import torch
from torch.nn import functional as F
from resnet224_models import build_model

BASE = Path(__file__).resolve().parent


def select_device(requested):
    device = requested
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable in this session. Run from a native Mac terminal or use a CUDA server.')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; install CUDA-enabled PyTorch in a GPU environment.')
    return device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    args = parser.parse_args()
    device = select_device(args.device)
    torch.set_num_threads(4)
    config = json.loads((BASE / 'config.json').read_text())
    with np.load(BASE / 'data.npz') as data:
        idx = data['train'][:config['batch_size']]
        x = torch.from_numpy(data['x'][idx]).to(device)
        y = torch.from_numpy(data['y_noise'][idx]).to(device)
    def synchronize():
        if device == 'mps':
            torch.mps.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()
    rows = []
    for width in [1, 8, 32, 64]:
        torch.manual_seed(42)
        model = build_model(config, width).to(device).train()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        seconds = []
        for step in range(5):
            synchronize()
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            synchronize()
            if step >= 2:
                seconds.append(time.perf_counter() - start)
        row = dict(width=width, device=device, batch_size=len(x),
                   parameters=sum(p.numel() for p in model.parameters()),
                   median_training_batch_seconds=float(np.median(seconds)))
        rows.append(row)
        print(json.dumps(row), flush=True)
        del model, optimizer
        if device == 'mps':
            torch.mps.empty_cache()
        elif device == 'cuda':
            torch.cuda.empty_cache()
    per_width = np.interp(np.square(config['widths']),
                          np.square([r['width'] for r in rows]),
                          [r['median_training_batch_seconds'] for r in rows])
    # Accounts for 293 images rather than five full batches; 15% approximate evaluation overhead.
    estimated = float(per_width.sum() * (293 / len(x)) * config['epochs'] * len(config['seeds']) * 1.15)
    report = dict(device=device, torch_version=torch.__version__, measurements=rows,
                  estimated_full_sweep_seconds=estimated,
                  estimate_note='Warm per-batch interpolation against squared base width, not an elapsed-time guarantee.')
    (BASE / f'benchmark_{device}.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(estimated_full_sweep_hours=estimated / 3600)), flush=True)


if __name__ == '__main__':
    main()
