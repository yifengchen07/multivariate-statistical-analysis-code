"""Fixed width/noise/seed sweep. All primary results use the last epoch."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from resnet224_models import build_model

HERE = Path(__file__).resolve().parent
BASE = HERE


def write_csv(path, rows):
    if not rows:
        return
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def main():
    global HERE
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='auto', choices=['auto', 'mps', 'cpu', 'cuda'])
    parser.add_argument('--output-dir', type=Path, default=BASE)
    args = parser.parse_args()
    HERE = args.output_dir.resolve()
    config_bytes = (HERE / 'config.json').read_bytes()
    config = json.loads(config_bytes)
    digest = hashlib.sha256(config_bytes).hexdigest()
    fingerprint_path = HERE / 'input_fingerprints.json'
    if fingerprint_path.exists():
        for relative, expected in json.loads(fingerprint_path.read_text()).items():
            if hashlib.sha256((HERE / relative).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'Frozen experiment input changed: {relative}')
    device = ('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')) if args.device == 'auto' else args.device
    if device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable in this execution context; use --device cpu or a host terminal.')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable.')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)
    data = np.load(HERE / 'data.npz')
    x = torch.from_numpy(data['x']).to(device)
    y = torch.from_numpy(data['y']).to(device)
    y_noise = torch.from_numpy(data['y_noise']).to(device)
    ids = {s: torch.from_numpy(data[s]).to(device) for s in ['train', 'validation', 'test']}
    with (HERE / 'splits.csv').open(encoding='utf-8-sig') as file:
        samples = list(csv.DictReader(file))
    runs = HERE / 'runs'
    runs.mkdir(exist_ok=True)
    metadata = dict(pid=os.getpid(), torch_version=torch.__version__, device=device,
                    config_sha256=digest, mps_available=torch.backends.mps.is_available(),
                    note='Seed controlled; reproducibility is not guaranteed bitwise across hardware/releases.')
    (HERE / 'runtime.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata), flush=True)
    all_results, predictions = [], []
    seeds = config['seeds']
    # Narrow-to-wide, noisy then clean, repeated for all predeclared initializations.
    jobs = [(seed, noise, width) for seed in seeds for noise in reversed(config['noise_rates']) for width in config['widths']]
    for run_number, (seed, noise, width) in enumerate(jobs, 1):
        tag = f'noise{int(noise * 100):02d}_seed{seed}_width{width:03d}'
        run_dir = runs / tag
        run_dir.mkdir(exist_ok=True)
        result_path = run_dir / 'result.json'
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result['config_sha256'] != digest:
                raise RuntimeError(f'Config differs from completed run: {tag}')
            all_results.append(result)
            with (run_dir / 'test_predictions.csv').open(encoding='utf-8-sig') as file:
                predictions.extend(list(csv.DictReader(file)))
            print(f'RESUMED {run_number}/{len(jobs)} {tag}', flush=True)
            continue
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device == 'mps':
            torch.mps.manual_seed(seed)
        model = build_model(config, width).to(device)
        parameters = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'], eta_min=.00001)
        train_target = y_noise if noise else y
        generator = torch.Generator().manual_seed(seed + 10000)
        history = []
        best_validation = float('inf')
        best_epoch = None
        start = time.perf_counter()
        print(f'START {run_number}/{len(jobs)} {tag} params={parameters}', flush=True)

        def evaluate(split, target):
            model.eval()
            idx = ids[split]
            with torch.inference_mode():
                logits = torch.cat([model(x[batch]) for batch in idx.split(config.get('eval_batch_size',128))])
                truth = target[idx]
                loss = F.cross_entropy(logits, truth).item()
                guess = logits.argmax(1)
                error = (guess != truth).float().mean().item()
            return loss, error, logits.detach().cpu().numpy()

        for epoch in range(1, config['epochs'] + 1):
            model.train()
            order = torch.randperm(len(ids['train']), generator=generator).to(device)
            train_indices = ids['train'][order]
            for batch in train_indices.split(config['batch_size']):
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x[batch]), train_target[batch])
                loss.backward()
                optimizer.step()
            scheduler.step()
            if epoch == 1 or epoch % 10 == 0 or epoch == config['epochs']:
                train_loss, train_error, train_logits = evaluate('train', train_target)
                val_loss, val_error, _ = evaluate('validation', y)
                if val_loss < best_validation:
                    best_validation, best_epoch = val_loss, epoch
                history.append(dict(epoch=epoch, train_observed_loss=train_loss,
                                    train_observed_error=train_error, validation_loss=val_loss,
                                    validation_error=val_error, lr=optimizer.param_groups[0]['lr']))
                write_csv(run_dir / 'history.csv', history)
                if epoch % 100 == 0 or epoch == 1:
                    print(f'  {tag} epoch={epoch} train_err={train_error:.4f} val_err={val_error:.4f} elapsed={time.perf_counter()-start:.1f}s', flush=True)
        train_loss, train_error, train_logits = evaluate('train', train_target)
        val_loss, val_error, _ = evaluate('validation', y)
        # The test set is queried once per final checkpoint, after fixed training.
        test_loss, test_error, test_logits = evaluate('test', y)
        train_guess = train_logits.argmax(1)
        train_clean = data['y'][data['train']]
        train_noisy = data['y_noise'][data['train']]
        corrupt = train_clean != train_noisy
        clean_train_error = float(np.mean(train_guess != train_clean))
        noisy_memorization = float(np.mean(train_guess[corrupt] == train_noisy[corrupt])) if noise else None
        test_truth = data['y'][data['test']]
        test_guess = test_logits.argmax(1)
        balanced_error = float(np.mean([np.mean(test_guess[test_truth == c] != c) for c in range(4)]))
        seconds = time.perf_counter() - start
        result = dict(run=tag, config_sha256=digest, seed=seed, noise_rate=noise, width=width,
                      parameters=parameters, epochs=config['epochs'], device=device, seconds=seconds,
                      train_observed_loss=train_loss, train_observed_error=train_error,
                      train_clean_error=clean_train_error, noisy_memorization=noisy_memorization,
                      validation_loss=val_loss, validation_error=val_error,
                      test_loss=test_loss, test_error=test_error, test_balanced_error=balanced_error,
                      interpolation_reached=bool(np.all(train_guess == (train_noisy if noise else train_clean))),
                      best_validation_epoch_diagnostic=best_epoch,
                      best_validation_loss_diagnostic=best_validation,
                      checkpoint_rule='last_epoch')
        checkpoint = dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                          width=width, blocks=config.get('blocks'), architecture=config['architecture'],
                          model_config=config, seed=seed, noise_rate=noise,
                          config_sha256=digest, mean=data['mean'], std=data['std'],
                          class_names=config['class_names'], result=result)
        torch.save(checkpoint, run_dir / 'model.pt')
        np.savez_compressed(run_dir / 'logits.npz', train=train_logits, test=test_logits)
        probs = torch.from_numpy(test_logits).softmax(1).numpy()
        run_predictions = []
        for j, idx in enumerate(data['test']):
            sample = samples[int(idx)]
            pred = dict(run=tag, seed=seed, noise_rate=noise, width=width,
                        sample_id=sample['sample_id'], relative_path=sample['relative_path'],
                        true_label=int(test_truth[j]), true_class=config['class_names'][int(test_truth[j])],
                        predicted_label=int(test_guess[j]), predicted_class=config['class_names'][int(test_guess[j])],
                        correct=bool(test_guess[j] == test_truth[j]), confidence=float(probs[j].max()))
            pred.update({f'p_{cls}': float(probs[j, c]) for c, cls in enumerate(config['class_names'])})
            run_predictions.append(pred)
        write_csv(run_dir / 'test_predictions.csv', run_predictions)
        result_path.write_text(json.dumps(result, indent=2))
        all_results.append(result)
        predictions.extend(run_predictions)
        write_csv(HERE / 'results.csv', all_results)
        write_csv(HERE / 'test_predictions.csv', predictions)
        print('DONE ' + json.dumps(result), flush=True)
        del model, optimizer, checkpoint
        if device == 'mps':
            torch.mps.empty_cache()
    # Refresh combined outputs also for a pure resume.
    write_csv(HERE / 'results.csv', all_results)
    write_csv(HERE / 'test_predictions.csv', predictions)
    print('SWEEP_COMPLETE', flush=True)
    (HERE / 'completed.json').write_text(json.dumps(dict(runs=len(all_results), device=device, config_sha256=digest)))


if __name__ == '__main__':
    main()
