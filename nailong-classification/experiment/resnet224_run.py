"""Portable guarded runner for a fixed 224px, 10%-noise ResNet-18 sweep."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import subprocess
import sys
import time
from benchmark import select_device

BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='auto', choices=['auto', 'mps', 'cpu', 'cuda'])
    parser.add_argument('--benchmark-only', action='store_true')
    args = parser.parse_args()
    lock = (BASE / 'pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for filename, digest in json.loads((BASE / 'input_fingerprints.json').read_text()).items():
        assert hashlib.sha256((BASE / filename).read_bytes()).hexdigest() == digest, filename
    device = select_device(args.device)
    log = (BASE / 'training.log').open('a', buffering=1)
    started = time.time()
    def execute(script, *arguments):
        process = subprocess.Popen([sys.executable, '-u', str(BASE / script), *arguments],
                                   cwd=BASE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, script)
    try:
        if not (BASE / f'benchmark_{device}.json').exists():
            execute('benchmark.py', '--device', device)
        if args.benchmark_only:
            return
        execute('train.py', '--device', device, '--output-dir', str(BASE))
        execute('report.py')
        (BASE / 'pipeline_completed.json').write_text(json.dumps(dict(
            status='completed', device=device, elapsed_seconds=time.time() - started)) + '\n')
    except Exception as error:
        (BASE / 'pipeline_failed.json').write_text(json.dumps(dict(
            status='failed', error=str(error), device=device, elapsed_seconds=time.time() - started)) + '\n')
        raise


if __name__ == '__main__':
    main()
