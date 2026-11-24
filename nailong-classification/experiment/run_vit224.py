"""Resume 224px width/seed sweep, validate, then regenerate introduction boards."""
from pathlib import Path
import json
import subprocess
import sys

BASE=Path(__file__).resolve().parent
OUT=BASE/'vit224'


def main():
    subprocess.run([sys.executable,str(BASE/'prepare_vit224.py')],check=True)
    subprocess.run([sys.executable,str(BASE/'train_sweep.py'),'--device','mps','--output-dir',str(OUT)],check=True)
    subprocess.run([sys.executable,str(BASE/'validate_results.py'),'--output-dir',str(OUT)],check=True)
    subprocess.run([sys.executable,str(BASE/'report_vit224.py')],check=True)
    (OUT/'pipeline_completed.json').write_text(json.dumps(dict(status='completed')))


if __name__=='__main__':
    main()
