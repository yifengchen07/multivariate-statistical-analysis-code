"""Short native-MPS speed/memory check before the full 224px sweep."""
from pathlib import Path
import json
import time
import numpy as np
import torch
from torch.nn import functional as F
from models import build_model

BASE = Path(__file__).resolve().parent
OUT = BASE/'vit224'


def main():
    torch.set_num_threads(4)
    assert torch.backends.mps.is_available()
    data=np.load(OUT/'data.npz')
    x=torch.from_numpy(data['x'][data['train'][:64]]).to('mps')
    y=torch.from_numpy(data['y_noise'][data['train'][:64]]).to('mps')
    config=json.loads((OUT/'config.json').read_text())
    rows=[]
    for d in [8,48,192]:
        model=build_model(config,d).to('mps')
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        times=[]
        for step in range(16):
            torch.mps.synchronize(); start=time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model(x),y)
            loss.backward();optimizer.step();torch.mps.synchronize()
            if step>=4:times.append(time.perf_counter()-start)
        item=dict(width=d,parameters=sum(p.numel() for p in model.parameters()),
                  median_step_seconds=float(np.median(times)),
                  estimated_400epoch_seconds=float(np.median(times)*5*400*1.15),
                  allocated_mb=torch.mps.current_allocated_memory()/2**20,
                  driver_mb=torch.mps.driver_allocated_memory()/2**20)
        rows.append(item);print(json.dumps(item),flush=True)
        del model,optimizer;torch.mps.empty_cache()
    (OUT/'benchmark.json').write_text(json.dumps(rows,indent=2))


if __name__=='__main__':
    main()
