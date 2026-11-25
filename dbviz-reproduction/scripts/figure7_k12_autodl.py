"""Export the four k=12 Figure 7 panels from the archived 500-epoch weights."""
import argparse
from contextlib import contextmanager
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile


WIDTH = 12
SEED = 1
EPOCHS = 500
GRID = 500
NOISES = (0, 20)
TRIPLETS = ((672, 766, 781), (75, 32, 304))
TRUTH = ((9, 8, 6), (1, 1, 1))
# 10,000 test examples × (1 - the reported Fig. 6 test error at k=12).
TEST_CORRECT = {0: 8575, 20: 6896}
# SHA-256 of the two original model.pt files exported from fig7-500.
EXPORTED_WEIGHTS = {
    0: "19763c7fb39cd096a3979180377ae33659d07fdc0e74e445c439928454a9e9a0",
    20: "01a003a9350a939396d5f06c28141216830644aad8288ee547e78694c3d21c1b",
}
BUNDLE = "figure7-k12-500-panels.zip"


def matches_test_count(noise, digest, correct, device):
    if digest == EXPORTED_WEIGHTS[20 - noise]:
        return False
    if correct == TEST_CORRECT[noise]:
        return True
    # The exported noisy model gave 6895/10000 on CPU and 6896/10000 in
    # the archived run. This observation applies only to these exact bytes.
    return noise == 20 and digest == EXPORTED_WEIGHTS[20] and device == "cpu" and correct == 6895


def check_pair_provenance(models):
    if models["20"]["test_correct"] != TEST_CORRECT[20]:
        if any(models[str(n)]["checkpoint_sha256"] != EXPORTED_WEIGHTS[n] for n in NOISES):
            raise ValueError("CPU 与归档计数不同时，两份权重均需与原实例导出文件的 SHA-256 一致。")


def model_roots(project, extra):
    base = project / "run" / "width"
    budget = base / "budget" / "epoch-500"
    # fig7-500 also staged the 11 model widths used in Figures 10/12.
    return [*extra, base / "fig7-500", budget / "replay", base / "paper", budget,
            base / "fig10-12-500-paper-sampling", project / "runs" / "width" / "measured"]


def exact_checkpoints(roots, noise, torch):
    """Yield exact-epoch state dicts; the original resume files store epoch internally."""
    seen = set()
    for root in roots:
        folder = root / "models" / f"k12-n{noise:02d}-s1"
        for filename in ("model.pt", "resume.pt"):
            path = folder / filename
            if not path.is_file() or path.resolve() in seen:
                continue
            seen.add(path.resolve())
            if filename == "model.pt":
                metric = folder / "metric.json"
                if not metric.is_file():
                    print(f"跳过 {path}（缺少同目录 metric.json）。", flush=True)
                    continue
                record = json.loads(metric.read_text(encoding="utf-8"))
                identity = tuple(record.get(field) for field in ("width", "noise_percent", "init_seed", "epochs"))
                if identity != (WIDTH, noise, SEED, EPOCHS):
                    print(f"跳过 {path}（元数据为 {identity}）。", flush=True)
                    continue
            # Hash and load the same bytes even if another process replaces a checkpoint.
            raw = path.read_bytes()
            state = torch.load(BytesIO(raw), map_location="cpu", weights_only=True)
            if filename == "resume.pt":
                if state.get("epoch") != EPOCHS:
                    print(f"跳过 {path}（第 {state.get('epoch')} 轮）。", flush=True)
                    continue
                state = state["weights"]
            yield path, state, sha256(raw).hexdigest()


def model_for_noise(roots, noise, torch, data, make_model):
    for path, state, digest in exact_checkpoints(roots, noise, torch):
        model = make_model(WIDTH).to(device=data.device, memory_format=torch.channels_last)
        model.load_state_dict(state, strict=True)
        model.eval()
        labels = torch.as_tensor(data.test_y, device=data.device)
        if len(labels) != 10_000 or len(data.true) != 50_000:
            raise ValueError("本实验需要完整 CIFAR-10 训练集和测试集。")
        hits = torch.zeros((), dtype=torch.int64, device=data.device)
        for start in range(0, len(labels), 128):
            part = slice(start, start + 128)
            hits += (model(data.norm(data.test[part])).argmax(1) == labels[part]).sum()
        correct = int(hits.item())
        device = str(data.device)
        if matches_test_count(noise, digest, correct, device):
            print(f"已核验 {path}，本次 {device} 正确 {correct}/10000，"
                  f"原图 6 归档正确 {TEST_CORRECT[noise]}/10000。", flush=True)
            return model, {"checkpoint_sha256": digest, "checkpoint_file": path.name,
                           "test_correct": correct, "test_samples": len(labels),
                           "archived_test_correct": TEST_CORRECT[noise], "device": device,
                           "torch_version": str(torch.__version__)}
        print(f"跳过 {path}（实测 {correct}/10000，已有图 6 记录为 {TEST_CORRECT[noise]}/10000）。", flush=True)
    raise ValueError(missing_weights(roots, noise))


def missing_weights(roots, noise):
    paths = "\n".join(f"  {root}/models/k12-n{noise:02d}-s1/" for root in roots)
    return (f"缺少可核验的 k=12、噪声 {noise}%、种子 1 的第 500 轮权重。已检查\n{paths}\n"
            "可用 --model-root 指定另一份含 models/ 的实验目录。")


def libraries(project):
    if (project / "width" / "measure.py").is_file():
        sys.path.insert(0, str(project / "width"))
        from measure import plane
        from resnet18k import make_resnet18k
        from training import Data
        return Data, make_resnet18k, plane, project / "data"
    if (project / "experiments" / "width" / "measure.py").is_file():
        sys.path.insert(0, str(project))
        from experiments.width.measure import plane
        from experiments.width.resnet18k import make_resnet18k
        from experiments.width.training import Data
        return Data, make_resnet18k, plane, project / "datasets"
    raise ValueError("请在 dbviz-stages 或 dbviz-reproduction 根目录运行，或用 --project 指定该目录。")


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")


def panels_for_noise(model, noise, data, plane):
    import numpy as np

    stem = f"fig07-k12-n{noise:02d}"
    records, members = [], {}
    for index, (triplet, truth) in enumerate(zip(TRIPLETS, TRUTH), start=1):
        ids = list(triplet)
        actual = data.true[ids].tolist()
        training_labels = (data.corrupt if noise else data.true)[ids].tolist()
        if actual != list(truth) or training_labels != actual:
            raise ValueError(f"三元组的真实标签或训练标签与原图不符：{ids}")
        print(f"计算 k=12、噪声 {noise}%、平面 {index}/2：{ids}，{GRID}×{GRID}。", flush=True)
        panel = plane(model, data, data.train, ids, GRID, .5, with_confidence=True)
        label, conf = panel["label"], panel["confidence"]
        xy, extent = panel["xy"], panel["extent"]
        if (label.shape != (GRID, GRID) or label.dtype != np.uint8 or label.max() >= 10
                or conf.shape != label.shape or not np.isfinite(conf).all() or conf.min() < 0 or conf.max() > 1
                or not np.array_equal(panel["ids"], ids) or xy.shape != (3, 2) or extent.shape != (4,)
                or not np.isfinite(xy).all() or not np.isfinite(extent).all()
                or not (extent[0] < extent[1] and extent[2] < extent[3])):
            raise ValueError(f"网格输出未通过校验：{stem} 平面 {index}")
        filename = f"{stem}-p{index}.npz"
        with BytesIO() as output:
            np.savez_compressed(output, **panel)
            members[f"measures/{filename}"] = output.getvalue()
        records.append({"file": filename, "ids": ids, "true_labels": actual, "training_labels": training_labels})
    metadata = {"figure": 7, "width": WIDTH, "noise_percent": noise, "complete": True, "panels": records}
    members[f"measures/{stem}.json"] = json_bytes(metadata)
    return members


def write_bundle(output, models, members):
    check_pair_provenance(models)
    manifest = {"schema": 1, "figure": 7, "width": WIDTH, "epochs": EPOCHS, "init_seed": SEED,
                "noise_percent": list(NOISES), "panel_grid": GRID, "widths": [1, 4, 7, 12, 20, 64],
                "models": models, "member_sha256": {k: sha256(v).hexdigest() for k, v in members.items()}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, prefix=output.name + ".", suffix=".part", delete=False) as f:
        temp = Path(f.name)
    try:
        with ZipFile(temp, "w", ZIP_DEFLATED, compresslevel=3) as bundle:
            bundle.writestr("manifest.json", json_bytes(manifest))
            for name, payload in members.items():
                bundle.writestr(name, payload)
        with ZipFile(temp) as bundle:
            if bundle.testzip() is not None or len(bundle.namelist()) != 7:
                raise ValueError("结果包未通过校验。")
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)
    print(f"图 7 的四个 k=12 面板已打包，请下载：{output}", flush=True)


@contextmanager
def project_lock(project):
    import fcntl

    # Share the project's existing training lock while using its GPU and checkpoints.
    if (project / "width").is_dir():
        path = project / "run" / "active.lock"
    else:
        path = project / "runs" / "width" / "measured" / ".lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("已有实验在使用这个项目，请等其结束再运行。") from None
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd(), help="云端 dbviz-stages 或复现仓库根目录")
    parser.add_argument("--model-root", type=Path, action="append", default=[], help="其他含 models/ 的实验目录，可重复")
    parser.add_argument("--data", type=Path, help="含 cifar-10-batches-py/ 的目录")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--plan", action="store_true", help="检查权重文件和轮数后退出")
    args = parser.parse_args()
    project = args.project.resolve()
    Data, make_model, plane, default_data = libraries(project)
    import torch

    roots = model_roots(project, [p.resolve() for p in args.model_root])
    if args.plan:
        missing = []
        for noise in NOISES:
            first = next(exact_checkpoints(roots, noise, torch), None)
            if first is None:
                missing.append(missing_weights(roots, noise))
            else:
                print(f"k=12，噪声 {noise}%，第 500 轮候选权重：{first[0]}", flush=True)
        if missing:
            raise ValueError("\n".join(missing))
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("当前实例的 CUDA 不可用。可在 GPU 实例上运行，或明确选择 --device cpu。")
    torch.set_num_threads(4)
    with project_lock(project), torch.inference_mode():
        data = Data(args.data or default_data, args.device)
        models, members = {}, {}
        for noise in NOISES:
            model, info = model_for_noise(roots, noise, torch, data, make_model)
            models[str(noise)] = info
            members.update(panels_for_noise(model, noise, data, plane))
            del model
        write_bundle(project / BUNDLE, models, members)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from None
