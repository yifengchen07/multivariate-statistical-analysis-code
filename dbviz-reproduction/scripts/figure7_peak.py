"""Render the 500-epoch Figure 7 variant with a newly measured k=12 column."""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import redraw
from figure7_k12_autodl import TEST_CORRECT, check_pair_provenance, matches_test_count
from matplotlib.backends.backend_pdf import PdfPages


WIDTHS = [1, 4, 7, 12, 20, 64]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def import_panels(path, baseline):
    arrays = redraw.load_figure7(baseline, redraw.PANEL_WIDTHS)
    with ZipFile(path) as bundle:
        meta = json.loads(bundle.read("manifest.json"))
        expected = {"schema": 1, "figure": 7, "width": 12, "epochs": 500, "init_seed": 1,
                    "noise_percent": [0, 20], "panel_grid": 500, "widths": WIDTHS}
        require(all(meta.get(k) == v for k, v in expected.items()), "k=12 评估设置与图 7 不符")
        members = {f"measures/fig07-k12-n{n:02d}{end}" for n in TEST_CORRECT
                   for end in (".json", "-p1.npz", "-p2.npz")}
        names = bundle.namelist()
        require(set(names) == members | {"manifest.json"} and len(names) == len(set(names)),
                "k=12 数据包的文件名或文件个数不符")
        hashes = meta["member_sha256"]
        require(set(hashes) == members, "k=12 数据包缺少文件摘要")
        blobs = {name: bundle.read(name) for name in members}
        require(all(sha(value) == hashes[name] for name, value in blobs.items()), "k=12 文件摘要校验失败")
        models = meta["models"]
        require(set(models) == {"0", "20"}, "缺少两个噪声条件的模型来源记录")
        for noise, hits in TEST_CORRECT.items():
            model = models[str(noise)]
            digest = str(model.get("checkpoint_sha256"))
            device = model.get("device")
            require(re.fullmatch("[0-9a-f]{64}", digest)
                    and matches_test_count(noise, digest, model.get("test_correct"), device),
                    f"噪声 {noise}% 的模型摘要或测试计数与 500 轮实验不符")
            if device is not None or model.get("test_correct") != hits:
                require(model.get("archived_test_correct") == hits and model.get("test_samples") == 10000
                        and isinstance(device, str) and device and isinstance(model.get("torch_version"), str)
                        and model["torch_version"], f"噪声 {noise}% 的复核来源记录不完整")
            elif "archived_test_correct" in model:
                require(model["archived_test_correct"] == hits, f"噪声 {noise}% 的归档计数不符")
            prefix = f"fig07-k12-n{noise:02d}"
            cell = json.loads(blobs[f"measures/{prefix}.json"])
            require((cell["figure"], cell["width"], cell["noise_percent"], cell["complete"]) == (7, 12, noise, True)
                    and len(cell["panels"]) == 2, f"评估记录不符：{prefix}")
            for group, item in enumerate(cell["panels"]):
                file = f"{prefix}-p{group + 1}.npz"
                ids, truth = arrays["ids"][group], arrays["true_labels"][group]
                require(item["file"] == file and item["ids"] == ids.tolist()
                        and item["true_labels"] == item["training_labels"] == truth.tolist(),
                        f"三元组或训练标签不符：{file}")
                row = 2 * group + (noise == 20)
                col = WIDTHS.index(12)
                with np.load(BytesIO(blobs["measures/" + file]), allow_pickle=False) as panel:
                    grid, xy, bounds = panel["label"], panel["xy"], panel["extent"]
                    require(grid.shape == (500, 500) and grid.dtype == np.uint8 and grid.max() < 10
                            and np.array_equal(panel["ids"], ids), f"类别网格或样本编号不符：{file}")
                    require(xy.shape == (3, 2) and bounds.shape == (4,) and np.isfinite(bounds).all(),
                            f"样本平面坐标无效：{file}")
                    low = bounds[[0, 2]]
                    span = bounds[[1, 3]] - low
                    require(np.all(span > 0), f"网格范围无效：{file}")
                    delta = span / 499
                    uv = (xy - low + delta / 2) / (span + delta)
                    require(np.allclose(uv, arrays["anchors"][row, col], rtol=0, atol=1e-5),
                            f"样本平面与归档图 7 不一致：{file}")
                    arrays["labels"][row, col] = grid
                    arrays["anchors"][row, col] = uv

    check_pair_provenance(models)
    arrays["widths"] = np.array(WIDTHS)
    receipt = dict(figure=7, epochs=500, init_seed=1, widths=WIDTHS, replaced_width=10, measured_width=12,
                   baseline_sha256=sha(baseline.read_bytes()), panels_zip_sha256=sha(path.read_bytes()),
                   checkpoints={n: record["checkpoint_sha256"] for n, record in models.items()},
                   test_correct={n: record["test_correct"] for n, record in models.items()},
                   archived_test_correct=TEST_CORRECT, validation=models, test_size=10000, member_sha256=hashes)
    return arrays, receipt


def render(panels, baseline, curves_path, out, font):
    arrays, receipt = import_panels(panels, baseline)
    curves = redraw.load_curves(curves_path) if curves_path is not None else None
    out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out) as tmp:
        stage = Path(tmp)
        redraw.configure_style(font, stage)
        combined_grid = stage / "figure7-k12.npz"
        np.savez_compressed(combined_grid, **arrays)
        fig = redraw.figure7(combined_grid, WIDTHS)
        try:
            title = "图 7　不同网络宽度下的决策区域（500 轮，含宽度 12）"
            fig.savefig(stage / "fig07-k12.pdf", metadata={"Title": title})
            fig.savefig(stage / "fig07-k12.png", dpi=180)
            if curves is not None:
                with PdfPages(stage / "figures-500epochs-k12-zh.pdf",
                              metadata={"Title": "图 6、7、10、12：500 轮复现实测（图 7 含宽度 12）"}) as pdf:
                    for n, draw in [(6, redraw.figure6), (7, None), (10, redraw.figure10), (12, redraw.figure12)]:
                        if n == 7:
                            pdf.savefig(fig)
                        else:
                            curve = draw(curves)
                            try:
                                pdf.savefig(curve)
                            finally:
                                redraw.plt.close(curve)
                receipt["curves_sha256"] = sha(curves_path.read_bytes())
        finally:
            redraw.plt.close(fig)
        (stage / "figure7-k12-receipt.json").write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        names = ["figure7-k12.npz", "fig07-k12.pdf", "fig07-k12.png", "figure7-k12-receipt.json"]
        if curves is not None:
            names.append("figures-500epochs-k12-zh.pdf")
        for name in names:
            (stage / name).replace(out / name)
            print((out / name).resolve())


def main():
    parser = argparse.ArgumentParser(description="将第 500 轮的 k=12 实测面板加入图 7")
    parser.add_argument("--panels", type=Path, required=True, help="由 figure7_k12_autodl.py 导出的面板 ZIP")
    parser.add_argument("--baseline", type=Path, default=redraw.ROOT / "data" / "figure7.npz")
    parser.add_argument("--curves", type=Path, help="提供 data/curves.csv 可同时生成四页合订 PDF")
    parser.add_argument("--out", type=Path, default=redraw.ROOT / "build" / "fig7-k12")
    parser.add_argument("--font", type=Path, help="可选的中文字体文件")
    args = parser.parse_args()
    render(args.panels, args.baseline, args.curves, args.out, args.font)


if __name__ == "__main__":
    main()
