"""Build the k=12 version of Figure 7 and its download package on the original AutoDL project."""
import argparse
from pathlib import Path
import os
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile


SCRIPTS = Path(__file__).resolve().parent
DATA = SCRIPTS.parent / "data"


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd(),
                        help="原云端训练项目 dbviz-stages 的目录")
    parser.add_argument("--panels", type=Path,
                        help="已经导出的 k=12 原始面板压缩包，可用于直接在 CPU 绘图")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                        help="用于已有权重的推理，默认 cuda")
    parser.add_argument("--model-root", type=Path, action="append", default=[],
                        help="另一个含 models/ 的权重目录，可重复")
    parser.add_argument("--data", type=Path, help="另一个含 CIFAR-10 数据的目录")
    args = parser.parse_args()
    project = args.project.resolve()
    if not project.is_dir():
        parser.error(f"找不到目录：{project}")
    if args.panels:
        panels = args.panels.resolve()
        if not panels.is_file():
            parser.error(f"找不到面板文件：{panels}")
    else:
        command = [sys.executable, "-u", str(SCRIPTS / "figure7_k12_autodl.py"),
                   "--project", str(project), "--device", args.device]
        for root in args.model_root:
            command.extend(["--model-root", str(root.resolve())])
        if args.data:
            command.extend(["--data", str(args.data.resolve())])
        subprocess.run(command, check=True)
        panels = project / "figure7-k12-500-panels.zip"

    target = project / "figure7-k12-results.zip"
    with TemporaryDirectory(prefix=".fig7-k12-", dir=project) as tmp:
        folder = Path(tmp)
        subprocess.run([sys.executable, "-u", str(SCRIPTS / "figure7_peak.py"),
                        "--panels", str(panels), "--baseline", str(DATA / "figure7.npz"),
                        "--curves", str(DATA / "curves.csv"), "--out", str(folder)], check=True)
        products = ["fig07-k12.pdf", "fig07-k12.png", "figure7-k12.npz",
                    "figure7-k12-receipt.json", "figures-500epochs-k12-zh.pdf"]
        with ZipFile(folder / "download.zip", "w", ZIP_DEFLATED, compresslevel=3) as archive:
            for name in products:
                archive.write(folder / name, name)
            archive.write(panels, "figure7-k12-500-panels.zip")
        with ZipFile(folder / "download.zip") as archive:
            failed = archive.testzip()
            if failed:
                raise RuntimeError(f"压缩包校验失败：{failed}")
        os.replace(folder / "download.zip", target)
    print(f"请在 JupyterLab 下载 {target}。", flush=True)


if __name__ == "__main__":
    run()
