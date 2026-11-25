"""Verify the 500-epoch measurements and compact plotting inputs; requires NumPy."""
import argparse
import csv
from io import BytesIO, StringIO
import json
from math import isclose
from pathlib import Path
from statistics import mean, median
from zipfile import ZipFile

import numpy as np


DATA = Path(__file__).resolve().parent
WIDTHS = [1, 2, 4, 6, 7, 8, 10, 12, 20, 40, 64]
PANEL_WIDTHS = [1, 4, 7, 10, 20, 64]
FIELDS = ["figure", "metric", "label_noise_pct", "width_k", "epochs", "source", "value"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(z, name):
    return json.loads(z.read(name))


def curves(z):
    name = "_".join(map(str, WIDTHS))
    rows = []
    for f in (6, 10, 12):
        chart = load(z, f"figures/fig{f:02d}-subset-{name}.json")
        require((chart["figure"], chart["widths"], chart["epochs"]) == (f, WIDTHS, 500), f"fig{f} settings")
        lookup = {(r["series"], r["k"], r["noise_percent"]): r["value"] for r in chart["measurements"]}
        require(len(lookup) == len(chart["measurements"]), f"duplicate entries: fig{f}")
        if f == 6:
            old_widths = [k for k in WIDTHS if k != 7]
            old = load(z, "figures/fig06-subset-" + "_".join(map(str, old_widths)) + ".json")
            require(old["epochs"] == 500 and old["widths"] == old_widths, "fig6 first 10 widths")
            tests = {(r["k"], r["noise_percent"]): r["value"] for r in old["measurements"]}
            for n in (0, 20):
                m = load(z, f"metrics/k07-n{n:02d}-s1.json")
                require((m["epochs"], m["width"], m["init_seed"], m["noise_percent"], m["test_samples"])
                        == (500, 7, 1, n, 10000), "fig6 k7 config")
                tests[7, n] = 1 - m["correct"] / m["test_samples"]
                require(tests[7, n] == m["test_error"], "fig6 k7 error")
        for k in WIDTHS:
            for n in (0, 20):
                if f == 6:
                    summary = {"test_error": tests[k, n]}
                else:
                    cell = load(z, f"measures/fig{f}-k{k:02d}-n{n:02d}.json")
                    require((cell["figure"], cell["width"], cell["noise_percent"], cell["complete"])
                            == (f, k, n, True), f"bad cell: {f}, {k}, {n}")
                    groups = ("random_train", "same_class_correct") if f == 10 else (
                        ("all", "correct", "mislabeled") if n else ("all",))
                    summary = {}
                    for g in groups:
                        values = cell[g]
                        require(len(values) == (1000 if f == 10 else 5000), f"sample count: {f}, {k}, {n}, {g}")
                        if f == 10:
                            key, value = g + "_mean", mean(values)
                        else:
                            kept = [v for v in values if v is not None]
                            require(len(kept) == cell[g + "_usable_images"], f"valid margins: {k}, {n}, {g}")
                            key, value = g + "_median", median(kept)
                        require(isclose(value, cell[key], abs_tol=1e-10, rel_tol=0), f"raw summary: {f}, {k}, {n}, {g}")
                        summary[key] = value
                for metric, expected in summary.items():
                    value = lookup[metric, k, n]
                    require(isclose(value, expected, abs_tol=1e-10, rel_tol=0), f"chart: {f}, {k}, {n}, {metric}")
                    rows.append((f, metric, n, k, 500, "reproduction_measured", value))
        require(len(lookup) == (22 if f == 6 else 44), f"point count: fig{f}")
    output = StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(FIELDS)
    writer.writerows(rows)
    return output.getvalue()


def panels(z):
    meta = load(z, "figures/fig07.json")
    require((meta["epochs"], meta["widths"], len(meta["measurements"])) == (500, PANEL_WIDTHS, 24), "fig7 settings")
    labels = np.empty((4, 6, 500, 500), dtype=np.uint8)
    anchors = np.empty((4, 6, 3, 2))
    ids = np.array([[672, 766, 781], [75, 32, 304]])
    truth = np.array([[9, 8, 6], [1, 1, 1]])
    noise = np.array([0, 20, 0, 20])
    seen = set()
    for record in meta["measurements"]:
        r, c = record["row"] - 1, record["column"] - 1
        require(0 <= r < 4 and 0 <= c < 6 and (r, c) not in seen, "fig7 duplicated/invalid panel")
        seen.add((r, c))
        p = r // 2
        expected = f"fig07-k{PANEL_WIDTHS[c]:02d}-n{noise[r]:02d}-p{p + 1}.npz"
        require((record["k"], record["noise_percent"], record["file"]) == (PANEL_WIDTHS[c], noise[r], expected), expected)
        require(record["true_labels"] == record["training_labels"] == truth[p].tolist(), "fig7 labels")
        with np.load(BytesIO(z.read("measures/" + expected)), allow_pickle=False) as panel:
            a = panel["label"]
            require(a.shape == (500, 500) and a.dtype == np.uint8 and a.max() < 10, "fig7 grid")
            labels[r, c] = a
            require(np.array_equal(panel["ids"], ids[p]) and record["ids"] == ids[p].tolist(), "fig7 ids")
            x0, x1, y0, y1 = panel["extent"]
            require(x1 > x0 and y1 > y0, "fig7 extents")
            # Extent records first and last grid point centres. imshow's
            # displayed bounds extend half a grid step beyond each centre.
            span = np.array([x1 - x0, y1 - y0])
            dx = span / 499
            uv = (panel["xy"] - [x0, y0] + dx / 2) / (span + dx)
            require(uv.shape == (3, 2) and np.all((uv >= 0) & (uv <= 1)), "fig7 anchors")
            anchors[r, c] = uv
    return dict(labels=labels, anchors=anchors, widths=np.array(PANEL_WIDTHS), ids=ids,
                true_labels=truth, noise_percent=noise)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="also regenerate curves.csv and figure7.npz from raw measurements")
    args = parser.parse_args()
    with ZipFile(DATA / "raw-measurements.zip") as z:
        require(z.testzip() is None, "ZIP checksum failed")
        require(z.read("settings.json") == (DATA / "settings.json").read_bytes(), "settings.json")
        text = curves(z)
        arrays = panels(z)
    path = DATA / "curves.csv"
    npz = DATA / "figure7.npz"
    if args.rebuild:
        path.write_text(text, encoding="utf-8")
        np.savez_compressed(npz, **arrays)
    require(path.read_text(encoding="utf-8") == text, "curves.csv")
    with np.load(npz, allow_pickle=False) as saved:
        require(set(saved.files) == set(arrays), "figure7.npz keys")
        for key, array in arrays.items():
            require(np.array_equal(array, saved[key]), "figure7.npz: " + key)
    print("Verified 110 curve points and 24 original 500 x 500 decision grids.")


if __name__ == "__main__":
    main()
