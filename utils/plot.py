"""把训练统计量（CSV）画成图。

约定：一个 CSV 代表一次实验，每行一个 epoch，列是各项统计量
（例如 data/vpg_stats.csv 的 epoch/ep_ret/ep_len/loss_pi/loss_v/kl/ent）。
多个 CSV 会画在同一张图上做对比。

用法：
    python utils/plot.py data/vpg_stats.csv
    python utils/plot.py data/vpg_stats.csv --keys ep_ret loss_v
    python utils/plot.py data/*.csv --legend vpg ppo --smooth 5
    python utils/plot.py data/vpg_stats.csv --list          # 只列出有哪些列
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

# 有些机器上 ~/.config/matplotlib 不可写，先给个可写的缓存目录，避免每次告警
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig"))

import matplotlib

matplotlib.use("Agg")  # 无显示器也能出图
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DEFAULT_OUT = "figures/plot.png"
# 不画的列：横轴、以及日志里常见的时间戳类列
SKIP_KEYS = {"epoch", "step", "steps", "time", "wall_time", "total_env_interacts"}


def load_csv(path):
    """读一个 CSV，返回 {列名: np.ndarray}，只保留能转成 float 的列。"""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} 是空文件")

    data = {}
    for key in rows[0]:
        try:
            data[key] = np.array([float(row[key]) for row in rows], dtype=np.float64)
        except (TypeError, ValueError):
            continue  # 非数值列（比如 exp_name）直接跳过
    if not data:
        raise ValueError(f"{path} 里没有可画的数值列")
    return data


def resolve_paths(inputs):
    """把命令行给的路径展开成 CSV 列表；目录会取里面的所有 *.csv。"""
    files = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            files.extend(sorted(path.glob("*.csv")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"找不到: {item}")
    if not files:
        raise FileNotFoundError("一个 CSV 都没找到")
    return files


def moving_average(y, window):
    """居中滑动平均，长度保持不变（两端用边缘值填充）。"""
    if window <= 1 or len(y) < 2:
        return y
    window = min(window, len(y))
    pad = window // 2
    padded = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")[: len(y)]


def pick_keys(runs, keys):
    """确定要画哪些列：用户指定优先，否则取所有 run 共有的数值列。"""
    common = set(runs[0]["data"].keys())
    for run in runs[1:]:
        common &= set(run["data"].keys())
    common -= SKIP_KEYS

    if keys:
        missing = [k for k in keys if k not in common]
        if missing:
            raise KeyError(
                f"这些列不存在或不是所有 run 都有: {missing}\n"
                f"可用列: {sorted(common)}"
            )
        return keys
    return sorted(common)


def plot_runs(runs, keys=None, out=DEFAULT_OUT, smooth=1, title=None, ncols=3):
    """runs: [{'label': str, 'data': {列名: ndarray}}, ...]"""
    keys = pick_keys(runs, keys)
    if not keys:
        raise ValueError("没有可画的列")

    ncols = max(1, min(ncols, len(keys)))
    nrows = (len(keys) + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.4 * ncols, 3.3 * nrows), squeeze=False
    )

    x_key = "epoch" if "epoch" in runs[0]["data"] else None

    for ax, key in zip(axes.ravel(), keys):
        for run in runs:
            y = run["data"].get(key)
            if y is None:
                continue
            x = run["data"].get(x_key) if x_key else np.arange(len(y))
            mask = np.isfinite(y)
            if mask.sum() < 2:
                continue
            ax.plot(
                x[mask],
                moving_average(y[mask], smooth),
                linewidth=1.6,
                label=run["label"],
            )

        ax.set_title(key, fontsize=10)
        ax.set_xlabel(x_key or "step", fontsize=9)
        ax.set_ylabel(key, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
        ax.margins(x=0.02)

    for ax in axes.ravel()[len(keys) :]:
        ax.set_visible(False)  # 多余的空位藏起来

    if len(runs) > 1:
        axes.ravel()[0].legend(fontsize=8)

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97) if title else None)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out, keys


def main(argv=None):
    parser = argparse.ArgumentParser(description="把实验 CSV 画成图")
    parser.add_argument("paths", nargs="*", help="CSV 文件或包含 CSV 的目录")
    parser.add_argument("--keys", "-k", nargs="*", help="要画的列，默认全部")
    parser.add_argument("--legend", "-l", nargs="*", help="每个 CSV 的图例名")
    parser.add_argument("--smooth", "-s", type=int, default=1, help="滑动平均窗口")
    parser.add_argument("--out", "-o", default=DEFAULT_OUT, help="输出 PNG 路径")
    parser.add_argument("--title", "-t", default=None)
    parser.add_argument("--ncols", type=int, default=3, help="子图列数")
    parser.add_argument("--list", action="store_true", help="只列出可画的列，不出图")
    args = parser.parse_args(argv)

    if not args.paths:
        parser.error("至少给一个 CSV 路径")

    files = resolve_paths(args.paths)

    if args.legend and len(args.legend) != len(files):
        parser.error(f"--legend 给了 {len(args.legend)} 个，但有 {len(files)} 个 CSV")

    runs = []
    for i, path in enumerate(files):
        label = args.legend[i] if args.legend else path.stem
        runs.append({"label": label, "data": load_csv(path)})

    if args.list:
        rows = len(next(iter(runs[0]["data"].values())))
        print(f"{files[0]}: {len(runs[0]['data'])} 列，{rows} 行")
        for key in runs[0]["data"]:
            print(f"  {key}")
        return

    out, keys = plot_runs(
        runs,
        keys=args.keys,
        out=args.out,
        smooth=args.smooth,
        title=args.title,
        ncols=args.ncols,
    )
    print(f"画了 {len(keys)} 个指标 × {len(runs)} 个 run: {', '.join(keys)}")
    print(f"图像已保存: {out}")


if __name__ == "__main__":
    main()
