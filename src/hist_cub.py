import itertools
import functools
import math
import multiprocessing
from pathlib import Path

import matplotlib
matplotlib.rcParams.update({'font.size': 24})
matplotlib.rcParams.update({
  "text.usetex": True,
  "text.latex.preamble": r"\usepackage{biolinum} \usepackage{libertineRoman} \usepackage{libertineMono} \usepackage{biolinum} \usepackage[libertine]{newtxmath}",
  'ps.usedistiller': "xpdf",
})

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tqdm
from scipy.stats import wasserstein_distance

import pose_estimation


def cub(x, a, b, c):
    x2 = x * x
    x3 = x2 * x

    y = a * x3 + b * x2 + c * x

    return y


def subsample(a, p=0.0005, seed=0):
    np.random.seed(seed)
    N = len(a)
    inds = np.random.choice(range(N), size=int(p * N))
    a = a[inds].copy()

    return a


def read_cos_opt(path, fname="cos_hist.npy"):
    cos_opt = []
    for p in Path(path).rglob(fname):
        d = np.load(p)
        cos_opt.append(d)

    cos_opt = np.array(cos_opt)

    return cos_opt


def plot_hist(cos_opt_dir, hist_smpl_fpath, params, out_dir, bins=10, xy=None):
    cos_opt = read_cos_opt(cos_opt_dir)
    angle_opt = np.arccos(cos_opt)
    angle_opt2 = cub(angle_opt, *params)

    cos_opt2 = np.cos(angle_opt2)
    cos_smpl = np.load(hist_smpl_fpath)
    # cos_smpl = subsample(cos_smpl)
    print(cos_smpl.shape)

    cos_smpl = np.clip(cos_smpl, -1, 1)

    cos_opt = angle_opt
    cos_opt2 = angle_opt2
    cos_smpl = np.arccos(cos_smpl)

    cos_opt = 180 / math.pi * cos_opt
    cos_opt2 = 180 / math.pi * cos_opt2
    cos_smpl = 180 / math.pi * cos_smpl
    max_range = 90  # math.pi / 2

    xticks = [0, 15, 30, 45, 60, 75, 90]
    for idx, bone in enumerate(pose_estimation.SKELETON):
        i, j = bone
        i_name = pose_estimation.KPS[i]
        j_name = pose_estimation.KPS[j]
        if i_name != "Left Upper Leg":
            continue

        name = f"{i_name}_{j_name}"

        gs = gridspec.GridSpec(2, 4)
        fig = plt.figure(tight_layout=True, figsize=(16, 8), dpi=300)

        ax0 = fig.add_subplot(gs[0, 0])
        ax0.hist(cos_smpl[:, idx], bins=bins, range=(0, max_range), density=True)
        ax0.set_xticks(xticks)
        ax0.tick_params(labelbottom=False, labelleft=True)

        ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
        ax1.hist(cos_opt[:, idx], bins=bins, range=(0, max_range), density=True)
        ax1.set_xticks(xticks)

        if xy is not None:
            ax2 = fig.add_subplot(gs[:, 1:3])
            ax2.plot(xy[0], xy[1], linewidth=8)
            ax2.plot(xy[0], xy[0], linewidth=4, linestyle="dashed")
            ax2.set_xticks(xticks)
            ax2.set_yticks(xticks)

        ax3 = fig.add_subplot(gs[0, 3], sharey=ax0)
        ax3.hist(cos_opt2[:, idx], bins=bins, range=(0, max_range), density=True)
        ax3.set_xticks(xticks)
        ax3.tick_params(labelbottom=False, labelleft=False)

        ax4 = fig.add_subplot(gs[1, 3], sharex=ax3, sharey=ax1)
        alpha = 0.5
        ax4.hist(cos_opt[:, idx], bins=bins, range=(0, max_range), density=True, label=r"$\mathcal{B}_i$", alpha=alpha)
        ax4.hist(cos_opt2[:, idx], bins=bins, range=(0, max_range), density=True, label=r"$f(\mathcal{B}_i)$", alpha=alpha)
        ax4.hist(cos_smpl[:, idx], bins=bins, range=(0, max_range), density=True, label=r"$\mathcal{A}_i$", alpha=alpha)
        ax4.set_xticks(xticks)
        ax4.tick_params(labelbottom=True, labelleft=False)
        ax4.legend()

        fig.savefig(out_dir / f"hist_{name}.png")
        plt.close()


def kldiv(p_hist, q_hist):
    wd = wasserstein_distance(p_hist, q_hist)

    return wd


def calc_histogram(x, bins=10, range=(0, 1)):
    h, _ = np.histogram(x, bins=bins, range=range, density=True)

    return h

def step(params, angles_opt, p_hist, bone_idx=None):
    if sum(params) > 1:
        return math.inf, params

    kl = 0
    for i, _ in enumerate(pose_estimation.SKELETON):
        if bone_idx is not None and i != bone_idx:
            continue

        angles_opt2 = cub(angles_opt[:, i], *params)
        if angles_opt2.max() > 1 or angles_opt2.min() < 0:
            kl = math.inf

            break

        q_hist = calc_histogram(angles_opt2)

        kl += kldiv(p_hist[i], q_hist)

    return kl, params


def optimize(cos_opt_dir, hist_smpl_fpath, bone_idx=None):
    cos_opt = read_cos_opt(cos_opt_dir)
    angles_opt = np.arccos(cos_opt) / (math.pi / 2)
    cos_smpl = np.load(hist_smpl_fpath)
    # cos_smpl = subsample(cos_smpl)
    print(cos_smpl.shape)
    cos_smpl = np.clip(cos_smpl, -1, 1)
    mask = cos_smpl <= 1
    assert np.all(mask), (~mask).mean()
    mask = cos_smpl >= 0
    assert np.all(mask), (~mask).mean()
    angles_smpl = np.arccos(cos_smpl) / (math.pi / 2)
    p_hist = [
        calc_histogram(angles_smpl[:, i])
        for i, _ in enumerate(pose_estimation.SKELETON)
    ]

    with multiprocessing.Pool(8) as p:
        results = list(
            tqdm.tqdm(
                p.imap_unordered(
                    functools.partial(step, angles_opt=angles_opt, p_hist=p_hist, bone_idx=bone_idx),
                    itertools.product(
                        np.linspace(0, 20, 100),
                        np.linspace(-20, 20, 200),
                        np.linspace(-20, 1, 100),
                    ),
                ),
                total=(100 * 200 * 100),
            )
        )

    kls, params = zip(*results)
    ind = np.argmin(kls)
    best_params = params[ind]

    print(kls[ind], best_params)

    inds = np.argsort(kls)
    for i in inds[:10]:
        print(kls[i])
        print(params[i])
        print()

    return best_params


def main():
    cos_opt_dir = "paper_single2_150mse"
    hist_smpl_fpath = "./data/hist_smpl.npy"
    # hist_smpl_fpath = "./testtest.npy"
    params = optimize(cos_opt_dir, hist_smpl_fpath)
    # params = (1.2121212121212122, -1.105527638190953, 0.787878787878789)
    # params = (0.20202020202020202, 0.30150753768844396, 0.3636363636363633)
    print(params)

    x = np.linspace(0, math.pi / 2, 100)
    y = cub(x / (math.pi / 2), *params) * (math.pi / 2)
    x = x * 180 / math.pi
    y = y * 180 / math.pi

    out_dir = Path("hists")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_hist(cos_opt_dir, hist_smpl_fpath, params, out_dir, xy=(x, y))

    plt.figure(figsize=(4, 4), dpi=300)
    plt.plot(x, y, linewidth=6)
    plt.plot(x, x, linewidth=2, linestyle="dashed")
    xticks = [0, 15, 30, 45, 60, 75, 90]
    plt.xticks(xticks)
    plt.yticks(xticks)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(out_dir / "new_out.png")


if __name__ == "__main__":
    main()
