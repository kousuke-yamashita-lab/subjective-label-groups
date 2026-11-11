#!/usr/bin/env python
"""Reproduce the numbers in the paper from the anonymised data. Writes results/summary.md.

Usage: python analysis/reproduce.py [--quick]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measures as M  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
S = M.SETS


def relabel(D, pid, flag):
    return {s: {**D[s], "jp": flag[pid[s]]} for s in S}


def person_index(D):
    persons = sorted({p for s in S for p in D[s]["person"]}); pi = {p: i for i, p in enumerate(persons)}
    pid = {s: np.array([pi[p] for p in D[s]["person"]]) for s in S}
    base = np.zeros(len(persons), bool)
    for s in S:
        base[pid[s]] = D[s]["jp"]
    return persons, pid, base


def strata(pid, base):
    """Group people by country and by the sets they answered."""
    combo = {}
    for i in range(len(base)):
        combo.setdefault((bool(base[i]),) + tuple(k for k, s in enumerate(S) if (pid[s] == i).any()), []).append(i)
    return [np.array(v) for v in combo.values()]


def half_draw(D, pid, cstrata, n_persons, rng, images=None):
    """Half of the raters without replacement in each stratum, and images with replacement within each set."""
    keep = np.zeros(n_persons, bool)
    for idx in cstrata:
        m = len(idx) // 2 + (rng.random() < .5 if len(idx) % 2 else 0)
        if m:
            keep[rng.choice(idx, int(m), replace=False)] = True
    out = {}
    for s in S:
        k = keep[pid[s]]
        pool = np.arange(10) if images is None else images[s]
        j = rng.choice(pool, len(pool))
        out[s] = {"mats": {c: v[k][:, j] for c, v in D[s]["mats"].items()}, "person": D[s]["person"][k], "jp": D[s]["jp"][k], "imgs": [D[s]["imgs"][x] for x in j]}
    return out


def interval(obs, draws):
    x = np.asarray(draws, float); x = x[np.isfinite(x)]
    lo, hi = np.percentile(x, [2.5, 97.5])
    return obs + lo - x.mean(), obs + hi - x.mean()


def model_interval(D, z, alpha, N_MODEL):
    # Model gap interval: raters and images, plus raters and images of the training data, with the penalty fixed.
    from sklearn.linear_model import Ridge
    t, _ = M.load_training("siglip2_base")
    tpos = {i: k for k, i in enumerate(z["train_image_ids"])}
    forms = {f: g.pivot(index="rater_id", columns="image_id", values="overall") for f, g in t.groupby("form_id")}
    rng = np.random.default_rng(101)
    gaps = []
    for _ in range(N_MODEL):
        Xs, ys = [], []
        for F in forms.values():
            A = F.to_numpy(float); ids = F.columns.to_numpy()
            r = rng.integers(0, A.shape[0], A.shape[0]); c = rng.integers(0, A.shape[1], A.shape[1])
            A = A[r][:, c]
            v = np.nanmean(A - np.nanmean(A, 1, keepdims=True), 0); v = v - np.nanmean(v)
            ok = ~np.isnan(v)
            Xs.append(z["train_X"][[tpos[i] for i in ids[c][ok]]]); ys.append(v[ok])
        m = Ridge(alpha=alpha).fit(np.concatenate(Xs).astype(float), np.concatenate(ys))
        pb = dict(zip(z["image_ids"], m.predict(z["X"].astype(float))))
        with np.errstate(all="ignore"):
            De = {}
            for s in S:
                k = []
                for j in (True, False):
                    w = np.where(D[s]["jp"] == j)[0]
                    k += list(rng.choice(w, max(len(w) // 2, 2), replace=False))
                k = np.array(k); c = rng.choice(10, 10)
                De[s] = {"mats": {"overall": D[s]["mats"]["overall"][k][:, c]}, "jp": D[s]["jp"][k], "imgs": [D[s]["imgs"][i] for i in c]}
            gaps.append(M.corrected_fit(De, pb, S, True) - M.corrected_fit(De, pb, S, False))
    return np.nanpercentile(gaps, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true")
    q = ap.parse_args().quick
    N_PERM, N_HALF, N_MODEL = (500, 500, 200) if q else (30000, 6000, 6000)
    RES.mkdir(exist_ok=True)
    D = M.load_ratings()
    persons, pid, base = person_index(D)
    cst = strata(pid, base)
    # Image features are not released. Without them, the released predictions and fixed results are used.
    use_features = M.has_features()
    fixed = json.load(open(M.DATA / "precomputed_model_results.json", encoding="utf-8"))
    if use_features:
        pred, alpha = M.fit_transfer_model("siglip2_base")
        _, z = M.load_training("siglip2_base")
    else:
        pred, alpha, z = M.load_predictions()["siglip2_base"], None, None

    p1 = lambda d, sets=S: M.icc(d, sets, True) - M.icc(d, sets, False)
    p2 = lambda d, sets=S: M.element_correlation(d, sets, True) - M.element_correlation(d, sets, False)
    p3 = lambda d, sets=S: M.element_correlation(d, sets, True, True) - M.element_correlation(d, sets, False, True)
    p5 = lambda d: M.corrected_fit(d, pred, S, True) - M.corrected_fit(d, pred, S, False)
    # The five comparisons that support the claims of the paper; Holm adjustment is over these five.
    simple = {"rho": p1, "H": p2, "H_res": p3, "1-rho*": lambda d: 1 - M.rho_star(d), "model": p5}
    names = list(simple)
    obs = {k: f(D) for k, f in simple.items()}

    # Permutation test: country labels are exchanged between raters, with images and the trained model held fixed.
    rng = np.random.default_rng(11)
    cnt = dict.fromkeys(names, 0); tot = dict.fromkeys(names, 0)
    for _ in range(N_PERM):
        flag = np.zeros(len(persons), bool); flag[rng.choice(len(persons), base.sum(), replace=False)] = True
        Dp = relabel(D, pid, flag)
        vals = {}
        for k, f in simple.items():
            with np.errstate(all="ignore"):
                vals[k] = f(Dp)
        for k, v in vals.items():
            if not np.isnan(v):
                tot[k] += 1; cnt[k] += (v >= obs[k]) if k == "1-rho*" else (abs(v) >= abs(obs[k]))
    raw = {k: (cnt[k] + 1) / (tot[k] + 1) for k in names}
    adj = dict(zip(names, M.holm([raw[k] for k in names])))

    # Half-sampling of raters and resampling of images.
    prior = pd.read_csv(M.DATA / "selection_scores.csv").set_index("image_id").selection_score
    x = {s: np.array([prior[i] for i in D[s]["imgs"]]) for s in S}
    mid = {s: np.sort(np.argsort(np.abs(x[s] - np.median(x[s])))[:5]) for s in S}
    flat = {}
    n_flat = {True: 0, False: 0}
    for s in S:
        Z = np.stack([D[s]["mats"][c] for c in M.ELEMENTS], -1)
        fl = (Z.max(-1) == Z.min(-1)).sum(1) >= 5
        for j in (True, False):
            n_flat[j] += int((fl & (D[s]["jp"] == j)).sum())
        k = ~fl
        flat[s] = {**D[s], "mats": {c: v[k] for c, v in D[s]["mats"].items()}, "person": D[s]["person"][k], "jp": D[s]["jp"][k]}
    keep_f = {s: np.isin(D[s]["person"], flat[s]["person"]) for s in S}

    q1 = {"rho_diff": p1, "H_diff": p2, "H_res_diff": p3, "rho_star": M.rho_star, "icc_jp": lambda d: M.icc(d, S, True), "icc_vn": lambda d: M.icc(d, S, False)}
    h = {k: [] for k in q1}
    rng = np.random.default_rng(20261107)
    for _ in range(N_HALF):
        Db = half_draw(D, pid, cst, len(persons), rng)
        for k, f in q1.items():
            with np.errstate(all="ignore"):
                h[k].append(f(Db))
    ci = {k: interval(q1[k](D), v) for k, v in h.items()}
    # Ratio of required raters (VN / JP). It does not depend on the target reliability R.
    # Interval: the same shifted percentile as above, on the log scale.
    kr = lambda a, b: M.required_raters(b) / M.required_raters(a)
    ratio_obs = kr(M.icc(D, S, True), M.icc(D, S, False))
    lr = np.log([kr(a, b) for a, b in zip(h["icc_jp"], h["icc_vn"])]); lr = lr[np.isfinite(lr)]
    ratio_ci = tuple(float(np.exp(np.log(ratio_obs) + np.percentile(lr, q) - lr.mean())) for q in (2.5, 97.5))

    # Sensitivity: images near the middle of the selection score, and without flat responses.
    Dmid = {s: {**D[s], "mats": {c: v[:, mid[s]] for c, v in D[s]["mats"].items()}, "imgs": [D[s]["imgs"][i] for i in mid[s]]} for s in S}
    mids, fH, fHr, fR = [], [], [], []
    rng = np.random.default_rng(20261108)
    for _ in range(N_HALF):
        mids.append(p1(half_draw(D, pid, cst, len(persons), rng, images=mid)))
        Db = half_draw(flat, {s: pid[s][keep_f[s]] for s in S}, cst, len(persons), rng)
        with np.errstate(all="ignore"):
            fH.append(p2(Db)); fHr.append(p3(Db)); fR.append(p1(Db))

    interval_model = tuple(model_interval(D, z, alpha, N_MODEL)) if use_features else tuple(fixed["model_gap_interval"])

    # Tables
    rs, rj, rv = M.between_group_parts(D)
    L = ["# Reproduced results", "", f"Permutations {N_PERM}, half-samples {N_HALF}. Image features: {'used' if use_features else 'not available; model training results are fixed values'}.", "",
         "| Measure | Japanese | Vietnamese | Holm p | 95% interval of the difference |", "|---|---|---|---|---|",
         f"| Agreement rho | {M.icc(D,S,True):.3f} | {M.icc(D,S,False):.3f} | {adj['rho']:.4f} | [{ci['rho_diff'][0]:+.3f}, {ci['rho_diff'][1]:+.3f}] |",
         f"| H | {M.element_correlation(D,S,True):.3f} | {M.element_correlation(D,S,False):.3f} | {adj['H']:.4f} | [{ci['H_diff'][0]:+.3f}, {ci['H_diff'][1]:+.3f}] |",
         f"| H_res | {M.element_correlation(D,S,True,True):.3f} | {M.element_correlation(D,S,False,True):.3f} | {adj['H_res']:.4f} | [{ci['H_res_diff'][0]:+.3f}, {ci['H_res_diff'][1]:+.3f}] |",
         f"| rho* (value and interval; p against 1) | {M.rho_star(D):.3f} | | {adj['1-rho*']:.4f} | [{ci['rho_star'][0]:+.3f}, {ci['rho_star'][1]:+.3f}] |",
         f"| Model fit | {M.corrected_fit(D,pred,S,True):.3f} | {M.corrected_fit(D,pred,S,False):.3f} | {adj['model']:.4f} | [{interval_model[0]:+.3f}, {interval_model[1]:+.3f}] |", "",
         "## Details", "",
          f"- By condition: blue shirt {M.icc(D,M.BLUE,True):.3f} / {M.icc(D,M.BLUE,False):.3f}, general {M.icc(D,M.GENERAL,True):.3f} / {M.icc(D,M.GENERAL,False):.3f}",
          f"- Rank correlation {rs:.3f}, reliability {rj:.3f} / {rv:.3f}",
          f"- Required raters (R=0.8): JP {M.required_raters(M.icc(D,S,True)):.1f} [{M.required_raters(ci['icc_jp'][1]):.1f}, {M.required_raters(ci['icc_jp'][0]):.1f}], VN {M.required_raters(M.icc(D,S,False)):.1f} [{M.required_raters(ci['icc_vn'][1]):.1f}, {M.required_raters(ci['icc_vn'][0]):.1f}], ratio VN/JP {ratio_obs:.1f} [{ratio_ci[0]:.1f}, {ratio_ci[1]:.1f}]",
          f"- Selection score vs consensus: JP {stats.spearmanr(np.concatenate([x[s]-x[s].mean() for s in S]), np.concatenate([M.consensus(D[s]['mats']['overall'][D[s]['jp']]) for s in S]))[0]:+.3f}, VN {stats.spearmanr(np.concatenate([x[s]-x[s].mean() for s in S]), np.concatenate([M.consensus(D[s]['mats']['overall'][~D[s]['jp']]) for s in S]))[0]:+.3f}",
          f"- 30 images near the middle of the score: {M.icc(Dmid,S,True):.3f} / {M.icc(Dmid,S,False):.3f}, difference {p1(Dmid):+.3f} [{interval(p1(Dmid),mids)[0]:+.3f}, {interval(p1(Dmid),mids)[1]:+.3f}]",
          f"- Flat responses: JP {n_flat[True]} of {int(sum(D[s]['jp'].sum() for s in S))}, VN {n_flat[False]} of {int(sum((~D[s]['jp']).sum() for s in S))}",
          f"  - H difference {p2(flat):+.3f} [{interval(p2(flat),fH)[0]:+.3f}, {interval(p2(flat),fH)[1]:+.3f}]",
          f"  - H_res difference {p3(flat):+.3f} [{interval(p3(flat),fHr)[0]:+.3f}, {interval(p3(flat),fHr)[1]:+.3f}]",
          f"  - rho {M.icc(flat,S,True):.3f} / {M.icc(flat,S,False):.3f}, difference {p1(flat):+.3f} [{interval(p1(flat),fR)[0]:+.3f}, {interval(p1(flat),fR)[1]:+.3f}]", "",
          "## Other image features (difference in corrected rank correlation)", ""]
    for f in M.FEATURES:
        pr = M.fit_transfer_model(f)[0] if use_features else M.load_predictions()[f]
        a, b = M.corrected_fit(D, pr, S, True), M.corrected_fit(D, pr, S, False)
        L.append(f"- {f}: {a:.3f} / {b:.3f}, {a-b:+.3f}")
    L += ["", "## By set", "", "| Set | JP raters | VN raters | rho difference | H difference |", "|---|---|---|---|---|"]
    L += [f"| {s} | {int(D[s]['jp'].sum())} | {int((~D[s]['jp']).sum())} | {p1(D,[s]):+.2f} | {p2(D,[s]):+.2f} |" for s in S]
    (RES / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
