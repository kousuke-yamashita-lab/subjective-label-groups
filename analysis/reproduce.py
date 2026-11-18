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


def imbalance_checks(D, n_draw=2000, n_per=5, seed=0):
    """Influence of the unequal numbers of raters per set: rho with each set removed,
    and rho with the raters of both groups resampled to n_per per set."""
    drop = {s: (M.icc(D, [x for x in S if x != s], True), M.icc(D, [x for x in S if x != s], False)) for s in S}
    rng = np.random.default_rng(seed)
    r = []
    for _ in range(n_draw):
        Db = {}
        for s in S:
            idx = []
            for j in (True, False):
                w = np.where(D[s]["jp"] == j)[0]
                idx += list(rng.choice(w, n_per, replace=False))
            idx = np.array(idx)
            Db[s] = {"mats": {"overall": D[s]["mats"]["overall"][idx]}, "jp": D[s]["jp"][idx], "imgs": D[s]["imgs"]}
        r.append((M.icc(Db, S, True), M.icc(Db, S, False)))
    return drop, np.array(r)


def aggregation_variants(D):
    """rho under other ways of combining the six sets (the paper averages the components before taking the ratio)."""
    out = {}
    for jp in (True, False):
        c = []
        for s in S:
            Y = D[s]["mats"]["overall"][D[s]["jp"] == jp]
            n, k = Y.shape
            vi, ve = M.variance_components(Y)
            msi = n * ((Y.mean(0) - Y.mean()) ** 2).sum() / (k - 1)
            c.append((vi, ve, (msi - ve) / n, (n - 1) * (k - 1)))
        c = np.array(c); vi, ve, raw, w = c.T
        out[jp] = {"Paper (negative variance set to 0)": vi.mean() / (vi + ve).mean(),
                   "Negative variance kept": raw.mean() / (raw + ve).mean(),
                   "Sets weighted by degrees of freedom": (vi * w).sum() / ((vi + ve) * w).sum(),
                   "Mean of the per-set rho": np.mean(vi / (vi + ve))}
    return out


SEED_EXTRA = 20261116


def matched_section(D, persons, pid, base, use_features, fixed, q):
    """Label reliability and model accuracy in the matched design (S3 left out, target 3 raters, training k other raters)."""
    if not use_features:
        return fixed["matched_section"] + ["", "Image features not available: the section above is the fixed output of a full run with the features."]
    SM, KS, MT = M.MATCHED_SETS, [1, 2, 3, 4], 3
    n_obs, n_b, n_p, inner = (40, 20, 20, 3) if q else (500, 400, 600, 6)
    z = np.load(M.PRIVATE / "features_siglip2_base.npz", allow_pickle=True)
    F = {i: v for i, v in zip(z["image_ids"], z["X"].astype(float))}
    X = {s: np.stack([F[i] for i in D[s]["imgs"]]) for s in SM}
    G = (("JP", True), ("VN", False))
    sb = lambda r, k: k * r / (1 + (k - 1) * r)

    def table(Dx, rng, n, ks, imgs=None, with_elements=True):
        out = {}
        with np.errstate(all="ignore"):
            for g, jp in G:
                Rm = M.target_reliability(Dx, SM, jp, ["overall"], MT)
                for k in ks:
                    v = np.array([M.matched_draw(Dx, X, jp, k, rng, imgs=imgs) for _ in range(n)], float)
                    lab, mod = np.nanmean(v[:, 0]), np.nanmean(v[:, 1])
                    out[(g, k)] = (lab ** 2 / Rm if Rm > 0 else np.nan, mod / np.sqrt(Rm) if Rm > 0 else np.nan, mod)
                if with_elements:
                    v = np.array([M.matched_draw(Dx, X, jp, 4, rng, cols=M.SPEC, imgs=imgs) for _ in range(n)], float)
                    out[(g, "el")] = float(np.nanmean(v[:, 1]))
        return out

    def summary(tb):
        return {"diff_k1": tb[("JP", 1)][1] - tb[("VN", 1)][1], "diff_k4": tb[("JP", 4)][1] - tb[("VN", 4)][1],
                "gain_diff": (tb[("VN", 4)][1] - tb[("VN", 1)][1]) - (tb[("JP", 4)][1] - tb[("JP", 1)][1]),
                "element_diff": tb[("JP", "el")] - tb[("VN", "el")]}

    rng = np.random.default_rng(SEED_EXTRA + 1)
    tb = table(D, rng, n_obs, KS)
    rho = {g: M.icc(D, SM, jp) for g, jp in G}
    L = ["", "## Label reliability and model accuracy (matched design)", "",
         f"Sets {', '.join(SM)} (S3 has too few raters). In both groups the target of a held-out set is the consensus of {MT} randomly drawn raters "
         "and the training labels are the consensus of k other raters per training set; the target persons are never used for training. "
         f"Ridge on SigLIP2-base features, leave-one-set-out; means of {n_obs} rater draws.", "",
         f"On these sets: rho {rho['JP']:.3f} / {rho['VN']:.3f}, required-rater ratio VN/JP {M.required_raters(rho['VN']) / M.required_raters(rho['JP']):.1f}; "
         f"rho_spec {M.element_specific_agreement(D, SM, True):.3f} / {M.element_specific_agreement(D, SM, False):.3f}.", "",
         "| Group | k | Label reliability (Spearman-Brown) | Label reliability (empirical) | Model accuracy (corrected) | Rank correlation (uncorrected) |", "|---|---|---|---|---|---|"]
    for g, _ in G:
        for k in KS:
            L.append(f"| {g} | {k} | {sb(rho[g], k):.2f} | {tb[(g, k)][0]:.2f} | {tb[(g, k)][1]:.3f} | {tb[(g, k)][2]:.3f} |")
    xs = np.sqrt([sb(rho[g], k) for g, _ in G for k in KS]); ys = [tb[(g, k)][1] for g, _ in G for k in KS]
    L += ["", f"Correlation between sqrt(label reliability) and model accuracy over the 8 conditions: {np.corrcoef(xs, ys)[0, 1]:.3f}. "
          "Accuracy / sqrt(label reliability): " + "; ".join(f"{g} " + ", ".join(f"{tb[(g, k)][1] / np.sqrt(sb(rho[g], k)):.2f}" for k in KS) for g, _ in G) + ".",
          f"Element-specific part, k = 4, rank correlation (uncorrected): JP {tb[('JP', 'el')]:.3f}, VN {tb[('VN', 'el')]:.3f} "
          f"(reliability of the {MT}-rater target {M.target_reliability(D, SM, True, M.SPEC, MT):.2f} / {M.target_reliability(D, SM, False, M.SPEC, MT):.2f})."]
    obs = summary(tb)
    rng = np.random.default_rng(SEED_EXTRA + 2); dr = {k: [] for k in obs}
    for _ in range(n_b):
        imgs = {s: rng.choice(10, 10) for s in SM}
        for k, v in summary(table(D, rng, inner, [1, 4], imgs=imgs)).items():
            dr[k].append(v)
    rng = np.random.default_rng(SEED_EXTRA + 3); cnt = dict.fromkeys(obs, 0); tot = dict.fromkeys(obs, 0)
    for _ in range(n_p):
        flag = np.zeros(len(persons), bool); flag[rng.choice(len(persons), int(base.sum()), replace=False)] = True
        for k, v in summary(table(relabel(D, pid, flag), rng, inner, [1, 4])).items():
            if np.isfinite(v):
                tot[k] += 1; cnt[k] += abs(v) >= abs(obs[k])
    L += ["", f"Uncertainty: interval from {n_b} resamples of the images within sets with new rater draws; p from {n_p} exchanges of group labels between raters (two-sided, not Holm-adjusted).", "",
          "| Statistic | Observed | 95% interval | p (valid permutations) |", "|---|---|---|---|"]
    names = {"diff_k1": "Accuracy, k = 1, JP - VN", "diff_k4": "Accuracy, k = 4, JP - VN", "gain_diff": "Gain from k = 1 to 4, VN - JP", "element_diff": "Element-specific part, k = 4, JP - VN (uncorrected)"}
    for k in obs:
        lo, hi = interval(obs[k], dr[k])
        L.append(f"| {names[k]} | {obs[k]:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {(cnt[k] + 1) / (tot[k] + 1):.4f} ({tot[k]}) |")
    if not q:
        fixed["matched_section"] = L
        (M.DATA / "precomputed_model_results.json").write_text(json.dumps(fixed, indent=1) + "\n", encoding="utf-8")
    return L


def within_group_section(D, use_features, fixed, q):
    """Agreement and model accuracy: models trained on the consensus of k raters per image within each group."""
    KS = [1, 2, 3, 5, None]
    n_draw = 30 if q else 300
    L = ["", "## Supplementary: all six sets with unequal numbers of raters", "",
         "Ridge on SigLIP2-base features, leave-one-set-out, trained on the consensus of k raters per training image; "
         "accuracy = rank correlation with the held-out consensus of all raters of the same group / sqrt(its reliability). "
         "Persons who rated the held-out set are not used for training. Mean [2.5%, 97.5%] over draws of the training raters.", "",
         "| k | Japanese | Vietnamese |", "|---|---|---|"]
    if use_features:
        z = np.load(M.PRIVATE / "features_siglip2_base.npz", allow_pickle=True)
        F = {i: v for i, v in zip(z["image_ids"], z["X"].astype(float))}
        X = {s: np.stack([F[i] for i in D[s]["imgs"]]) for s in S}
        rng = np.random.default_rng(SEED_EXTRA)
        res = {}
        for k in KS:
            lab = "all" if k is None else str(k)
            res[lab] = {}
            for g, jp in (("JP", True), ("VN", False)):
                v = np.array([M.within_group_accuracy(D, X, jp, k, rng) for _ in range(1 if k is None else n_draw)])
                res[lab][g] = [float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    else:
        res = fixed["within_group_accuracy"]
    for lab, r in res.items():
        f = lambda a: f"{a[0]:.3f}" if lab == "all" else f"{a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}]"
        L.append(f"| {lab} | {f(r['JP'])} | {f(r['VN'])} |")
    if not use_features:
        L += ["", "Image features not available: the values above are fixed results of a full run with the features."]
    return L


def pilot_section(D, q):
    """Proportion of simulated pilots that reach the same decisions as the full data."""
    n_pilot = 200 if q else 2000
    rj, rv = M.icc(D, S, True), M.icc(D, S, False)
    L = ["", "## Size of a pilot study", "",
         f"Full data: rho {rj:.3f} / {rv:.3f}, required-rater ratio VN/JP {M.required_raters(rv) / M.required_raters(rj):.1f}. "
         f"Proportion of {n_pilot} simulated pilots that reach the same decision as the full data.", "",
         "| Pilot | Images | rho_VN < rho_JP | Ratio >= 2 | Median ratio [IQR] | H_VN > H_JP | H_res_VN > H_res_JP | rho_spec_VN < rho_spec_JP | All four directions |",
         "|---|---|---|---|---|---|---|---|---|"]
    for n_sets in (1, 2, 4, 6):
        for n in (3, 5, 8):
            rng = np.random.default_rng(SEED_EXTRA + 100 * n_sets + n)
            a = b = c = d = e = g = 0; ratios = []
            for _ in range(n_pilot):
                if n_sets == 6:
                    sets = S
                elif n_sets % 2 == 0:
                    sets = list(rng.choice(M.BLUE, n_sets // 2, replace=False)) + list(rng.choice(M.GENERAL, n_sets // 2, replace=False))
                else:
                    sets = list(rng.choice(S, n_sets, replace=False))
                P = M.pilot_subset(D, sets, n, rng)
                with np.errstate(all="ignore"):
                    pj, pv = M.icc(P, sets, True), M.icc(P, sets, False)
                    hj, hv = M.element_correlation(P, sets, True), M.element_correlation(P, sets, False)
                    qj, qv = M.element_correlation(P, sets, True, True), M.element_correlation(P, sets, False, True)
                    sj, sv = M.element_specific_agreement(P, sets, True), M.element_specific_agreement(P, sets, False)
                r = M.required_raters(pv) / M.required_raters(pj) if pj > 0 else np.nan
                ratios.append(r if np.isfinite(r) else 1e9)
                ok1 = pv < pj; ok2 = ((pv <= 0) or (np.isfinite(r) and r >= 2)) if pj > 0 else False
                ok3 = hv > hj; ok4 = qv > qj; ok5 = sv < sj
                a += ok1; b += ok2; c += ok3; d += ok4; g += ok5; e += (ok1 and ok3 and ok4 and ok5)
            qq = np.percentile(ratios, [25, 50, 75]); fq = lambda x: ">100" if x > 100 else f"{x:.1f}"
            L.append(f"| {n_sets} set{'s' if n_sets > 1 else ''} x {n} raters per group | {10 * n_sets} | {a / n_pilot:.2f} | {b / n_pilot:.2f} | "
                     f"{fq(qq[1])} [{fq(qq[0])}, {fq(qq[2])}] | {c / n_pilot:.2f} | {d / n_pilot:.2f} | {g / n_pilot:.2f} | {e / n_pilot:.2f} |")
    L += ["", "Raters per set are capped by availability (the smallest set has 6 Japanese and 5 Vietnamese raters)."]
    return L


def stability_section(D):
    """Rank correlation, over raters who answered both conditions, between their H in the blue-shirt and the general set."""
    h = M.rater_element_correlation(D)
    jp = {p: bool(D[s]["jp"][i]) for s in S for i, p in enumerate(D[s]["person"])}
    L = ["", "## Stability of each rater's H across the two conditions", "",
         "Each rater's H is computed within one set (10 images). Spearman correlation between the blue-shirt and the general value, "
         "raters who answered both conditions only; interval from 1000 resamplings of raters.", "",
         "| Group | Raters | Rank correlation [95% interval] | p |", "|---|---|---|---|"]
    rng = np.random.default_rng(0)
    for name, flag in (("JP", True), ("VN", False)):
        ps = sorted(p for p in jp if jp[p] == flag and (p, "blue") in h and (p, "general") in h)
        a = np.array([h[(p, "blue")] for p in ps]); b = np.array([h[(p, "general")] for p in ps])
        rho, pv = stats.spearmanr(a, b)
        bt = []
        for _ in range(1000):
            i = rng.integers(0, len(a), len(a))
            with np.errstate(all="ignore"):
                bt.append(stats.spearmanr(a[i], b[i])[0])
        bt = np.array(bt); bt = bt[np.isfinite(bt)]
        L.append(f"| {name} | {len(ps)} | {rho:+.2f} [{np.percentile(bt, 2.5):+.2f}, {np.percentile(bt, 97.5):+.2f}] | {pv:.3f} |")
    return L


def image_split_section(D):
    """Exploratory check of how much each image splits the raters (supplementary; not part of the Holm family)."""
    within_rank = lambda vs: np.concatenate([stats.rankdata(v) for v in vs])

    def per_group(jp, rows=None):
        v, c = [], []
        for s in S:
            Y = D[s]["mats"]["overall"][D[s]["jp"] == jp]
            if rows is not None:
                Y = Y[rows[s]]
            v.append(M.image_disagreement(Y)); c.append(M.consensus(Y))
        return v, c

    L = ["", "## Image-level disagreement (exploratory)", "",
         "Disagreement of an image: variance of the residual after removing rater levels and the image consensus. "
         "Values are ranked within each set before pooling the 60 images, because the raters differ between sets.", "",
         "| Group | Split-half rank correlation (mean of 2000) | Corrected to all raters (Spearman-Brown) |", "|---|---|---|"]
    rng = np.random.default_rng(20261109)
    for jp, name in ((True, "JP"), (False, "VN")):
        rs = []
        for _ in range(2000):
            A, B = {}, {}
            for s in S:
                n = int((D[s]["jp"] == jp).sum()); p = rng.permutation(n); h = n // 2
                A[s], B[s] = p[:h], p[h:2 * h]
            if any(len(A[s]) < 2 for s in S):
                continue
            rs.append(stats.spearmanr(within_rank(per_group(jp, A)[0]), within_rank(per_group(jp, B)[0]))[0])
        r = float(np.mean(rs))
        L.append(f"| {name} | {r:+.2f} | {2 * r / (1 + r):+.2f} |")
    vj, cj = per_group(True); vv, cv = per_group(False)
    r, p = stats.spearmanr(within_rank(vj), within_rank(vv))
    a = stats.spearmanr(within_rank([np.abs(x) for x in cv]), within_rank(vj))[0]
    b = stats.spearmanr(within_rank([np.abs(x) for x in cj]), within_rank(vv))[0]
    L += ["", f"- JP vs VN disagreement, rank correlation over 60 images: {r:+.2f} (p = {p:.3f})",
          f"- Disagreement vs |consensus of the other group|: JP {a:+.2f}, VN {b:+.2f}"]
    return L


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
    p6 = lambda d, sets=S: M.element_specific_agreement(d, sets, True) - M.element_specific_agreement(d, sets, False)
    # The six comparisons that support the claims of the paper; Holm adjustment is over these six.
    simple = {"rho": p1, "H": p2, "H_res": p3, "rho_spec": p6, "1-rho*": lambda d: 1 - M.rho_star(d), "model": p5}
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

    q1 = {"rho_diff": p1, "H_diff": p2, "H_res_diff": p3, "rho_star": M.rho_star, "icc_jp": lambda d: M.icc(d, S, True), "icc_vn": lambda d: M.icc(d, S, False),
          "rho_spec_diff": p6, "spec_jp": lambda d: M.element_specific_agreement(d, S, True), "spec_vn": lambda d: M.element_specific_agreement(d, S, False)}
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
    mids, fH, fHr, fR, fS = [], [], [], [], []
    rng = np.random.default_rng(20261108)
    for _ in range(N_HALF):
        mids.append(p1(half_draw(D, pid, cst, len(persons), rng, images=mid)))
        Db = half_draw(flat, {s: pid[s][keep_f[s]] for s in S}, cst, len(persons), rng)
        with np.errstate(all="ignore"):
            fH.append(p2(Db)); fHr.append(p3(Db)); fR.append(p1(Db)); fS.append(p6(Db))

    interval_model = tuple(model_interval(D, z, alpha, N_MODEL)) if use_features else tuple(fixed["model_gap_interval"])

    # Tables
    rs, rj, rv = M.between_group_parts(D)
    L = ["# Reproduced results", "", f"Permutations {N_PERM}, half-samples {N_HALF}. Image features: {'used' if use_features else 'not available; model training results are fixed values'}.", "",
         "| Measure | Japanese | Vietnamese | Holm p | 95% interval of the difference |", "|---|---|---|---|---|",
         f"| Agreement rho | {M.icc(D,S,True):.3f} | {M.icc(D,S,False):.3f} | {adj['rho']:.4f} | [{ci['rho_diff'][0]:+.3f}, {ci['rho_diff'][1]:+.3f}] |",
         f"| H | {M.element_correlation(D,S,True):.3f} | {M.element_correlation(D,S,False):.3f} | {adj['H']:.4f} | [{ci['H_diff'][0]:+.3f}, {ci['H_diff'][1]:+.3f}] |",
         f"| H_res | {M.element_correlation(D,S,True,True):.3f} | {M.element_correlation(D,S,False,True):.3f} | {adj['H_res']:.4f} | [{ci['H_res_diff'][0]:+.3f}, {ci['H_res_diff'][1]:+.3f}] |",
         f"| Element-specific agreement rho_spec | {M.element_specific_agreement(D,S,True):.3f} [{ci['spec_jp'][0]:.3f}, {ci['spec_jp'][1]:.3f}] | {M.element_specific_agreement(D,S,False):.3f} [{ci['spec_vn'][0]:.3f}, {ci['spec_vn'][1]:.3f}] | {adj['rho_spec']:.4f} | [{ci['rho_spec_diff'][0]:+.3f}, {ci['rho_spec_diff'][1]:+.3f}] |",
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
          f"  - rho {M.icc(flat,S,True):.3f} / {M.icc(flat,S,False):.3f}, difference {p1(flat):+.3f} [{interval(p1(flat),fR)[0]:+.3f}, {interval(p1(flat),fR)[1]:+.3f}]",
          f"  - rho_spec {M.element_specific_agreement(flat,S,True):.3f} / {M.element_specific_agreement(flat,S,False):.3f}, difference {p6(flat):+.3f} [{interval(p6(flat),fS)[0]:+.3f}, {interval(p6(flat),fS)[1]:+.3f}]",
          "- rho_spec by element (JP / VN): " + "; ".join(f"{e} {M.icc(D,S,True,'spec_'+e):.2f} / {M.icc(D,S,False,'spec_'+e):.2f}" for e in M.ELEMENTS), "",
          "## Other image features (difference in corrected rank correlation)", ""]
    for f in M.FEATURES:
        pr = M.fit_transfer_model(f)[0] if use_features else M.load_predictions()[f]
        a, b = M.corrected_fit(D, pr, S, True), M.corrected_fit(D, pr, S, False)
        L.append(f"- {f}: {a:.3f} / {b:.3f}, {a-b:+.3f}")
    agg = aggregation_variants(D)
    L += ["", "## Ways of combining the sets", "", "| Method | Japanese | Vietnamese | Difference |", "|---|---|---|---|"]
    L += [f"| {m} | {agg[True][m]:.3f} | {agg[False][m]:.3f} | {agg[True][m] - agg[False][m]:+.3f} |" for m in agg[True]]
    drop, eq = imbalance_checks(D)
    L += ["", "## Unequal numbers of raters per set", "", "| Condition | Japanese | Vietnamese | Difference |", "|---|---|---|---|"]
    L += [f"| {s} removed | {a:.3f} | {v:.3f} | {a - v:+.3f} |" for s, (a, v) in drop.items()]
    L += [f"| Five raters per set, resampled (mean of {len(eq)}) | {eq[:, 0].mean():.3f} | {eq[:, 1].mean():.3f} | {(eq[:, 0] - eq[:, 1]).mean():+.3f} |",
          "", f"Draws in which the Japanese group is higher: {int((eq[:, 0] > eq[:, 1]).sum())} of {len(eq)}"]
    sd_c = lambda s, jp: np.sqrt(M.variance_components(D[s]["mats"]["overall"][D[s]["jp"] == jp])[0])
    L += ["", "## By set", "",
          "sigma_c: square root of the image-consensus variance component (0-100 scale). Score SD: SD of the selection score of the set's images.", "",
          "| Set | JP raters | VN raters | rho JP | rho VN | rho difference | H difference | Score SD | sigma_c JP | sigma_c VN |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    L += [f"| {s} | {int(D[s]['jp'].sum())} | {int((~D[s]['jp']).sum())} | {M.icc(D, [s], True):.2f} | {M.icc(D, [s], False):.2f} | "
          f"{p1(D,[s]):+.2f} | {p2(D,[s]):+.2f} | {np.std(x[s], ddof=1):.1f} | {sd_c(s, True):.1f} | {sd_c(s, False):.1f} |" for s in S]
    L += matched_section(D, persons, pid, base, use_features, fixed, q)
    L += within_group_section(D, use_features, fixed, q)
    L += pilot_section(D, q)
    L += stability_section(D)
    L += image_split_section(D)
    (RES / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
