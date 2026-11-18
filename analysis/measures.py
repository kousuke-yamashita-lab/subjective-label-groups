"""Data loading and the statistics reported in the paper."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV

DATA = Path(__file__).resolve().parents[1] / "data"
PRIVATE = DATA / "private"     # image features are not released
ELEMENTS = ["color", "pattern", "silhouette", "material", "accessory", "styling"]
SETS = ["S1", "S2", "S3", "S4", "S5", "S6"]
BLUE, GENERAL = SETS[:3], SETS[3:]
ALPHAS = np.logspace(-3, 3, 25)
FEATURES = ["siglip2_base", "siglip_so400m", "clip_l14", "fashion_clip", "vit_in21k"]


def load_ratings(path=None):
    """Return {set_id: {"mats": {name: raters x images}, "person": ids, "jp": mask, "imgs": ids}}."""
    df = pd.read_csv(path or DATA / "ratings.csv")
    out = {}
    for s, g in df.groupby("set_id"):
        raters = sorted(g.rater_id.unique()); imgs = sorted(g.image_id.unique())
        mats = {c: g.pivot(index="rater_id", columns="image_id", values=c).loc[raters, imgs].to_numpy(float) for c in ["overall"] + ELEMENTS}
        Z = np.stack([mats[c] for c in ELEMENTS], -1)
        for j, c in enumerate(ELEMENTS):
            # element-specific part: the element rating minus the rater's mean over the six elements for the image
            mats["spec_" + c] = Z[..., j] - Z.mean(-1)
        group = g.drop_duplicates("rater_id").set_index("rater_id").group
        out[s] = {"mats": mats, "person": np.array(raters), "jp": np.array([group[r] == "JP" for r in raters]), "imgs": imgs}
    return out


def variance_components(Y):
    """Two-way additive decomposition: variance of the image consensus and of the residual."""
    n, k = Y.shape
    E = Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + Y.mean()
    ms_e = (E ** 2).sum() / ((n - 1) * (k - 1))
    ms_i = n * ((Y.mean(0) - Y.mean()) ** 2).sum() / (k - 1)
    return max((ms_i - ms_e) / n, 0.0), ms_e


def icc(D, sets, jp, col="overall"):
    """ICC(C,1): components are averaged over the sets before taking the ratio."""
    c = np.array([variance_components(D[s]["mats"][col][D[s]["jp"] == jp]) for s in sets if (D[s]["jp"] == jp).sum() >= 2])
    return c[:, 0].mean() / c.sum(1).mean()


def required_raters(rho, target=0.8):
    return target * (1 - rho) / ((1 - target) * rho) if rho > 0 else np.inf


def element_correlation(D, sets, jp, remove_image=False):
    """H (level removed) or H_res (level and image consensus removed), pooled over raters and sets."""
    rows = []
    for s in sets:
        m = D[s]["jp"] == jp
        if m.sum() < 2:
            continue
        X = np.stack([D[s]["mats"][c][m] for c in ELEMENTS], -1).astype(float)
        X = X - X.mean(1, keepdims=True)
        if remove_image:
            X = X - X.mean(0, keepdims=True)
        rows.append(X.reshape(-1, 6))
    with np.errstate(all="ignore"):
        return np.corrcoef(np.concatenate(rows).T)[np.triu_indices(6, 1)].mean()


def image_disagreement(Y):
    """Per-image variance of the residual after removing rater levels and the image consensus.

    The (n-1)(k-1) degrees of freedom of the residual are shared equally among the k images.
    """
    n, k = Y.shape
    E = Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + Y.mean()
    return (E ** 2).sum(0) / ((n - 1) * (k - 1) / k)


def consensus(Y):
    """Image consensus: remove each rater's level, average over raters, centre within the set."""
    v = (Y - Y.mean(1, keepdims=True)).mean(0)
    return v - v.mean()


def _reliability(comps):
    vi, ve, n = (np.mean([c[i] for c in comps]) for i in range(3))
    return vi / (vi + ve / n)


def between_group_parts(D, sets=SETS):
    cj, cv, rj, rv = [], [], [], []
    for s in sets:
        A, B = D[s]["mats"]["overall"][D[s]["jp"]], D[s]["mats"]["overall"][~D[s]["jp"]]
        if len(A) < 2 or len(B) < 2:
            return np.nan, np.nan, np.nan
        cj.append(consensus(A)); cv.append(consensus(B))
        rj.append((*variance_components(A), len(A))); rv.append((*variance_components(B), len(B)))
    return stats.spearmanr(np.concatenate(cj), np.concatenate(cv))[0], _reliability(rj), _reliability(rv)


def rho_star(D, sets=SETS):
    """Rank correlation of the group consensuses, corrected for their reliability."""
    r, a, b = between_group_parts(D, sets)
    return r / np.sqrt(a * b) if a > 0 and b > 0 else np.nan


def corrected_fit(D, pred, sets, jp):
    """Corrected rank correlation between model predictions and one group's consensus."""
    ts, ps, comps = [], [], []
    for s in sets:
        Y = D[s]["mats"]["overall"][D[s]["jp"] == jp]
        if len(Y) < 2:
            continue
        ts.append(consensus(Y))
        p = np.array([pred[i] for i in D[s]["imgs"]]); ps.append(p - p.mean())
        comps.append((*variance_components(Y), len(Y)))
    R = _reliability(comps)
    return stats.spearmanr(np.concatenate(ps), np.concatenate(ts))[0] / np.sqrt(R) if R > 0 else np.nan


def load_training(feature):
    t = pd.read_csv(DATA / "train_ratings.csv")
    z = np.load(PRIVATE / f"features_{feature}.npz", allow_pickle=True)
    return t, z


def has_features():
    return (PRIVATE / "features_siglip2_base.npz").exists()


def load_predictions():
    """Released per-image predictions of the transfer model, one column per image feature."""
    df = pd.read_csv(DATA / "model_predictions.csv").set_index("image_id")
    return {f: df[f].to_dict() for f in FEATURES}


def training_target(t):
    t = t.copy()
    t["yc"] = t.overall - t.groupby(["form_id", "rater_id"]).overall.transform("mean")
    c = t.groupby(["form_id", "image_id"]).yc.mean()
    return c - c.groupby(level=0).transform("mean")


def fit_transfer_model(feature="siglip2_base"):
    """Ridge on frozen image features, trained on the separate Japanese ratings. Returns {image_id: prediction}."""
    t, z = load_training(feature)
    target = training_target(t)
    pos = {i: k for k, i in enumerate(z["train_image_ids"])}
    X = z["train_X"][[pos[i] for _, i in target.index]]
    model = RidgeCV(alphas=ALPHAS).fit(X, target.values)
    return dict(zip(z["image_ids"], model.predict(z["X"]))), float(model.alpha_)


def holm(ps):
    ps = np.asarray(ps); order = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for i, k in enumerate(order):
        run = max(run, min(1, (len(ps) - i) * ps[k])); adj[k] = run
    return adj


def within_group_accuracy(D, X, jp, k, rng):
    """Leave-one-set-out accuracy of a ridge model trained on the consensus of k raters per training image.

    X: {set_id: image features (10 x d)}. The target of the held-out set is the consensus of all raters of the same
    group on that set; persons who rated the held-out set are excluded from the training raters. k=None uses every
    available rater. Returns the rank correlation divided by sqrt(reliability of the held-out consensus)."""
    ps, ts, comps = [], [], []
    for te in SETS:
        m_te = D[te]["jp"] == jp
        Yte = D[te]["mats"]["overall"][m_te]
        excl = list(set(D[te]["person"][m_te]))
        Xtr, ytr = [], []
        for tr in SETS:
            if tr == te:
                continue
            idx = np.where((D[tr]["jp"] == jp) & ~np.isin(D[tr]["person"], excl))[0]
            if len(idx) == 0:
                continue
            pick = idx if k is None else rng.choice(idx, min(k, len(idx)), replace=False)
            Xtr.append(X[tr]); ytr.append(consensus(D[tr]["mats"]["overall"][pick]))
        model = RidgeCV(alphas=ALPHAS).fit(np.concatenate(Xtr), np.concatenate(ytr))
        p = model.predict(X[te]); ps.append(p - p.mean()); ts.append(consensus(Yte))
        comps.append((*variance_components(Yte), len(Yte)))
    return stats.spearmanr(np.concatenate(ps), np.concatenate(ts))[0] / np.sqrt(_reliability(comps))


def pilot_subset(D, sets, n, rng):
    """A simulated pilot: the given sets, with n raters per group in each set drawn without replacement."""
    out = {}
    for s in sets:
        keep = np.zeros(len(D[s]["person"]), bool)
        for jp in (True, False):
            idx = np.where(D[s]["jp"] == jp)[0]
            keep[rng.choice(idx, min(n, len(idx)), replace=False)] = True
        out[s] = {**D[s], "mats": {c: v[keep] for c, v in D[s]["mats"].items()}, "person": D[s]["person"][keep], "jp": D[s]["jp"][keep]}
    return out


MATCHED_SETS = [s for s in SETS if s != "S3"]   # S3 has too few raters for the matched design
SPEC = ["spec_" + c for c in ELEMENTS]


def element_specific_agreement(D, sets, jp):
    """Mean over the six elements of ICC(C,1) of the element-specific part."""
    return float(np.mean([icc(D, sets, jp, c) for c in SPEC]))


def target_reliability(D, sets, jp, cols, m):
    """Reliability of the consensus of m raters, with variance components averaged over sets and columns."""
    c = np.array([variance_components(D[s]["mats"][col][D[s]["jp"] == jp]) for s in sets for col in cols])
    return c[:, 0].mean() / (c[:, 0].mean() + c[:, 1].mean() / m)


def matched_draw(D, X, jp, k, rng, cols=("overall",), sets=None, m=3, imgs=None):
    """One draw of the matched design. For each held-out set the target is the consensus of m randomly drawn raters;
    training labels are the consensus of k other raters per training set, and the m target persons are never used
    for training. Returns (label_r, model_r): the Pearson correlation between the k-rater and the m-rater consensus
    of the same set, and the rank correlation between the predictions and the m-rater consensus, each pooled over
    the held-out sets and the columns. X: {set_id: features}; imgs: optional {set_id: image indices} (resampling)."""
    sets = MATCHED_SETS if sets is None else sets
    la, lb, ps, ts = [], [], [], []
    for te in sets:
        idx = np.where(D[te]["jp"] == jp)[0]
        if len(idx) < m + 1:
            return np.nan, np.nan
        perm = rng.permutation(idx); tgt, oth = perm[:m], perm[m:m + k]
        ci = np.arange(X[te].shape[0]) if imgs is None else imgs[te]
        excl = list(D[te]["person"][tgt])
        Xtr, Ytr = [], []
        for tr in sets:
            if tr == te:
                continue
            pool = np.where((D[tr]["jp"] == jp) & ~np.isin(D[tr]["person"], excl))[0]
            if len(pool) == 0:
                continue
            pick = rng.choice(pool, min(k, len(pool)), replace=False)
            cj = np.arange(X[tr].shape[0]) if imgs is None else imgs[tr]
            Xtr.append(X[tr][cj]); Ytr.append(np.stack([consensus(D[tr]["mats"][c][pick][:, cj]) for c in cols], 1))
        if not Xtr:
            return np.nan, np.nan
        P = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(np.concatenate(Xtr), np.concatenate(Ytr)).predict(X[te][ci])
        P = P.reshape(len(ci), -1)
        for j, c in enumerate(cols):
            Y = D[te]["mats"][c][:, ci]
            ct = consensus(Y[tgt]); la.append(consensus(Y[oth])); lb.append(ct)
            ps.append(P[:, j] - P[:, j].mean()); ts.append(ct)
    with np.errstate(all="ignore"):
        return np.corrcoef(np.concatenate(la), np.concatenate(lb))[0, 1], stats.spearmanr(np.concatenate(ps), np.concatenate(ts))[0]


def rater_element_correlation(D):
    """H of each rater in each set: mean correlation between the six level-removed element ratings over the set's images.
    Returns {(person, condition): value} with condition "blue" or "general"."""
    out = {}
    for s in SETS:
        X = np.stack([D[s]["mats"][c] for c in ELEMENTS], -1).astype(float)
        for r, person in enumerate(D[s]["person"]):
            x = X[r] - X[r].mean(0, keepdims=True)
            with np.errstate(all="ignore"):
                out[(person, "blue" if s in BLUE else "general")] = np.nanmean(np.corrcoef(x.T)[np.triu_indices(6, 1)])
    return out
