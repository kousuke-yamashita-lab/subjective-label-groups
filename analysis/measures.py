"""Data loading and the statistics reported in the paper."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV

DATA = Path(__file__).resolve().parents[1] / "data"
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


def score_gap(D, sets=SETS):
    return np.mean([D[s]["mats"]["overall"][D[s]["jp"]].mean() - D[s]["mats"]["overall"][~D[s]["jp"]].mean() for s in sets])


def accessory_weight_gap(D, sets=SETS):
    """Difference in the standardised ridge weight of the accessory element on the overall consensus."""
    def cons(jp):
        return {c: np.concatenate([consensus(D[s]["mats"][c][D[s]["jp"] == jp].astype(float)) for s in sets]) for c in ["overall"] + ELEMENTS}
    z = lambda a: (a - a.mean()) / a.std()
    a, b = cons(True), cons(False)
    X = np.column_stack([(z(a[k]) + z(b[k])) / 2 for k in ELEMENTS]); X = (X - X.mean(0)) / X.std(0)
    w = lambda y: np.linalg.solve(X.T @ X + 5 * np.eye(6), X.T @ z(y))
    return (w(a["overall"]) - w(b["overall"]))[ELEMENTS.index("accessory")]


def load_training(feature):
    t = pd.read_csv(DATA / "train_ratings.csv")
    z = np.load(DATA / f"features_{feature}.npz", allow_pickle=True)
    return t, z


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


class LeaveOneSetOutRidge:
    """Fast ridge that predicts each set from the other five (same criterion as RidgeCV)."""

    def __init__(self, X):
        self.X = {s: np.asarray(v, dtype=np.float64) for s, v in X.items()}; self.pre = {}
        for s in SETS:
            Xt = np.concatenate([self.X[t] for t in SETS if t != s]); mu = Xt.mean(0); Xc = Xt - mu
            v, Q = np.linalg.eigh(Xc @ Xc.T)
            self.pre[s] = (Xc, mu, v, Q, int(np.argmax(np.abs(Q.T @ np.ones(len(Q))))))

    def predict(self, teach):
        out = {}
        for s in SETS:
            Xc, mu, v, Q, dim0 = self.pre[s]
            y = np.concatenate([teach[t] for t in SETS if t != s]); ym = y.mean()
            Qy = Q.T @ (y - ym); best, bc = np.inf, None
            for a in ALPHAS:
                w = 1 / (v + a); w[dim0] = 0
                c = Q @ (w * Qy); g = (Q ** 2) @ w
                e = ((c / g) ** 2).mean()
                if e < best:
                    best, bc = e, c
            p = (self.X[s] - mu) @ (Xc.T @ bc) + ym
            out[s] = p - p.mean()
        return out


def same_group_vs_transfer(D, fast):
    """(fit of own-group models JP - VN, JP model on JP - JP model on VN), both by leave-one-set-out."""
    def teach(jp):
        out = {}
        for s in SETS:
            Y = D[s]["mats"]["overall"][D[s]["jp"] == jp]
            if len(Y) < 1:
                return None
            out[s] = consensus(Y)
        return out
    tj, tv = teach(True), teach(False)
    if tj is None or tv is None:
        return np.nan, np.nan
    pj, pv = fast.predict(tj), fast.predict(tv)

    def score(pred, jp):
        ts, ps, comps = [], [], []
        for s in SETS:
            Y = D[s]["mats"]["overall"][D[s]["jp"] == jp]
            if len(Y) < 2:
                return np.nan
            ts.append(consensus(Y)); ps.append(pred[s] - pred[s].mean()); comps.append((*variance_components(Y), len(Y)))
        R = _reliability(comps)
        return stats.spearmanr(np.concatenate(ps), np.concatenate(ts))[0] / np.sqrt(R) if R > 0 else np.nan
    jj = score(pj, True)
    return jj - score(pv, False), jj - score(pj, False)
