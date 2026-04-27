import numpy as np
import pandas as pd

# ============================================================
# BASIC NORMALIZATION
# ============================================================

def zscore(series):
    return (series - series.mean())/(series.std() + 1e-12)

# ============================================================
# INVARIANT / ANOMALY SCORING
# ============================================================

def enrich_invariants(df):
    df = df.copy()

    if "sigma" not in df.columns:
        df["sigma"] = df["resolvent_peak"]/(df["mu1"] + 1e-12)

    if "chi" not in df.columns:
        df["chi"] = df["ipr"]/(df["entropy"] + 1e-12)

    if "omega" not in df.columns:
        df["omega"] = df["resolvent_peak"]*df["ipr"]

    return df

def anomaly_scores(df):
    df = enrich_invariants(df)

    score = (
        np.abs(zscore(df["sigma"])) +
        np.abs(zscore(df["chi"])) +
        np.abs(zscore(df["omega"]))
    )

    df["anomaly_score"] = score
    return df.sort_values("anomaly_score", ascending=False)

# ============================================================
# TOPOLOGICAL ISOLATION
# ============================================================

def density_outliers(df):
    X = df[["mu1","ipr","entropy","resolvent_peak","gamma"]].values

    radii = []
    for i in range(len(X)):
        d = np.linalg.norm(X - X[i], axis=1)
        d = np.sort(d)[1:6]
        radii.append(np.mean(d))

    df = df.copy()
    df["isolation_radius"] = radii
    return df.sort_values("isolation_radius", ascending=False)

# ============================================================
# STRUCTURAL DESCRIPTORS OF pi
# ============================================================

def descriptor_vector(pi):
    pi = np.array(pi, dtype=float)
    n = len(pi)
    d = np.diff(pi)

    local_max = 0
    local_min = 0

    for i in range(1, n-1):
        if pi[i] > pi[i-1] and pi[i] > pi[i+1]:
            local_max += 1
        if pi[i] < pi[i-1] and pi[i] < pi[i+1]:
            local_min += 1

    center = pi[n//2]
    edge = 0.5*(pi[0] + pi[-1])

    return {
        "mean_weight": float(np.mean(pi)),
        "std_weight": float(np.std(pi)),
        "total_variation": float(np.sum(np.abs(d))),
        "roughness": float(np.std(d)),
        "local_max_count": int(local_max),
        "local_min_count": int(local_min),
        "center_ratio": float(center/(np.mean(pi)+1e-12)),
        "edge_ratio": float(edge/(np.mean(pi)+1e-12)),
        "asymmetry": float(np.sum(np.abs(pi - pi[::-1]))/(np.sum(np.abs(pi))+1e-12)),
        "gradient_skew": float(np.mean(d)),
        "bottleneck_depth": float(np.min(pi)/(np.mean(pi)+1e-12))
    }

def descriptor_frame(df):
    rows = []
    for _, row in df.iterrows():
        rows.append(descriptor_vector(row["pi"]))
    return pd.DataFrame(rows)

# ============================================================
# CORRELATION LAW FITTING
# ============================================================

def correlation_laws(df):
    D = descriptor_frame(df)
    targets = ["anomaly_score","sigma","chi","omega","resolvent_peak"]

    out = {}

    for t in targets:
        vals = {}
        if t not in df.columns:
            continue

        for c in D.columns:
            corr = np.corrcoef(D[c], df[t])[0,1]
            if np.isnan(corr):
                corr = 0.0
            vals[c] = float(corr)

        vals = dict(sorted(vals.items(), key=lambda x: abs(x[1]), reverse=True))
        out[t] = vals

    return out

# ============================================================
# INEQUALITY WINDOW MINING
# ============================================================

def persistent_windows(df):
    D = descriptor_frame(df)
    out = {}

    for col in D.columns:
        out[col] = {
            "q10": float(D[col].quantile(0.10)),
            "q50": float(D[col].quantile(0.50)),
            "q90": float(D[col].quantile(0.90))
        }

    return out

# ============================================================
# CONJECTURE SYNTHESIS
# ============================================================

def build_conjectures(corrs, wins):
    out = []

    rp = corrs.get("resolvent_peak", {})
    an = corrs.get("anomaly_score", {})
    om = corrs.get("omega", {})

    if rp.get("bottleneck_depth",0) < -0.55:
        out.append("Persistent high-resolvent families prefer deep bottleneck suppression.")

    if rp.get("edge_ratio",0) > 0.45:
        out.append("Boundary-enhanced weight profiles correlate with transient amplification.")

    if an.get("asymmetry",0) > 0.35:
        out.append("Reflection asymmetry strongly predicts anomaly persistence.")

    if om.get("total_variation",0) > 0.40:
        out.append("Localization-amplification product rises with oscillatory variation.")

    if wins["center_ratio"]["q90"] < 0.75:
        out.append("Persistent anomaly families exhibit central weight depression.")

    if wins["bottleneck_depth"]["q90"] < 0.65:
        out.append("Extreme anomaly zones systematically avoid shallow minima.")

    return out

# ============================================================
# MASTER DISCOVERY PIPELINE
# ============================================================

def run_discovery(df, frac=0.01):
    ranked = anomaly_scores(df)

    n = max(1, int(frac*len(ranked)))
    anoms = ranked.head(n)

    topo = density_outliers(anoms)

    iso_cut = topo["isolation_radius"].quantile(0.80)
    persistent = topo[topo["isolation_radius"] >= iso_cut]

    corrs = correlation_laws(persistent)
    wins = persistent_windows(persistent)
    conjectures = build_conjectures(corrs, wins)

    return {
        "ranked": ranked,
        "persistent": persistent,
        "correlations": corrs,
        "windows": wins,
        "conjectures": conjectures
    }
