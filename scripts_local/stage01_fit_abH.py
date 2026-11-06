# =====================================================================
# 01 - TREND AND HURST PROFILE
#
#       mu(x,t) = m_x(t) * mu_1(x,t)
#
#   m_x(t) = exp(a_x + b_x (t - T0)), penalised Poisson MLE with
#   b_x <= B_MAX < 0, so every age improves.
#
#   H_x from the rescaled range of the stage-1 log residuals, smoothed
#   across age by a low-order polynomial.
#
#   Writes one parameter table per sex with a, b, H_raw and H filled in
#   and lam, sigma left empty for the deviation fit.
# =====================================================================
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import argparse

# ============================== CONFIG ===============================
DATA_DIR  = r"C:\Users\sever\Documents\Tu Delft\Thesis\Italy_mortality\data"
OUT_DIR   = r"C:\Users\sever\Documents\Tu Delft\Thesis\Italy_mortality\outputs"
PARAM_DIR = os.path.join(OUT_DIR, "params")

FILES   = dict(mx="Mx_1x1_{c}.txt",
               deaths="Deaths_1x1_{c}.txt",
               exposure="Exposures_1x1_{c}.txt")

SEXES          = ["Male", "Female"]
AGE_LO, AGE_HI = 30, 105
FIT_START      = 1970

p = argparse.ArgumentParser(description="trend and Hurst profile")
p.add_argument("-c", "--country", default="IT")
p.add_argument("-e", "--end", type=int, default=2023)
args, _ = p.parse_known_args()

COUNTRY = args.country
FIT_END = args.end
T0      = FIT_START

PEN_A, PEN_B = 10.0, 1000.0
B_MAX        = -1e-4

H_DEG           = 2
H_FLOOR, H_CEIL = 0.505, 0.95

MIN_DEATHS = 0.5

SEX_COL = {"Female": 0, "Male": 1}
AGES    = np.arange(AGE_LO, AGE_HI + 1)
NA      = len(AGES)

os.makedirs(PARAM_DIR, exist_ok=True)


# ============================== LOADING ==============================
def load_hmd(path):
    """Parse an HMD 1x1 file into {(year, age): (female, male, total)}."""
    val = lambda s: np.nan if s == "." else float(s)
    out = {}
    for line in open(path):
        p = line.split()
        if len(p) == 5 and p[0].isdigit():         # data rows only
            out[(int(p[0]), int(p[1].rstrip("+")))] = tuple(map(val, p[2:]))
    return out


def build_matrix(dct, sex, years, ages):
    """One sex column of a loaded file on the (age, year) grid."""
    col = SEX_COL[sex]
    M = np.full((len(ages), len(years)), np.nan)
    for j, yr in enumerate(years):
        for i, ag in enumerate(ages):
            rec = dct.get((yr, int(ag)))
            if rec is not None:
                M[i, j] = rec[col]
    return M


def load_country(country, years, ages):
    """Rates, deaths, exposure and the usable-cell mask, per sex.

    A cell is usable when rate, deaths and exposure are finite and
    positive and deaths clear MIN_DEATHS.
    """
    src = {k: load_hmd(os.path.join(DATA_DIR, f.format(c=country)))
           for k, f in FILES.items()}
    out = {}
    for sex in SEXES:
        MX = build_matrix(src["mx"], sex, years, ages)
        DD = build_matrix(src["deaths"], sex, years, ages)
        EE = build_matrix(src["exposure"], sex, years, ages)
        ok = (np.isfinite(MX) & (MX > 0) & np.isfinite(DD)
              & (DD >= MIN_DEATHS) & np.isfinite(EE) & (EE > 0))
        out[sex] = dict(MX=MX, DD=DD, EE=EE, ok=ok)
    return out


def empty_frame(country, sexes, ages, years):
    """Parameter table, one row per (sex, age), all parameters NaN."""
    rows = [dict(country=country, sex=s, age=int(a),
                 fit_start=int(years[0]), fit_end=int(years[-1]),
                 T0=int(T0), n_years=np.nan,
                 a=np.nan, b=np.nan, H_raw=np.nan, H=np.nan,
                 lam=np.nan, sigma=np.nan)
            for s in sexes for a in ages]
    return pd.DataFrame(rows)


# ================================ TREND ==============================
def diffgrad(v, w):
    """Gradient of w * sum_i (v_{i+1} - v_i)^2."""
    g = np.zeros_like(v)
    d = np.diff(v)
    g[:-1] -= 2.0 * w * d
    g[1:]  += 2.0 * w * d
    return g


def fit_trend(D, E, ok, u):
    """Penalised Poisson MLE of a_x, b_x with b_x <= B_MAX < 0.

    D ~ Poisson(E exp(a_x + b_x u)), squared first differences on a and
    on b across age.  Convex under a box constraint, analytic gradient.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        logM = np.where(ok, np.log(np.maximum(D / np.maximum(E, 1e-12),
                                              1e-12)), 0.0)
    a0 = np.zeros(NA)
    b0 = np.zeros(NA)
    for i in range(NA):
        m_ = ok[i]
        X = np.column_stack([np.ones(m_.sum()), u[m_]])
        bb, *_ = np.linalg.lstsq(X, logM[i, m_], rcond=None)
        a0[i], b0[i] = bb[0], min(float(bb[1]), B_MAX)

    def objgrad(p):
        a, b = p[:NA], p[NA:]
        eta  = np.clip(a[:, None] + b[:, None] * u[None, :], -30.0, 3.0)
        lam  = E * np.exp(eta)
        ll   = np.sum(np.where(ok, D * eta - lam, 0.0))
        res  = np.where(ok, D - lam, 0.0)
        ga   = -res.sum(axis=1) + diffgrad(a, PEN_A)
        gb   = -(res * u[None, :]).sum(axis=1) + diffgrad(b, PEN_B)
        pen  = (PEN_A * np.sum(np.diff(a) ** 2)
                + PEN_B * np.sum(np.diff(b) ** 2))
        return -ll + pen, np.concatenate([ga, gb])

    bnds = [(None, None)] * NA + [(None, B_MAX)] * NA
    r = minimize(objgrad, np.concatenate([a0, b0]), jac=True,
                 method="L-BFGS-B", bounds=bnds,
                 options=dict(maxiter=10000, maxfun=40000,
                              ftol=1e-14, gtol=1e-10))
    return r.x[:NA], r.x[NA:]


def trend_matrix(a, b, u):
    """m_x(t) on the (age, year) grid."""
    return np.exp(a[:, None] + b[:, None] * u[None, :])


# ================================ HURST ==============================
def rs_hurst(x):
    """Single-scale rescaled range, log(R/S)/log(n).

    Even-length padding and divisor-count truncation, so the value
    matches pracma::hurstexp to within 1e-9.
    """
    x = np.asarray(x, float).copy()
    N = x.size
    if N % 2 != 0:
        x = np.append(x, (x[N - 2] + x[N - 1]) / 2.0)
        N += 1

    def divisors(n, n0=50):
        lo, hi = n0, n // 2
        r = (np.arange(lo, hi + 1) if lo <= hi
             else np.arange(lo, hi - 1, -1))
        return r[n % r == 0]

    N0 = min(int(np.floor(0.99 * N)), N - 1)
    N1, dv = N0, divisors(N0)
    for i in range(N0 + 1, N + 1):
        dw = divisors(i)
        if dw.size > dv.size:
            N1, dv = i, dw

    xx = x[:N1]
    z  = np.cumsum(xx - xx.mean())
    rs = (z.max() - z.min()) / np.std(xx, ddof=1)
    return float(np.log(rs) / np.log(xx.size))


def smooth_hurst(ages, H_raw, deg=H_DEG):
    """Low-order polynomial in age through the raw estimates."""
    return np.polyval(np.polyfit(ages, H_raw, deg), ages)


def fit_hurst(MX, ok, a, b, u, ages):
    """Rescaled range per age on the stage-1 log residuals, then smooth."""
    H_raw = np.array([rs_hurst(np.log(MX[i, ok[i]]) - (a[i] + b[i] * u[ok[i]]))
                      for i in range(len(ages))])
    return H_raw, np.clip(smooth_hurst(ages, H_raw), H_FLOOR, H_CEIL)


def save_frame(df, country, sex, end):
    path = os.path.join(PARAM_DIR, f"params_{country}_{sex}_{end}.csv")
    df[df.sex == sex].to_csv(path, index=False)
    return path


# ================================ RUN ================================
years = np.arange(FIT_START, FIT_END + 1)
u     = (years - T0).astype(float)

DATA   = load_country(COUNTRY, years, AGES)
params = empty_frame(COUNTRY, SEXES, AGES, years)

for sex in SEXES:
    d = DATA[sex]
    a_x, b_x = fit_trend(np.nan_to_num(d["DD"]),
                             np.nan_to_num(d["EE"], nan=1.0), d["ok"], u)
    H_raw, H_x = fit_hurst(d["MX"], d["ok"], a_x, b_x, u, AGES)

    m = params.sex == sex
    params.loc[m, "a"]       = a_x
    params.loc[m, "b"]       = b_x
    params.loc[m, "H_raw"]   = H_raw
    params.loc[m, "H"]       = H_x
    params.loc[m, "n_years"] = d["ok"].sum(axis=1)

    print(save_frame(params, COUNTRY, sex, FIT_END))