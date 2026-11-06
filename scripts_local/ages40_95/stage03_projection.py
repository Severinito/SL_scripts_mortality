# =====================================================================
# 03 - PROJECTION: RICCATI RATES, SIMULATED FANS, BACKTEST, SURFACES
#
#   mu1_hat(x,t) = m_x(t) mu_1(x,t),   mu2_hat(x,t) = m_x(t) mu_2(x,t)
#
#   Both are affine in the lifted state xi = (Y; Z; 1) with mu_2
#   appended, so for a coefficient k on the source
#
#       E[exp(-k int_t^T mu_hat ds) | xi_t] = exp(beta_k(tau) . xi_t),
#
#   with beta = (psi; psi_2) solving, from beta(0) = 0,
#
#     d psi  /d tau = A' psi + [1/2 sig_1^2 (v . psi)^2 + delta psi_2] h,
#     d psi_2/d tau = 1/2 sig_2^2 psi_2^2 - gamma psi_2 - k m(T - tau),
#
#   the national case dropping psi_2 and loading -k m on h directly.
#
#   The origin is the training end: tau = 0 there, horizon n is calendar
#   end + n, and every solve and every path starts at the filtered state
#   xi_end, so the dispersion shown is future diffusion alone.
#
#   Backtest: parameters trained to 2003, scored on 2004-2018 against
#   the national rates, since the observed series is national.  Bands
#   there come from simulation, and the central intervals in PI_LEVELS
#   are saved per age and year so the figures can be redrawn without
#   simulating again.
#
#   Projection: parameters trained to 2023, carried to 2073 by Riccati
#   alone.  No paths are simulated over that horizon: the projected
#   rates are expectations, and the dispersion exhibit is the backtest.
#
#   sigma_2 = sigma_1: both deviations are dimensionless and live on the
#   same scale, so the portfolio diffusion is read from the national fit.
# =====================================================================
import os
import time
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.special import gamma as GAMMA
import matplotlib.pyplot as plt
import argparse

from numba import njit


# ============================== CONFIG ===============================
DATA_DIR  = r"C:\Users\sever\Documents\Tu Delft\Thesis\Italy_mortality\data"
OUT_DIR   = r"C:\Users\sever\Documents\Tu Delft\Thesis\Italy_mortality\outputs\ages40_95"
PARAM_DIR = os.path.join(OUT_DIR, "params")

FILES   = dict(mx="Mx_1x1_{c}.txt",
               deaths="Deaths_1x1_{c}.txt",
               exposure="Exposures_1x1_{c}.txt")

SEXES          = ["Male", "Female"]
AGE_LO, AGE_HI = 40, 95
FIT_START      = 1970
T0             = FIT_START

p = argparse.ArgumentParser(description="projection, backtest and surfaces")
p.add_argument("-c", "--country", default="IT")
p.add_argument("-e", "--end", type=int, default=2023,
               help="projection origin, the training end of the table")
p.add_argument("--no-lrd", dest="lrd", action="store_false",
               help="read the params_..._nolrd tables")
args, _ = p.parse_known_args()

BT_END,   BT_H   = 2003, 15        # backtest origin, scored 2004-2018
PROJ_END, PROJ_H = args.end, 50    # projection origin

COUNTRY = args.country
LRD     = args.lrd
SUFFIX  = "_lrd" if LRD else "_nolrd"

THETA      = 1.0
MIN_DEATHS = 0.5

N_NODES    = 96
ETA0, ETAN = 1e-6, 1e6
DT         = 1.0
N_GL       = 12
P0_STEPS   = 1.0

RTOL, ATOL = 1e-6, 1e-9
N_PATHS    = 20000
MC_KSUB    = 16
MC_SEED    = 42
PI_LEVELS  = [0.95, 0.99, 0.995]   # central intervals kept from the paths
PI_MAIN    = 0.995                 # the one drawn and scored for coverage

BT_AGES   = list(range(40, 96))
PLOT_AGES = [45, 55, 65, 75, 85, 95]
SURF_AGES = list(range(40, 101, 2))
HIST_FROM = FIT_START

SEX_COL = {"Female": 0, "Male": 1}
SEXCOL  = {"Male": "#1f77b4", "Female": "#ff7f0e"}
AGES    = np.arange(AGE_LO, AGE_HI + 1)
NA      = len(AGES)

GLX, GLW = np.polynomial.legendre.leggauss(N_GL)
GLV      = 0.5 * (GLX + 1.0)
GLW2     = np.ascontiguousarray(0.5 * GLW)


def pi_tag(p):
    """Column suffix for a central interval, 0.995 -> '995'."""
    return f"{p:.3f}".rstrip("0").rstrip(".").replace("0.", "")


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
    """Rates, deaths and exposure on the (age, year) grid, per sex."""
    src = {k: load_hmd(os.path.join(DATA_DIR, f.format(c=country)))
           for k, f in FILES.items()}
    out = {}
    for sex in SEXES:
        MX = build_matrix(src["mx"], sex, years, ages)
        DD = build_matrix(src["deaths"], sex, years, ages)
        EE = build_matrix(src["exposure"], sex, years, ages)
        out[sex] = dict(MX=MX, DD=DD, EE=EE)
    return out


def load_params(country, sex, end):
    """Parameter table for one training end, ordered by age."""
    path = os.path.join(PARAM_DIR, f"params_{country}_{sex}_{end}{SUFFIX}.csv")
    return pd.read_csv(path).sort_values("age").reset_index(drop=True)


def deviation_path(MX, DD, EE, a, b, u):
    """Observed deviation mu_1 = M / m, NaN where the cell is unusable."""
    with np.errstate(divide="ignore", invalid="ignore"):
        mu1 = MX / np.exp(a[:, None] + b[:, None] * u[None, :])
    bad = ~(np.isfinite(MX) & (MX > 0) & np.isfinite(DD)
            & (DD >= MIN_DEATHS) & np.isfinite(EE) & (EE > 0))
    mu1[bad] = np.nan
    return mu1


# ========================== LIFT AND FILTER ==========================
def lift_nodes(H, N=N_NODES, eta0=ETA0, etaN=ETAN):
    """Quantise the kernel measure, mass-weighted nodes."""
    CG  = GAMMA(H + 0.5) * GAMMA(1.5 - H)
    eta = eta0 * (etaN / eta0) ** (np.arange(N + 1) / N)
    a, b = eta[:-1], eta[1:]
    w = (b ** (1.5 - H) - a ** (1.5 - H)) / ((1.5 - H) * CG)
    x = ((b ** (2.5 - H) - a ** (2.5 - H)) / ((2.5 - H) * CG)) / w
    return x, w


def build_system(x, w, lam, mu0, theta=THETA, dt=DT):
    """Affine system for the lifted deviation, state (Y; Z; 1).

    The level enters through the last entry of h, so mu_1 = mu_1(0) +
    w' Z and the state starts empty at (0, ..., 0, 1).
    """
    n  = x.size
    ns = 2 * n + 1
    A  = np.zeros((ns, ns))
    A[:n, :n]           = -np.diag(x)
    A[:n, n:2 * n]      = -lam * np.outer(np.ones(n), w)
    A[:n, -1]           = lam * (theta - mu0)
    A[n:2 * n, :n]      = np.eye(n)
    A[n:2 * n, n:2 * n] = -np.diag(x)
    h = np.concatenate([np.zeros(n), w, [mu0]])
    v = np.concatenate([np.ones(n), np.zeros(n + 1)])

    Phi = expm(A * dt)
    Mq  = np.zeros((N_GL, ns))
    hE  = np.zeros((N_GL, ns))
    for j, vv in enumerate(GLV):
        Ev    = expm(A * (vv * dt))
        Mq[j] = Ev @ v
        hE[j] = Ev.T @ h
    hE = hE[::-1].copy()
    return (np.ascontiguousarray(Phi), np.ascontiguousarray(h), A, v,
            np.ascontiguousarray(Mq), np.ascontiguousarray(hE))


def initial_state(sigma, sys):
    """State starts at mu_1(0) with empty memory; P0 is one step's noise."""
    Phi, h, A, v, Mq, hE = sys
    ns = Phi.shape[0]
    s0 = np.zeros(ns)
    s0[-1] = 1.0
    sig2 = float(sigma) ** 2
    P0 = np.zeros((ns, ns))
    for g in range(N_GL):
        P0 += (P0_STEPS * sig2 * GLW2[g]) * np.outer(Mq[g], Mq[g])
    return s0, P0, sig2


@njit(cache=True, fastmath=True)
def kf_state(y, mm, R, obs, h, Phi, Mq, hE, wq, sig2, s0, P0, St):
    """Kalman recursion, recording the filtered state each year."""
    T  = y.shape[0]
    ns = Phi.shape[0]
    ng = Mq.shape[0]
    s  = s0.copy()
    P  = P0.copy()
    Mw = np.empty((ng, ns))
    for t in range(T):
        if t > 0:
            z = hE @ s
            for g in range(ng):
                zz = z[g]
                if zz < 1e-10:
                    zz = 1e-10
                cg = np.sqrt(sig2 * wq[g] * zz)
                for i in range(ns):
                    Mw[g, i] = cg * Mq[g, i]
            P = (Phi @ P) @ Phi.T + Mw.T @ Mw
            s = Phi @ s
        if obs[t]:
            L  = mm[t]
            PH = (P @ h) * L
            S  = R[t] + L * (h @ PH)
            if S > 0.0 and np.isfinite(S):
                r = y[t] - L * (h @ s)
                s = s + PH * (r / S)
                P = P - np.outer(PH, PH) / S
                P = 0.5 * (P + P.T)
        for j in range(ns):
            St[t, j] = s[j]


def build_bundles(country, sex, end):
    """One bundle per age: trend, lift arrays, parameters, origin state.

    The parameter table stores parameters only, so the filtered state at
    the training end is recovered by re-running the filter on the window
    that ends there.  mu_2 is placed at its equilibrium (delta/gamma)
    mu_1, matching the anchor under which delta and gamma were fitted.
    """
    df = load_params(country, sex, end)
    a_x, b_x = df["a"].to_numpy(), df["b"].to_numpy()
    H_x, lam = df["H"].to_numpy(), df["lam"].to_numpy()
    sig      = df["sigma"].to_numpy()
    dlt, gmm = df["delta"].to_numpy(), df["gamma"].to_numpy()

    years = np.arange(FIT_START, end + 1)
    u     = (years - T0).astype(float)
    d     = load_country(country, years, AGES)[sex]
    mu1   = deviation_path(d["MX"], d["DD"], d["EE"], a_x, b_x, u)
    m     = np.exp(a_x[:, None] + b_x[:, None] * u[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(np.isfinite(d["DD"]) & (d["DD"] > 0),
                     d["MX"] ** 2 / np.maximum(d["DD"], MIN_DEATHS), 1.0)

    lifts, bundles = {}, {}
    for i, age in enumerate(AGES):
        age = int(age)
        obs = np.isfinite(mu1[i])
        key = round(float(H_x[i]), 6)
        if key not in lifts:
            lifts[key] = lift_nodes(key)
        x, w = lifts[key]
        sys = build_system(x, w, float(lam[i]), float(mu1[i][obs][0]))
        Phi, h, A, v, Mq, hE = sys
        s0, P0, sig2 = initial_state(float(sig[i]), sys)
        St = np.zeros((len(years), Phi.shape[0]))
        kf_state(np.ascontiguousarray(np.nan_to_num(d["MX"][i], nan=1.0)),
                 np.ascontiguousarray(m[i]),
                 np.ascontiguousarray(np.nan_to_num(R[i], nan=1.0)),
                 np.ascontiguousarray(obs), h, Phi, Mq, hE, GLW2, sig2,
                 np.ascontiguousarray(s0), np.ascontiguousarray(P0), St)

        s_end = St[-1].copy()
        z1    = float(h @ s_end)
        bundles[age] = dict(age=age, end=int(end),
                            a=float(a_x[i]), b=float(b_x[i]), T0=T0,
                            A=np.ascontiguousarray(A),
                            h=np.ascontiguousarray(h),
                            v=np.ascontiguousarray(v),
                            sig1_2=sig2, sig2_2=sig2,
                            H=float(H_x[i]), lam=float(lam[i]),
                            delta=float(dlt[i]), gamma=float(gmm[i]),
                            s_nat=s_end, z1=z1,
                            mu2=float(dlt[i]) / float(gmm[i]) * z1)
    return bundles


# =========================== RICCATI SYSTEM ==========================
def m_age(B, year):
    """Deterministic trend of one age at a calendar year."""
    return float(np.exp(B["a"] + B["b"] * (float(year) - B["T0"])))


def riccati_rhs(tau, beta, B, t_ref, T, k, national):
    """Derivative of the backward system at one point in tau.

    tau runs backward from beta(0) = 0 at the horizon T, so the trend
    inside the source is evaluated at calendar t_ref + T - tau.  The
    quadratic term uses the noise loading v of the lift, and the
    coupling delta psi_2 loads on h, whose last slot is frozen at one.
    """
    A, h, v = B["A"], B["h"], B["v"]
    ns   = A.shape[0]
    psi  = beta[:ns]
    m_n  = m_age(B, t_ref + T - tau)
    quad = 0.5 * B["sig1_2"] * float(v @ psi) ** 2
    if national:
        return A.T @ psi + (quad - k * m_n) * h
    psi2  = beta[ns]
    dpsi  = A.T @ psi + (quad + B["delta"] * psi2) * h
    dpsi2 = 0.5 * B["sig2_2"] * psi2 * psi2 - B["gamma"] * psi2 - k * m_n
    return np.concatenate([dpsi, [dpsi2]])


def riccati_survival(B, t_ref, horizons, national, k=1.0):
    """E[exp(-k int mu_hat ds)] at every horizon, one integration each.

    The source m(T - tau) depends on the horizon T, so each horizon
    needs its own backward solve.
    """
    ns  = B["A"].shape[0]
    dim = ns if national else ns + 1
    hz  = np.asarray(sorted(horizons), float)
    out = np.empty(hz.size)
    for j, T in enumerate(hz):
        sol = solve_ivp(riccati_rhs, (0.0, float(T)), np.zeros(dim),
                        args=(B, t_ref, float(T), k, national),
                        method="Radau", rtol=RTOL, atol=ATOL)
        beta = sol.y[:, -1]
        expo = float(beta[:ns] @ B["s_nat"])
        if not national:
            expo += beta[ns] * B["mu2"]
        out[j] = np.exp(expo)
    return out


def annual_q(Ep):
    """One-year death probabilities from consecutive survival factors."""
    prev = np.concatenate([[1.0], Ep[:-1]])
    return np.clip(1.0 - Ep / prev, 0.0, 1.0)


# ======================= MONTE CARLO SIMULATION ======================
def simulate(B, n_years, n_paths=N_PATHS, ksub=MC_KSUB, seed=MC_SEED):
    """Simulate survival factors for both intensities on one scenario set.

        xi   <- e^{A dt} xi + sig_1 sqrt(max(mu_1, 0)) v dW,
        mu_2 <- mu_2 + (delta mu_1 - gamma mu_2) dt
                     + sig_2 sqrt(max(mu_2, 0)) dW~,

    with mu_1 = h . xi and the two noises independent.  The square roots
    are truncated at zero so the scheme stays defined at the boundary.
    Hazards accumulate as int m(t) mu dt with the trend at the substep
    midpoint, and both factors are recorded at the year marks, so the
    national and portfolio bases come from the same paths.
    """
    rng  = np.random.default_rng(seed)
    h, v = B["h"], B["v"]
    s1, s2 = np.sqrt(B["sig1_2"]), np.sqrt(B["sig2_2"])
    d, g   = B["delta"], B["gamma"]

    dt  = 1.0 / ksub
    sq  = np.sqrt(dt)
    PhT = np.ascontiguousarray(expm(B["A"] * dt).T)

    xi  = np.tile(B["s_nat"], (n_paths, 1))
    mu2 = np.full(n_paths, B["mu2"])
    h1  = np.zeros(n_paths)
    h2  = np.zeros(n_paths)
    p1  = np.empty((n_paths, n_years))
    p2  = np.empty((n_paths, n_years))

    for yr in range(n_years):
        for st in range(ksub):
            mu1 = xi @ h
            mt  = m_age(B, B["end"] + yr + (st + 0.5) * dt)
            h1 += mt * np.maximum(mu1, 0.0) * dt
            h2 += mt * np.maximum(mu2, 0.0) * dt
            dW  = rng.standard_normal(n_paths) * sq
            dWp = rng.standard_normal(n_paths) * sq
            xi  = xi @ PhT + (s1 * np.sqrt(np.maximum(mu1, 0.0))
                              * dW)[:, None] * v[None, :]
            mu2 = (mu2 + (d * mu1 - g * mu2) * dt
                   + s2 * np.sqrt(np.maximum(mu2, 0.0)) * dWp)
        p1[:, yr] = np.exp(-h1)
        p2[:, yr] = np.exp(-h2)
    return p1, p2


def path_q(paths):
    """Per-path one-year death probabilities from survival factors.

    The hazards are non-decreasing, so q lies in [0, 1] without clipping.
    """
    prev = np.concatenate([np.ones((paths.shape[0], 1)), paths[:, :-1]],
                          axis=1)
    return 1.0 - paths / prev


# ================================ RUN ================================
BT_YEARS   = [BT_END + n for n in range(1, BT_H + 1)]
PROJ_YEARS = [PROJ_END + n for n in range(1, PROJ_H + 1)]

obs_years = np.arange(HIST_FROM, max(BT_YEARS[-1], PROJ_END) + 1)
OBS = load_country(COUNTRY, obs_years, AGES)
oi  = {int(y): j for j, y in enumerate(obs_years)}


def observed_q(sex, age, years):
    """Observed one-year death probability, q = 1 - exp(-m)."""
    return np.array([1.0 - np.exp(-OBS[sex]["MX"][age - AGE_LO, oi[y]])
                     for y in years])


# wall-clock per component, summed over ages and sexes
TIMES = {}


def tick(key, t_start, n=1):
    """Add the time since t_start to TIMES[key], with n solves or runs."""
    dt = time.perf_counter() - t_start
    s, c = TIMES.get(key, (0.0, 0))
    TIMES[key] = (s + dt, c + n)


# ---------------------- part A: backtest 2004-2018 -------------------
BT, ERR = {}, []
for sex in SEXES:
    t0  = time.time()
    tp  = time.perf_counter()
    bnd = build_bundles(COUNTRY, sex, BT_END)
    tick("bundles (filter)", tp)
    ages = [a for a in BT_AGES if a in bnd]
    print(f"{sex}: {len(ages)} ages, bundles in {time.time() - t0:.0f}s",
          flush=True)

    q1  = np.full((len(ages), BT_H), np.nan)
    q2  = np.full((len(ages), BT_H), np.nan)
    gap = np.full((len(ages), BT_H), np.nan)
    qlo = {p: np.full((len(ages), BT_H), np.nan) for p in PI_LEVELS}
    qhi = {p: np.full((len(ages), BT_H), np.nan) for p in PI_LEVELS}

    t0 = time.time()
    hz = list(range(1, BT_H + 1))
    for i, age in enumerate(ages):
        B = bnd[age]
        tp    = time.perf_counter()
        Ep1   = riccati_survival(B, BT_END, hz, True)
        tick("riccati backtest national", tp)
        tp    = time.perf_counter()
        q1[i] = annual_q(Ep1)
        q2[i] = annual_q(riccati_survival(B, BT_END, hz, False))
        tick("riccati backtest portfolio", tp)
        tp    = time.perf_counter()
        s1, s2 = simulate(B, BT_H)
        tick("MC backtest", tp)
        qs = path_q(s1)
        for pl in PI_LEVELS:
            a = 0.5 * (1.0 - pl)
            qlo[pl][i], qhi[pl][i] = np.quantile(qs, [a, 1.0 - a], axis=0)
        gap[i] = s1.mean(axis=0) / Ep1 - 1.0
        if (i + 1) % 10 == 0:
            print(f"  {sex} {i + 1}/{len(ages)} ages, "
                  f"{time.time() - t0:.0f}s", flush=True)

    obs = np.array([observed_q(sex, a, BT_YEARS) for a in ages])
    ok  = np.isfinite(obs) & (obs > 0)
    BT[sex] = dict(ages=ages, q1=q1, q2=q2, qlo=qlo, qhi=qhi, gap=gap,
                   obs=obs, ok=ok)

    e = q1[ok] - obs[ok]
    row = dict(sex=sex, basis="mu1",
               rmse=float(np.sqrt(np.mean(e ** 2))),
               mape=float(np.mean(np.abs(e / obs[ok]))) * 100.0,
               bias=float(np.mean(e / obs[ok])) * 100.0,
               ratio_mu2_mu1=float(np.mean(q2[ok] / q1[ok])))
    for pl in PI_LEVELS:
        row[f"cover{pi_tag(pl)}"] = 100.0 * (
            ((obs >= qlo[pl]) & (obs <= qhi[pl])) & ok).sum() / ok.sum()
    ERR.append(row)
    print(f"{sex}: MC against Riccati, mean |gap| "
          f"{np.nanmean(np.abs(gap)) * 100:.3f}%, max "
          f"{np.nanmax(np.abs(gap)) * 100:.3f}%", flush=True)

err = pd.DataFrame(ERR)
print(err.round(4).to_string(index=False))
err.to_csv(os.path.join(
    OUT_DIR, f"backtest_error_{COUNTRY}_{BT_END}{SUFFIX}.csv"), index=False)

rows = []
for sex in SEXES:
    R = BT[sex]
    for i, a in enumerate(R["ages"]):
        for j, y in enumerate(BT_YEARS):
            r = dict(sex=sex, age=int(a), year=int(y),
                     observed=R["obs"][i, j],
                     q_national=R["q1"][i, j],
                     q_portfolio=R["q2"][i, j])
            for pl in PI_LEVELS:
                r[f"lo{pi_tag(pl)}"] = R["qlo"][pl][i, j]
                r[f"hi{pi_tag(pl)}"] = R["qhi"][pl][i, j]
            rows.append(r)
pd.DataFrame(rows).to_csv(os.path.join(
    OUT_DIR, f"backtest_qx_{COUNTRY}_{BT_END}{SUFFIX}.csv"), index=False)

HIST_BT = list(range(HIST_FROM, BT_END + 1))
nc = 3
nr = int(np.ceil(len(PLOT_AGES) / nc))
fig, axs = plt.subplots(nr, nc, figsize=(5.4 * nc, 3.9 * nr),
                        squeeze=False)
for p, age in enumerate(PLOT_AGES):
    ax = axs[p // nc, p % nc]
    for sex in SEXES:
        R = BT[sex]
        i = R["ages"].index(age)
        c = SEXCOL[sex]
        ax.plot(HIST_BT, observed_q(sex, age, HIST_BT), "-", lw=1.1,
                color=c, label=sex)
        ax.plot(BT_YEARS, R["obs"][i], "-", lw=1.1, color=c)
        ax.fill_between(BT_YEARS, R["qlo"][PI_MAIN][i],
                        R["qhi"][PI_MAIN][i], color=c, alpha=0.20, lw=0)
        ax.plot(BT_YEARS, R["q1"][i], "--", lw=1.6, color=c)
    ax.axvspan(BT_END + 0.5, BT_YEARS[-1], color="0.5", alpha=0.10, lw=0)
    ax.axvline(BT_END + 0.5, color="k", lw=0.8)
    ax.set_xlim(HIST_FROM, BT_YEARS[-1])
    ax.set_ylim(bottom=0.0)
    ax.set_title(f"Age {age}", fontweight="bold")
    ax.set_xlabel("calendar year")
    ax.set_ylabel(r"one-year death probability $q_x$")
    ax.grid(alpha=0.3)
    if p == 0:
        ax.legend(fontsize=8)
handles = [plt.Line2D([], [], color="0.3", lw=1.1, label="observed"),
           plt.Line2D([], [], color="0.3", lw=1.6, ls="--",
                      label="projected national"),
           plt.Rectangle((0, 0), 1, 1, color="0.4", alpha=0.20,
                         label=f"{PI_MAIN:.1%} interval")]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8)
fig.suptitle(f"Backtest of the national $q_x$: training {HIST_FROM}-"
             f"{BT_END}, projected {BT_YEARS[0]}-{BT_YEARS[-1]}",
             fontweight="bold")
fig.tight_layout(rect=(0, 0.045, 1, 1))
plt.show()

# -------------------- part B: projection to 2073 ---------------------
SURF = {}
for sex in SEXES:
    tp   = time.perf_counter()
    bnd  = build_bundles(COUNTRY, sex, PROJ_END)
    tick("bundles (filter)", tp)
    ages = sorted(set(SURF_AGES + PLOT_AGES) & set(bnd))
    s1 = np.full((len(ages), PROJ_H), np.nan)
    s2 = np.full((len(ages), PROJ_H), np.nan)
    t0 = time.time()
    hz = list(range(1, PROJ_H + 1))
    for i, age in enumerate(ages):
        B = bnd[age]
        tp    = time.perf_counter()
        s1[i] = annual_q(riccati_survival(B, PROJ_END, hz, True))
        tick("riccati projection national", tp)
        tp    = time.perf_counter()
        s2[i] = annual_q(riccati_survival(B, PROJ_END, hz, False))
        tick("riccati projection portfolio", tp)
        if (i + 1) % 10 == 0:
            print(f"  {sex} surface {i + 1}/{len(ages)}, "
                  f"{time.time() - t0:.0f}s", flush=True)
    SURF[sex] = dict(ages=ages, q1=s1, q2=s2,
                     H=np.array([bnd[a]["H"] for a in ages]))

pd.DataFrame([dict(sex=s, age=a, year=y,
                   q_national=SURF[s]["q1"][i, j],
                   q_portfolio=SURF[s]["q2"][i, j])
              for s in SEXES for i, a in enumerate(SURF[s]["ages"])
              for j, y in enumerate(PROJ_YEARS)]
             ).to_csv(os.path.join(
                 OUT_DIR,
                 f"projection_qx_{COUNTRY}_{PROJ_END}{SUFFIX}.csv"),
                 index=False)

tim = pd.DataFrame([dict(component=k, total_s=s, n=c, per_call_ms=1e3 * s / c)
                    for k, (s, c) in TIMES.items()])
print(f"\ntiming ({COUNTRY}{SUFFIX}, MC {N_PATHS} paths x {MC_KSUB} "
      f"substeps x {BT_H} years)")
print(tim.round(3).to_string(index=False), flush=True)
tim.assign(country=COUNTRY, hurst=SUFFIX.lstrip("_"), n_paths=N_PATHS,
           ksub=MC_KSUB).to_csv(os.path.join(
               OUT_DIR, f"timing_{COUNTRY}{SUFFIX}.csv"), index=False)

HIST_PR = list(range(HIST_FROM, PROJ_END + 1))
fig, axs = plt.subplots(nr, nc, figsize=(5.4 * nc, 3.9 * nr),
                        squeeze=False)
for p, age in enumerate(PLOT_AGES):
    ax = axs[p // nc, p % nc]
    for sex in SEXES:
        R = SURF[sex]
        i = R["ages"].index(age)
        c = SEXCOL[sex]
        ax.plot(HIST_PR, observed_q(sex, age, HIST_PR), "-", lw=1.1,
                color=c, label=f"{sex} observed")
        ax.plot(PROJ_YEARS, R["q1"][i], "--", lw=1.6, color=c,
                label=f"{sex} national")
        ax.plot(PROJ_YEARS, R["q2"][i], "-.", lw=1.6, color=c,
                label=f"{sex} portfolio")
    ax.axvspan(PROJ_END + 0.5, PROJ_YEARS[-1], color="0.5", alpha=0.12,
               lw=0)
    ax.axvline(PROJ_END + 0.5, color="k", lw=0.8)
    ax.set_xlim(HIST_FROM, PROJ_YEARS[-1])
    ax.set_ylim(bottom=0.0)
    ax.set_title(f"Age {age}", fontweight="bold")
    ax.set_xlabel("calendar year")
    ax.set_ylabel(r"one-year death probability $q_x$")
    ax.grid(alpha=0.3)
    if p == 0:
        ax.legend(fontsize=7, ncol=2)
hh = np.concatenate([SURF[s]["H"] for s in SEXES])
fig.suptitle(f"One-year death probability by age: observed and "
             f"projected to {PROJ_YEARS[-1]}\nnational and portfolio at "
             f"estimated $H$ (median {np.median(hh):.2f}, range "
             f"{hh.min():.2f}-{hh.max():.2f})", fontweight="bold")
fig.tight_layout()
plt.show()

for sex in SEXES:
    R = SURF[sex]
    keep = [i for i, a in enumerate(R["ages"]) if a in SURF_AGES]
    ag   = [R["ages"][i] for i in keep]
    X, Y = np.meshgrid(PROJ_YEARS, ag)

    fig = plt.figure(figsize=(12, 4.8))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot_surface(X, Y, R["q2"][keep], cmap="viridis", linewidth=0,
                    antialiased=True, rstride=1, cstride=1)
    ax.set_title(f"Insurer portfolio - {sex}")
    ax.set_xlabel("calendar year")
    ax.set_ylabel("age")
    ax.set_zlabel(r"$q_x$")
    ax.view_init(elev=22, azim=-60)

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.plot_surface(X, Y, R["q1"][keep] - R["q2"][keep], cmap="RdBu_r",
                    linewidth=0, antialiased=True, rstride=1, cstride=1)
    ax.set_title(f"National - portfolio - {sex}")
    ax.set_xlabel("calendar year")
    ax.set_ylabel("age")
    ax.set_zlabel(r"$\Delta q_x$")
    ax.view_init(elev=22, azim=-60)

    fig.suptitle(f"Projected mortality from end-{PROJ_END} to "
                 f"{PROJ_YEARS[-1]}", fontweight="bold")
    fig.tight_layout()
    plt.show()