# =====================================================================
# 02 - DEVIATION AND PORTFOLIO PARAMETERS
#
#       mu(x,t) = m_x(t) * mu_1(x,t),
#       mu_2_hat(x,t) = m_x(t) * mu_2(x,t)
#
#   NATIONAL.  mu_1 is a mean-theta CIR deviation driven through the
#   fractional kernel of index H_x, in the Harms-Stefanovits lift
#
#       dY = (-X Y - lam 1 w' Z + lam (theta - mu_1(0)) 1) dt
#            + sig sqrt(mu_1) 1 dW
#       dZ = (Y - X Z) dt,      mu_1 = mu_1(0) + w' Z.
#
#   lam:  least squares on the observed path.  Taking lam out of A
#         leaves a forcing lam (theta - mu_1), so with the state run at
#         lam = 1 and no noise along the observed path,
#
#             d mu_1(t_k) = lam * d (w' Z_{t_k}) + noise,
#
#         a regression through the origin.  sigma enters only the
#         residual, so the two stages use disjoint information.
#   sigma: joint penalised QML on the Kalman filter at the fitted lam,
#         observation variance M^2 / D.
#
#   PORTFOLIO.  The book shares the trend and reverts to the national
#   deviation,
#
#       d mu_2 = (delta_x mu_1 - gamma_x mu_2) dt + sig_2 sqrt(mu_2) dW~.
#
#   Deaths are Poisson with rate E m_x(t) mu_2, so the likelihood needs
#   E[mu_2]; taking expectations annihilates the stochastic integral and
#   leaves d E[mu_2]/dt = delta mu_1 - gamma E[mu_2], exactly.  sig_2
#   therefore cannot enter this stage.  With mu_1 held across the year,
#   variation of constants gives the exact recursion
#
#       mu_2(t+1) = e^{-gamma} mu_2(t)
#                   + delta mu_1(t) (1 - e^{-gamma}) / gamma,
#
#   started at the equilibrium (delta/gamma) mu_1(0), so the likelihood
#   identifies delta/gamma while the Gaussian process priors across age
#   pin the split and carry ages without exposure.
#
#   Reads the table from script 01 and the portfolio extract, and fills
#   lam, sigma, delta, gamma and ratio.
# =====================================================================
import os
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import expm, solveh_banded
from scipy.special import gamma as GAMMA
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

p = argparse.ArgumentParser(description="deviation and portfolio parameters")
p.add_argument("-c", "--country", default="IT")
p.add_argument("-e", "--end", type=int, default=2023)
p.add_argument("--no-lrd", dest="lrd", action="store_false",
               help="force H = 1/2 and write params_..._nolrd.csv "
                    "(default writes params_..._lrd.csv)")
args, _ = p.parse_known_args()

COUNTRY = args.country
LRD     = args.lrd
FIT_END = args.end
SUFFIX  = "_lrd" if LRD else "_nolrd"

T0             = FIT_START

THETA      = 1.0
MIN_DEATHS = 0.5

N_NODES    = 96
ETA0, ETAN = 1e-6, 1e6
DT         = 1.0

RHO_OLS = 10.0                 # lam penalty, as a multiple of median(a_x)

PEN_SIG        = 100.0
SIG_LO, SIG_HI = 1e-4, 1.50
N_GL           = 12
P0_STEPS       = 1.0

MAXITER   = 800
FTOL      = 1e-12
GTOL      = 1e-7
FD_EPS    = 1e-5
HEARTBEAT = 25

# portfolio
PORT_LABEL   = "CQP_2Q26"
PORT_FILE    = r"C:\Users\sever\Documents\Tu Delft\Thesis\Italy_mortality\data\portfolio_exposure_CQP_2Q26.csv"
PORT_YEARS   = [2023, 2024, 2025]     # 2022 and 2026 are partial years
MIN_EXPOSURE = 500.0
ANCHOR_BURN  = 1

DELTA0, GAMMA0       = 1.0, 1.0
TAU_DELTA, ELL_DELTA = 0.05, 10.0
TAU_GAMMA, ELL_GAMMA = 0.05, 10.0
JITTER       = 1e-8
DELTA_BOUNDS = (1e-3, 5.0)
GAMMA_BOUNDS = (5e-2, 5.0)

SEX_COL = {"Female": 0, "Male": 1}
AGES    = np.arange(AGE_LO, AGE_HI + 1)
NA      = len(AGES)

GLX, GLW = np.polynomial.legendre.leggauss(N_GL)
GLV      = 0.5 * (GLX + 1.0)
GLW2     = np.ascontiguousarray(0.5 * GLW)
LN_SIG   = (float(np.log(SIG_LO)), float(np.log(SIG_HI)))


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
    """Table from script 01, ordered by age, aligned to AGES."""
    path = os.path.join(PARAM_DIR, f"params_{country}_{sex}_{end}.csv")
    df = pd.read_csv(path).sort_values("age").reset_index(drop=True)
    if not LRD:
        df["H"] = 0.5
    return df, df["a"].to_numpy(), df["b"].to_numpy(), df["H"].to_numpy()


def load_extract(path, years, min_exposure):
    """Portfolio cells from the extract, per sex.

    Keeps the requested calendar years and cells clearing the exposure
    floor, so a handful of person-years cannot drive an age.
    """
    df = pd.read_csv(path)
    df = df[df.year.isin(years) & (df.exposure_years >= min_exposure)]
    obs = {s: {} for s in SEXES}
    for r in df.itertuples(index=False):
        obs[r.sex][(int(r.age), int(r.year))] = (float(r.deaths),
                                                 float(r.exposure_years))
    return obs


def deviation_path(MX, DD, EE, a, b, u):
    """Observed deviation mu_1 = M / m, NaN where the cell is unusable.

    A cell is unusable when the rate, deaths or exposure are absent or
    non-positive, or deaths fall below MIN_DEATHS.  Downstream code reads
    the NaNs directly; there is no separate mask.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        mu1 = MX / np.exp(a[:, None] + b[:, None] * u[None, :])
    bad = ~(np.isfinite(MX) & (MX > 0) & np.isfinite(DD)
            & (DD >= MIN_DEATHS) & np.isfinite(EE) & (EE > 0))
    mu1[bad] = np.nan
    return mu1


# =============================== LIFT ================================
def lift_nodes(H, N=N_NODES, eta0=ETA0, etaN=ETAN):
    """Quantise the kernel measure, mass-weighted nodes.

    Cells of a geometric partition of [eta0, etaN] carry the mass of
    nu(dx) = x^{-(H-1/2)} / (Gamma(H+1/2) Gamma(3/2-H)) dx, each placed
    at its own barycentre, so that K(t) = sum_i w_i t exp(-eta_i t).
    """
    CG  = GAMMA(H + 0.5) * GAMMA(1.5 - H)
    eta = eta0 * (etaN / eta0) ** (np.arange(N + 1) / N)
    a, b = eta[:-1], eta[1:]
    w = (b ** (1.5 - H) - a ** (1.5 - H)) / ((1.5 - H) * CG)
    x = ((b ** (2.5 - H) - a ** (2.5 - H)) / ((2.5 - H) * CG)) / w
    return x, w


def step_coef(x, dt=DT):
    """Zero-order-hold coefficients of the lam-free block system."""
    e  = np.exp(-x * dt)
    p  = np.where(x > 0.0, x, 1.0)
    d1 = np.where(x > 0.0, (1.0 - e) / p, dt)
    d2 = np.where(x > 0.0, (1.0 - (1.0 + x * dt) * e) / p ** 2, 0.5 * dt ** 2)
    f  = np.where(x > 0.0, dt * e, dt)
    return e, d1, d2, f


# ========================= LAMBDA BY LEAST SQUARES ====================
def ols_sums(mu1, x, w, theta=THETA):
    """Regression sums for one age.

    Runs the state at lam = 1 with no noise along the observed path,

        Y_{k+1} = e Y_k + d1 (theta - mu_1(t_k)),
        Z_{k+1} = f Y_k + e Z_k + d2 (theta - mu_1(t_k)),
        G_k     = w' Z_k,

    so mu_1(t_k) = mu_1(0) + lam G_k + noise.  Differencing removes
    mu_1(0), leaving d mu_1 = lam dG + noise, with a = sum dG^2 and
    b = sum dG dmu_1.  Years without an observation force the state with
    zero and are dropped from the sums.
    """
    e, d1, d2, f = step_coef(x)
    obs = np.isfinite(mu1)
    Y = np.zeros_like(x)
    Z = np.zeros_like(x)
    G = np.zeros(mu1.size)
    for k in range(mu1.size - 1):
        gap  = theta - mu1[k] if obs[k] else 0.0
        Y, Z = e * Y + d1 * gap, f * Y + e * Z + d2 * gap
        G[k + 1] = w @ Z
    dG, dm = np.diff(G), np.diff(mu1)
    keep = np.isfinite(dm)
    dG, dm = dG[keep], dm[keep]
    return float(np.dot(dG, dG)), float(np.dot(dG, dm))


def fit_lambda(A, B, rho_rel=RHO_OLS):
    """Solve (diag(A) + rho D'D) lam = B across age.

    D is the first difference on the age index, so the penalty is
    rho sum (lam_{x+1} - lam_x)^2, with rho quoted relative to the median
    of A and therefore free of the scale of G.  Tridiagonal, one solve.
    """
    rho = rho_rel * float(np.median(A))
    n   = A.size
    d   = A.copy()
    d[0]    += rho
    d[-1]   += rho
    d[1:-1] += 2.0 * rho
    ab = np.zeros((2, n))
    ab[0, 1:] = -rho
    ab[1, :]  = d
    return solveh_banded(ab, B, lower=False)


# ========================= LIFTED AFFINE SYSTEM =======================
def build_system(x, w, lam, mu0, theta=THETA, dt=DT):
    """Affine system for the lifted deviation at one age.

    State (Y; Z; 1) of dimension 2N+1, with the level carried in the last
    slot so the drift is linear:

        A = [[-X, -lam 1 w', lam (theta - mu0) 1],
             [ I,      -X,             0        ],
             [ 0,       0,             0        ]],
        h = (0, w, mu0),    v = (1, 0, 0).

    Returned alongside are the two quadrature arrays of the exact
    one-step conditional variance,

        Mq[j] = e^{A v_j} v,   hE[j] = h' e^{A (dt - v_j)},
        Q(s)  = sig^2 sum_j GLW2[j] (hE[j] . s) outer(Mq[j], Mq[j]),

    with hE stored reversed, exact because the Gauss-Legendre nodes are
    symmetric about dt/2.
    """
    n  = x.size
    ns = 2 * n + 1
    A  = np.zeros((ns, ns))
    A[:n, :n]       = -np.diag(x)
    A[:n, n:2 * n]  = -lam * np.outer(np.ones(n), w)
    A[:n, -1]       = lam * (theta - mu0)
    A[n:2 * n, :n]  = np.eye(n)
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
def kf_nll(y, mm, R, obs, h, Phi, Mq, hE, wq, sig2, s0, P0):
    """Kalman recursion, negative log-likelihood.

    Predicts every year and updates only where obs is true, so gaps need
    no special handling.
    """
    T  = y.shape[0]
    ns = Phi.shape[0]
    ng = Mq.shape[0]
    s  = s0.copy()
    P  = P0.copy()
    Mw = np.empty((ng, ns))
    ll = 0.0
    LOG2PI = np.log(2.0 * np.pi)
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
            if S <= 0.0 or not np.isfinite(S):
                return 1e10
            r  = y[t] - L * (h @ s)
            ll += -0.5 * (r * r / S + np.log(S) + LOG2PI)
            s = s + PH * (r / S)
            P = P - np.outer(PH, PH) / S
            P = 0.5 * (P + P.T)
    if not np.isfinite(ll):
        return 1e10
    return -ll


@njit(cache=True, fastmath=True)
def kf_path(y, mm, R, obs, h, Phi, Mq, hE, wq, sig2, s0, P0, Zf, St):
    """Same recursion, recording the filtered deviation and state.

    Run once per age after the fit, since the portfolio stage needs the
    path and the terminal state that the likelihood discards.
    """
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
        Zf[t] = h @ s
        for j in range(ns):
            St[t, j] = s[j]


def age_nll(sigma, sys, dat):
    """Scalar objective for one age, shared by the warm start and the fit."""
    y, mm, R, obs = dat
    Phi, h, A, v, Mq, hE = sys
    s0, P0, sig2 = initial_state(sigma, sys)
    return kf_nll(y, mm, R, obs, h, Phi, Mq, hE, GLW2, sig2,
                  np.ascontiguousarray(s0), np.ascontiguousarray(P0))


def fit_sigma(dat, sys, idx):
    """Joint penalised fit of log sigma over all ages at lam fixed.

    The likelihood is separable across age and the penalty couples
    neighbours, so the ages are solved together.  Only the likelihood
    partials are differenced, one filter call each; the penalty gradient
    is analytic.
    """
    n  = len(idx)
    p0 = np.full(n, np.log(0.05))
    for k, i in enumerate(idx):
        r = minimize(lambda t: age_nll(float(np.exp(t[0])), sys[i], dat[i]),
                     [p0[k]], method="L-BFGS-B", bounds=[LN_SIG],
                     options=dict(maxiter=60, ftol=1e-10))
        p0[k] = r.x[0]

    def objgrad(p):
        f = 0.0
        g = np.zeros(n)
        for k, i in enumerate(idx):
            f0 = age_nll(float(np.exp(p[k])), sys[i], dat[i])
            if f0 >= 1e10:
                return 1e12, np.zeros(n)
            f += f0
            g[k] = (age_nll(float(np.exp(p[k] + FD_EPS)), sys[i], dat[i])
                    - f0) / FD_EPS
        d = np.diff(p)
        f += PEN_SIG * float(np.sum(d ** 2))
        g[:-1] -= 2.0 * PEN_SIG * d
        g[1:]  += 2.0 * PEN_SIG * d
        return f, g

    nit = [0]
    t0  = time.time()

    def cb(p):
        nit[0] += 1
        if nit[0] % HEARTBEAT == 0:
            print(f"  iter {nit[0]:>4}  {time.time() - t0:>6.0f}s", flush=True)

    r = minimize(objgrad, p0, jac=True, method="L-BFGS-B",
                 bounds=[LN_SIG] * n, callback=cb,
                 options=dict(maxiter=MAXITER, maxfun=20 * MAXITER,
                              ftol=FTOL, gtol=GTOL))
    return np.exp(r.x)


def filtered_paths(dat, sys, sigma, idx, years):
    """Filtered deviation and forward pieces, per fitted age.

    Years inside the fitting window come from the filtered path; later
    years are propagated by s <- Phi s, the drift being carried in the
    frozen slot of the state.
    """
    nat = {}
    for i in idx:
        Phi, h, A, v, Mq, hE = sys[i]
        s0, P0, sig2 = initial_state(float(sigma[i]), sys[i])
        y, mm, R, obs = dat[i]
        Zf = np.zeros(len(years))
        St = np.zeros((len(years), Phi.shape[0]))
        kf_path(y, mm, R, obs, h, Phi, Mq, hE, GLW2, sig2,
                np.ascontiguousarray(s0), np.ascontiguousarray(P0), Zf, St)
        nat[int(AGES[i])] = dict(Z=Zf, sT=St[-1].copy(), Phi=Phi, h=h)
    return nat


def z1_series(nat_age, years, year0=FIT_START, end=FIT_END):
    """National deviation over a window, filtered then propagated."""
    out = np.empty(len(years))
    s   = nat_age["sT"].copy()
    last = end
    for j, y in enumerate(years):
        if y <= end:
            out[j] = nat_age["Z"][y - year0]
        else:
            while last < y:
                s = nat_age["Phi"] @ s
                last += 1
            out[j] = float(nat_age["h"] @ s)
    return out


# ================= PORTFOLIO: delta, gamma BY MAP ====================
def mu2_mean(delta, gamma, z1):
    """Exact mean path of mu_2, started at its equilibrium.

    Solves d E[mu_2]/dt = delta mu_1 - gamma E[mu_2] over one year with
    mu_1 held constant, by variation of constants.  Vectorised over ages:
    delta and gamma are (n,), z1 is (n, T).
    """
    a = np.exp(-gamma)
    b = (1.0 - a) / gamma
    mu2 = np.empty_like(z1)
    mu2[:, 0] = delta / gamma * z1[:, 0]
    for t in range(z1.shape[1] - 1):
        mu2[:, t + 1] = a * mu2[:, t] + delta * z1[:, t] * b
    return mu2


def neg_log_post(params, z1, m, D, E, ok, Kd_inv, Kg_inv, d0, g0):
    """Negative of the MAP objective, with its gradient.

    Poisson term D log(E m mu_2) - E m mu_2 summed over cells carrying
    exposure, minus the two Gaussian process quadratic forms across age.
    The mean path and its sensitivities to delta and gamma are propagated
    by the same recursion, so the gradient is exact and costs two extra
    rows per year rather than one path per parameter.
    """
    n = z1.shape[0]
    delta, gamma = params[:n], params[n:]

    a  = np.exp(-gamma)
    b  = (1.0 - a) / gamma
    db = (gamma * a - (1.0 - a)) / gamma ** 2

    mu2 = np.empty_like(z1)
    dd  = np.empty_like(z1)
    dg  = np.empty_like(z1)
    mu2[:, 0] = delta / gamma * z1[:, 0]
    dd[:, 0]  = z1[:, 0] / gamma
    dg[:, 0]  = -delta * z1[:, 0] / gamma ** 2
    for t in range(z1.shape[1] - 1):
        mu2[:, t + 1] = a * mu2[:, t] + delta * z1[:, t] * b
        dd[:, t + 1]  = a * dd[:, t] + z1[:, t] * b
        dg[:, t + 1]  = -a * mu2[:, t] + a * dg[:, t] + delta * z1[:, t] * db

    offs = np.where(ok, E * m, 0.0)
    m2   = np.maximum(mu2, 1e-12)
    lp   = float(np.sum(np.where(ok, D * np.log(np.maximum(offs, 1e-300) * m2)
                                 - offs * m2, 0.0)))
    wgt  = np.where(ok, D / m2 - offs, 0.0)
    gd   = (wgt * dd).sum(axis=1)
    gg   = (wgt * dg).sum(axis=1)

    rd, rg = delta - d0, gamma - g0
    lp -= 0.5 * (rd @ (Kd_inv @ rd)) + 0.5 * (rg @ (Kg_inv @ rg))
    gd -= Kd_inv @ rd
    gg -= Kg_inv @ rg
    return -lp, -np.concatenate([gd, gg])


def fit_delta_gamma(nat, obs_sex, a_x, b_x):
    """MAP fit of (delta_x, gamma_x) over the national age grid.

    The window starts ANCHOR_BURN years before the first scored year, so
    the national deviation has moved before the first residual.  Ages
    with no portfolio exposure are carried by the priors, pulled towards
    (DELTA0, GAMMA0) far from data and blended towards fitted values near
    the edge of the book.
    """
    years = list(range(min(PORT_YEARS) - ANCHOR_BURN, max(PORT_YEARS) + 1))
    idx   = {y: j for j, y in enumerate(years)}
    tt    = np.array(years, float) - T0
    ages  = [int(a) for a in AGES if int(a) in nat]
    n, T  = len(ages), len(years)

    z1 = np.empty((n, T))
    m  = np.empty((n, T))
    D  = np.zeros((n, T))
    E  = np.zeros((n, T))
    ok = np.zeros((n, T), bool)
    pos = {int(a): k for k, a in enumerate(AGES)}
    for k, age in enumerate(ages):
        z1[k] = z1_series(nat[age], years)
        m[k]  = np.exp(a_x[pos[age]] + b_x[pos[age]] * tt)
        for y in PORT_YEARS:
            cell = obs_sex.get((age, y))
            if cell is not None:
                D[k, idx[y]], E[k, idx[y]] = cell
                ok[k, idx[y]] = True

    af = np.array(ages, float)
    dif = af[:, None] - af[None, :]
    Kd_inv = np.linalg.inv(TAU_DELTA ** 2 * np.exp(-0.5 * (dif / ELL_DELTA) ** 2)
                           + JITTER * np.eye(n))
    Kg_inv = np.linalg.inv(TAU_GAMMA ** 2 * np.exp(-0.5 * (dif / ELL_GAMMA) ** 2)
                           + JITTER * np.eye(n))
    d0 = DELTA0 * np.ones(n)
    g0 = GAMMA0 * np.ones(n)

    res = minimize(neg_log_post, np.concatenate([d0, g0]), jac=True,
                   args=(z1, m, D, E, ok, Kd_inv, Kg_inv, d0, g0),
                   method="L-BFGS-B",
                   bounds=[DELTA_BOUNDS] * n + [GAMMA_BOUNDS] * n,
                   options=dict(maxiter=5000, maxfun=20000,
                                ftol=1e-10, gtol=1e-7))

    delta   = np.full(NA, np.nan)
    gamma   = np.full(NA, np.nan)
    pdeaths = np.zeros(NA)
    pexp    = np.zeros(NA)
    for k, age in enumerate(ages):
        i = pos[age]
        delta[i], gamma[i] = res.x[k], res.x[n + k]
        pdeaths[i] = float(D[k][ok[k]].sum())
        pexp[i]    = float(E[k][ok[k]].sum())
    return delta, gamma, pdeaths, pexp


def save_frame(df, country, sex, end):
    path = os.path.join(PARAM_DIR, f"params_{country}_{sex}_{end}{SUFFIX}.csv")
    df.to_csv(path, index=False)
    return path


# ================================ RUN ================================
years = np.arange(FIT_START, FIT_END + 1)
u     = (years - T0).astype(float)
DATA  = load_country(COUNTRY, years, AGES)
PORT  = load_extract(PORT_FILE, PORT_YEARS, MIN_EXPOSURE)

for sex in SEXES:
    d = DATA[sex]
    df, a_x, b_x, H_x = load_params(COUNTRY, sex, FIT_END)
    mu1 = deviation_path(d["MX"], d["DD"], d["EE"], a_x, b_x, u)
    m   = np.exp(a_x[:, None] + b_x[:, None] * u[None, :])

    lifts = {}
    A_ols = np.empty(NA)
    B_ols = np.empty(NA)
    for i in range(NA):
        key = round(float(H_x[i]), 6)
        if key not in lifts:
            lifts[key] = lift_nodes(key)
        x, w = lifts[key]
        A_ols[i], B_ols[i] = ols_sums(mu1[i], x, w)
    lam = fit_lambda(A_ols, B_ols)

    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(np.isfinite(d["DD"]) & (d["DD"] > 0),
                     d["MX"] ** 2 / np.maximum(d["DD"], MIN_DEATHS), 1.0)
    dat = [None] * NA
    sys = [None] * NA
    idx = list(range(NA))
    for i in idx:
        obs = np.isfinite(mu1[i])
        x, w = lifts[round(float(H_x[i]), 6)]
        sys[i] = build_system(x, w, float(lam[i]), float(mu1[i][obs][0]))
        dat[i] = (np.ascontiguousarray(np.nan_to_num(d["MX"][i], nan=1.0)),
                  np.ascontiguousarray(m[i]),
                  np.ascontiguousarray(np.nan_to_num(R[i], nan=1.0)),
                  np.ascontiguousarray(obs))

    print(f"{sex}: {len(idx)} ages", flush=True)
    sigma = fit_sigma(dat, sys, idx)

    nat = filtered_paths(dat, sys, sigma, idx, years)
    delta, gamma, pdeaths, pexp = fit_delta_gamma(nat, PORT[sex], a_x, b_x)

    df["lam"]           = lam
    df["sigma"]         = sigma
    df["delta"]         = delta
    df["gamma"]         = gamma
    df["ratio"]         = delta / gamma
    df["port_deaths"]   = pdeaths
    df["port_exposure"] = pexp
    df["n_years"]       = np.isfinite(mu1).sum(axis=1)
    print(save_frame(df, COUNTRY, sex, FIT_END), flush=True)