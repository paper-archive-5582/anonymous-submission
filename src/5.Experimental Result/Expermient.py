# 0. Libraries
import re
import time
import subprocess
import numpy as np
import random
import matplotlib.pyplot as plt
from fpylll import IntegerMatrix, GSO, FPLLL, BKZ, LLL
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2

from sage.all import randint
from utility import cbd_eta, dgb_sigma

def cal_GSO_basis(basis):
    """
    The function that calculate the Gram-Scmidt Orthogonaliztion basis of input basis
    
        return GSO basis
    
    :param basis: Input basis
    """
    dim = basis.nrows
    bkz_basis = IntegerMatrix(dim,dim)

    # basis copy
    for i in range(dim):
        for j in range(dim):
            bkz_basis[i,j] = basis[i,j]

    M = GSO.Mat(bkz_basis, flags=GSO.INT_GRAM)
    _ = M.update_gso()

    GSO_basis = np.zeros((dim, dim))

    # b_1 = b_1^*
    b1 = [basis[0, i] for i in range(dim)]
    GSO_basis[0, :] = b1
    
    for i in range(1, dim):
        bi = np.array([basis[i, j] for j in range(dim)], dtype=np.float64)

        for j in range(i):
            bi -= M.get_mu(i,j) * GSO_basis[j,:]
        GSO_basis[i,:] = bi

    return GSO_basis

GSO_basis = cal_GSO_basis(B_bkz)

# -----------------------------
# Helpers: convert basis to numpy
# -----------------------------
def _basis_to_numpy_rows(B):
    """
    Convert a reduced basis to a numpy float matrix A (rows are basis vectors).
    Supports:
      - fpylll IntegerMatrix (has .nrows and index access)
      - numpy array-like
    """
    if hasattr(B, "nrows") and hasattr(B, "__getitem__"):
        d = int(B.nrows)
        A = np.array([[float(B[i, j]) for j in range(d)] for i in range(d)], dtype=np.float64)
        return A
    # assume array-like
    A = np.array(B, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Basis must be a square (d x d) matrix (row-basis convention).")
    return A

# -----------------------------
# Build orthonormal GS directions for row-basis using QR on A^T
# -----------------------------
def _orthonormal_q_from_row_basis(A_rows):
    """
    A_rows: numpy float matrix with rows = basis vectors (d x d).
    Returns list of q_j (numpy vectors), where q_j correspond to the normalized GS directions
    in the same order as the row basis.
    """
    Q, R = np.linalg.qr(A_rows.T)  # A^T = Q R
    d = A_rows.shape[0]
    qs = [Q[:, j].copy() for j in range(d)]
    return qs

# -----------------------------
# Sampling: DG / CBD test vectors t in Z^d (no normalization)
# -----------------------------
def _sample_dg_integer(d, sigma, rng):
    # Approximate discrete Gaussian by rounding a normal sample
    x = rng.normal(loc=0.0, scale=float(sigma), size=d)
    return np.rint(x).astype(np.int64)

def _sample_cbd_integer(d, eta, rng):
    """
    Centered binomial CBD(eta): sum_{i=1..eta} b_i - sum_{i=1..eta} b'_i, b_i ~ Bernoulli(1/2)
    """
    eta = int(eta)
    b1 = rng.integers(0, 2, size=(d, eta), dtype=np.int8)
    b2 = rng.integers(0, 2, size=(d, eta), dtype=np.int8)
    return (b1.sum(axis=1) - b2.sum(axis=1)).astype(np.int64)

def sigma_eff_for_dist(dist, sigma_dg=3.19, eta_cbd=3):
    dist = (dist or "").lower()
    if dist in ["dg", "gaussian", "discrete_gaussian"]:
        return float(sigma_dg)
    if dist in ["cbd", "centered_binomial"]:
        # Var(CBD(eta)) = eta/2
        return float(np.sqrt(float(eta_cbd) / 2.0))
    raise ValueError("dist must be 'dg' or 'cbd'.")

# -----------------------------
# Chi-square CDF
# -----------------------------
def chi2_cdf(x, k):
    if x <= 0.0:
        return 0.0
    a = k / 2.0
    return float(mp.gammainc(a, 0, x / 2.0, regularized=True))

def empirical_cdf(samples):
    xs = np.sort(np.array(samples, dtype=np.float64))
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys

# -----------------------------
# Main: validate chi-square surrogate from reduced basis only
# -----------------------------
def validate_tail_projection_chi2(
    B_final,
    tau=None,
    dist="dg",
    N=500,
    sigma_dg=3.19,
    eta_cbd=3,
    seed=0,
    make_plot=True,
    point_size=2,
):
    """
    Input:
      - B_final: reduced basis (row-basis convention), fpylll IntegerMatrix or numpy (d x d).
      - tau: tail dimension. If None, default tau = min(60, d)  (reasonable toy default).
      - dist: 'dg' or 'cbd' for test-vector coordinates (no normalization).
      - N: number of trials
      - sigma_dg / eta_cbd: distribution parameters
      - seed: RNG seed
      - make_plot: overlay ECDF(S) with chi2_tau CDF

    Output:
      dict with: d, tau, sigma_eff, KS distance, mean/var of S, plus arrays.
    """
    A = _basis_to_numpy_rows(B_final)
    d = A.shape[0]
    if tau is None:
        tau = min(60, d)  # default if user doesn't supply
    tau = int(tau)
    if not (1 <= tau <= d):
        raise ValueError(f"tau must satisfy 1 <= tau <= d (got tau={tau}, d={d}).")

    rng = np.random.default_rng(int(seed))
    sigma_eff = sigma_eff_for_dist(dist, sigma_dg=sigma_dg, eta_cbd=eta_cbd)

    qs = _orthonormal_q_from_row_basis(A)
    j0 = d - tau  # tail starts at d-tau (row-GS order)

    # sample -> project -> collect S = ||proj||^2 / sigma_eff^2
    S = np.empty(int(N), dtype=np.float64)
    for i in range(int(N)):
        if dist.lower().startswith("dg"):
            t = _sample_dg_integer(d, sigma=sigma_dg, rng=rng).astype(np.float64)
        else:
            t = _sample_cbd_integer(d, eta=eta_cbd, rng=rng).astype(np.float64)

        # projection onto tail subspace span{q_{j0},...,q_{d-1}}
        s2 = 0.0
        for j in range(j0, d):
            z = float(np.dot(qs[j], t))
            s2 += z * z

        S[i] = s2 / (sigma_eff * sigma_eff)

    # ECDF vs model CDF on the same x-grid (use xs of ECDF)
    xs, ys_emp = empirical_cdf(S)
    ys_model = np.array([chi2_cdf(x, tau) for x in xs], dtype=np.float64)

    # KS distance (sup norm between CDFs)
    ks = float(np.max(np.abs(ys_emp - ys_model)))

    out = {
        "d": int(d),
        "tau": int(tau),
        "dist": dist,
        "N": int(N),
        "sigma_eff": float(sigma_eff),
        "S": S,
        "ecdf_x": xs,
        "ecdf_y": ys_emp,
        "model_y": ys_model,
        "KS": ks,
        "mean_S": float(np.mean(S)),
        "var_S": float(np.var(S)),
        "mean_chi2": float(tau),      # E[chi2_tau] = tau
        "var_chi2": float(2 * tau),   # Var[chi2_tau] = 2*tau
    }

    if make_plot:
        plt.figure(figsize=(6, 8))
        plt.scatter(xs, ys_emp, s=point_size, color="red", label=r"Empirical CDF of $S$" + f"({dist})")
        plt.scatter(xs, ys_model, s=point_size, color="blue", label=fr"Model CDF: $\chi^2_{{{tau}}}$")
        plt.grid(True, alpha=0.3)
        plt.xlabel(r"$S=\|\mathrm{proj}_{U_{\mathrm{tail}}}(t)\|^2 / \sigma_{\mathrm{eff}}^2$")
        plt.ylabel("CDF")
        #plt.title(f"ECDF vs chi-square model (dist={dist}, d={d}, tau={tau}, KS={ks:.3f})")
        plt.rc('xtick', labelsize=15)  
        plt.rc('ytick', labelsize=15)  
        plt.rc('legend', fontsize=15)
        plt.rc('axes', labelsize=20)   
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()

    return out

if __name__=="__main__":
    from compute_final_basis import compute_final_basis
    # parameter
    n = 100
    m = 100
    q = 257
    beta = 40
    
    B_final = computefinal_basis(n, m, q, beta)

    # plotting
    res = validate_tail_projection_chi2(B_bkz, tau=beta+k_mid, dist="dg", N=500, seed=1)
    res = validate_tail_projection_chi2(B_bkz, tau=beta+k_mid, dist="cbd", N=500, seed=1)
