import math

# -----------------------------
# Given cost/quality primitives
# -----------------------------

def delta_0f(k):
    """
    Root-Hermite factor δ_0(k).
    Small values are experimentally determined, otherwise Chen13 asymptotic.
    """
    small = (
        ( 2, 1.02190),
        ( 5, 1.01862),
        (10, 1.01616),
        (15, 1.01485),
        (20, 1.01420),
        (25, 1.01342),
        (28, 1.01331),
        (40, 1.01295),
    )

    k = float(k)
    if k <= 2:
        return 1.0219
    elif k < 40:
        for i in range(1, len(small)):
            if small[i][0] > k:
                return small[i - 1][1]
        return small[-1][1]
    elif k == 40:
        return small[-1][1]
    else:
        # (k/(2*pi*e) * (pi*k)^(1/k))^(1/(2*(k-1)))
        return (k / (2 * math.pi * math.e) * (math.pi * k) ** (1.0 / k)) ** (1.0 / (2.0 * (k - 1.0)))


def d4f(beta):
    beta = float(beta)
    denom = math.log(beta / (2 * math.pi * math.e))
    if denom <= 0:
        return 0.0
    return max(beta * math.log(4.0 / 3.0) / denom, 0.0)


def BKZ_time(beta, dim):
    """
    log2(time) model you pasted:
      Mat = 5.46*(2*dim - beta)* 2^(a*(beta - d4f(beta)) + b)
    """
    a = 0.29613500308205365
    b = 20.387885985467914
    beta = float(beta)
    dim = float(dim)
    val = 5.46 * (2.0 * dim - beta) * (2.0 ** (a * (beta - d4f(beta)) + b))
    return math.log(val, 2)


def HKZ_time(beta):
    """
    log2(time) for HKZ at dimension 'beta' in your model:
      log2( 2^(a*(beta - d4f(beta)) + b) ) = a*(beta-d4f)+b
    """
    a = 0.29613500308205365
    b = 20.387885985467914
    beta = float(beta)
    return math.log(2.0 ** (a * (beta - d4f(beta)) + b), 2)



def find_k_for_time_match(beta, dim, k_max=None, tol=1e-9):
    """
    Define k by enforcing  T_BKZ(beta, dim) = 2*T_HKZ(beta+k).

    Since HKZ_time(x) is (weakly) increasing in x in this model, we pick
    the *smallest integer k >= 0* such that:
        2*HKZ_time(beta+k) >= BKZ_time(beta, dim)
    (Up to numerical tolerance).
    """
    if k_max is None:
        # Safe cap: you can raise this if you expect huge k
        k_max = int(max(0, 5 * dim))

    target = BKZ_time(beta, dim)
    base = int(beta)

    for k in range(0, k_max + 1):
        if HKZ_time(base + k)+1 + tol >= target:
            return k

    raise RuntimeError(f"Could not find k up to k_max={k_max} for beta={beta}, dim={dim}.")




# ---------- ZGSA (Z-shape) building blocks ----------

def gh_dim1(d: float) -> float:
    return math.sqrt(d / (2.0 * math.pi * math.e))

def alpha_beta(beta: int) -> float:
    beta = float(beta)
    if beta <= 1:
        raise ValueError("beta must be >= 2")
    return gh_dim1(beta) ** (2.0 / (beta - 1.0))

def zgsa_m(q: float, beta: int) -> float:
    # m = 1/2 + ln(q) / (2 ln(alpha_beta))
    q = float(q)
    a = alpha_beta(beta)
    return 0.5 + math.log(q) / (2.0 * math.log(a))

def zgsa_bstar_norm(i: int, d: int, r: int, q: float, beta: int) -> float:
    """
    ZGSA Z-shape:
      q                                     if i <= r - m
      sqrt(q) * alpha_beta^{(d-1-2i)/2}      if r - m < i < r + m - 1
      1                                     if i >= r + m - 1
    """
    if not (0 <= i < d):
        raise IndexError("i out of range")
    if not (0 <= r <= d):
        raise ValueError("r must satisfy 0 <= r <= d")
    q = float(q)
    if q <= 0:
        raise ValueError("q must be > 0")

    a = alpha_beta(beta)
    m = zgsa_m(q, beta)

    left  = r - m
    right = r + m - 1.0

    if i <= left:
        return q
    if i >= right:
        return 1.0

    exp_ = (float(d) - 1.0 - 2.0 * float(i)) / 2.0
    return math.sqrt(q) * (a ** exp_)

def zgsa_profile_zshape(d: int, r: int, q: float, beta: int):
    return [zgsa_bstar_norm(i, d, r, q, beta) for i in range(d)]

def tail_logdet_from_profile(profile, t: int) -> float:
    """log( Π_{j=d-t}^{d-1} ||b_j^*|| ) computed safely."""
    if t <= 0 or t > len(profile):
        raise ValueError("invalid t")
    s = 0.0
    for x in profile[-t:]:
        # x is positive by construction
        s += math.log(float(x))
    return s


# ---------- Your "improved" factor (keep yours, but log-safe) ----------

def A_extra_log(beta: int, k: int, delta_beta: float) -> float:
    """
    log(A_extra) where
      A_extra = (sqrt(t)**(-k/t *ln t/beta)  * delta_beta^{ beta*k/t}
    """
    beta = float(beta)
    k = float(k)
    t = beta + k
    if t <= 1:
        return 0.0

    term1 = sqrt(t)**(-k/t* ln(t/beta))
    if term1 <= 0:
        return float("-inf")

    logA1 = math.log(sqrt(t)**(-k/t* ln(t/beta)))
    logA2 = ( (beta * k / t)) * math.log(float(delta_beta))
    return logA1 + logA2


# ---------- ZGSA-based sigma threshold ----------

def sigma_threshold_zgsa(beta: int, k: int, d: int, r: int, q: float,
                         delta_beta: float,
                         use_improved: bool = True):
    """
    Base threshold (log):
        log(threshold_base) = (1/t)*log(det_tail) - 1/2*log(2πe)
    where det_tail = product of last t GS norms from ZGSA profile.
    Then optionally add log(A_extra).
    """
    beta_i = int(beta)
    k_i = int(k)
    d_i = int(d)
    r_i = int(r)
    q_f = float(q)

    t = beta_i + k_i
    prof = zgsa_profile_zshape(d_i, r_i, q_f, beta_i)
    log_det_tail = tail_logdet_from_profile(prof, t)

    log_base = (log_det_tail / float(t)) - 0.5 * math.log(2.0 * math.pi * math.e)
    log_factor = A_extra_log(beta_i, k_i, delta_beta) if use_improved else 0.0
    log_thr = RR(log_base + log_factor)

    
    return {
        "t": t,
        "log_det_tail": log_det_tail,
        "log_base_threshold": log_base,
        "log_A_extra": log_factor,
        "log_threshold": log_thr,
        "profile": prof,  
    }


def lhs_for_secret_dist(s_dist: str, n: int, m: int, sigma_e: float, dim: int, sigma_s: float) -> float:
    """
    LHS used in the success test.

    s_dist:
      - "same" (default): original behavior, lhs = sigma_e
      - "binary" or "ternary": sqrt(n + 1 + m^2*sigma_e) / sqrt(dim)
      - "gaussian" (discrete Gaussian): sqrt(n^2*sigma_s + 1 + m^2*sigma_e) / sqrt(dim)

    NOTE: you asked to keep e governed by sigma (sigma_e).
    """
    s = (s_dist or "same").lower()

    if s in ["same", "orig", "original"]:
        return float(sigma_e)

    if s == "binary":
        return math.sqrt(n + 1.0 + m * (sigma_e**2) ) / math.sqrt(dim)

    if s == "ternary":
        return math.sqrt(n * (sigma_s ** 2) + 1.0 + m * (sigma_e**2) ) / math.sqrt(dim)

    if s in ["gaussian", "dg", "discrete_gaussian", "discrete-gaussian"]:
        return math.sqrt((sigma_s ** 2) * n + 1.0 + (sigma_e ** 2) * m) / math.sqrt(dim)

    raise ValueError(f"Unknown s_dist={s_dist}. Use one of: same, binary, ternary, gaussian.")




# ---------- Wrapper search: minimal beta ----------

def find_min_beta_zgsa(n, m, logq, sigma,
                      beta_min=2, beta_max=900,
                      k_max=None,
                      require_beta_ge_40=False,
                      use_improved=True,
                      s_dist = "same",
                      sigma_s = 3.19
                      ):
    """
    Conventions (as you used):
      q = 2^{logq}
      m chosen by your heuristic using delta_0f(beta)
      d = n + m + 1
      r = m   (# of q-vectors; typical q-ary embedding)
    Compare in log-domain: log(sigma) < log(threshold).
    """
    n = int(n)
    m = int(m)
    logq = float(logq)
    sigma = lhs_for_secret_dist(s_dist, n, m, sigma, n+m+1, sigma_s)
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    log_sigma = math.log(sigma)

    start = int(beta_min)
    if require_beta_ge_40:
        start = max(start, 40)

    for beta in range(start, int(beta_max) + 1):
        delta_beta = float(delta_0f(beta))
        ld = math.log(delta_beta, 2)
        if ld <= 0:
            continue
        width = round(zgsa_m(2.0 ** logq, beta)*2-1)
        # your m heuristic        
        if m <= 0:
            continue

        d = int(n + m + 1)
        r = int(m)
        q = 2.0 ** logq

        k = find_k_for_time_match(beta, width, k_max=k_max)

        info = sigma_threshold_zgsa(beta, k, d, r, q, delta_beta, use_improved=1)
        
        
        if log_sigma < info["log_threshold"]:
            if zgsa_bstar_norm( n+m-beta-k, d, r, 2** logq, beta) > 2*2**log_sigma:
                return {
                    "beta": beta,
                    "k": k,
                    "t": info["t"],
                    "n": n,
                    "m": m,
                    "d": d,
                    "r": r,
                    "q": q,
                    "w": width,
                    "sigma": sigma,
                    "log_threshold": info["log_threshold"],
                    "log_base_threshold": info["log_base_threshold"],
                    "log_A_extra": info["log_A_extra"],
                    "delta_beta": delta_beta,
                    "T_BKZ_log2": BKZ_time(beta, width),
                    "T_HKZ_log2": HKZ_time(beta + k),
                    "Final_cost": math.log(2**BKZ_time(beta,width) + 2*2**HKZ_time(beta + k),2)
                }

    return None

# -----------------------------
# Example usage (edit numbers)
# -----------------------------
if __name__ == "__main__":
    n = 256*6
    m = 256*5
    logq = log(8380417,2)
    sigma = sqrt(2)
    
    find_min_beta_zgsa(n,m,logq,sigma,beta_min=600, beta_max=950, k_max=None,require_beta_ge_40=True, s_dist = "same", sigma_s = 1)
