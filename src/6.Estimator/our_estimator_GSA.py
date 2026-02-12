import math
from sage.all import RR, ZZ, binomial, cached_function

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


# --------------------------------------------
# Your derived pieces: find k and check success
# --------------------------------------------

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


def success_rhs(beta, k, dim):
    """
    Compute RHS of your success condition:

    sigma < sqrt(beta)/(2*pi*e) * (sqrt(t))^{-k/t*(ln t/beta)} * delta_beta^{-n - m + k + beta*k/t}

    where:
      t = beta + k
      delta_beta = delta_0f(beta)
      d = dim  (we take dim = n+m+1 per your statement)
    """
    beta_f = float(beta)
    k_f = float(k)
    d = float(dim)

    t = beta_f + k_f
    if t <= 1:
        return 0.0

    delta_beta = float(delta_0f(beta_f))
    # log is natural log here
    logt = math.log(t)
    if logt <= 0:
        return 0.0


    term1 = math.log(math.sqrt(beta_f) / (2.0 * math.pi * math.e), 2.0)
    #term2 = 0.5 * (-k_f / t * math.log((t/ beta_f)) ) * math.log(t, 2.0)  
    term2 = 0.5 * (k_f/t) * math.log(k_f/t, 2.0)
    exponent = (-d + k_f + (beta_f * k_f / t))
    term3 = exponent * math.log(delta_beta , 2.0)
    #term4 = term2 * (delta_beta ** (beta_f * k_f / t))
    
    return term1 + term2 + term3 



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

    if s in ["binary", "ternary"]:
        return math.sqrt(n *  (sigma_s**2)  + 1.0 + m * (sigma_e**2) ) / math.sqrt(dim)

    if s in ["gaussian", "dg", "discrete_gaussian", "discrete-gaussian"]:
        return math.sqrt((sigma_s ** 2) * n + 1.0 + (sigma_e ** 2) * m) / math.sqrt(dim)

    raise ValueError(f"Unknown s_dist={s_dist}. Use one of: same, binary, ternary, gaussian.")



def find_min_beta(n, logq, sigma,
                  beta_min=2, beta_max=500,
                  k_max=None,
                  require_beta_ge_40=False, s_dist= "same", sigma_s = 3.19):
    """
    Search the smallest beta such that the success condition holds,
    with k determined by time-matching:
      T_BKZ(beta, n+m+1) = 2*T_HKZ(beta+k).

    Inputs:
      - n, m, logq, sigma: provided by you (logq currently unused in the inequality you gave)
      - beta_min/beta_max: search range
      - require_beta_ge_40: if True, starts search at max(beta_min, 40)
        (sometimes you may want this because delta_0f has a step table up to 40)

    Returns dict with beta, k, t, rhs, and the two times.
    """
    

    start = beta_min
    if require_beta_ge_40:
        start = max(start, 40)

    # Compute scaling factor w = σ_s / σ_e or σ_s / σ_e or 1 
    if sigma[0] > sigma[1]:
        w = sigma[0] / sigma[1]
        sigma = sigma[0]
    elif sigma[0] < sigma[1]:
        w = sigma[0] / sigma[1]
        sigma = sigma[1]
    else:
        w = 1
        sigma = math.sqrt(sigma[1])

    for beta in range(int(start), int(beta_max) + 1):
        delta_0 = delta_0f(beta)
        m = round(math.sqrt((n+1)*logq/ math.log(delta_0,2) ) - n-1)
        dim = int(n + m + 1)
        k = find_k_for_time_match(beta, dim, k_max=k_max)
        # Using Bai-Galbraith embedding, the determinant of lattice is q^{m/d}w^{(n+1)/d}
        rhs = success_rhs(beta, k, dim) + (logq *m /dim + math.log2(w)*(n+1)/dim)
        lhs = math.log(lhs_for_secret_dist(s_dist, n, m, sigma, dim, sigma_s), 2.0)
        if lhs < rhs:
            return {
              "beta": beta,
                "k": k,
                "t": beta + k,
                "rhs": rhs,
                "lhs": lhs,
                "sigma_e": sigma,
                "s_dist": s_dist,
                "m": m,
                "dim": dim,
                "T_BKZ": BKZ_time(beta, dim),
                "T_HKZ": HKZ_time(beta + k),
                "delta_beta": delta_0f(beta),
                "Total_cost": RR(math.log(2**BKZ_time(beta, dim) + 2*2**(HKZ_time(beta+k)),2))
            }

    return None




#========================================================================================== bdd
def success_rhs_bdd(beta, k, dim):
    """
    Compute RHS of your success condition:

    sigma < sqrt(beta)/(2*pi*e) * (beta/(t*log t))^{k/(2*t)} * delta_beta^{-d  + k + beta*k/t}

    where:
      t = beta + k
      delta_beta = delta_0f(beta)
      d = dim  (we take dim = n+m+1 per your statement)
    """
    beta_f = float(beta)
    k_f = float(k)
    d = float(dim)

    t = beta_f + k_f
    if t <= 1:
        return 0.0

    delta_beta = float(delta_0f(beta_f))
    # log is natural log here
    logt = math.log(t)
    if logt <= 0:
        return 0.0

    term1 = math.log(math.sqrt(beta_f) / (2.0 * math.pi * math.e), 2.0)
    term2 = 0
    exponent = (-d + k_f )
    term3 = exponent * math.log(delta_beta, 2.0)
    #term4 = term2 * (delta_beta ** (beta_f * k_f / t))
    return term1 + term2 + term3


def find_min_beta_bdd(n, logq, sigma,
                  beta_min=2, beta_max=500,
                  k_max=None,
                  require_beta_ge_40=False, s_dist= "same", sigma_s = 3.19):
    """
    Search the smallest beta such that the success condition holds,
    with k determined by time-matching:
      T_BKZ(beta, n+m+1) = T_HKZ(beta+k).

    Inputs:
      - n, m, logq, sigma: provided by you (logq currently unused in the inequality you gave)
      - beta_min/beta_max: search range
      - require_beta_ge_40: if True, starts search at max(beta_min, 40)
        (sometimes you may want this because delta_0f has a step table up to 40)

    Returns dict with beta, k, t, rhs, and the two times.
    """
    

    start = beta_min
    if require_beta_ge_40:
        start = max(start, 40)

    # Compute scaling factor w = σ_s / σ_e or σ_s / σ_e or 1 
    if sigma[0] > sigma[1]:
        w = sigma[0] / sigma[1]
        sigma = sigma[0]
    elif sigma[0] < sigma[1]:
        w = sigma[0] / sigma[1]
        sigma = sigma[1]
    else:
        w = 1
        sigma = sigma[1]
    
    for beta in range(int(start), int(beta_max) + 1):
        delta_0 = delta_0f(beta)
        m = round(math.sqrt((n+1)*logq/ math.log(delta_0,2) ) - n-1)
        dim = int(n + m + 1)
        k = find_k_for_time_match(beta, dim, k_max=k_max)
        # Using Bai-Galbraith embedding, the determinant of lattice is q^{m/d}w^{(n+1)/d}
        rhs = success_rhs_bdd(beta, k, dim) + (logq *m /dim + math.log2(w)*(n+1)/dim)
        lhs = math.log(lhs_for_secret_dist(s_dist, n, m, sigma, dim, sigma_s), 2.0)
        if lhs < rhs:
            return {
                "beta": beta,
                "k": k,
                "t": beta + k,
                "rhs": rhs,
                "sigma": sigma,
                "dim": dim,
                "T_BKZ": BKZ_time(beta, dim),
                "T_HKZ": HKZ_time(beta + k),
                "delta_beta": delta_0f(beta),
            }

    return None


#========================================================================================== usvp
def success_rhs_usvp(beta, k, dim):
    """
    Compute RHS of your success condition:

    sigma < sqrt(beta)/(2*pi*e) * (beta/(t*log t))^{k/(2*t)} * delta_beta^{-d  + k + beta*k/t}

    where:
      t = beta + k
      delta_beta = delta_0f(beta)
      d = dim  (we take dim = n+m+1 per your statement)
    """
    beta_f = float(beta)
    k_f = float(k)
    d = float(dim)

    t = beta_f + k_f
    if t <= 1:
        return 0.0

    delta_beta = float(delta_0f(beta_f))
    # log is natural log here
    logt = math.log(t)
    if logt <= 0:
        return 0.0

    term1 = math.log(math.sqrt(beta_f) / (2.0 * math.pi * math.e), 2.0)
    term2 = 0
    exponent = (-d)
    term3 = exponent * math.log(delta_beta, 2.0)
    #term4 = term2 * (delta_beta ** (beta_f * k_f / t))
    return term1 + term2 + term3



def find_min_beta_usvp(n, logq, sigma,
                  beta_min=2, beta_max=500,
                  k_max=None,
                  require_beta_ge_40=False, s_dist= "same", sigma_s = 3.19):
    """
    Search the smallest beta such that the success condition holds,
    with k determined by time-matching:
      T_BKZ(beta, n+m+1) = T_HKZ(beta+k).

    Inputs:
      - n, m, logq, sigma: provided by you (logq currently unused in the inequality you gave)
      - beta_min/beta_max: search range
      - require_beta_ge_40: if True, starts search at max(beta_min, 40)
        (sometimes you may want this because delta_0f has a step table up to 40)

    Returns dict with beta, k, t, rhs, and the two times.
    """
    

    start = beta_min
    if require_beta_ge_40:
        start = max(start, 40)

    # Compute scaling factor w = σ_s / σ_e or σ_s / σ_e or 1 
    if sigma[0] > sigma[1]:
        w = sigma[0] / sigma[1]
        sigma = sigma[0]
    elif sigma[0] < sigma[1]:
        w = sigma[0] / sigma[1]
        sigma = sigma[1]
    else:
        w = 1
        sigma = sigma[1]
    
    for beta in range(int(start), int(beta_max) + 1):
        delta_0 = delta_0f(beta)
        m = round(math.sqrt((n+1)*logq/ math.log(delta_0,2) ) - n-1)
        dim = int(n + m + 1)
        k = find_k_for_time_match(beta, dim, k_max=k_max)
        # Using Bai-Galbraith embedding, the determinant of lattice is q^{m/d}w^{(n+1)/d}
        rhs = success_rhs_usvp(beta, k, dim) + (logq *m /dim + math.log2(w)*(n+1)/dim)
        lhs = math.log(lhs_for_secret_dist(s_dist, n, m, sigma, dim, sigma_s), 2.0)
        if lhs < rhs:
            return {
                "beta": beta,
                "k": k,
                "t": beta + k,
                "rhs": rhs,
                "sigma": sigma,
                "dim": dim,
                "T_BKZ": BKZ_time(beta, dim),
                "T_HKZ": HKZ_time(beta + k),
                "delta_beta": delta_0f(beta),
            }

    return None




# -----------------------------
# Example usage (edit numbers)
# -----------------------------
if __name__ == "__main__":
   
    n =  256*4
    logq = math.log(8380417,2)
    # σ = (σ_s, σ_e) 
    sigma = (1, 1)
    #find_min_beta_usvp(n, logq, sigma, beta_min=2, beta_max=900, require_beta_ge_40=False)
    #find_min_beta_bdd(n, logq, sigma, beta_min=2, beta_max=900, require_beta_ge_40=False)
    print(find_min_beta(n, logq, sigma, beta_min=350, beta_max=550, require_beta_ge_40=False))
