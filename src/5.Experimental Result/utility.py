import numpy as np
import secrets
from lattice_estimator.estimator import RC
from lattice_estimator.estimator.cost import Cost
from lattice_estimator.estimator.lwe_parameters import LWEParameters
from sage.all import *
from sage.all import ZZ
from sage.stats.distributions.discrete_gaussian_integer import DiscreteGaussianDistributionIntegerSampler
from fpylll import IntegerMatrix, LLL, BKZ

def log2(x):
     return log(x, 2.0)

def cbd_eta(eta : int):
    """
    Kyber-style centered binomial distribution CBD(eta):
        x = sum_{i=1, ..., eta} b_i - sum_{1, ..., eta} b'_i, where b_i, b'_i ~ Bernoulli(1/2)
    
    Returns an integer in [-eta, eta].

    :param eta: CBD parameter controlling the distribution's spread.
    """
    a = sum(secrets.randbits(1) for _ in range(eta))
    b = sum(secrets.randbits(1) for _ in range(eta))

    return ZZ(a - b)

def dgb_sigma(sigma : float, c : float = 0.0):
    """
    Sample from an integer discrete Gaussian centered at c with width sigma
    
    :param sigma: The standard deviation controlling how spread out the discrete Gaussian smaples are around the center.
    :param c: The center of the distribution, i.e., the real-valued shift the samples are concentrated around.
    """
    D = DiscreteGaussianDistributionIntegerSampler(sigma, c=c)
    return ZZ(D())

def deltaf(beta):
    """
    Calculate the root Hermite factor δ_β from block size β.

    :param beta: Block size of BKZ algorithm.
    """
    
    small = (
            (2, 1.02190),  # LLL-like baseline
            (5, 1.01862),
            (10, 1.01616),
            (15, 1.01485),
            (20, 1.01420),
            (25, 1.01342),
            (28, 1.01331),
            (40, 1.01295),
        )
    
    if beta <= 2:
            return RR(1.0219)
    elif beta < 40:
        for i in range(1, len(small)):
            if small[i][0] > beta:
                return RR(small[i - 1][1])
    elif beta == 40:
        return RR(small[-1][1])
    else:
        return RR(beta / (2 * pi * e) * (pi * beta) ** (1 / beta)) ** (1 / (2 * (beta - 1)))

def costf(beta, d, cost_model = RC.MATZOV, B=None):
    """
    Return cost of BKZ-β or SVP-β using cost_model(e.g., MATZOV, etc...).
    
    :param cost_model: Cost model which use for computing the complextiy of BKZ-β or SVP-β.
    :param beta: Block size of BKZ algorithm. In SVP-β, the block size and dimension of lattice is equal.
    :param d: The dimension of lattice.
    :param B: Bit-size of entries.
    """

    # convenience: instantiate static classes if needed
    if isinstance(cost_model, type):
         cost_model = cost_model()
    
    cost = cost_model(beta, d, B)
    delta = deltaf(beta)
    cost = Cost(rop=cost, red=cost, delta=delta, beta=beta, d=d)

    return cost

def find_k(beta, d, B=None, cost_model=RC.MATZOV):
    """
    Find positive integer k such that Cost(BKZ-β) = Cost(SVP-(β+k))

    :param beta: Block size of BKZ algorithm.
    :param d: The dimension of lattice.
    :param B: Bit-size of entries.
    :param cost_model: Cost model which use for computing the complextiy of BKZ-β or SVP-β.
    """

    BKZ_cost = cost_model(beta, d, B)

    SVP_cost = cost_model(beta, beta, B)

    k = 0
    while SVP_cost < BKZ_cost:
        k +=1

        SVP_cost = cost_model(beta+k, beta+k, B)

    return k

def im_to_numpy(A):
    m, n = A.nrows, A.ncols
    X = np.zeros((m,n), dtype=object)
    for i in range(m):
        for j in range(n):
            X[i, j] = int(A[i,j])
    return X.astype(np.int64) if np.max(np.abs(X.astype(object))) < 2 ** 62 else x

def numpy_to_im(X):
    m, n = X.shape
    A = IntegerMatrix(m, n)

    for i in range(m):
        for j in range(n):
            A[i,j] = int(X[i,j])
    return A

def augment_with_identity(A):
    m, n = A.nrows, A.ncols
    Aug = IntegerMatrix(m, n+m)
    
    for i in range(m):
        for j in range(n):
            Aug[i,j] = A[i,j]
        # Right block I
        Aug[i,n+i] = 1
    return Aug

def split_aug(Aug, left_ncols):
    m, n = Aug.nrows, Aug.ncols
    Left = IntegerMatrix(m, left_ncols)
    Right = IntegerMatrix(m, n - left_ncols)
    for i in range(m):
        for j in range(left_ncols):
            Left[i,j] = Aug[i,j]
        for j in range(n-left_ncols):
            Right[i, j] = Aug[i, left_ncols + j]
    return Left, Right

def qr_R_from_row_basis(B_row_int):

    B_row = np.array([[int(B_row_int[i,j]) for j in range(B_row_int.ncols)] for i in range(B_row_int.nrows)], dtype=np.float64)

    B_col = B_row.T
    Q, R = np.linalg.qr(B_col)
    return Q, R

def unimodular_invers_Z(U):

    try:
        import sympy as sp
        M = sp.Matrix([[int(U[i,j]) for j in range(U.ncols)] for i in range(U.nrows)])
        Minv = M.inv() # exact for det ±1
        return np.array(Minv.tolist(), dtype=object)
    except Exception as e:
        raise RuntimeError("Need sympy (or sage) to invert unimodular matrix exactly.") from e


if __name__ == "__main__":
    D = DiscreteGaussianDistributionIntegerSampler(3.19,0)
    DG = dgb_sigma(3.19, 0) 
    print(DG)

    print(f"log_2(4) = {log2(4)}")
    print(f"β = 50 → δ_β = {deltaf(50)}")
    print(f"dimension = 500, β = 120, cost model = MATZOV \n<cost>\n{costf(120, 500, RC.MATZOV)}")
    print(f"dimension = 500, β = 120 → Cost(BKZ-β) = Cost(SVP-(β+{find_k(120, 500)}))")
