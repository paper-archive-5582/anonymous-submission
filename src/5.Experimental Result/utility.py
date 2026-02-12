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
    """
    Compute base-2 logarithm using Sage's log.

    :param x: A Sage/Python numeric value.
    :return: log_2(x).
    """
     return log(x, 2.0)

def cbd_eta(eta : int):
    """
    Kyber-style centered binomial distribution CBD(eta):
        x = sum_{i=1, ..., eta} b_i - sum_{1, ..., eta} b'_i, where b_i, b'_i ~ Bernoulli(1/2)
    
    Returns an integer in [-eta, eta].

    :param eta: CBD parameter controlling the distribution's spread.
    :return: A Sage integer sample from CBD(eta).
    """
    a = sum(secrets.randbits(1) for _ in range(eta))
    b = sum(secrets.randbits(1) for _ in range(eta))

    return ZZ(a - b)

def dgb_sigma(sigma : float, c : float = 0.0):
    """
    Sample from an integer discrete Gaussian centered at c with width sigma
    
    :param sigma: The standard deviation controlling how spread out the discrete Gaussian smaples are around the center.
    :param c: The center of the distribution, i.e., the real-valued shift the samples are concentrated around.
    :return: A Sage integer sample from D_{Z, sigma, c}.
    """
    D = DiscreteGaussianDistributionIntegerSampler(sigma, c=c)
    return ZZ(D())

def deltaf(beta):
    """
    Calculate the root Hermite factor δ_β from block size β.

    - For small β, use a piecewise table (coarse heuristic).
    - For β > 40, use the standard asymptotic approximation.
    
    :param beta: Block size of BKZ algorithm.
    :return: δ_β as a Sage real (RR).
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
    
    This wraps a lattice-estimator cost model (e.g., MATZOV) and attaches:
      - rop/red cost estimates,
      - δ_β (root Hermite factor),
      - β and d metadata.
      
    :param cost_model: Cost model which use for computing the complextiy of BKZ-β or SVP-β.
    :param beta: Block size of BKZ algorithm. In SVP-β, the block size and dimension of lattice is equal.
    :param d: The dimension of lattice.
    :param B: Bit-size of entries.
    :return: A Cost(...) object with fields rop/red/delta/beta/d
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
    :return: The smallest integer k >= 0 satisfying the inequality above.
    """

    BKZ_cost = cost_model(beta, d, B)

    SVP_cost = cost_model(beta, beta, B)

    k = 0
    while SVP_cost < BKZ_cost:
        k +=1

        SVP_cost = cost_model(beta+k, beta+k, B)

    return k

def im_to_numpy(A):
    """
    Convert an fpylll IntegerMatrix to a NumPy array.

    This is useful for interoperability (e.g., QR decomposition, NumPy linear algebra).
    The function attempts to return int64 if values fit safely; otherwise it keeps Python objects.

    :param A: fpylll IntegerMatrix of shape (m, n).
    :return: NumPy array of shape (m, n). dtype is int64 if entries fit, else dtype=object.
    """
    m, n = A.nrows, A.ncols
    X = np.zeros((m,n), dtype=object)
    for i in range(m):
        for j in range(n):
            X[i, j] = int(A[i,j])
    return X.astype(np.int64) if np.max(np.abs(X.astype(object))) < 2 ** 62 else x

def numpy_to_im(X):
    """
    Convert a NumPy array to an fpylll IntegerMatrix.

    :param X: NumPy array of shape (m, n) containing integer-like entries.
    :return: fpylll IntegerMatrix with the same shape and values.
    """ 
    m, n = X.shape
    A = IntegerMatrix(m, n)

    for i in range(m):
        for j in range(n):
            A[i,j] = int(X[i,j])
    return A

def augment_with_identity(A):
    """
    Augment an IntegerMatrix A with an identity block on the right: [A | I_m].

    This is commonly used to track unimodular transformations implicitly, e.g.,
    when performing row operations/reductions on the augmented matrix and then
    extracting the transformation from the right block.

    :param A: fpylll IntegerMatrix of shape (m, n).
    :return: fpylll IntegerMatrix of shape (m, n+m) equal to [A | I_m].
    """
    m, n = A.nrows, A.ncols
    Aug = IntegerMatrix(m, n+m)
    
    for i in range(m):
        for j in range(n):
            Aug[i,j] = A[i,j]
        # Right block I
        Aug[i,n+i] = 1
    return Aug

def split_aug(Aug, left_ncols):
    """
    Split an augmented matrix [Left | Right] into its left and right blocks.

    :param Aug: fpylll IntegerMatrix of shape (m, n_total).
    :param left_ncols: Number of columns to take for the left block.
    :return: (Left, Right) where Left has shape (m, left_ncols) and Right has shape (m, n_total-left_ncols).
    """
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
    """
    Compute QR decomposition from a basis stored as row vectors (fpylll IntegerMatrix).

    We first convert the row-basis to a NumPy array, then transpose to obtain a column-basis,
    and finally compute QR on the column-basis:
        B_col = (B_row)^T = Q R.

    :param B_row_int: fpylll IntegerMatrix whose rows are basis vectors.
    :return: (Q, R) from NumPy QR decomposition of B_col.
    """
    B_row = np.array([[int(B_row_int[i,j]) for j in range(B_row_int.ncols)] for i in range(B_row_int.nrows)], dtype=np.float64)

    B_col = B_row.T
    Q, R = np.linalg.qr(B_col)
    return Q, R

def unimodular_invers_Z(U):
    """
    Compute the exact inverse of a unimodular integer matrix U (det(U)=±1).

    This uses SymPy for exact rational/integer matrix inversion.
    The output is returned as a NumPy array with dtype=object to preserve exactness.

    :param U: fpylll IntegerMatrix assumed to be unimodular (det ±1).
    :return: NumPy array representing U^{-1} exactly (dtype=object).
    :raises RuntimeError: If SymPy is not available or inversion fails.
    """
    try:
        import sympy as sp
        M = sp.Matrix([[int(U[i,j]) for j in range(U.ncols)] for i in range(U.nrows)])
        Minv = M.inv() # exact for det ±1
        return np.array(Minv.tolist(), dtype=object)
    except Exception as e:
        raise RuntimeError("Need sympy (or sage) to invert unimodular matrix exactly.") from e

