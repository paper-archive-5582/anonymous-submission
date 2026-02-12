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

# 0. Make LWE sample
def gen_LWE_sample(n, m, q, Xs = "Gaussian", Xe = "Gaussian"):
    """
    Generate the LWE sample(A, b = As + e)
    
    :param n: Dimension of A.
    :param m: The number of samples.
    :param q: The modular of each element of LWE instance.
    :param Xs: Distribution of secret vector. ("Gaussian" or "CBD")
    :param Xe: Distribution of error vector. ("Gaussian" or "CBD")
    :return: Tuple (A, b) where A ∈ Z_q^{m×n} and b ∈ Z_q^{m×1}.
    """

    # Generate A ∈ Z_q^{m×n}, where (A)_{i,j} ~ Uniform
    A = IntegerMatrix(m, n)

    for i in range(m):
        for j in range(n):
            A[i, j] = randint(0, q-1)

    # Gennerate secret vector s ∈ Xs^n, where Xs is Gaussian or centered binomial
    s = IntegerMatrix(n, 1)
    
    if Xs == "Gaussian":
        for i in range(n):
            s[i,0] = dgb_sigma(3.19, 0)
    elif Xs == "CBD":
        for i in range(n):
            s[i,0] = cbd_eta(3)
    
    # Gennerate secret vector e ∈ Xe^m, where Xs ie Gaussian or centered binomial
    e = IntegerMatrix(m, 1)
    
    if Xe == "Gaussian":
        e[i,0] = dgb_sigma(3.19, 0)
    elif Xe == "CBD":
        e[i,0] = cbd_eta(3.19, 0)

    # Compute b = As + e
    b = A*s
    
    for i in range(m):
        b[i,0] = (b[i,0] + e[i,0]) % q

    return (A, b)
        
# 1. Generate BKZ-β reduced basis
def gen_BKZ_reduced_primal_basis(LWEsample, beta, q):
    """
    The function generating a BKZ-β reduced basis from LWE sample.
    
    :param LWEsample: LWE samples (A,b=As + e)
    :param block_size: Block size from BKZ-β algorithm.
    :param modular: The modular of each element of basis.
    :return: BKZ-β reduced basis as fpylll IntegerMatrix of shape (d, d).
    """
    
    A = LWEsample[0]
    b = LWEsample[1]
    m, n = A.nrows, A.ncols
    print(f"Shape of A : {m,n}")
    dim = n + m + 1

    # Construct primal embedding lattice basis
    basis = IntegerMatrix(dim, dim)

    for i in range(n+1):
        basis[i,i] = 1

    for i in range(m):
        basis[n+1+i, 0] = b[i,0]
        basis[n+1+i,n+1+i] = q

        for j in range(n):

            basis[n+1+i, 1+j] = A[i,j]

    basis = matrix(ZZ, basis)
    basis_T = basis.T
    basis_T = IntegerMatrix.from_matrix(basis_T)

    # Make .txt file
    with open("basis.txt", "w") as f:
        f.write("[")
        for i in range(dim):
            f.write("[")
            f.write(" ".join(str(int(basis_T[i,j])) for j in range(dim)) + "]\n")
        f.write("]")

    # Perform LLL before BKZ
    res_lll = subprocess.run([
        "../local/bin/fplll",
        "-a", "lll",
        "basis.txt",
        "-of", "b",
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    
    output_lll = res_lll.stdout.strip()
    rows = re.findall(r'\[([^\]]*)\]', output_lll[1:])
    data = [np.fromstring(row, dtype=np.int64, sep=' ') for row in rows]
    B_lll = np.vstack(data)

    # Make .txt file
    with open("basis_lll.txt", "w") as f:
        f.write("[")
        for i in range(dim):
            f.write("[")
            f.write(" ".join(str(int(B_lll[i,j])) for j in range(dim)) + "]\n")
        f.write("]")

    # Perform BKZ-β to B
    result = subprocess.run([
        "../local/bin/fplll",
        "-a", "bkz",
        "-b", str(beta),
        "basis_lll.txt",
        "-s", "../local/strategies/default.json",
        "-bkzautoabort",
        "-of", "b",
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)

    #err = result.stderr.strip()
    #print(err)
    output = result.stdout.strip()
    rows = re.findall(r'\[([^\]]*)\]', output[1:])
    data = [np.fromstring(row, dtype=np.int64, sep=' ') for row in rows]
    B_bkz = np.vstack(data)

    res = IntegerMatrix(dim, dim)

    for i in range(dim):
        for j in range(dim):
            res[i,j] = B_bkz[i,j]

    return res

from utility import augment_with_identity, split_aug

def HKZ_reduction_with_Aug(A):
    """
    Apply (full-block) BKZ with block size equal to the dimension as a practical HKZ surrogate,
    while tracking the unimodular transformation via augmentation.

    :param A: fpylll IntegerMatrix of shape (m, m). Must be square.
    :return: (A_red, U) where:
             - A_red: reduced left block (m×m),
             - U: unimodular matrix (m×m) such that A_red = U * A (row-operation convention).
    """
    m, n = A.nrows, A.ncols
    assert m == n, "We assume only square of the tail reduced blocks"
    Aug = augment_with_identity(A)
    m_Aug, n_Aug = Aug.nrows, Aug.ncols
    # make .txt file
    with open("basis_HKZ.txt", "w") as f:
        f.write("[")
        for i in range(m_Aug):
            f.write("[")
            f.write(" ".join(str(int(Aug[i,j])) for j in range(n_Aug)) + "]\n")
        f.write("]")

    result = subprocess.run([
        "../local/bin/fplll",
        "-a", "bkz",
        "-b", str(m),
        "basis_HKZ.txt",
        "-s", "../local/strategies/default.json",
        "-bkzautoabort",
        "-of", "b",
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)

    err = result.stderr.strip()
    print(err)
    # Make result numpy matrix
    output = result.stdout.strip()
    rows = re.findall(r'\[([^\]]*)\]', output[1:])
    data = [np.fromstring(row, dtype=np.int64, sep=' ') for row in rows]
    B_hkz = np.vstack(data)
    #print(np.shape(B_hkz))
    res = IntegerMatrix(m_Aug, n_Aug)

    for i in range(m_Aug):
        for j in range(n_Aug):
            res[i,j] = B_hkz[i,j]

    A_red, U = split_aug(res, left_ncols = n)

    return A_red, U

# 0. Set the LWE instance
n = 100
m = 100
q = 257
beta = 40

def compute_final_basis(n, m, q, beta):
    
    """
    Run the full "strengthened primal pipeline" on a generated LWE instance:

      0) Generate LWE sample (A, b).
      1) Build and BKZ-β reduce the primal embedding basis.
      2) Choose k by time-matching (Cost(BKZ-β) ≈ Cost(SVP-(β+k))).
      3) Extract the middle projected block and apply dual-HKZ-like reduction:
         - take a middle t-dimensional block from the R-factor (QR/GSO coordinates),
         - invert to obtain a dual basis block,
         - reduce it with HKZ (via BKZ with full block size),
         - lift the unimodular transform back to update the original BKZ-reduced basis.
      4) Apply tail HKZ reduction on the last t-dimensional tail block and lift back.

    This function returns the updated basis after:
      BKZ-β → middle dual-HKZ preprocessing → terminal tail HKZ.

    :param n: LWE secret dimension (A has n columns).
    :param m: Number of LWE samples (A has m rows).
    :param q: Modulus for the embedding lattice construction.
    :param beta: BKZ block size β for the initial reduction stage.
    :return: Final updated basis (fpylll IntegerMatrix) after the pipeline.
    """
    # 0-1. Generate the LWE sample (A, b=As+e)
    lwe_sample = gen_LWE_sample(n, m, q)
    
    # 0-2. Compute the BKZ-β reduced primal lattice using LWE sample
    B_bkz = gen_BKZ_reduced_primal_basis(lwe_sample, beta, q)
    
    # 0-3. Find k such that COST(BKZ-β) = COST(SVP-(β+k))
    # We set the dimension of primal lattice is n+m+1
    k = find_k(beta, n+m+1)
    
    print("step 0 finish")
    
    # 1. Extract the middle part which is applied the dual HKZ reduction
    Q, R = qr_R_from_row_basis(B_bkz)
    
    t = beta + k
    i0 = n + m + 1 - (2*beta + k) # equivalently i0 = d - (t + k)
    V = R[i0:, i0:i0+t]
    Qm, Rm = np.linalg.qr(V)
    
    M_row_real = Rm.T
    
    # 1-1. Compute inverse matrix of M to make dual HKZ-reduced basis
    D_row_real = np.linalg.inv(M_row_real)
    
    # Integerize with scale factor
    scale_middle = 2 ** 40
    scale_dual=2**40
    scale_tail=2**40
    
    M_int = np.rint(scale_middle * M_row_real).astype(object)
    D_int = np.rint(scale_dual * D_row_real).astype(object)
    
    # Replace the type numpy array to IntegerMatrix in fpylll
    M_IM = numpy_to_im(M_int)
    D_IM = numpy_to_im(D_int)
    
    # 1-2. HKZ-reduction to Dual basis D
    D_red, U = HKZ_reduction_with_Aug(D_IM)
    
    Uinv = unimodular_invers_Z(U)
    
    # 1-3. Updating the BKZ-reduced basis using the result of dual HKZ-reduction
    block = np.array([[int(B_bkz[i0+i,j]) for j in range(n+m+1)] for i in range(t)], dtype=object)
    new_block = (Uinv @ block)
    for i in range(t):
      for j in range(n+m+1):
          B_bkz[i0+i,j] = int(new_block[i,j])
    
    print("step 1 finish")
    
    # 2. Apply HKZ-reduction to Tail part
    Q2, R2 = qr_R_from_row_basis(B_bkz)
    j0 = n + m + 1 - t
    R_tail = R2[j0:, j0:]
    
    # 2-1. Integerize the tail part basis
    Tail_row_real = R_tail.T
    Tail_int = np.rint(scale_tail * Tail_row_real).astype(object)
    Tail_IM = numpy_to_im(Tail_int)
    
    # 2-2. Apply HKZ-reduction to tail part
    Tail_red, V = HKZ_reduction_with_Aug(Tail_IM)
    
    # 2-3. Updating the BKZ-reduved basis using the result of Tail HKZ-reduction
    V_np = np.array([[int(V[i,j]) for j in range(V.ncols)] for i in range(V.nrows)], dtype = object)
    tail_block = np.array([[int(B_bkz[j0+i,j]) for j in range(n+m+1)] for i in range(t)], dtype=object)
    new_tail_block = (V_np @ tail_block)
    for i in range(t):
      for j in range(n+m+1):
          B_bkz[j0+i, j] = int(new_tail_block[i,j])
    
    print("step 2 finish")
    
    return B_bkz

