import re
import math
import random
import numpy as np
import subprocess

def random_orthonormal_basis(n: int, k: int | None = None, seed: int | None = None) -> np.ndarray:

    rng = np.random.default_rng(int(seed))
    k = n if k is None else k
    A = rng.standard_normal((n, k))
    Q, R = np.linalg.qr(A)

    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    Q = Q * signs  
    return Q
def cal_bstars_norm(dimension, block_size, volume):

    gh_beta = math.sqrt(block_size / (2 * math.pi * math.e))
    alpha_beta = math.pow(gh_beta, 2/(block_size - 1))

    b_stars_list = []
    for i in range(dimension):
        bi_star = math.pow(math.sqrt(alpha_beta), dimension - 1 - 2*i) * math.pow(determinant, 1/dimension)
        b_stars_list.append(bi_star)

    return b_stars_list

def make_GSA_basis(b_stars_list, dimension):
    
    GSA_basis = np.zeros((dimension, dimension), dtype = np.float64)

    for i in range(dimension):
        GSA_basis[i][i] = b_stars_list[i]
        for j in range(i+1, dimension):
            R_ij = random.uniform(-b_stars_list[i]/2, b_stars_list[i]/2)
            GSA_basis[i][j] = R_ij

    return GSA_basis

def HKZ_reduction_square_matrix(B, scale=2**30):
    dim = np.shape(B)[0]

    B_trans = B.T

    # basis = IntegerMatrix(dim, dim)

    # for i in range(dim):
    #     for j in range(dim):
    #         basis[i,j] = B[i,j]

    with open("basis_HKZ.txt", "w") as f:
        f.write("[")
        for i in range(dim):
            f.write("[")
            f.write(" ".join(str(int(B_trans[i,j] * scale)) for j in range(dim)) + "]\n")
        f.write("]")

    result = subprocess.run([
        "../local/bin/fplll",
        "-a", "hkz",
        "basis_HKZ.txt",
        "-of", "b",
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)

    output = result.stdout.strip()
    rows = re.findall(r'\[([^\]]*)\]', output[1:])
    data = [np.fromstring(row, dtype=np.int64, sep=' ') for row in rows]
    B_hkz = np.vstack(data)
    res = np.zeros((dim, dim), dtype=np.float64)

    for i in range(dim):
        for j in range(dim):
            res[i,j] = B_hkz[i,j] / scale

    return res

def qr_R_from_row_basis(B_row_int, n):

    B_row = np.array([[int(B_row_int[i,j]) for j in range(n)] for i in range(n)], dtype=np.float64)

    B_col = B_row.T
    Q, R = np.linalg.qr(B_col)
    s = np.sign(np.diag(R))
    Q = Q * s
    R = (s[:, None]) * R
    return Q, R

def dual_HKZ_log_tail_volume(B_hkz, n, k):

    _, R = qr_R_from_row_basis(B_hkz, n)
    
    log_tail_volume = 0
    log_profile = [math.log(R[i,i]) for i in range(n)]
    log_total_volume = np.sum(log_profile)

    for i in range(k):
        log_tail_volume += -math.log(R[i,i])

    return log_tail_volume, log_total_volume

if __name__ == "__main__":
  dim = 60
  q = 257
  beta = 30
  k = 1

  volume = math.pow(q, (d-1)//2)
  
  Q = random_orthonormal_basis(d, seed=0)

  GSA_profile = cal_bstars_norm(d, beta, volume)
  
  GSA_basis = make_GSA_basis(GSA_profile, d)

  input_basis = Q @ GSA_basis
  
  B_hkz = HKZ_reduction_square_matrix(input_basis)

  log_tail_volume, log_total_volume = dual_HKZ_log_tail_volume(B_hkz, d,k)
  heu = k * math.log(math.sqrt(k/d)) + k * (-math.log(volume)) / d

  print(f"Logscale_Tail_Volume:{log_tail_volume}")
  print(f"Heuristic Lemma5 : {heu}")
