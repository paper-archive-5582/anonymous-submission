import re
import math
import random
import numpy as np
import subprocess
import matplotlib.pyplot as plt

def random_orthonormal_basis(n: int, k: int | None = None, seed: int | None = None) -> np.ndarray:
    """
    Generate a random orthonormal matrix Q using QR decomposition.

    :param n: Ambient dimension.
    :param k: Number of orthonormal columns to generate.
              If None, returns an n×n random orthogonal matrix.
              If k < n, returns an n×k matrix with orthonormal columns.
    :param seed: RNG seed for reproducibility.
    :return: Orthonormal matrix Q of shape (n, k), satisfying Q^T Q = I_k.
    """
    rng = np.random.default_rng(int(seed))
    k = n if k is None else k

    # A ~ N(0,1)^{n×k}
    A = rng.standard_normal((n, k))

    # QR: A = Q R, with Q having orthonormal columns
    Q, R = np.linalg.qr(A)

    # Fix sign ambiguity so that diag(R) is nonnegative (standard QR convention tweak).
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    Q = Q * signs  
    return Q

def cal_bstars_norm(dimension, block_size, volume):
    """
    Build a GSA-profile Gram–Schmidt length profile (||b_i^*||).

    This models Gram–Schmidt norms as a geometric progression governed by BKZ block size β.

    :param dimension: Lattice dimension d.
    :param block_size: BKZ block size β used in the model.
    :param volume: Lattice determinant det(L) (i.e., fundamental volume).
    :return: List of length d containing the modeled values ||b_i^*|| for i=0..d-1.
    """
    # Gaussian heuristic constant for SVP in dimension β: sqrt(β/(2πe))
    gh_beta = math.sqrt(block_size / (2 * math.pi * math.e))
    
    # Convert GH constant into the "alpha" parameter used by your geometric progression.
    alpha_beta = math.pow(gh_beta, 2/(block_size - 1))

    b_stars_list = []
    for i in range(dimension):
        # GSA profile: ||b_i^*|| ≈ (sqrt(alpha))^{d-1-2i} * det(L)^{1/d}
        bi_star = math.pow(math.sqrt(alpha_beta), dimension - 1 - 2*i) * math.pow(volume, 1/dimension)
        b_stars_list.append(bi_star)

    return b_stars_list

def make_GSA_basis(b_stars_list, dimension):
    """
    Construct a synthetic upper-triangular basis matrix consistent with a target diagonal profile.

    Diagonal: GSA profile values.
    Upper triangle: random values in [-b_i^*/2, b_i^*/2] to mimic size-reduction bounds.

    :param b_stars_list: Target diagonal profile (length = dimension).
    :param dimension: Matrix dimension.
    :return: Upper-triangular matrix of shape (dimension, dimension).
    """
    GSA_basis = np.zeros((dimension, dimension), dtype = np.float64)

    for i in range(dimension):
        GSA_basis[i][i] = b_stars_list[i]
        for j in range(i+1, dimension):
            R_ij = random.uniform(-b_stars_list[i]/2, b_stars_list[i]/2)
            GSA_basis[i][j] = R_ij

    return GSA_basis

def HKZ_reduction_square_matrix(B, scale=2**30):
    """
    Perform HKZ reduction by calling external fplll on a square basis matrix.

    Procedure:
      - Scale and convert to integers.
      - Write basis to a file in fplll bracket format.
      - Run: fplll -a hkz <file> -of b
      - Parse the output basis and rescale back to floats.

    :param B: Input basis matrix of shape (d, d) (interpreted later as a row-basis).
    :param scale: Multiplicative scale to map float entries to integers.
    :return: HKZ-reduced basis matrix of shape (d, d), scaled back to floats.
    """
    dim = np.shape(B)[0]

    # fplll expects basis vectors as rows in the bracket format;
    # we transpose so that we write B_trans[i, :] as one row in the file.
    B_trans = B.T

    # Write scaled integer basis to file
    with open("basis_HKZ.txt", "w") as f:
        f.write("[")
        for i in range(dim):
            f.write("[")
            f.write(" ".join(str(int(B_trans[i,j] * scale)) for j in range(dim)) + "]\n")
        f.write("]")
        
    # Call fplll HKZ reduction
    result = subprocess.run([
        "../local/bin/fplll",
        "-a", "hkz",
        "basis_HKZ.txt",
        "-of", "b",
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    
    # Parse fplll output: [[...]\n [...]\n ...]
    output = result.stdout.strip()
    rows = re.findall(r'\[([^\]]*)\]', output[1:])
    data = [np.fromstring(row, dtype=np.int64, sep=' ') for row in rows]
    B_hkz = np.vstack(data)
    res = np.zeros((dim, dim), dtype=np.float64)

    # Scale back to floats
    for i in range(dim):
        for j in range(dim):
            res[i,j] = B_hkz[i,j] / scale

    return res

def qr_R_from_row_basis(B_row_int, n):
    """
    Compute QR decomposition consistent with treating the input as a row-basis.

    We form B_col = B_row^T, then compute QR(B_col) = Q R. The diagonal of R is
    related to Gram–Schmidt lengths (depending on convention). We also fix signs
    so diag(R) is nonnegative.

    :param B_row_int: Basis matrix (n×n), interpreted as row vectors.
    :param n: Dimension.
    :return: (Q, R) where Q is orthogonal and R is upper-triangular with diag(R) >= 0.
    """
    B_row = np.array([[int(B_row_int[i,j]) for j in range(n)] for i in range(n)], dtype=np.float64)

    B_col = B_row.T
    Q, R = np.linalg.qr(B_col)
    s = np.sign(np.diag(R))
    Q = Q * s
    R = (s[:, None]) * R
    return Q, R

def dual_HKZ_log_tail_volume(B_hkz, n, k):
    """
    Compute log-volume quantities for a 'dual-tail' expression using the QR R-diagonal.

    - log_total_volume := sum_i log(R_ii)
    - log_tail_volume  := sum_{i=0}^{k-1} -log(R_ii)

    Note: This uses indices 0..k-1 (the head of R). Depending on your convention,
    what you call "tail" might correspond to the last k entries instead.

    :param B_hkz: HKZ-reduced basis matrix (n×n).
    :param n: Dimension.
    :param k: Tail length parameter.
    :return: (log_tail_volume, log_total_volume)
    """
    _, R = qr_R_from_row_basis(B_hkz, n)
    
    log_tail_volume = 0
    log_profile = [math.log(R[i,i]) for i in range(n)]
    log_total_volume = np.sum(log_profile)

    # Since we want dual-HKZ tail volume
    for i in range(k):
        log_tail_volume += -math.log(R[i,i])

    return log_tail_volume, log_total_volume

if __name__ == "__main__":
    count = 12

    dim = [60+i for i in range(count+1)]

    hkz_dict = {}
    q = 257
    beta = 30
    k=10

    for d in dim:
        volume = math.pow(q, (d-1)//2)

        Q = random_orthonormal_basis(d, seed=0)

        GSA_profile = cal_bstars_norm(d, beta, volume)
        
        GSA_basis = make_GSA_basis(GSA_profile, d)

        input_basis = Q @ GSA_basis
        
        B_hkz = HKZ_reduction_square_matrix(input_basis)

        
        hkz_dict[d] = {}
        hkz_dict[d]["volume"] = volume
        hkz_dict[d]["Q"] = Q
        hkz_dict[d]["GSA_profile"] = GSA_profile
        hkz_dict[d]["GSA_basis"] = GSA_basis
        hkz_dict[d]["input_basis"] = input_basis
        hkz_dict[d]["B_hkz"] = B_hkz

        print(f"{d} dimension process complete")

    # plotting
    # 1. Fixed dim=60 and k ∈{1, ..., 30}
    x_k = np.linspace(1,30,30)
    dim_k = 60
    y_real_k = [dual_HKZ_log_tail_volume(hkz_dict[dim_k]["B_hkz"], dim_k,i)[0] for i in range(1,31)]
    y_heur_k = [i * math.log(math.sqrt(i/dim_k)) + i * (-math.log(volume)) / d for i in range(1,31)]

    plt.plot(x_k, y_real_k, label="real",marker="^", markersize=3, linewidth=1.8, color="blue")
    plt.plot(x_k, y_heur_k, label="heur",marker="s", markersize=3, linewidth=1.8, color="red")
    
    # grid behind lines
    ax = plt.gca()
    ax.set_axisbelow(True)           
    ax.grid(True, which="major", linestyle="--", alpha=0.5)
    
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", alpha=0.25)
    
    plt.rc('xtick', labelsize=15) 
    plt.rc('ytick', labelsize=15)
    plt.legend(fontsize = 20)
    plt.tight_layout()
    plt.savefig(f"Result from Fixed dim={dim_k}.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # 2. Fixed k=10 and dim ∈ {60, ..., 72}
    x_dim = np.linspace(60,60+count,count+1)
    k_dim = 10
    y_real_d = [dual_HKZ_log_tail_volume(hkz_dict[i]["B_hkz"], i,k_dim)[0] for i in range(60,60+count+1)]
    y_heur_d = [k_dim * math.log(math.sqrt(k_dim/i)) + k_dim * (-math.log(volume)) / i for i in range(60,60+count+1)]

    plt.plot(x_dim,y_real_d, label="real",marker="^", markersize=3, linewidth=1.8, color="blue")
    plt.plot(x_dim,y_heur_d, label="heur",marker="s", markersize=3, linewidth=1.8, color="red")
    
    # grid behind lines
    ax = plt.gca()
    ax.set_axisbelow(True)           
    ax.grid(True, which="major", linestyle="--", alpha=0.5)
    
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", alpha=0.25)
    
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.legend(fontsize = 20)
    plt.tight_layout()
    plt.savefig(f"Result from Fixed k={k_dim}.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
