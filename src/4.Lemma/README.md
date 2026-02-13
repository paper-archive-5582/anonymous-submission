# 4. Lemma — Dual-HKZ tail volume vs. heuristic

This folder documents an empirical comparison between

- **real (measured)** log tail-volume behavior after HKZ reduction, and  
- a **heuristic expression** used as a surrogate bound in our lemma-level analysis.

All plots referenced below are stored in `Result/`. The experiment driver is `tail_vol_HKZ.py`.

---

## Files

- `tail_vol_HKZ.py`  
  Generates synthetic lattice bases (via a random orthonormal transform and a GSA-style diagonal profile), runs HKZ reduction by calling an external `fplll` binary, and measures a log tail-volume proxy using the QR `R`-diagonal.

- `Result/`
  - `Lemma3_fixed_dim.png`
  - `Lemma3_fixed_k.png`

---

## What is being measured?

### 1) “real” (empirical) log tail volume

Given an HKZ-reduced basis `B_hkz`, the script computes a QR decomposition and uses the diagonal entries `R_ii` to form

- `log_tail_volume = sum_{i=0}^{k-1} -log(R_ii)`

exactly as implemented in the code. 

> Note on indexing/convention.  
> The current implementation uses indices `i = 0..k-1` (i.e., the first `k` diagonal entries).  
> A code comment explicitly notes that, depending on the “tail” convention, one may need to use the **last** `k` entries instead. 

### 2) “heur” (heuristic expression)

The heuristic baseline plotted in red is computed as

- `heu = k * log(sqrt(k/d)) + k * (-log(volume))/d`

---

## Plots (stored in `Result/`)

### A. Fixed dimension, varying k

- Setting: `dim = 60`, `k = 1..30`
- Blue: `real`, Red: `heur`

The plotting loop appears here.
![Fixed dim (d=60), varying k](Result/Lemma3_fixed_dim.png)

### B. Fixed k, varying dimension

- Setting: `k = 10`, `dim = 60..72`
- Blue: `real`, Red: `heur`

The plotting loop appears here.

![Fixed k (k=10), varying dim](Result/Lemma3_fixed_k.png)

---

## Requirements

- Python 3.10+
- `numpy`, `matplotlib`
- External `fplll` binary  
  The script invokes `../local/bin/fplll`; adjust the path for your environment.

## How to run
```python
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
```
