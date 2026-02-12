# 5.ExperimentResult

This directory contains the experimental code for:
1) running a strengthened primal-embedding pipeline for LWE (BKZ → middle dual-HKZ-like preprocessing → terminal tail HKZ), and  
2) validating a chi-square surrogate model for normalized tail-projection energy.

> Row-basis convention: bases are stored as row vectors throughout.

---

## Files

- `compute_final_basis.py`  
  End-to-end pipeline: generate LWE sample, build primal embedding basis, run LLL+BKZ-β, apply middle dual-HKZ-like reduction and terminal tail HKZ, and return the updated basis.  
  (See `compute_final_basis()` and the k-selection via time-matching.)  :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}

- `Expermient.py`  
  Validation script for the chi-square surrogate:
  sample DG/CBD integer targets, project onto tail subspace induced by the reduced basis, normalize by σ_eff, and compare ECDF with χ²_τ CDF (incl. KS distance). :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}

- `utility.py`  
  Helpers: sampling (CBD/DG), cost/time-matching `find_k`, QR helper for row-basis, unimodular inverse lifting, conversion helpers, etc. :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5} :contentReference[oaicite:6]{index=6}

---

## Requirements

- SageMath (the code imports `sage.all` and uses Sage integer/linear algebra).  
- Python packages:
  - `numpy`, `matplotlib`
  - `fpylll`
  - `sympy` (exact inverse of unimodular matrices)
  - `mpmath` (for χ² CDF; `gammainc`)
  - `lattice-estimator` (MATZOV cost model used in `find_k`) :contentReference[oaicite:7]{index=7} :contentReference[oaicite:8]{index=8}

- External binary:
  - `fplll` is invoked at `../local/bin/fplll`
  - BKZ strategy JSON at `../local/strategies/default.json` :contentReference[oaicite:9]{index=9} :contentReference[oaicite:10]{index=10}

Recommended layout:

