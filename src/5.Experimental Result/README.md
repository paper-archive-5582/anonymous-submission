# 5.ExperimentResult

This directory contains the experimental code for:

1) Running a strengthened **primal-embedding** reduction pipeline for LWE  
   (**BKZ → middle dual-HKZ-like preprocessing → terminal tail HKZ**), and  

2) Empirically validating a **chi-square surrogate model** for the normalized tail-projection energy
\[
S \;=\; \|\mathrm{proj}_{U_{\mathrm{tail}}}(t)\|^2 / \sigma_{\mathrm{eff}}^2,
\]
using both CBD- and DG-sampled integer test vectors.

> **Convention:** All bases are stored as **row vectors** (row-basis convention).

---

## Contents

- `compute_final_basis.py`  
  End-to-end pipeline:
  generates an LWE instance, constructs the primal embedding lattice basis, runs LLL + BKZ-β (via external `fplll`),
  then applies a **middle dual-HKZ-like step** and a **terminal tail HKZ** step, and returns the updated basis. 

- `Expermient.py`  
  Validation of the chi-square surrogate:
  samples integer target-like vectors (DG or CBD), projects them onto the **tail subspace** induced by the reduced basis,
  forms \(S\), and compares the **empirical CDF** of \(S\) to the model CDF \(\chi^2_\tau\) (also reports KS distance). 

- `utility.py`  
  Helpers:
  - DG/CBD samplers (`dgb_sigma`, `cbd_eta`),
  - cost/time-matching `find_k` (MATZOV estimator),
  - QR helper for row-basis (`qr_R_from_row_basis`),
  - exact unimodular inverse (`unimodular_invers_Z`),
  - conversion helpers and augmentation utilities.

- `Result/`  
  Experimental plots (including chi-square validation figures shown below).
### CBD test vectors
![Empirical CDF of S(cbd) vs chi-square model](Result/Experiement_Result_cbd.png)

### DG test vectors
![Empirical CDF of S(dg) vs chi-square model](Result/Experiment_Result_dg.png)
---

## Requirements

### Core
- SageMath (the code imports `sage.all` and uses Sage integer utilities).
- Python packages:
  - `numpy`, `matplotlib`
  - `fpylll`
  - `sympy` (exact inversion of unimodular matrices)
  - `mpmath` (regularized incomplete gamma for χ² CDF) 
  - `lattice-estimator` (MATZOV cost model used in `find_k`)

### External binary
- `fplll` is called at `../local/bin/fplll`
- BKZ strategy JSON at `../local/strategies/default.json` 

## How to run

