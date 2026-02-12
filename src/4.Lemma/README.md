# 4. Lemma — Dual-HKZ tail volume vs. heuristic

This folder documents an empirical comparison between

- **real (measured)** log tail-volume behavior after HKZ reduction, and  
- a **heuristic expression** used as a surrogate bound in our lemma-level analysis.

All plots referenced below are stored in `Result/`. The experiment driver is `tail_vol_HKZ.py`. :contentReference[oaicite:0]{index=0}

---

## Files

- `tail_vol_HKZ.py`  
  Generates synthetic lattice bases (via a random orthonormal transform and a GSA-style diagonal profile), runs HKZ reduction by calling an external `fplll` binary, and measures a log tail-volume proxy using the QR `R`-diagonal. :contentReference[oaicite:1]{index=1} :contentReference[oaicite:2]{index=2}

- `Result/`
  - `Lemma3_fixed_dim.png`
  - `Lemma3_fixed_k.png`

---

## What is being measured?

### 1) “real” (empirical) log tail volume

Given an HKZ-reduced basis `B_hkz`, the script computes a QR decomposition and uses the diagonal entries `R_ii` to form

- `log_tail_volume = sum_{i=0}^{k-1} -log(R_ii)`

exactly as implemented in the code. :contentReference[oaicite:3]{index=3}

> ⚠️ Note on indexing/convention.  
> The current implementation uses indices `i = 0..k-1` (i.e., the *first* `k` diagonal entries).  
> A code comment explicitly notes that, depending on the “tail” convention, one may need to use the **last** `k` entries instead. :contentReference[oaicite:4]{index=4}

### 2) “heur” (heuristic expression)

The heuristic baseline plotted in red is computed as

- `heu = k * log(sqrt(k/d)) + k * (-log(volume))/d` :contentReference[oaicite:5]{index=5}

---

## Plots (stored in `Result/`)

### A. Fixed dimension, varying k

- Setting: `dim = 60`, `k = 1..30`
- Blue: `real`, Red: `heur`

The plotting loop appears here. :contentReference[oaicite:6]{index=6}

![Fixed dim (d=60), varying k](Result/Lemma3_fixed_dim.png)

### B. Fixed k, varying dimension

- Setting: `k = 10`, `dim = 60..72`
- Blue: `real`, Red: `heur`

The plotting loop appears here. :contentReference[oaicite:7]{index=7}

![Fixed k (k=10), varying dim](Result/Lemma3_fixed_k.png)

---

## How to run

### Requirements

- Python 3.10+
- `numpy`, `matplotlib`
- External `fplll` binary  
  The script invokes `../local/bin/fplll`; adjust the path for your environment. :contentReference[oaicite:8]{index=8}

