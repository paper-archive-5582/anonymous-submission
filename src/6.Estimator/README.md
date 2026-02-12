# Lattice Estimators (GSA / ZGSA)

This folder contains two Python scripts that estimate the minimum BKZ block size **β** required for (toy / modeled) primal-embedding style lattice attacks, using either:

- **GSA (Geometric Series Assumption)**: models the Gram–Schmidt spectrum by a global geometric progression.
- **ZGSA (Z-shaped GSA)**: models the spectrum with a *q-limited head* and a *flattened tail* (often observed in q-ary/primal-embedding lattices), yielding a typically less stringent tail success condition than plain GSA.

## Files

- `our_estimator_GSA.py`  
  GSA-based estimator: builds a GSA profile and searches for the smallest **β** that satisfies the modeled success condition.

- `our_estimator_ZGSA.py`  
  ZGSA-based estimator: builds a Z-shaped spectrum/profile and searches for the smallest **β** under the ZGSA-tail determinant / success condition.

## Requirements

- Python 3.10+
- Standard library: `math`, `random`, `time`, `re`, `subprocess` (depending on what you run)

> **Note (SageMath vs pure Python):**  
> If your scripts use Sage types/functions (e.g., `RR`, `RealNumber`, Sage-style `log`), run them inside a Sage environment **or** replace them with standard Python equivalents (`float`, `math.log`, etc.).

## Quick Start

Run the estimators directly:

```bash
python our_estimator_GSA.py
python our_estimator_ZGSA.py
