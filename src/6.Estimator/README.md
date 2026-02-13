# Lattice Estimators (GSA / ZGSA)

This folder contains two Python scripts that estimate the minimum BKZ block size **β**
required for primal-embedding style lattice attacks, using either:

- **GSA (Geometric Series Assumption)**: models the log Gram–Schmidt profile by a global geometric progression.
- **ZGSA (Z-shaped GSA)**: models the log profile with a *q-limited head* and a *flattened tail* (often observed in q-ary / primal-embedding lattices), which can yield a less strict tail success condition than plain GSA.

## Files

- `our_estimator_GSA.py`  
  GSA-based estimator: builds a GSA profile and searches for the smallest **β** that satisfies the modeled success condition.

- `our_estimator_ZGSA.py`  
  ZGSA-based estimator: builds a Z-shaped spectrum/profile and searches for the smallest **β** under the ZGSA tail determinant / success condition.

## Requirements

- Python 3.10+
- Standard library: `math`, `random`, `time`, `re`, `subprocess` (depending on what you run)

> **Note (SageMath vs pure Python):**  
> If your scripts use Sage types/functions (e.g., `RR`, `RealNumber`, Sage-style `log`), run them inside a Sage environment
> **or** replace them with standard Python equivalents (`float`, `math.log`, etc.).

## How to use

Edit the parameters in the `if __name__ == "__main__":` block of each script
(e.g., `n`, `logq`, `sigma`, `beta_min`, `beta_max`) to match your target setting, then run:

```bash
sage our_estimator_GSA.py
sage our_estimator_ZGSA.py
```
## Examples
### Example: ZGSA estimator (n=1024, m=1024, logq=26)

In `our_estimator_ZGSA.py`, set the `__main__` block like this:

```python
import math

if __name__ == "__main__":
    n = 1024
    m = 1024
    logq = 26
    sigma = math.sqrt(2/3)

    print(
        find_min_beta_zgsa(
            n, m, logq, sigma,
            beta_min=300,
            beta_max=950,
            k_max=None,
            require_beta_ge_40=True,
            s_dist="dg",
            sigma_s=math.sqrt(2/3),
        )
    )
```
Run:
```bash
sage our_estimator_ZGSA.py
```
Result:
```bash
>>> {'beta': 312, 'k': 48, 't': 360, 'n': 1024, 'm': 1024, 'd': 2049, 'r': 1024, 'logq': 26.0, 'w': 1929, 'sigma': 0.8165961962005822, 'log_threshold': -0.188764351510164, 'log_base_threshold': -0.25021177778280435, 'log_A_extra': 0.061447426272640215, 'delta_beta': 1.0047172039849062, 'T_BKZ_log2': 117.87349461009258, 'T_HKZ_log2': 116.93510267051546, 'Final_cost': 118.90462747493645}
```

