# mylib-cosmology

Personal Python utilities for cosmological background calculations. The
repository contains flat $\Lambda$CDM, CPL, and linear interacting dark-energy
(LIDE) background models, together with optional Cobaya interfaces.

This is research code maintained primarily for personal use, reproducibility,
and remote backup. The API may change without notice. Results intended for
precision analyses should be independently verified.

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/IffyZh/mylib-cosmology.git
cd mylib-cosmology
python -m pip install -e .
```

For the optional Cobaya integration:

```bash
python -m pip install -e ".[cobaya]"
```

For local tests:

```bash
python -m pip install -e ".[test]"
python -m pytest
```

## Minimal example

```python
import numpy as np
from mylib import LIDE

model = LIDE(
    H_0=70.0,
    O_m0=0.30,
    O_b0=0.0493,
    O_r0=9.27e-5,
    Ksi_1=0.005,
    Ksi_2=-0.001,
    omega=-1.0,
)

z = np.linspace(0.0, 10.0, 200)
a = 1.0 / (1.0 + z)
H = np.sqrt(model.Hsquare(a))
rho_dm = model.Rho_dm(a)
rho_lambda = model.Rho_L(a)
```

## Implemented models

- Flat $\Lambda$CDM background evolution
- CPL background evolution
- Linear interacting dark energy with
  $Q=3H(\xi_1\rho_\Lambda+\xi_2\rho_{\rm dm})$
- Exact analytic LIDE solutions for $\Delta>0$, $\Delta=0$, and $\Delta<0$
- Regular treatment of the $\xi_2=0$ limit
- Cosmological distances and approximate sound-horizon calculations
- Optional Cobaya theory and CMB distance-prior interfaces

## Validation

The tests compare every analytic LIDE branch with the independent matrix
exponential solution. They also cover the $\xi_2=0$ cases and the
$\Lambda$CDM limit.

## License

MIT License. See [LICENSE](LICENSE).

## Contact

Yifei Zhao — yfzhaoiffy@gmail.com
