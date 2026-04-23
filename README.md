# SSRPC

Reference implementation of two noise-robust algorithms for ergodic
time-series analysis:

- **STPDP (Sum of Trend Power and Detrended Power)** — noise-robust
  estimator of the mean orbital period.
- **SSRPC (State Space Reconstruction with Principal Components)** —
  noise-robust state space reconstruction that enables consistent
  estimation of the maximal Lyapunov exponent even at high noise levels.

See the accompanying [paper](#citation) for the full theory.

The figures below illustrate the key advantage of SSRPC: the slope of
the divergence curve (the MLE) stays consistent at heavy noise levels,
whereas the conventional method does not.

<p float="left">
  <img src="figures/ssrpc.jpg?raw=true" width="49.0%" />
  <img src="figures/conventional.jpg?raw=true" width="49.0%" />
</p>

## Installation

The repository includes [NoLiTSA](https://github.com/manu-mannattil/nolitsa)
as a git submodule. Clone recursively so it is pulled in:

```bash
git clone --recursive https://github.com/Lee-Jun-Hyuk-37/SSRPC.git
cd SSRPC
```

Then create a Python environment. With conda:

```bash
conda create -n ssrpc python=3.10
conda activate ssrpc
pip install -e .
```

Or with a plain virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

`pip install -e .` installs `numpy`, `scipy`, `numba`, `matplotlib`, and
`scikit-learn`, then registers the `ssrpc` package in editable mode.

## Usage

### One-call functional API

```python
import numpy as np
import ssrpc

# `data` is a 1-D scalar time series with sampling period `sample`.
result = ssrpc.estimate_mle(
    data,
    sample=0.01,
    maxt=500,
    linear_region=(0, 200),   # slope of divergence over these indices
)

print(result.omega)       # mean orbital period (STPDP)
print(result.dim)         # retained principal-component count
print(result.mle)         # maximal Lyapunov exponent
print(result.divergence)  # ndarray of length maxt
print(result.time)        # matching time axis
```

`estimate_mle` auto-estimates `omega` when it is not supplied. Pass
`omega=...` explicitly to reuse a previously computed value, and
`dimension=...` to skip the STPDP validity test and keep exactly that
many leading components.

### Class-based API

The original class-based API is preserved and can be mixed with the
functional helpers:

```python
from ssrpc import STPDP, SSRPC

omega = STPDP(data, max_length=500).calculate_omega()

model = SSRPC(data, sample=0.01, omega=omega)
model.reconstruct()
model.calculate_divergence(maxt=500)
model.plot_divergence(expected=1.5)
lyap = model.mle((0, 200))
```

### Backwards compatibility

Scripts that used `from SSRPC import SSRPC` / `from STPDP import STPDP`
continue to work — the top-level modules are re-exporting shims onto the
new `ssrpc` package.

## Notes

- Importing `ssrpc` (or either shim) transparently patches a
  memory-leak in NoLiTSA's `@jit`-compiled distance functions. The
  replacements are bit-for-bit equivalent to the upstream versions and
  remove hundreds of MB of leaked memory per MLE run.
- `SSRPC.calculate_divergence` and `SSRPC.mle` now raise `RuntimeError`
  if called before the prerequisite step; the original `print`-only
  warnings were easy to miss.

## Citation

```bibtex
@article{lee2023ssrpc,
  title={Noise-robust estimation of the maximal Lyapunov exponent based on state space reconstruction with principal components},
  author={Lee, Jun Hyuk and Park, Il Seung and Ahn, Jooeun},
  journal={Chaos, Solitons \& Fractals},
  volume={174},
  pages={113916},
  year={2023},
  publisher={Elsevier}
}
```

Published in **Chaos, Solitons & Fractals**. [Link](https://www.sciencedirect.com/science/article/pii/S0960077923008172)

## Acknowledgement

I appreciate M. Mannattil for the wonderful [NoLiTSA](https://github.com/manu-mannattil/nolitsa) repository. 
Thanks to co-authors Il Seung Park and Jooeun Ahn for their contributions to this study.
