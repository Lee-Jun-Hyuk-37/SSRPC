# SSPRC

This repository contains implementation of two algorithms STPDP and SSRPC.
The implementation code for each algorithm can be found in STPDP.py and SSRPC.py, and examples of how to use them are available in demo.ipynb.
Detailed explanations and theoretical backgrounds of the algorithms are provided in the [paper](#publication)

- **STPDP (Sum of Trend Power and Detrended Power)** algorithm
enables noise-robust estimation of the mean orbital period for ergodic time series data.

- **SSRPC (State Space Reconstruction with Principal Components)** algorithm
enables noise-robust state space reconstruction.
By utilizing this algorithm, valid, consistent, and noise-robust estimation of maximal Lyapunov exponent is available.

The following figures show noise-robustness, which is the primary advantage of SSRPC.
Despite a significant level of noise, the slope of linear region of the divergence graph which represents maximal Lyapunov exponent remains consistent, in contrast to the conventional method.  

<p float="left">
  <img src="figures/ssrpc.jpg?raw=true" width="49.0%" />
  <img src="figures/conventional.jpg?raw=true" width="49.0%" />
</p>

## Environment

This repository requires NoLiTSA, Numpy, Scipy, Numba, Matplotlib, and Scikit-learn

- NoLiTSA is included as a submodule in this repository. It will be automatically installed when you clone this repository using the command below.

  ```commandline
  git clone --recursive https://github.com/Lee-Jun-Hyuk-37/SSRPC.git
  ```

- For more information about NoLiTSA, refer [here](https://github.com/manu-mannattil/nolitsa/tree/master#installation).

- You can install all libraries except for NoLiTSA using pip or conda. By following the commands below, you can install all the necessary libraries at once.

  ```commandline
  pip install -r requirements.txt
  ```

  or

  ```commandline
  conda create --name SSRPC --file requirements.txt
  conda activate SSRPC
  ```

## Publication

Published in **Chaos, Solitons & Fractals**. [Link](https://www.sciencedirect.com/science/article/pii/S0960077923008172)

## Acknowledgement

I appreciate M. Mannattil for wonderful Github repository, [NoLiTSA](https://github.com/manu-mannattil/nolitsa)  
Thanks to co-authors Il Seung Park and Jooeun Ahn for their contributions to this study.
