"""Physical and mathematical constants used throughout jaxlibxc.

Translated from libxc maple/util.mpl.
"""

import numpy as np

# Speed of light in atomic units
M_C = 137.0359996287515

# Dimensions (3D by default)
DIMENSIONS = 3

# Wigner-Seitz radius factor: rs = RS_FACTOR / n^(1/3)
RS_FACTOR = (3.0 / (4.0 * np.pi)) ** (1.0 / 3.0)

# Reduced gradient scaling: s = X2S * x, where x = |grad n| / n^(4/3)
X2S = 1.0 / (2.0 * (6.0 * np.pi**2) ** (1.0 / 3.0))
X2S_2D = 1.0 / (2.0 * (4.0 * np.pi) ** 0.5)

# Exchange energy prefactor: E_x^LDA = -X_FACTOR_C * n^(4/3)
X_FACTOR_C = 3.0 / 8.0 * (3.0 / np.pi) ** (1.0 / 3.0) * 4.0 ** (2.0 / 3.0)
X_FACTOR_2D_C = 8.0 / (3.0 * np.sqrt(np.pi))

# Kinetic energy prefactor
K_FACTOR_C = 3.0 / 10.0 * (6.0 * np.pi**2) ** (2.0 / 3.0)

# LDA exchange factor (negative for exchange)
LDA_X_FACTOR = -X_FACTOR_C

# GE and PBE parameters
MU_GE = 10.0 / 81.0
MU_PBE = 0.06672455060314922 * (np.pi**2) / 3.0
KAPPA_PBE = 0.8040

# f_zeta normalization: 2^(4/3) - 2
F_ZETA_DENOM = 2.0 ** (4.0 / 3.0) - 2.0

# PW92 f''(0) normalization constant
FZ20 = 1.709921

# Default thresholds (matching libxc)
DEFAULT_DENS_THRESHOLD = 1e-15
DEFAULT_ZETA_THRESHOLD = 1e-10
DEFAULT_SIGMA_THRESHOLD = 1e-15
DEFAULT_TAU_THRESHOLD = 1e-20
