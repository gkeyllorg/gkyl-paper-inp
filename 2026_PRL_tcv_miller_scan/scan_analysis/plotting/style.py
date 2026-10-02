"""
Matplotlib style defaults for scan-analysis plots.

Import this module (or call ``apply()``) before creating figures to get
a consistent look.
"""

import matplotlib.pyplot as plt

_STYLE = 'seaborn-v0_8-whitegrid'
_RC_UPDATES = {"text.usetex": False, "font.size": 14}


def apply():
    """Apply the default plot style."""
    plt.style.use(_STYLE)
    plt.rcParams.update(_RC_UPDATES)


# Auto-apply on first import
apply()
