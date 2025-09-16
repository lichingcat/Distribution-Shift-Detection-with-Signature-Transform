from dataclasses import dataclass
from typing import List, Sequence, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

@dataclass
class Regime:
    """
    A single regime (segment) in a piecewise AR(p) process.

    length : number of points in this regime (after burn-in).
    phi    : AR coefficients [phi_1, ..., phi_p].
    mu     : target unconditional mean within this regime (if stationary).
             Implemented via intercept c = mu*(1 - sum(phi)).
    sigma  : innovation std for white noise in this regime.
    trend  : deterministic linear trend added per step since regime start.
             If trend = 0.01, mean drifts +0.01 each step within the regime.
    label  : optional name for bookkeeping.
    """
    length: int
    phi: Sequence[float]
    mu: float = 0.0
    sigma: float = 1.0
    trend: float = 0.0
    label: Optional[str] = None

class PiecewiseARProcess:
    def __init__(
        self, 
        regimes: List[Regime],
        burn_in: int = 200,
        random_state: Optional[Union[int, np.random.Generator]] = None
    ):  
        if len(regimes) == 0:
            raise ValueError("At least one regime must be specified.")
        self.regimes = regimes
        orders = set(len(r.phi) for r in regimes)
        # Ensure consistent AR order across regimes
        orders = {len(r.phi) for r in regimes}
        if len(orders) != 1:
            raise ValueError(f"All regimes must share the same AR order p. Got orders: {orders}")
        self.p = len(regimes[0].phi)  # AR order
        self.burn_in = max(0, int(burn_in))
        self._rng = np.random.default_rng(random_state) if not isinstance(random_state, np.random.Generator) else random_state
        # Stationarity checks (warn only)
        for idx, r in enumerate(self.regimes):
            if not self._is_stationary(np.asarray(r.phi, dtype=float)):
                warnings.warn(
                    f"Regime {idx} (label={r.label}) AR coefficients appear non-stationary. "
                    "Simulation will proceed, but 'mu' won’t be an unconditional mean.",
                    RuntimeWarning,
                )

    @staticmethod
    def _is_stationary(phi: Sequence[float]) -> bool:
        """Check if AR coefficients correspond to a stationary process."""
        p = len(phi)
        if p == 0:
            return True  # AR(0) is white noise, hence stationary
        # Characteristic polynomial coefficients
        coeffs = [1.0] + [-a for a in phi]
        roots = np.roots(coeffs)
        return np.all(np.abs(roots) > 1.0)
    
    def get_change_points(self) -> List[int]:
        """Get the indices of change points (end of each regime)."""
        change_points = []
        cumulative_length = 0
        for regime in self.regimes[:-1]:  # Exclude last regime
            cumulative_length += regime.length
            change_points.append(cumulative_length)
        return change_points
    
    def total_length(self) -> int:
        """Total length of the generated time series (excluding burn-in)."""
        return sum(r.length for r in self.regimes)
    
    def _simulate_segment(
        self, 
        y: np.ndarray,
        t_start: int,
        r: Regime, 
        p: int
    ) -> None:
        """Simulate one regime into y starting at index t_start, for r.length steps."""
        phi = np.asarray(r.phi, dtype=float)
        c = r.mu * (1.0 - phi.sum())
        for k in range(r.length):
            t = t_start + k
            # previous p values:
            prev = y[t - np.arange(1, p + 1)]
            eps = self._rng.normal(0.0, r.sigma)
            y[t] = c + r.trend * k + float(phi @ prev) + eps
    def sample(
        self,
        return_regimes: bool = True,
        initial_level: Optional[float] = None
    ):
        n = self.total_length()
        p = self.p # AR order
        # Allocate with room for burn-in
        y = np.zeros(n + self.burn_in + p, dtype=float)
        # Initialize history
        init = self.regimes[0].mu if initial_level is None else initial_level
        y[:p] = init

        # Burn-in using first regime (trend disabled during burn-in for stability)
        r0 = self.regimes[0]
        phi0 = np.asarray(r0.phi, dtype=float)
        c0 = r0.mu * (1.0 - phi0.sum())
        for t in range(p, p + self.burn_in):
            prev = y[t - np.arange(1, p + 1)]
            eps = self._rng.normal(0.0, r0.sigma)
            # no trend during burn-in
            y[t] = c0 + float(phi0 @ prev) + eps

        # Main simulation by segments
        t_cursor = p + self.burn_in
        labels = []
        for r in self.regimes:
            self._simulate_segment(y, t_cursor, r, p)
            label = r.label if r.label is not None else f"Regime {len(labels)+1}"
            labels.extend([label] * r.length)
            t_cursor += r.length
        
        # Strip burn-in and initial p values
        out = y[p + self.burn_in: p + self.burn_in + n]
        labels = np.array(labels)  # in case burn-in truncated some regimes


        if return_regimes:
            return out, labels
        
        return out
    
# Example usage
if __name__ == "__main__":
    regimes = [
        Regime(length=200, phi=[0.6],  mu=0.0, sigma=1.0, trend=0.0, label="baseline"),
        Regime(length=200, phi=[0.6],  mu=3.0, sigma=1.0, trend=0.01, label="mean+trend"),
        Regime(length=200, phi=[0.6],  mu=3.0, sigma=3.0, trend=0.01, label="vol_jump"),
        Regime(length=200, phi=[0.2],  mu=1.0, sigma=1.2, trend=-0.005, label="dyn_change"),
    ]
    gen = PiecewiseARProcess(regimes, burn_in=300, random_state=42)

    # Generate with a daily date index
    series, labs = gen.sample(return_regimes=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    plt.plot(series)

    for cp in gen.get_change_points():
        ax.axvline(cp, ls="--", alpha=0.6)
    ax.set_title("Piecewise AR(1) with change points")
    ax.set_xlabel("")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig("figures/piecewise_ar.png")
    plt.show()
