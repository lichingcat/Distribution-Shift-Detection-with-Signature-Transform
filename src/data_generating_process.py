from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Optional, Tuple, Union
import numpy as np
import pandas as pd
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


class ChangePointARGenerator:
    """
    Piecewise-stationary AR(p) simulator with user-defined change points.

    y_t = c_r + trend_r * j + sum_{i=1}^p phi_{r,i} * y_{t-i} + eps_t,
    where c_r = mu_r * (1 - sum(phi_r)), j is steps since the regime start,
    and eps_t ~ N(0, sigma_r^2) within regime r.

    Features:
    - Multiple regimes with changes in mean (mu), dynamics (phi), volatility (sigma),
      and optional linear trend per regime.
    - Burn-in to reduce dependence on initial conditions.
    - Optional pandas DateTimeIndex.
    - Stationarity checks per regime (warning if violated).

    Notes:
    - If a regime’s phi violates stationarity, the theoretical "mu" is not an
      unconditional mean; we still simulate but warn you.
    """

    def __init__(
        self,
        regimes: List[Regime],
        burn_in: int = 200,
        random_state: Optional[Union[int, np.random.Generator]] = None,
    ):
        if len(regimes) == 0:
            raise ValueError("Provide at least one regime.")
        self.regimes = regimes
        # Ensure consistent AR order across regimes
        orders = {len(r.phi) for r in regimes}
        if len(orders) != 1:
            raise ValueError(f"All regimes must share the same AR order p. Got orders: {orders}")
        self.p = next(iter(orders))
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
    def _is_stationary(phi: np.ndarray) -> bool:
        """Check AR(p) stationarity: roots of 1 - phi1 z - ... - phip z^p must lie outside unit circle."""
        # Polynomial: 1 - phi1*z - phi2*z^2 - ... - phip*z^p
        poly = np.r_[1.0, -phi]
        roots = np.roots(poly)
        return np.all(np.abs(roots) > 1.0)

    def get_change_points(self) -> List[int]:
        """Return regime start indices (0-based, in the generated main series)."""
        cps = []
        t = 0
        for r in self.regimes:
            cps.append(t)
            t += r.length
        return cps  # last point is overall start of each regime

    def total_length(self) -> int:
        return int(sum(r.length for r in self.regimes))

    def _simulate_segment(
        self,
        y: np.ndarray,
        t_start: int,
        r: Regime,
        p: int,
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
        start: Optional[Union[str, pd.Timestamp]] = None,
        freq: Optional[str] = None,
        as_pandas: bool = True,
        initial_level: Optional[float] = None,
    ) -> Union[np.ndarray, pd.Series, Tuple[Union[np.ndarray, pd.Series], pd.Series]]:
        """
        Generate one realization.

        Parameters
        ----------
        return_regimes : if True, also return a series of regime labels per time index.
        start, freq    : if provided, return a pandas Series with DateTimeIndex (start, freq).
        as_pandas      : if True and (start & freq provided), return pandas objects; otherwise NumPy array.
        initial_level  : optional initial value y_{-1}=...=y_{-p}=initial_level; if None, use first regime mu.

        Returns
        -------
        y or (y, labels)
            y: length = total_length() time series.
            labels: regime label per time step (string), helpful for plotting/inspection.
        """
        n = self.total_length()
        p = self.p
        # Allocate with room for burn-in
        y = np.zeros(n + self.burn_in + p, dtype=float)

        # Initialize history y[-1],...,y[-p]
        init = self.regimes[0].mu if initial_level is None else float(initial_level)
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
            # assign labels
            lab = r.label if r.label is not None else f"regime_{len(set(labels))+1}"
            labels.extend([lab] * r.length)
            t_cursor += r.length

        # Strip burn-in and padding
        out = y[p + self.burn_in : p + self.burn_in + n]
        labels = np.array(labels, dtype=object)

        if start is not None and freq is not None and as_pandas:
            idx = pd.date_range(start=pd.to_datetime(start), periods=n, freq=freq)
            y_series = pd.Series(out, index=idx, name="y")
            if return_regimes:
                lab_series = pd.Series(labels, index=idx, name="regime")
                return y_series, lab_series
            return y_series
        else:
            if return_regimes:
                return out, labels
            return out

    @classmethod
    def from_change_points(
        cls,
        change_points: List[int],
        params: List[dict],
        burn_in: int = 200,
        random_state: Optional[Union[int, np.random.Generator]] = None,
    ) -> "ChangePointARGenerator":
        """
        Alternate constructor using change-point indices and parameter dicts.

        change_points: sorted list of 0-based start indices of each regime (in the main series).
                       Must start with 0. Example: [0, 200, 450] (3 regimes).
        params      : list of dicts with keys: 'phi' (Sequence[float]) and optional
                      'mu', 'sigma', 'trend', 'label', and 'length' (ignored if using change_points).
        """
        if len(change_points) < 1 or change_points[0] != 0:
            raise ValueError("change_points must start at 0.")
        if len(params) != len(change_points):
            raise ValueError("params and change_points lengths must match.")

        lengths = []
        for i in range(len(change_points)):
            if i < len(change_points) - 1:
                lengths.append(change_points[i + 1] - change_points[i])
            else:
                # last regime length must be provided in params or we'll raise:
                if "length" not in params[i]:
                    raise ValueError("Provide 'length' for the last regime when using change_points.")
                lengths.append(int(params[i]["length"]))

        regimes = []
        for i, L in enumerate(lengths):
            d = params[i].copy()
            phi = d.pop("phi")
            regimes.append(Regime(length=L, phi=phi, **d))
        return cls(regimes=regimes, burn_in=burn_in, random_state=random_state)


# Example usage
if __name__ == "__main__":
    regimes = [
        Regime(length=200, phi=[0.6],  mu=0.0, sigma=1.0, trend=0.0, label="baseline"),
        Regime(length=200, phi=[0.6],  mu=3.0, sigma=1.0, trend=0.01, label="mean+trend"),
        Regime(length=200, phi=[0.6],  mu=3.0, sigma=3.0, trend=0.01, label="vol_jump"),
        Regime(length=200, phi=[0.2],  mu=1.0, sigma=1.2, trend=-0.005, label="dyn_change"),
    ]
    gen = ChangePointARGenerator(regimes, burn_in=300, random_state=42)

    # Generate with a daily date index
    y, labs = gen.sample(return_regimes=True, start="2024-01-01", freq="D", as_pandas=True)

    # Plot (optional)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    y.plot(ax=ax, lw=1)
    for cp in gen.get_change_points()[1:]:
        ax.axvline(y.index[cp], ls="--", alpha=0.6)
    ax.set_title("Piecewise AR(1) with change points")
    ax.set_xlabel("")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()
