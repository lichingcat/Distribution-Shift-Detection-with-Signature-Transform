from dataclasses import dataclass
from typing import List, Sequence, Optional, Tuple, Union
import numpy as np
import warnings
from piecewiseAR import PiecewiseARProcess
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class RegimeGARCH:
    """
    A regime with AR(p) mean dynamics and GARCH(1,1) volatility.

    length : number of points in this regime (after burn-in)
    phi    : AR coefficients [phi_1, ..., phi_p]
    mu     : target unconditional mean for the AR part (if stationary)
    trend  : linear drift added per step within this regime
    label  : optional name
    omega, alpha, beta : GARCH(1,1) parameters for h_t = omega + alpha*eps_{t-1}^2 + beta*h_{t-1}
                         Conditions (typical): omega > 0, alpha >= 0, beta >= 0, alpha+beta < 1
    sigma  : (optional) used only as a fallback initial variance if alpha+beta >= 1; otherwise ignored
    """
    length: int
    phi: Sequence[float]
    mu: float = 0.0
    trend: float = 0.0
    label: Optional[str] = None

    # GARCH(1,1)
    omega: float = 0.1
    alpha: float = 0.05
    beta: float = 0.9

    # Fallback/compat
    sigma: float = 1.0

class PiecewiseARGARCH(PiecewiseARProcess):
    """
    Piecewise AR(p) with regime-wise GARCH(1,1) conditional variance.
    """

    def __init__(
        self,
        regimes: List[RegimeGARCH],
        burn_in: int = 200,
        random_state: Optional[Union[int, np.random.Generator]] = None,
        vol_carryover: bool = True,
        var_floor: float = 1e-10,
    ):
        # Validate AR(p) order consistency & stationarity via parent
        super().__init__(regimes=regimes, burn_in=burn_in, random_state=random_state)
        self.vol_carryover = bool(vol_carryover)
        self.var_floor = float(var_floor)

        # Light validation of GARCH params
        for i, r in enumerate(self.regimes):
            if r.omega <= 0 or r.alpha < 0 or r.beta < 0:
                raise ValueError(f"Regime {i}: require omega>0, alpha>=0, beta>=0.")
            if r.alpha + r.beta >= 1:
                warnings.warn(
                    f"Regime {i} (label={r.label}): alpha+beta >= 1 implies non-stationary volatility. "
                    "Simulation will proceed, but unconditional variance does not exist.",
                    RuntimeWarning,
                )

    def _unconditional_variance(self, regime: RegimeGARCH) -> float:
        """Compute the unconditional variance of the GARCH(1,1) process if it exists."""
        if regime.alpha + regime.beta < 1:
            return regime.omega / (1.0 - regime.alpha - regime.beta)
        else:
            return np.nan  # No finite unconditional variance
    
    def sample(
        self,
        return_regimes: bool = True,
        return_vol: bool = False,
        initial_values: Optional[Sequence[float]] = None,
    ):
        n = self.total_length()
        p = self.p

        y = np.zeros(n + self.burn_in + p, dtype=float)
        eps = np.zeros_like(y)
        h = np.zeros_like(y)  # Conditional variances

        init = self.regimes[0].mu if initial_values is None else initial_values
        y[:p] = init

        r0 = self.regimes[0]
        h0 = max(self.var_floor, self._unconditional_variance(r0))
        h[:p] = h0
        eps[:p] = 0.0

        phi0 = np.asarray(r0.phi, dtype=float)
        c0 = r0.mu * (1.0 - phi0.sum())
        for t in range(p, p + self.burn_in):
            # GARCH recursion
            h[t] = np.maximum(r0.omega + r0.alpha * (eps[t - 1] ** 2) + r0.beta * h[t - 1], self.var_floor)
            z = self._rng.normal(0.0, 1.0)
            eps[t] = np.sqrt(h[t]) * z
            prev = y[t - np.arange(1, p + 1)]
            y[t] = c0 + float(phi0 @ prev) + eps[t]
        
        # Main simulation over regimes
        t_cursor = p + self.burn_in
        labels = []

        # Carry-over state from burn-in end
        last_h = h[t_cursor - 1]
        last_eps = eps[t_cursor - 1]

        for ridx, r in enumerate(self.regimes):
            phi = np.asarray(r.phi, dtype=float)
            c = r.mu * (1.0 - phi.sum())

            # Reset or carry variance across regime boundary
            if self.vol_carryover:
                # keep last_h / last_eps
                pass
            else:
                last_h = max(self._uncond_var(r), self.var_floor)
                last_eps = 0.0

            for k in range(r.length):
                t = t_cursor + k
                # GARCH recursion
                h[t] = np.maximum(r.omega + r.alpha * (last_eps ** 2) + r.beta * last_h, self.var_floor)
                z = self._rng.normal(0.0, 1.0)
                eps[t] = np.sqrt(h[t]) * z

                prev = y[t - np.arange(1, p + 1)]
                y[t] = c + r.trend * k + float(phi @ prev) + eps[t]

                # advance state
                last_h = h[t]
                last_eps = eps[t]

            lab = r.label if r.label is not None else f"Regime {len(labels)+1}"
            labels.extend([lab] * r.length)
            t_cursor += r.length

        # Slice out padding + burn-in
        y_out = y[p + self.burn_in : p + self.burn_in + n]
        h_out = h[p + self.burn_in : p + self.burn_in + n]
        labels = np.array(labels)

        if return_regimes and return_vol:
            return y_out, labels, h_out
        if return_regimes:
            return y_out, labels
        if return_vol:
            return y_out, h_out
        return y_out
    

if __name__ == "__main__":
    regimes = [
        RegimeGARCH(length=300, phi=[0.6], mu=0.0, trend=0.0, label="calm",
                    omega=0.02, alpha=0.05, beta=0.90),
        RegimeGARCH(length=300, phi=[0.6], mu=2.0, trend=0.005, label="higher mean + sticky vol",
                    omega=0.05, alpha=0.10, beta=0.85),
        RegimeGARCH(length=300, phi=[0.2], mu=1.0, trend=0.0, label="faster AR decay + high vol",
                    omega=0.10, alpha=0.15, beta=0.80),
    ]
    gen = PiecewiseARGARCH(regimes, burn_in=300, random_state=42, vol_carryover=True)

    y, labs, h = gen.sample(return_regimes=True, return_vol=True)

    

    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    axes[0].plot(y, lw=1)
    for cp in gen.get_change_points():
        axes[0].axvline(cp, ls="--", alpha=0.5)
    axes[0].set_title("Piecewise AR(p) with GARCH(1,1) volatility")

    axes[1].plot(np.sqrt(h), lw=1)  # conditional std dev
    for cp in gen.get_change_points():
        axes[1].axvline(cp, ls="--", alpha=0.5)
    axes[1].set_ylabel("cond. σ_t")
    axes[1].set_xlabel("t")
    plt.tight_layout()
    plt.show()
