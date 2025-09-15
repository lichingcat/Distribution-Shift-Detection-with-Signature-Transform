# etf_sim.py
import numpy as np
import pandas as pd

def trading_minutes(start_date, end_date, tz="America/New_York"):
    days = pd.bdate_range(start_date, end_date, tz=tz)
    intraday = pd.date_range("09:30", "16:00", freq="1min").time
    idx = pd.DatetimeIndex(
        [pd.Timestamp.combine(d.date(), t, tzinfo=d.tz) for d in days for t in intraday]
    )
    return idx

def simulate_index_returns(n, mu_annual=0.08, vol_annual=0.18,
                           vol_phi=0.98, vol_of_vol=0.25, seed=7):
    """
    Minute returns r_t = mu*dt + sigma_t * eps_t
    log(sigma_t) = c + phi*log(sigma_{t-1}) + eta_t  (volatility clustering)
    """
    rng = np.random.default_rng(seed)
    minutes_per_year = 252 * 390
    mu_dt = mu_annual / minutes_per_year

    # initialize around target minute vol
    target_sigma = vol_annual / np.sqrt(minutes_per_year)
    log_sigma = np.empty(n)
    log_sigma[0] = np.log(target_sigma)
    eta = rng.normal(scale=vol_of_vol / np.sqrt(minutes_per_year), size=n)  # tiny minute shocks

    c = (1 - vol_phi) * np.log(target_sigma)
    for t in range(1, n):
        log_sigma[t] = c + vol_phi * log_sigma[t-1] + eta[t]
    sigma = np.exp(log_sigma)
    eps = rng.normal(size=n)
    r = mu_dt + sigma * eps
    return r, sigma

def monthly_dividends(index, annual_yield=0.013, seed=7):
    """
    Simple monthly dividend on last business day of month (paid 1pm timestamp).
    Returns a Series 'div' in dollars per share.
    """
    rng = np.random.default_rng(seed)
    div = pd.Series(0.0, index=index)
    # last business minute of each month ~ 15:59
    months = pd.PeriodIndex(index.tz_convert(None)).to_timestamp("M")
    last_bdays = pd.bdate_range(index.min().date(), index.max().date(), freq="BM")
    last_minutes = []
    for d in last_bdays:
        day_slice = index[index.date == d.date()]
        if len(day_slice):
            last_minutes.append(day_slice[ -1])  # 15:59
    # target annual yield split into 12 payments with small noise
    monthly = annual_yield / 12.0
    for ts in last_minutes:
        div.at[ts] = monthly * (1 + rng.normal(0, 0.1))  # small randomness
    div.clip(lower=0.0, inplace=True)
    return div

def simulate_etf(start_date="2024-01-02", end_date="2024-02-14",
                 start_nav=500.0, expense_ratio=0.0009,
                 premium_phi=0.6, premium_sigma_bps=3.0,
                 vol_annual=0.18, mu_annual=0.08,
                 seed=42):
    """
    Return a DataFrame with: open, high, low, close, volume, nav, premium_pct, te_1d, dividend
    """
    idx = trading_minutes(start_date, end_date)
    n = len(idx)
    rng = np.random.default_rng(seed)

    # Index/NAV path (continuous)
    r_idx, sigma = simulate_index_returns(n, mu_annual=mu_annual, vol_annual=vol_annual, seed=seed)
    minutes_per_year = 252 * 390
    fee_dt = expense_ratio / minutes_per_year
    nav = np.empty(n)
    nav[0] = start_nav
    for t in range(1, n):
        nav[t] = nav[t-1] * np.exp(r_idx[t] - fee_dt)

    # Dividends: reduce NAV on ex-div minute, track cash distribution
    div_series = monthly_dividends(idx, annual_yield=0.013, seed=seed)
    for i, ts in enumerate(idx):
        if div_series.iat[i] > 0:
            nav[i:] -= div_series.iat[i]  # ex-div drop

    # Premium/discount process (mean-reverting to 0, in %)
    prem = np.empty(n)
    prem[0] = 0.0
    prem_sigma = premium_sigma_bps / 10000.0
    for t in range(1, n):
        prem[t] = premium_phi * prem[t-1] + rng.normal(scale=prem_sigma)

    price = nav * (1.0 + prem)  # ETF market price (mid)

    # Within-bar OHLC from micro-moves around mid
    # Use bar range proportional to sigma (volatility) and mean-reverting noise
    # Make high/low symmetric around the mid of open/close.
    open_ = np.empty(n)
    close = np.empty(n)
    high = np.empty(n)
    low = np.empty(n)
    # seed open/close with consecutive mids
    open_[0] = price[0]
    for t in range(1, n):
        open_[t] = price[t-1]
    close[:] = price

    # Intrabar range ~ k * sigma
    k = 6.0  # scale from minute sigma to bar range
    rng_u = rng.uniform(size=n)
    ranges = np.maximum(0.01, k * sigma * open_)  # dollars

    for t in range(n):
        lo = min(open_[t], close[t]) - rng_u[t] * ranges[t] * 0.5
        hi = max(open_[t], close[t]) + (1 - rng_u[t]) * ranges[t] * 0.5
        low[t] = max(0.01, lo)
        high[t] = hi

    # Volume with U-shaped intraday pattern + activity with abs returns
    minutes = np.arange(n) % 390
    u_shape = (np.exp(-((minutes - 10) / 50) ** 2) + np.exp(-((minutes - 380) / 50) ** 2)) + 0.4
    rets = np.diff(np.log(close), prepend=np.log(close[0]))
    vol_level = 2e5 * u_shape * (0.3 + 10 * np.abs(rets))  # baseline + spike on big moves
    volume = rng.poisson(lam=np.maximum(100, vol_level)).astype(int)

    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
        "nav": nav, "premium_pct": prem * 100, "dividend": div_series.values
    }, index=idx)

    # Daily tracking error (close-to-close % difference vs NAV) — computed at session end
    session = df.index.floor("D")
    last_row = df.groupby(session).tail(1)
    te = (last_row["close"].pct_change() - last_row["nav"].pct_change()) * 100  # in %
    df["te_1d"] = np.nan
    df.loc[last_row.index, "te_1d"] = te.values

    return df

if __name__ == "__main__":
    df = simulate_etf(start_date="2025-08-01", end_date="2025-09-12",
                      start_nav=650.0, expense_ratio=0.0009,
                      premium_phi=0.5, premium_sigma_bps=2.0,
                      vol_annual=0.17, mu_annual=0.09, seed=123)

    print(df.head(10))
    print("\nColumns:", list(df.columns))
    # Save if you want
    # df.to_csv("synthetic_spy_like_1min.csv")
