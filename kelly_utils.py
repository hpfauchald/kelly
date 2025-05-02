import numpy as np
import time
import os
from pathlib import Path

def tic():
    global start_time
    start_time = time.perf_counter()

def toc():
    elapsed = time.perf_counter() - start_time
    print(f"Elapsed time: {elapsed:.4f} seconds")

def save_plot(fig, fig_number, folder="figs", prefix="fig", ext="png"):
    Path(folder).mkdir(parents=True, exist_ok=True)  # Create folder if it doesn't exist
    filename = f"{prefix}{fig_number}.{ext}"
    full_path = os.path.join(folder, filename)
    fig.savefig(full_path)
    print(f"Plot saved to {full_path}")

def simulate_regime_vol_with_jumps(
    N,
    p_LL=0.99,
    p_HH=0.80,
    vol_low_annual=0.15,
    vol_high_annual=0.60,
    rho=0.95,
    std_lns=0.05,
    jump_prob=0.03,
    jump_size=0.4,
    periods=360,
):
    """
    Simulate regime-switching log-volatility with occasional jumps.
    Returns (lns, regime, jumps) of length N.
    """
    regime = np.zeros(N, dtype=int)
    lns = np.zeros(N)
    jumps = np.zeros(N)

    mu_low = np.log(vol_low_annual / np.sqrt(periods))
    mu_high = np.log(vol_high_annual / np.sqrt(periods))

    regime[0] = 0
    lns[0] = mu_low

    for t in range(1, N):
        if regime[t - 1] == 0:
            regime[t] = 0 if np.random.rand() < p_LL else 1
        else:
            regime[t] = 1 if np.random.rand() < p_HH else 0

        mu = mu_low if regime[t] == 0 else mu_high
        lns[t] = mu + rho * (lns[t - 1] - mu) + std_lns * np.random.randn()

        if np.random.rand() < jump_prob:
            magnitude = np.random.rand() * jump_size
            if jumps[t - 1] > 0:
                magnitude *= 1.5
            if np.random.rand() < 0.1:
                magnitude += 1.0
            lns[t] += magnitude
            jumps[t] = magnitude

    return lns, regime, jumps


def drawdown(P):
    """
    Compute drawdown (as negative percentage) from a price or wealth series.
    Returns an array of same length as P.
    """
    peak = np.maximum.accumulate(P)
    dd = P / peak - 1.0
    return dd


def rebalancing_returns(R, w, w_volTarget, Rp, RvolT, rebalance_points, step, tc=0.001):
    """
    Compute rebalanced returns for Kelly and volatility-targeting strategies.

    Args:
        R: array of log returns
        w: array of Kelly weights
        w_volTarget: array of volatility-targeting weights
        Rp: preallocated array for Kelly strategy returns (modified in place)
        RvolT: preallocated array for vol-targeting strategy returns (modified in place)
        rebalance_points: list of time indices where rebalancing occurs
        step: number of time periods between rebalances
        tc: transaction cost (default 0.1%)

    Returns:
        None (Rp and RvolT are modified in place)
    """
    for idx, i in enumerate(rebalance_points):
        prev_ret = R[(i - step) : i].sum()
        this_ret = R[i : (i + step)].sum()

        w_eff = w[i - 1] * (1 + prev_ret) / (1 + w[i - 1] * prev_ret)
        trade = abs(w[i] - w_eff)
        Rp[idx] = w[i] * this_ret - trade * tc

        wv_eff = (
            w_volTarget[i - 1] * (1 + prev_ret) / (1 + w_volTarget[i - 1] * prev_ret)
        )
        trade_vol = abs(w_volTarget[i] - wv_eff)
        RvolT[idx] = w_volTarget[i] * this_ret - trade_vol * tc


def stats(R_ts, Rebalancing):
    mean = R_ts.mean() * Rebalancing
    sd = R_ts.std() * np.sqrt(Rebalancing)
    sr = mean / sd
    wealth_path = np.cumprod(1 + R_ts)
    return sr, wealth_path[-1], drawdown(wealth_path).min()


def kelly(
    n_years,
    periods,
    burn_in,
    p_LL,
    p_HH,
    vol_low_annual,
    vol_high_annual,
    rho,
    std_lns,
    jump_prob,
    jump_size,
    c,
    m,
    annual_ret,
    sharpe,
    f,
    min_leverage,
    max_leverage,
    Rebalancing,
    scale_long,
    use_momentum,
    use_TC,
    return_paths
):
    """
    Simulates asset returns under a regime-switching volatility model and evaluates:
    - Buy-and-hold
    - Kelly allocation
    - Volatility-targeting allocation

    Args:
        use_momentum (bool): Whether to add momentum signal in expected returns.
        use_TC (bool): Whether to include transaction costs in rebalancing.
        scale_long (str): Rescale long-only returns to match volatility ("same_avg_vol", etc.).

    Returns:
        dict: Sharpe ratio, final wealth, and max drawdown for each strategy."""

    # 1) Simulate
    N = n_years * periods
    T = N + burn_in

    p = np.zeros(T)
    r = np.zeros(T)
    E_r = np.zeros(T)
    momentum = np.zeros(T)

    lns, regime, jumps = simulate_regime_vol_with_jumps(
        T,
        p_LL,
        p_HH,
        vol_low_annual,
        vol_high_annual,
        rho,
        std_lns,
        jump_prob,
        jump_size,
        periods,
    )

    # 2) Build path
    for i in range(periods + 1, T):
        sigma = np.exp(lns[i])
        momentum[i] = p[i - 2] - p[i - (periods + 1)]

        if sharpe == "constant":
            E_r[i] = c * sigma
        else:
            E_r[i] = annual_ret / periods

        if use_momentum and regime[i] == 0:
            E_r[i] += m * momentum[i]

        r[i] = E_r[i] + sigma * np.random.randn()
        p[i] = p[i - 1] + r[i]

    # 3) Discard burn-in
    p = p[burn_in:] - p[burn_in]
    r = r[burn_in:]
    E_r = E_r[burn_in:]
    momentum = momentum[burn_in:]
    lns = lns[burn_in:]

    # 4) Levels & returns
    P = np.exp(p)
    R = np.exp(r) - 1
    vol = np.exp(lns)
    E_R = E_r + 0.5 * vol**2
    E_R2 = vol**2

    # Weights
    w = np.clip(f * (E_R / E_R2), min_leverage, max_leverage)
    w_volTarget = np.clip(np.mean(vol) / vol, min_leverage, max_leverage)

    # 5) Rebalancing
    step = periods // Rebalancing
    rebalance_points = list(range(step, len(R) - step, step))
    steps = len(rebalance_points)
    Rp = np.zeros(steps)
    RvolT = np.zeros(steps)

    if use_TC:
        rebalancing_returns(R, w, w_volTarget, Rp, RvolT, rebalance_points, step)
    else:
        for idx, i in enumerate(rebalance_points):
            this_ret = R[i : (i + step)].sum()
            Rp[idx] = w[i] * this_ret
            RvolT[idx] = w_volTarget[i] * this_ret

    # 6) Buy-and-hold monthly
    monthly_long = P[step::step] / P[:-step:step] - 1

    if scale_long == "same_avg_vol":
        monthly_long *= Rp.std() / monthly_long.std()
    elif scale_long == "vol_targeting":
        monthly_long *= Rp.std() / monthly_long.std()

    # 7) Stats
    sl, wl, dl = stats(monthly_long, Rebalancing)
    sk, wk, dk = stats(Rp, Rebalancing)
    sv, wv, dv = stats(RvolT, Rebalancing)

    results = {
    "sharpe_long": sl,
    "sharpe_kelly": sk,
    "sharpe_volTarget": sv,
    "final_wealth_long": wl,
    "final_wealth_kelly": wk,
    "final_wealth_volTarget": wv,
    "max_dd_volTarget": dv,
    "max_dd_long": dl,
    "max_dd_kelly": dk,
}

    if return_paths:
        results.update({
            "volatility_path": vol * np.sqrt(periods),  # Annualized
            "log_prices": p,
            "wealth_paths": {
                "buy_and_hold": np.cumprod(1 + monthly_long),
                "kelly": np.cumprod(1 + Rp),
                "vol_target": np.cumprod(1 + RvolT),
            }
        })

    return results

