import numpy as np
import time
import os
from pathlib import Path
import pandas as pd

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
    return_paths, 
    tc, 
    expected_return = "knwown", 
    return_vol_relation = "yes", 
    expected_vol = "known"
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
    
    hist_r   = pd.Series(r).rolling(window=2520).mean().to_numpy()

    # 3) Discard burn-in
    p = p[burn_in:] - p[burn_in]
    r = r[burn_in:]
    E_r = E_r[burn_in:]
    momentum = momentum[burn_in:]
    lns = lns[burn_in:]
    hist_r = hist_r[burn_in:]

    # 4) Levels & returns
    P = np.exp(p)
    R = np.exp(r) - 1
    vol = np.exp(lns)

    if expected_return == "known":
        E_R = E_r + 0.5 * vol**2
    elif expected_return == "unknown": 
        E_R = hist_r + 0.5*vol**2

    if expected_vol == "known":
        E_R2 = vol**2
    elif expected_vol == "unknown":
        E_R2 = "SETT INN FUNKSJON FOR VOL ESTIMERING"

    # Weights
    w = np.clip(f * (E_R / E_R2), min_leverage, max_leverage)
    w_volTarget = np.clip(np.mean(vol) / vol, min_leverage, max_leverage)

    # 5) Rebalancing
    step = periods // Rebalancing
    rebalance_points = list(range(step, len(R) - step, step))
    steps = len(rebalance_points)

    if use_TC:
        Rp, Rp_tc = rebalanced_returns(R, w, Rebalancing, periods, tc)
        RvolT, RvolT_tc_ = rebalanced_returns(R, w_volTarget, Rebalancing, periods, tc)
    else:
        Rp = np.zeros(steps)
        RvolT = np.zeros(steps)
        for idx, i in enumerate(rebalance_points):
            this_ret = max((np.prod(1 + R[i : i+step]) - 1),-1)
            Rp[idx] = max(w[i] * this_ret,-1)
            RvolT[idx] = max(w_volTarget[i] * this_ret,-1)

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
        "kelly_weights": w,
        "vol_target_weights": w_volTarget,
        "wealth_paths": {
            "buy_and_hold": np.cumprod(1 + monthly_long),
            "kelly": np.cumprod(1 + Rp),
            "vol_target": np.cumprod(1 + RvolT),
        },
        "returns": {
            "Actual expectation": (E_r + 0.5 * vol**2), 
            "Used expectation": E_R
        }
    })


    return results

def simulate_two_asset_returns(
    periods: int,
    T: int,
    lns: np.ndarray,
    regime: np.ndarray,
    p_eq: np.ndarray,
    p_b: np.ndarray,
    sharpe: str,
    c_eq: float,
    c_b: float,
    m: float,
    annual_ret_eq: float,
    annual_ret_b: float,
    partial_vol: float,
    rho_r: float, 
):
    """
    Simulate two‐asset returns with regime‐switching and momentum.

    Returns:
      p_eq, p_b       : updated price series (length T)
      r_eq, r_b       : return series
      E_r_eq, E_r_b   : expected excess‐return series
      momentum_eq, momentum_b : momentum series (12-month log returns)
    """
    # pre‐allocate
    momentum_eq = np.zeros(T)
    momentum_b  = np.zeros(T)
    E_r_eq      = np.zeros(T)
    E_r_b       = np.zeros(T)
    r_eq        = np.zeros(T)
    r_b         = np.zeros(T)

    for i in range(periods + 1, T):
        sigma = np.exp(lns[i])

        # 12-month log return momentum
        momentum_eq[i] = p_eq[i-2] - p_eq[i-(periods + 1)]
        momentum_b[i]  = p_b[i-2]  - p_b[i-(periods + 1)]

        # expected‐return depending on regime & sharpe setting
        if sharpe == "constant":
            if regime[i] == 0:
                E_r_eq[i] = c_eq * sigma   + m * momentum_eq[i]
                E_r_b[i]  = c_b  * sigma * partial_vol   + m * momentum_b[i]
            else:
                E_r_eq[i] = c_eq * sigma
                E_r_b[i]  = c_b  * sigma * partial_vol
        elif sharpe == "variable":
            if regime[i] == 0:
                E_r_eq[i] = annual_ret_eq/periods + m * momentum_eq[i]
                E_r_b[i]  = annual_ret_b/periods  + m * momentum_b[i]
            else:
                E_r_eq[i] = annual_ret_eq/periods
                E_r_b[i]  = annual_ret_b/periods

        # generate random shocks & returns
        ε1 = np.random.randn()
        ε2 = np.random.randn()
        r_eq[i] = E_r_eq[i] + sigma * ε1
        r_b[i]  = E_r_b[i]  + sigma * partial_vol * (
                     rho_r * ε1 + np.sqrt(1 - rho_r**2) * ε2
                  )

        # update prices
        p_eq[i] = p_eq[i-1] + r_eq[i]
        p_b[i]  = p_b[i-1] + r_b[i]

    return p_eq, p_b, r_eq, r_b, E_r_eq, E_r_b, momentum_eq, momentum_b


def rebalanced_returns(
     R: np.ndarray,
    w: np.ndarray,
    rebalancing: int,
    periods: int = 12,
    tc: float = 0.001
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute rebalanced returns (net of transaction costs) and trading costs
    for a single‐strategy weight series, rebalancing a given number of times per year.

    Args:
        R            : array of log returns, length T
        w            : array of target weights, length T
        rebalancing  : times per year to rebalance (e.g. 12=monthly, 4=quarterly)
        periods      : number of periods in one year (default 12 for monthly data)
        tc           : transaction cost per unit traded (default 0.1%)

    Returns:
        Rp           : np.ndarray of net returns for each rebalance interval
        trade_costs  : np.ndarray of transaction costs at each rebalance
    """
    T = len(R)
    step = periods // rebalancing
    if step < 1:
        raise ValueError(
            f"Rebalancing frequency ({rebalancing}) must be ≤ periods ({periods})."
        )

    # build the rebalance index points: step, 2*step, 3*step, … up to T
    rebalance_points = np.arange(step, T, step)
    n = len(rebalance_points)

    Rp = np.zeros(n)
    trade_costs = np.zeros(n)

    for idx, i in enumerate(rebalance_points):
        # cumulative return over the last interval
        prev_ret = max((np.prod(1 + R[i - step : i])-1).sum(), -1)
        # return over the upcoming interval
        this_ret = max((np.prod(1 + R[i : min(i + step, T)])-1).sum(), -1)

        # “effective” weight just before rebalancing
        w_eff = w[i - 1] * (1 + prev_ret) / (1 + w[i - 1] * prev_ret)

        # compute trade amount and cost
        trade = abs(w[i] - w_eff)
        cost = trade * tc
        trade_costs[idx] = cost

        # net return after paying transaction cost
        Rp[idx] = max(w[i] * this_ret - cost, -1)

    return Rp, trade_costs

import cvxpy as cp

def kelly_no_shorts(mu: np.ndarray, Sigma: np.ndarray):
    """
    Solve max { mu^T w - 0.5 w^T Sigma w }  s.t. w >= 0.
    Returns the optimal w (no further clipping needed).
    """
    n = len(mu)
    w = cp.Variable(n)
    objective = cp.Maximize(mu.T @ w - 0.5 * cp.quad_form(w, Sigma))
    constraints = [w >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver = cp.OSQP, verbose = False)
    return w.value



def MC_func(
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
    scale_long
):
    """
    Run one Monte Carlo path and return metrics for three strategies:
    - buy-and-hold (long)
    - regime-based Kelly
    - volatility-targeting

    Returns dict with sharpe, final_wealth, max_drawdown for each.
    """
    # 1) Simulate
    N = n_years * periods
    T = N + burn_in

    p        = np.zeros(T)
    r        = np.zeros(T)
    E_r      = np.zeros(T)
    momentum = np.zeros(T)

    lns, regime, jumps = simulate_regime_vol_with_jumps(
        T, p_LL, p_HH,
        vol_low_annual, vol_high_annual,
        rho, std_lns,
        jump_prob, jump_size,
        periods
    )

    # 2) Build path
    for i in range(periods+1, T):
        sigma       = np.exp(lns[i])
        momentum[i] = p[i-2] - p[i-(periods+1)]

        if sharpe == "constant":
            E_r[i] = c * sigma + (m * momentum[i] if regime[i] == 0 else 0)
        else:
            base   = annual_ret / periods
            E_r[i] = base + (m * momentum[i] if regime[i] == 0 else 0)

        r[i] = E_r[i] + sigma * np.random.randn()
        p[i] = p[i-1] + r[i]

    # 3) Discard burn-in
    p        = p[burn_in:] - p[burn_in]
    r        = r[burn_in:]
    E_r      = E_r[burn_in:]
    momentum = momentum[burn_in:]
    lns      = lns[burn_in:]

    # 4) Levels & returns
    P    = np.exp(p)
    R    = np.exp(r) - 1
    vol  = np.exp(lns)
    E_R  = E_r + 0.5 * vol**2
    E_R2 = vol**2

    # Weights
    w           = np.clip(f * (E_R / E_R2), min_leverage, max_leverage)
    w_volTarget = np.clip(np.mean(vol) / vol, min_leverage, max_leverage)

    # 5) Rebalancing
    step  = periods // Rebalancing
    steps = n_years * Rebalancing
    Rp      = np.zeros(steps)
    RvolT   = np.zeros(steps)

    for idx, i in enumerate(range(step, n_years*periods, step)):
        prev_ret = R[(i-step):i].sum()
        this_ret = R[i:(i+step)].sum()

        w_eff   = w[i-1]*(1+prev_ret)/(1+w[i-1]*prev_ret)
        trade   = abs(w[i] - w_eff)
        Rp[idx] = w[i]*this_ret - trade*(0.1/100)

        wv_eff      = w_volTarget[i-1]*(1+prev_ret)/(1+w_volTarget[i-1]*prev_ret)
        trade_vol   = abs(w_volTarget[i] - wv_eff)
        RvolT[idx]  = w_volTarget[i]*this_ret - trade_vol*(0.1/100)

    # 6) Buy-and-hold monthly
    monthly_long = P[step::step] / P[:-step:step] - 1

    if scale_long == "same_avg_vol":
        monthly_long *= (Rp.std() / monthly_long.std())
    elif scale_long == "vol_targeting":
        monthly_long *= (Rp.std() / monthly_long.std())

    # 7) Stats
    sl, wl, dl = stats(monthly_long)
    sk, wk, dk = stats(Rp)
    sv, wv, dv = stats(RvolT)

    return {
        "sharpe_long": sl,
        "sharpe_kelly": sk,
        "sharpe_volTarget": sv,
        "final_wealth_long": wl,
        "final_wealth_kelly": wk,
        "final_wealth_volTarget": wv,
        "max_dd_volTarget": dv,
        "max_dd_long": dl,
        "max_dd_kelly": dk
    }


def simulate_kelly(
    n_years,
    periods,
    burn_in,
    f,
    Rebalancing,
    tc_rate,
    SR_eq,
    SR_b,
    partial_vol,
    m,
    sharpe_mode,
    short_constraint,
    annual_ret_eq,
    annual_ret_b,
    rho_r,
    p_LL,
    p_HH,
    vol_low_annual,
    vol_high_annual,
    rho,
    std_lns,
    jump_prob,
    jump_size, 
    max_leverage
):
    """
    Run a single simulation of the Kelly strategy and an equal-weight (EW) strategy for two assets.
    All model parameters are passed explicitly.
    Returns metrics for both Kelly and EW portfolios plus transaction costs.
    """
    # 1) Setup timeline
    N = n_years * periods
    T = N + burn_in

    # 2) Simulate regimes & volatility
    lns, regime, jumps = simulate_regime_vol_with_jumps(
        T, p_LL, p_HH, vol_low_annual, vol_high_annual,
        rho, std_lns, jump_prob, jump_size, periods
    )
    vol = np.exp(lns)

    # 3) Simulate prices & raw returns
    p_eq, p_b, r_eq, r_b, E_r_eq, E_r_b, _, _ = simulate_two_asset_returns(
        periods=periods,
        T=T,
        lns=lns,
        regime=regime,
        p_eq=np.zeros(T),
        p_b=np.zeros(T),
        sharpe=sharpe_mode,
        c_eq=SR_eq/np.sqrt(periods),
        c_b=SR_b/np.sqrt(periods),
        m=m,
        annual_ret_eq=annual_ret_eq,
        annual_ret_b=annual_ret_b,
        partial_vol=partial_vol,
        rho_r=rho_r
    )

    # 4) Discard burn-in
    r_eq = r_eq[burn_in:]
    r_b = r_b[burn_in:]
    vol_trim = vol[burn_in:]

    # 5) Expected returns & variances
    E_R_eq = E_r_eq[burn_in:] + 0.5 * vol_trim**2
    E_R_b = E_r_b[burn_in:] + 0.5 * (vol_trim * partial_vol)**2
    E_R2_eq = vol_trim**2
    E_R2_b = (vol_trim * partial_vol)**2

    # 6) (Kelly) weights over time
    W = np.zeros((N,2))
    for t in range(N):
        E_R = np.array([E_R_eq[t], E_R_b[t]])
        cov = rho_r * np.sqrt(E_R2_eq[t]) * np.sqrt(E_R2_b[t])
        SIGMA = np.array([[E_R2_eq[t], cov],[cov, E_R2_b[t]]])
        raw = (kelly_no_shorts(E_R,SIGMA) if short_constraint else np.linalg.solve(SIGMA,E_R))

        total = np.abs(raw).sum()

        if total > max_leverage:
            W[t] = raw * (max_leverage / total)
        else: 
            W[t] = raw

    # 7) Rebalanced log returns & transaction costs for Kelly
    net_eq, tc_eq = rebalanced_returns(r_eq, W[:,0]*f, Rebalancing, periods, tc_rate)
    net_b , tc_b  = rebalanced_returns(r_b,  W[:,1]*f, Rebalancing, periods, tc_rate)

    Kelly_r = net_eq + net_b
    Kelly_R = np.exp(Kelly_r) - 1
    Kelly_sharpe, Kelly_FinalW, Kelly_maxDD = stats(Kelly_R, Rebalancing)[:3]
    tc_eq_total = tc_eq.sum()
    tc_b_total  = tc_b.sum()

    # 8) Equal-weight (50/50) strategy
    w_eq = np.full(N, 0.5)
    net_eq_EW, tc_eq_EW = rebalanced_returns(r_eq, w_eq, Rebalancing, periods, tc_rate)
    net_b_EW , tc_b_EW  = rebalanced_returns(r_b,  w_eq, Rebalancing, periods, tc_rate)

    EW_r = net_eq_EW + net_b_EW
    EW_R = np.exp(EW_r) - 1
    EW_sharpe, EW_FinalW, EW_maxDD = stats(EW_R, Rebalancing)[:3]
    tc_eq_EW_total = tc_eq_EW.sum()
    tc_b_EW_total  = tc_b_EW.sum()

    return {
        'Kelly_sharpe': Kelly_sharpe,
        'Kelly_FinalW': Kelly_FinalW,
        'Kelly_maxDD':  Kelly_maxDD,
        'tc_eq_Kelly':  tc_eq_total,
        'tc_b_Kelly':   tc_b_total,
        'EW_sharpe':    EW_sharpe,
        'EW_FinalW':    EW_FinalW,
        'EW_maxDD':     EW_maxDD,
        'tc_eq_EW':     tc_eq_EW_total,
        'tc_b_EW':      tc_b_EW_total
    }
