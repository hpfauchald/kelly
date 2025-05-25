import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import statsmodels.api as sm
import os

def get_first_and_last_day_in_period(date_list, period='M'):
    """
    Given a Series of datetime64, return the index positions of the first and last
    observation in each period.

    Parameters:
    - date_list: pd.Series of datetime64[ns]
    - period: string ('M' for month, 'Y' for year, etc.)

    Returns:
    - first_idx: integer positions of first date in each period
    - last_idx: integer positions of last date in each period
    """
    # Create a dummy DataFrame for grouping
    df = pd.DataFrame({'date': date_list})
    periods = df['date'].dt.to_period(period)

    # Use integer positions for return values
    first_idx = df.groupby(periods).head(1).index.to_numpy()
    last_idx = df.groupby(periods).tail(1).index.to_numpy()

    return first_idx, last_idx

def aggregate_returns(original_returns, date_list, period='M'):
    """
    Aggregate returns over time by compounding: ∏(1 + r) - 1

    Parameters:
    - original_returns: pd.DataFrame or pd.Series of returns (daily, monthly, etc.)
    - date_list: pd.Series of datetime64[ns], same length as original_returns
    - period: 'M' for monthly, 'Y' for yearly, etc.

    Returns:
    - aggregated_returns: np.ndarray of shape (nPeriods, nAssets)
    """
    # Ensure 2D structure for original_returns
    if isinstance(original_returns, pd.Series):
        original_returns = original_returns.to_frame()

    n_assets = original_returns.shape[1]

    # Get first and last day index per period
    first_idx, last_idx = get_first_and_last_day_in_period(date_list, period)

    n_periods = len(first_idx)
    aggregated_returns = np.zeros((n_periods, n_assets))

    for i in range(n_periods):
        first = first_idx[i]
        last = last_idx[i]

        window = original_returns.iloc[first:last + 1]  # include last row
        aggregated_returns[i, :] = (1 + window).prod(axis=0).values - 1

    return aggregated_returns

def aggregate_variance(original_var, date_list, period='M'):
    """
    Aggregate variance or volatility proxy over time by summing.

    Parameters:
    - original_var: pd.DataFrame or pd.Series of daily variances or RV
    - date_list: pd.Series of datetime64[ns], same length as original_var
    - period: 'M' for monthly, 'Y' for yearly, etc.

    Returns:
    - aggregated_var: np.ndarray of shape (nPeriods, nAssets)
    """
    if isinstance(original_var, pd.Series):
        original_var = original_var.to_frame()

    n_assets = original_var.shape[1]
    first_idx, last_idx = get_first_and_last_day_in_period(date_list, period)

    n_periods = len(first_idx)
    aggregated_var = np.zeros((n_periods, n_assets))

    for i in range(n_periods):
        first = first_idx[i]
        last = last_idx[i]

        window = original_var.iloc[first:last + 1]
        aggregated_var[i, :] = window.sum(axis=0).values

    return aggregated_var


def drawdown(P):
    """
    Compute drawdown (as negative percentage) from a price or wealth series.
    Returns an array of same length as P.
    """
    peak = np.maximum.accumulate(P)
    dd = P / peak - 1.0
    return dd

def stats(R_ts, Rebalancing):
    mean = R_ts.mean() * Rebalancing                # Annualized return
    sd = R_ts.std() * np.sqrt(Rebalancing)          # Annualized standard deviation
    sr = mean / sd                                  # Sharpe ratio
    wealth_path = np.cumprod(1 + R_ts)              # NAV path
    return sr, wealth_path.iloc[-1], drawdown(wealth_path).min()

def fit_har_model(df, rv_col='RV'):
    df = df.copy()

    # Step 1: Compute log RV and lags (local variables only)
    log_RV = np.log(df[rv_col])
    log_RV_d = log_RV.shift(1)
    log_RV_w = log_RV.rolling(window=5, min_periods=5).mean().shift(1)
    log_RV_m = log_RV.rolling(window=22, min_periods=22).mean().shift(1)

    # Step 2: Prepare regression dataset
    har_df = pd.DataFrame({
        'log_RV': log_RV,
        'log_RV_d': log_RV_d,
        'log_RV_w': log_RV_w,
        'log_RV_m': log_RV_m
    }).dropna()

    X = sm.add_constant(har_df[['log_RV_d', 'log_RV_w', 'log_RV_m']])
    y = har_df['log_RV']
    model = sm.OLS(y, X).fit()

    # Step 3: Predict log RV and convert back to variance scale
    full_X = sm.add_constant(pd.DataFrame({
        'log_RV_d': log_RV_d,
        'log_RV_w': log_RV_w,
        'log_RV_m': log_RV_m
    }))
    log_RV_pred = model.predict(full_X).shift(1)
    df['RV_pred'] = np.exp(log_RV_pred)

    return df, model

