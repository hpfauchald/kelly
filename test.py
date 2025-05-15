import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from kelly_utils import kelly, tic, toc

# USER SETTINGS
periods = 252
n_years = 30
burn_in = periods * 30
max_leverage = 10.0
min_leverage = 0.0
f = 1
Rebalancing = 252
scale_long = "none"
sharpe = "constant"
annual_ret = 0.06
expected_return = "known"
return_vol_relation = False
expected_vol = "known"
estimation_period = 30
deviation = 0.00
rebalance_fraction = 1.0
SR = 0.42
m = 0.00
tc = 0.000
use_momentum = False
return_paths = True

p_LL = 0.995
p_HH = 0.99
vol_low_annual = 0.12
vol_high_annual = 0.3
rho = 0.95
std_lns = 0.05
jumps_per_year = 0.5
jump_prob = 1 - np.exp(-jumps_per_year * (1 / periods))
jump_size = 0.30
c = SR / np.sqrt(periods)

n_sims = 1000
output_file = "simulation_summary.xlsx"

def run_simulation(sim):
    res = kelly(
        n_years, periods, burn_in, p_LL, p_HH, vol_low_annual, vol_high_annual, rho, std_lns, jump_prob, jump_size,
        c, m, annual_ret, sharpe, f, min_leverage, max_leverage, Rebalancing,
        scale_long, use_momentum, return_paths, tc, estimation_period, deviation, rebalance_fraction,
        expected_return, return_vol_relation, expected_vol
    )
    
    # Collect metrics
    return {
        'sim': sim,
        'kelly_final_wealth': res['final_wealth_kelly'],
        'bh_final_wealth':    res['final_wealth_long'],
        'vol_final_wealth':   res['final_wealth_volTarget'],
        'kelly_log_wealth': np.log(res['final_wealth_kelly']),
        'bh_log_wealth':    np.log(res['final_wealth_long']),   
        'vol_log_wealth':   np.log(res['final_wealth_volTarget']),
        'kelly_max_dd':     res['max_dd_kelly'],
        'bh_max_dd':        res['max_dd_long'],
        'vol_max_dd':       res['max_dd_volTarget'],
        'kelly_sharpe':     res['sharpe_kelly'],
        'bh_sharpe':        res['sharpe_long'],
        'vol_sharpe':       res['sharpe_volTarget'],
        "mod_sharpe_long":      res["mod_sharpe_long"],
        "mod_sharpe_kelly":     res["mod_sharpe_kelly"],
        "mod_sharpe_volTarget": res["mod_sharpe_volTarget"],
        'kelly weights':        np.mean(res['kelly_weights']),
        'vol_target_weights':   np.mean(res['vol_target_weights']),
        'Mean volatility': res['volatility_path'].mean()
    }

if __name__ == "__main__":
    tic()

    # Use tqdm with joblib
    with tqdm_joblib(tqdm(total=n_sims, desc="Running simulations", ncols=200)) as progress_bar:
        results = Parallel(n_jobs=6)(
            delayed(run_simulation)(sim) for sim in range(n_sims)
        )

    toc()

    # Save to Excel
    data = pd.DataFrame(results).set_index("sim")
    data.describe().round(4).to_excel(output_file)
    print(f"Table saved to {output_file}")
