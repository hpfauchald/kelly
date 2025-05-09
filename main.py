import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange
from kelly_utils import kelly, save_plot, drawdown, bootstrap_ci, bootstrap_distribution, tic, toc
import seaborn as sns

# USER SETTINGS

periods = 252
n_years = 30

burn_in = 1000

max_leverage = 10.0
min_leverage = 0.0

f = 0.75

Rebalancing = 12  # Times per year

scale_long = "none"  # options: "none", "same_avg_vol", "vol_targeting"

sharpe = "variable"  # options: "constant", "variable"

annual_ret = 0.05

SR = 0.40  # Expected annualized sharpe ratio.

m = 0.01  # degree of momentum

use_momentum = False
use_TC = False
return_paths = False

# Parameters for volatility simulations

p_LL = 0.999
p_HH = 0.99
vol_low_annual = 0.15
vol_high_annual = 0.40
rho = 0.95
std_lns = 0.05
jump_prob = 1 / periods
jump_size = 0.30

c = SR / np.sqrt(periods)

## End of user settings
tic()

n_simulations = 1000
results_list = []
vol_paths = []
wealth_kelly = []
wealth_volT = []
wealth_long = []

for _ in trange(n_simulations, desc="Running simulations"):
    result = kelly(
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
        return_paths=False,
    )
    # Keep only scalar statistics
    results_list.append(
        {
            "sharpe_long": result["sharpe_long"],
            "sharpe_kelly": result["sharpe_kelly"],
            "sharpe_volTarget": result["sharpe_volTarget"],
            "mod_sharpe_long": result["mod_sharpe_long"],
            "mod_sharpe_kelly": result["mod_sharpe_kelly"],
            "mod_sharpe_volTarget": result["mod_sharpe_volTarget"],
            "final_wealth_long": result["final_wealth_long"],
            "final_wealth_kelly": result["final_wealth_kelly"],
            "final_wealth_volTarget": result["final_wealth_volTarget"],
            "max_dd_long": result["max_dd_long"],
            "max_dd_kelly": result["max_dd_kelly"],
            "max_dd_volTarget": result["max_dd_volTarget"],
        }
    )

wealth_long = [result["final_wealth_long"] for result in results_list]
wealth_kelly = [result["final_wealth_kelly"] for result in results_list]
wealth_volT = [result["final_wealth_volTarget"] for result in results_list]

dd_long = [result["max_dd_long"] for result in results_list]
dd_kelly = [result["max_dd_kelly"] for result in results_list]
dd_volT = [result["max_dd_volTarget"] for result in results_list]

# Convert to DataFrame
df = pd.DataFrame(results_list)

mean_results = df.mean().round(4)
median_results = df.median().round(4)
table = pd.DataFrame(
    {
        "Buy-and-Hold": {
            "Sharpe": mean_results["sharpe_long"],
            "Modified Sharpe": mean_results["mod_sharpe_long"], 
            "Final Wealth": median_results["final_wealth_long"],
            "Max Drawdown": mean_results["max_dd_long"],
        },
        "Kelly": {
            "Sharpe": mean_results["sharpe_kelly"],
            "Modified Sharpe": mean_results["mod_sharpe_kelly"], 
            "Final Wealth": median_results["final_wealth_kelly"],
            "Max Drawdown": mean_results["max_dd_kelly"],
        },
        "Vol Targeting": {
            "Sharpe": mean_results["sharpe_volTarget"],
            "Modified Sharpe": mean_results["mod_sharpe_volTarget"], 
            "Final Wealth": median_results["final_wealth_volTarget"],
            "Max Drawdown": mean_results["max_dd_volTarget"],
        },
    }
)
print(table)

# Generate bootstrap distributions
bootstrap_long = bootstrap_distribution(dd_long, stat_func=np.mean)
bootstrap_kelly = bootstrap_distribution(dd_kelly, stat_func=np.mean)
bootstrap_volT = bootstrap_distribution(dd_volT, stat_func=np.mean)

# Combine for seaborn
boot_data = [bootstrap_long, bootstrap_kelly, bootstrap_volT]
labels = ["Buy-and-Hold", "Kelly", "Vol Targeting"]

plt.figure(figsize=(12, 6))

# Standard boxplot without adjusting whiskers
sns.boxplot(data=boot_data, showfliers=True)
plt.xticks([0, 1, 2], labels)
plt.title("Bootstrapped Wealth Distributions")
plt.ylabel("Final Wealth")
#plt.yscale("log")
plt.show()

toc()