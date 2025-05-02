import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange
from kelly_utils import kelly, save_plot, drawdown, tic, toc

# USER SETTINGS

periods = 360
n_years = 30

burn_in = 1000

max_leverage = 10.0
min_leverage = 0.0

f = 0.75

Rebalancing = 12  # Times per year

scale_long = "none"  # options: "none", "same_avg_vol", "vol_targeting"

sharpe = "consant"  # options: "constant", "variable"

annual_ret = 0.05

SR = 0.40  # Expected annualized sharpe ratio.

m = 0.00  # degree of momentum

use_momentum = False
use_TC = False
return_paths = True

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

n_simulations = 10
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
        return_paths=True,
    )
    # Keep only scalar statistics
    results_list.append(
        {
            "sharpe_long": result["sharpe_long"],
            "sharpe_kelly": result["sharpe_kelly"],
            "sharpe_volTarget": result["sharpe_volTarget"],
            "final_wealth_long": result["final_wealth_long"],
            "final_wealth_kelly": result["final_wealth_kelly"],
            "final_wealth_volTarget": result["final_wealth_volTarget"],
            "max_dd_long": result["max_dd_long"],
            "max_dd_kelly": result["max_dd_kelly"],
            "max_dd_volTarget": result["max_dd_volTarget"],
        }
    )
    vol_paths.append(result["volatility_path"])
    wealth_kelly.append(result["wealth_paths"]["kelly"])
    wealth_volT.append(result["wealth_paths"]["vol_target"])
    wealth_long.append(result["wealth_paths"]["buy_and_hold"])

# Volatility
vol_array = np.vstack(vol_paths)  # shape: (n_simulations, time_steps)
avg_vol = vol_array.mean(axis=0)

# Wealth
W_kelly = np.vstack(wealth_kelly)
W_volT = np.vstack(wealth_volT)
W_long = np.vstack(wealth_long)

# Averages
avg_kelly = W_kelly.mean(axis=0)
avg_volT = W_volT.mean(axis=0)
avg_long = W_long.mean(axis=0)

# Apply drawdown to each row (simulation)
DD_kelly = np.array([drawdown(w) for w in W_kelly])
DD_volT = np.array([drawdown(w) for w in W_volT])
DD_long = np.array([drawdown(w) for w in W_long])

avg_dd_kelly = DD_kelly.mean(axis=0)
avg_dd_volT = DD_volT.mean(axis=0)
avg_dd_long = DD_long.mean(axis=0)


# Convert to DataFrame
df = pd.DataFrame(results_list)

mean_results = df.mean().round(4)
table = pd.DataFrame(
    {
        "Buy-and-Hold": {
            "Sharpe": mean_results["sharpe_long"],
            "Final Wealth": mean_results["final_wealth_long"],
            "Max Drawdown": mean_results["max_dd_long"],
        },
        "Kelly": {
            "Sharpe": mean_results["sharpe_kelly"],
            "Final Wealth": mean_results["final_wealth_kelly"],
            "Max Drawdown": mean_results["max_dd_kelly"],
        },
        "Vol Targeting": {
            "Sharpe": mean_results["sharpe_volTarget"],
            "Final Wealth": mean_results["final_wealth_volTarget"],
            "Max Drawdown": mean_results["max_dd_volTarget"],
        },
    }
)
print(table)

## Figure 1
fig = plt.figure(figsize=(10, 5))
plt.plot(avg_vol, label="Average Volatility", color="blue")
plt.title("Volatility Over Time (Monte Carlo Average)")
plt.xlabel("Time")
plt.ylabel("Annualized Volatility")
plt.grid(True)
plt.legend()
plt.tight_layout()
save_plot(fig, fig_number=1)  # Save as figs/fig1.png
plt.close()

## Figure 2
fig = plt.figure(figsize=(10, 5))

# Kelly
plt.plot(avg_kelly, label="Kelly", color="blue")

# Volatility Targeting
plt.plot(avg_volT, label="Vol Targeting", color="orange")

# Buy and Hold
plt.plot(avg_long, label="Buy and Hold", color="green", linestyle="--")

plt.title("Average Wealth Path Over Time (Monte Carlo)")
plt.xlabel("Rebalancing Periods")
plt.ylabel("Wealth")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.yscale("log")

save_plot(fig, fig_number=2)
plt.close()

## Figure 3
fig = plt.figure(figsize=(10, 5))

# Kelly
plt.plot(avg_dd_kelly, label="Kelly", color="blue")

# Vol Targeting
plt.plot(avg_dd_volT, label="Vol Targeting", color="orange")

# Buy and Hold
plt.plot(avg_dd_long, label="Buy and Hold", color="green", linestyle="--")

plt.title("Average Drawdown Path Over Time (Monte Carlo)")
plt.xlabel("Rebalancing Periods")
plt.ylabel("Drawdown")
plt.grid(True)
plt.legend()
plt.tight_layout()

save_plot(fig, fig_number=3)
plt.close()

toc()