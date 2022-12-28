from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
import matplotlib.pyplot as plt





env = "Breakout-v0"
experiments = ["PPO", "ModifiedPPO1f", "ModifiedPPO10f10b10i10r"]
timesteps = 1000_000
log_dirs = ["data/{}/{}".format(env, ex) for ex in experiments]
plot_results(log_dirs, timesteps, results_plotter.X_TIMESTEPS, "{}".format(env))
plt.show()