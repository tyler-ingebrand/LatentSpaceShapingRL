import os

from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import DQN, PPO
from ModifiedDQN.modified_dqn import ModifiedDQN
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
import matplotlib.pyplot as plt

from ModifiedDQN.modified_ppo import ModifiedPPO

timesteps = 250_000
env_name = "Breakout-v0"


# create log dirs
experiment_name = "ModifiedPPO10f10b10i10r"
log_dir = "data/{}/{}".format(env_name, experiment_name)
os.makedirs(log_dir, exist_ok=True)
env = make_atari_env(env_name, n_envs=4, seed=0, monitor_dir=log_dir)
env = VecFrameStack(env, n_stack=4)
# model = DQN("CnnPolicy", env, verbose=0, buffer_size=20_000, train_freq=(100, "step"))
model = ModifiedPPO("CnnPolicy", env, verbose=0, forward_weight=10.0, inverse_weight=10.0, backward_weight=10.0, reward_weight=10.0)
model.learn(total_timesteps=timesteps, progress_bar=True)

import distutils.dir_util

from_dir = log_dir
to_dir = "{}{}".format(log_dir, "copy")
distutils.dir_util.copy_tree(from_dir, to_dir)

# print estimates and loss
model.test = True
model.n_epochs = 1
model.learn(total_timesteps=10_000, progress_bar=False)

# create log dirs
# experiment_name = "ModifiedDQN2"
# log_dir = "data/{}/{}".format(env_name, experiment_name)
# os.makedirs(log_dir, exist_ok=True)
# env = make_atari_env(env_name, n_envs=4, seed=0, monitor_dir=log_dir)
# env = VecFrameStack(env, n_stack=4)
# model = ModifiedDQN("CnnPolicy", env, verbose=0, buffer_size=20_000, train_freq=(100, "step"))
# model.learn(total_timesteps=timesteps, callback=ProgressBarCallback())


# plot results
#plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "{} {}".format(experiment_name, env_name))
#plt.show()


# run episodes and show policy
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()



