# LatentSpaceShapingRL

This project was designed to test if enforcing certain properties on the latent space of a value function would improve data efficiency.
I took inspiration from MuZero, which uses a forward dynamics model and a reward model to improve learning along with MCTS. These models are with respect to the latent
state, not the observation. Therefore, their latent space is guranteed to be useful for predicting forward dynamics and reward, since it is part of the loss function,
which is not typically true in RL. RL only requires the latent space to predict value.

This begs the question, does enforcing the latent space to contain the information for forward dynamics improve learning on its own?

To test this, I modified DQN from stable baselines 3 to include various dynamics models. These models are based on the latent state space, which is produced from a CNN. 
The models ensure that the latent space is consistent with dynamics, IE you can use the latent space and action to predict the next latent space (forward dynamics). I 
also include backward dynamics (s',a) -> s and inverse dynamics (s,s') -> a along with a learned reward function. 

I use these models for nothing except modifying the loss function. Therefore, the latent space is enforced to contain the information required for dynamics predictions. 

## Results


![image](https://user-images.githubusercontent.com/105821676/209851103-4fb71d35-1ba6-4ee1-b2e6-df2d1fec1286.png)

In the above graph, I compare DQN on Breakout-v0 for the following 3 variations:
- No latent space augmentation,  IE normal DQN
- Augment latent space with forward dynamics
- Augment latent space with forward, backward, inverse, and reward function. 

Notably, there is no noticable difference between the three. Other experiments varying the hyperparameters found the same results. Therefore, I conclude incorporating 
latent space requirements via dynamics models does not improve learning. The dynamics models must actually be used in order for the benefit to appear. 
