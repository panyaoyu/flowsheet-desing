import numpy as np
import matplotlib.pyplot as plt 
import torch
import os
import copy
import time

from agent import PPO
from env import Flowsheet
from Simulation import *



cwd = os.getcwd()
sim = Simulation("DME_prod.bkp", cwd)


env_kwargs = {
        "sim": sim,
        "pure": 0.99, 
        "max_iter": 15, 
        "inlet_specs": [25.0, 1, {"DME": 0, "WATER": 0.2*261.5, "METHANOL": 0.8*261.5}]
    }

env = Flowsheet(**env_kwargs)

   

    # Hyperparameters
kwargs = {
    "state_dim": env.observation_space.shape[0], 
    "action_dim": env.action_space.n, 
    "env_with_Dead": True,
    "gamma": 0.99, 
    "gae_lambda": 0.95, 
    "policy_clip": 0.2, 
    "n_epochs": 10, 
    "net_width": 64, 
    "lr": 2.5e-4, 
    "l2_reg": 0.5, 
    "batch_size": 64, 
    "entropy_coef": 0.025,
    "adv_normalization": True, 
    "entropy_coef_decay": 0.9
}


model = PPO(**kwargs)

model.load_best()
scores = []

for i in range(3):
    (obs, sin) = env.reset()
    mask_vec = env.action_masks(sin, True)
    actions = []
    score = 0
    while True:
        action, probs = model.evaluate(obs, mask_vec)
        obs, reward, done, info, sin = env.step(action, sin)
        mask_vec = env.action_masks(sin)
        score += reward
        actions.append(action)

        if done:
            print(f"Done, points: {score}")
            env.render()
            break
    
    scores.append(score)

print(f"Mean score: {np.mean(scores)}")
sim.CloseAspen()