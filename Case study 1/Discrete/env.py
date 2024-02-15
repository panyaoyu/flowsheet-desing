import numpy as np
import matplotlib.pyplot as plt 
import torch
import os
import copy
import time

from gym import Env
from gym.spaces import Discrete, Box, Tuple, Dict
from gym.utils import seeding

from operations import CSTR, Mixer, Flash_recycle


class Flowsheet(Env):
    def __init__(self, conv, max_iteras):

        # Characteristics of the environment
        self.d_actions = 1 + 9 + 6
        self.conv = conv
        self.max_iteras = max_iteras
        self.actions_list = []

        self.Cao = 1.
        self.To = 600.
        self.Fo = 100.

        self.size_dict = {
            1: (5.5, 5.5),
            2: (5.5, 6.625),
            3: (5.5, 7.75),
            4: (6.625, 5.5),
            5: (6.625, 6.625),
            6: (6.625, 7.75),
            7: (7.75, 5.5),
            8: (7.75, 6.625),
            9: (7.75, 7.75)

        }

        self.q_dict = {
            10: 0.2,
            11: 0.25,
            12: 0.3,
            13: 0.35,
            14: 0.4,
            15: 0.45
        }

        # Flowsheet
        self.flowsheet_dict = {}
        self.info = {}        
        self.avail_actions = np.ones(self.d_actions, )
        self.cstr_count = 0
        self.mixer_count = 0
        self.rf_count = 0


        # Action declaration
        self.action_space = Discrete(self.d_actions)

        # Observation
        self.low = np.zeros((5,))
        self.high = np.ones((5,))

        self.observation_space = Box(low=self.low, high=self.high, dtype=np.float32)
        self.mask_vec = np.ones(self.d_actions, dtype=bool)
        
        
        self.reset()
        self.seed()
    
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    

    def step(self, action):
        Ca, T, F, self.iters, x_prev = self.state
        T, F, self.iters = 700*T, self.Fo*F, self.iters*self.max_iteras
        self.iters += 1
        Ca_prev = copy.copy(Ca)

        # Action decision and rewards

        # --------------------- Mixer -----------------------------
        if action == 0:
            self.mixer_count += 1
            self.avail_actions[0] = 0
            self.flowsheet_dict["M"] = (Ca, T, F)
            self.actions_list.append("M")
            
            mixer = Mixer([Ca, T, F])
            Ca, T, F = mixer.mix()
            self.info[f"M{self.mixer_count}"] = (Ca, T, F)

            cost = -0.1

        
            # --------------------- CSTR --------------------------------
        elif action in range(1, 10):
            self.cstr_count += 1
            self.avail_actions[action] -= 1

            D, H = self.size_dict[action]

            self.flowsheet_dict[f"C{self.cstr_count}"] = (D, H)
            self.actions_list.append(f"C{self.cstr_count}")


            cstr = CSTR([Ca, T, F], D, H)
            Ca, T, F = cstr.steady_state()
            self.info[f"C{self.cstr_count}"] = [(D, H), (Ca, T, F)]

            
            cost = -((D/7.75)**(1.05) + (H/(7.75))**(0.82))/2


        # --------------------- Flash with recycle ----------------------
        elif action in range(10, self.d_actions):
            self.rf_count += 1 
            self.avail_actions[10:] = 0
            self.avail_actions[0] = 1

            q = self.q_dict[action]

            rec = Flash_recycle(q, [Ca, T, F], self.flowsheet_dict)
            Ca, T, F = rec.recycle()
            self.flowsheet_dict.clear()
            self.info[f"R+F{self.rf_count}"] = [q, (Ca, T, F)]
            self.actions_list.append("R+F")

            #fcost = lambda D, H: 101.9*D**(1.066)*H**(0.802)*(2.18)
            cost = -0.5*(1+q)
        
        
        if Ca < Ca_prev and "M" in self.actions_list:
            self.avail_actions[10:] = 1
            self.actions_list.clear()
        
        self.mask_vec = self.action_masks()

        x = (self.Cao - Ca)/self.Cao
        bonus = (x-x_prev)
        

        # Completion and reward
        reward = cost + bonus

        if self.iters >= self.max_iteras:
            self.done = True

            if x < self.conv:
                reward -= 10*(self.conv - x)
                pass
        
        else:
            if x >= self.conv:
                self.done = True
                reward += 0.5*(self.max_iteras - self.iters)        
        
        self.state = np.array([Ca, T/700, F/100, self.iters/10, x], dtype=np.float32)
        
        # Return step information
        return self.state, reward, self.done, self.info



    def render(self):
        for i in self.info:
            print(f"{i}: {self.info[i]}")

    
    def action_masks(self):
        v1 = np.ones((self.d_actions,), dtype=np.int32)*self.avail_actions
        mask_vec = np.where(v1 > 0, 1, 0)
        mask_vec = np.array(mask_vec, dtype=bool)
        return mask_vec



    def reset(self):
        # Reset all instances
        self.iters = 0
        self.mask_vec = np.ones((self.d_actions), dtype=np.bool8)
        self.mask_vec[10:] = False
        self.state = np.array([self.Cao, self.To/700, self.Fo/100, self.iters/10, 0], dtype=np.float32)

        self.avail_actions = np.zeros((self.d_actions), dtype=np.int32)
        self.avail_actions[0] = 1
        self.avail_actions[1:10] = 3
                
        self.flowsheet_dict.clear()
        self.info.clear()
        self.actions_list.clear()
        self.done = False
        self.cstr_count = 0
        self.mixer_count = 0
        self.rf_count = 0
        
        return self.state