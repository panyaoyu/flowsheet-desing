import numpy as np
import copy
from gym import Env
from gym.spaces import Discrete, Box, Tuple, Dict
from gym.utils import seeding
from operations import CSTR, Mixer, Flash_recycle



class Flowsheet(Env):
    def __init__(self, conv, max_iteras, D_dims, H_dims):

        # Characteristics of the environment
        self.d_actions = 3
        self.conv = conv
        self.max_iteras = max_iteras
        self.actions_list = []
        self.D_min, self.D_max = D_dims
        self.H_min, self.H_max = H_dims

        self.Cao = 1.
        self.To = 600.
        self.Fo = 100.

        # Flowsheet
        self.flowsheet_dict = {}
        self.info = {}
        self.avail_actions = np.array([1,8,0])
        self.cstr_count = 0
        self.mixer_count = 0
        self.rf_count = 0


        # Action declaration
        self.action_space = Dict({
            "discrete": Discrete(self.d_actions), 
            "continuous": Box(low=np.array([0., 0., 0.]), high=np.array([1., 1., 1.]), dtype=np.float32) 
        })


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

        d_action = action["discrete"]
        c_action = action["continuous"]
        c_action = self.interpolation(np.array(c_action))
        D, H, q = c_action

        # Action decision and rewards

        # --------------------- Mixer -----------------------------
        if d_action == 0:
            self.mixer_count += 1
            self.avail_actions[0] = 0
            self.flowsheet_dict["M"] = (Ca, T, F)
            self.actions_list.append("M")
            
            mixer = Mixer([Ca, T, F])
            Ca, T, F = mixer.mix()
            self.info[f"M{self.mixer_count}"] = (Ca, T, F)

            cost = -0.1


        # --------------------- CSTR --------------------------------
        elif d_action == 1:
            self.cstr_count += 1
            self.avail_actions[1] -= 1
            
            self.flowsheet_dict[f"C{self.cstr_count}"] = (D, H)
            self.actions_list.append(f"C{self.cstr_count}")


            cstr = CSTR([Ca, T, F], D, H)
            Ca, T, F = cstr.steady_state()
            self.info[f"C{self.cstr_count}"] = [(D, H), (Ca, T, F)]

            cost = -((D/self.D_max)**(1.05) + (H/(self.H_max))**(0.82))/2

        
        # --------------------- Flash with recycle ----------------------
        elif d_action == 2:
            self.rf_count += 1 
            self.avail_actions[2] = 0
            self.avail_actions[0] = 1

            rec = Flash_recycle(q, [Ca, T, F], self.flowsheet_dict)
            Ca, T, F = rec.recycle()
            self.flowsheet_dict.clear()
            self.info[f"R+F{self.rf_count}"] = [q, (Ca, T, F)]
            self.actions_list.append("R+F")

            cost = -(1+q)*0.5
        
        
        if Ca < Ca_prev and "M" in self.actions_list:
            self.avail_actions[2] = 1
            self.actions_list.clear()
        
        self.mask_vec = self.action_masks()


        x = (self.Cao - Ca)/self.Cao
        bonus = x-x_prev


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
        
        
        self.state = np.array([Ca, T/700, F/self.Fo, self.iters/10, x], dtype=np.float32)
        
        # Return step information
        return self.state, reward, self.done, self.info

    
    def action_masks(self):
        v1 = np.ones((self.d_actions,), dtype=np.int32)*self.avail_actions
        mask_vec = np.where(v1 > 0, 1, 0)
        mask_vec = np.array(mask_vec, dtype=bool)
        return mask_vec

    
    def render(self):
        for i in self.info:
            print(f"{i}: {self.info[i]}")


    def interpolation(self, c_action):
        D, H, q = c_action

        D = np.interp(D, [0, 1], (self.D_min, self.D_max))
        H = np.interp(H, [0, 1], (self.H_min, self.H_max))
        q = np.interp(q, [0, 1], (0.15, 0.5))

        y =  D, H, q

        return y


    def reset(self):
        # Reset all instances
        self.iters = 0
        self.mask_vec = np.array([1,1,0], dtype=bool)
        self.state = np.array([self.Cao, self.To/700, self.Fo/self.Fo, self.iters/10, 0], dtype=np.float32)
                
        self.flowsheet_dict.clear()
        self.info.clear()
        self.actions_list.clear()
        self.done = False
        self.avail_actions = np.array([1,10,0], dtype=np.int32)
        self.cstr_count = 0
        self.mixer_count = 0
        self.rf_count = 0
        
        return self.state