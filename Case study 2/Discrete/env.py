import os
import numpy as np
import time
from Simulation import *
import copy
from gym import Env
from gym.spaces import Discrete, Box, Dict
from gym.utils import seeding

class Flowsheet(Env):
    def __init__(self, sim, pure, max_iter, inlet_specs):

        # Establish connection with ASPEN
        self.sim = sim

        # Characteristics of the environment
        self.d_actions = 2 + 6 + 18 + 9 + 27 + 27 + 81
        self.pure = pure
        self.max_iter = max_iter
        self.iter = 0
        self.actions_list = []
        self.water_pure = False
        self.dme_pure = False
        self.dme_extra_added = False
        self.value_step = "pre"


        # Declare the initial flowrate conditions
        self.inlet_specs = inlet_specs
        self.Cao = self.inlet_specs[2]["METHANOL"]


        # Flowsheet
        self.info = {}
        self.avail_actions = np.zeros(self.d_actions, )


        # Actions dictionaries

        # Heater
        self.heaters = {2: 150, 3: 275, 4: 400}

        # Cooler
        self.coolers = {5: 5, 6: 25, 7: 50}

        # PFR
        self.pfrs = {8: (0.5, 6.5), 9: (0.5, 9.25), 10: (0.5, 12),
                11: (2, 6.5), 12: (2, 9.25), 13: (2, 12),
                14: (3.5, 6.5), 15: (3.5, 9.25), 16: (3.5, 12)}
        
        # Adiabatic PFR
        self.apfrs = {17: (0.5, 6.5), 18: (0.5, 9.25), 19: (0.5, 12),
                20: (2, 6.5), 21: (2, 9.25), 22: (2, 12),
                23: (3.5, 6.5), 24: (3.5, 9.25), 25: (3.5, 12)}
        
        # Column 
        self.columns = {26: (15, 80), 27: (15, 85), 28: (15, 95),
                   29: (20, 80), 30: (20, 85), 31: (20, 95),
                   32: (25, 80), 33: (25, 85), 34: (25, 95)}
        
        # Column with recycle
        self.columns_r = {35: (15, 20, 0.7), 36: (15, 20, 0.85), 37: (15, 20, 0.95),
                     38: (15, 40, 0.7), 39: (15, 40, 0.85), 40: (15, 40, 0.95),
                     41: (15, 60, 0.7), 42: (15, 60, 0.85), 43: (15, 60, 0.95),
                     44: (20, 20, 0.7), 45: (20, 20, 0.85), 46: (20, 20, 0.95),
                     47: (20, 40, 0.7), 48: (20, 40, 0.85), 49: (20, 40, 0.95),
                     50: (20, 60, 0.7), 51: (20, 60, 0.85), 52: (20, 60, 0.95),
                     53: (25, 20, 0.7), 54: (25, 20, 0.85), 55: (25, 20, 0.95),
                     56: (25, 40, 0.7), 57: (25, 40, 0.85), 58: (25, 40, 0.95),
                     59: (25, 60, 0.7), 60: (25, 60, 0.85), 61: (25, 60, 0.95)}

        # Tri-column
        self.tricolumns = {62: (15, 80, 20), 63: (15, 80, 40), 64: (15, 80, 60),
                      65: (15, 85, 20), 66: (15, 85, 40), 67: (15, 85, 60),
                      68: (15, 95, 20), 69: (15, 95, 40), 70: (15, 95, 60),
                      71: (20, 80, 20), 72: (20, 80, 40), 73: (20, 80, 60),
                      74: (20, 85, 20), 75: (20, 85, 40), 76: (20, 85, 60),
                      77: (20, 95, 20), 78: (20, 95, 40), 79: (20, 95, 60),
                      80: (25, 80, 20), 81: (25, 80, 40), 82: (25, 80, 60),
                      83: (25, 85, 20), 84: (25, 85, 40), 85: (25, 85, 60),
                      86: (25, 95, 20), 87: (25, 95, 40), 88: (25, 95, 60)}
        
        # Tri-column recycle
        self.tricolumns_r = {89: (15, 80, 20, 0.7), 90: (15, 80, 20, 0.85), 91: (15, 80, 20, 0.95),
                        92: (15, 80, 40, 0.7), 93: (15, 80, 40, 0.85), 94: (15, 80, 40, 0.95),
                        95: (15, 80, 60, 0.7), 96: (15, 80, 60, 0.85), 97: (15, 80, 60, 0.95),
                        98: (15, 85, 20, 0.7), 99: (15, 85, 20, 0.85), 100: (15, 85, 20, 0.95),
                        101: (15, 85, 40, 0.7), 102: (15, 85, 40, 0.85), 103: (15, 85, 40, 0.95),
                        104: (15, 85, 60, 0.7), 105: (15, 85, 60, 0.85), 106: (15, 85, 60, 0.95),
                        107: (15, 95, 20, 0.7), 108: (15, 95, 20, 0.85), 109: (15, 95, 20, 0.95),
                        110: (15, 95, 40, 0.7), 111: (15, 95, 40, 0.85), 112: (15, 95, 40, 0.95),
                        113: (15, 95, 60, 0.7), 114: (15, 95, 60, 0.85), 115: (15, 95, 60, 0.95),
                        116: (20, 80, 20, 0.7), 117: (20, 80, 20, 0.85), 118: (20, 80, 20, 0.95),
                        119: (20, 80, 40, 0.7), 120: (20, 80, 40, 0.85), 121: (20, 80, 40, 0.95),
                        122: (20, 80, 60, 0.7), 123: (20, 80, 60, 0.85), 124: (20, 80, 60, 0.95),
                        125: (20, 85, 20, 0.7), 126: (20, 85, 20, 0.85), 127: (20, 85, 20, 0.95),
                        128: (20, 85, 40, 0.7), 129: (20, 85, 40, 0.85), 130: (20, 85, 40, 0.95),
                        131: (20, 85, 60, 0.7), 132: (20, 85, 60, 0.85), 133: (20, 85, 60, 0.95),
                        134: (20, 95, 20, 0.7), 135: (20, 95, 20, 0.85), 136: (20, 95, 20, 0.95),
                        137: (20, 95, 40, 0.7), 138: (20, 95, 40, 0.85), 139: (20, 95, 40, 0.95),
                        140: (20, 95, 60, 0.7), 141: (20, 95, 60, 0.85), 142: (20, 95, 60, 0.95),
                        143: (25, 80, 20, 0.7), 144: (25, 80, 20, 0.85), 145: (25, 80, 20, 0.95),
                        146: (25, 80, 40, 0.7), 147: (25, 80, 40, 0.85), 148: (25, 80, 40, 0.95),
                        149: (25, 80, 60, 0.7), 150: (25, 80, 60, 0.85), 151: (25, 80, 60, 0.95),
                        152: (25, 85, 20, 0.7), 153: (25, 85, 20, 0.85), 154: (25, 85, 20, 0.95),
                        155: (25, 85, 40, 0.7), 156: (25, 85, 40, 0.85), 157: (25, 85, 40, 0.95),
                        158: (25, 85, 60, 0.7), 159: (25, 85, 60, 0.85), 160: (25, 85, 60, 0.95),
                        161: (25, 95, 20, 0.7), 162: (25, 95, 20, 0.85), 163: (25, 95, 20, 0.95),
                        164: (25, 95, 40, 0.7), 165: (25, 95, 40, 0.85), 166: (25, 95, 40, 0.95),
                        167: (25, 95, 60, 0.7), 168: (25, 95, 60, 0.85), 169: (25, 95, 60, 0.95)}
        
        
        self.mixer_count = 0
        self.hex_count = 0
        self.cooler_count = 0
        self.pump_count = 0
        self.reac_count = 0
        self.column_count = 0
        

        # Action declaration
        self.action_space = Discrete(self.d_actions)


        # Observation
        self.low = np.zeros((6,))
        self.high = np.ones((6,))
        self.observation_space = Box(low=self.low, high=self.high, dtype=np.float32)


        self.dme_out = 0


        self.reset()
        self.seed()
    
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_outputs(self, sout):
        T = sout.get_temp()
        P = sout.get_press()
        Fw = sout.get_molar_flow("WATER")
        Fm = sout.get_molar_flow("METHANOL")
        Fdme = sout.get_molar_flow("DME")
        out_list = [T, P, Fm, Fw, Fdme]

        return out_list
     
        

    def step(self, action, sin):
        self.iter += 1
      
        # ----------------------------------------- Mixer -----------------------------------------
        if action == 0:
            self.mixer_count += 1
            self.avail_actions[0] = 0
            self.actions_list.append(f"M{self.mixer_count}")

            mixer = Mixer(f"M{self.mixer_count}", sin)
            sout = mixer.mix()

            self.sim.EngineRun()
            
            if self.sim.Convergence():
                self.info[f"M{self.mixer_count}"] = self.get_outputs(sout)

                # Costs --> normalized cost approximation 
                f_cost = -0.1
                v_cost = -0 # Variable cost (heat)
                cost = f_cost + v_cost # Total cost
        
        # ----------------------------------------- Pump -----------------------------------------
        elif action == 1:
            self.pump_count += 1
            self.avail_actions[2] = 0
            self.actions_list.append(f"PP{self.pump_count}")

            pp = Pump(f"PP{self.pump_count}", 10, sin)

            sout = pp.pump()
            self.sim.EngineRun()

            if self.sim.Convergence():
                        
                self.info[f"PP{self.pump_count}"] = self.get_outputs(sout)

                # Costs --> normalized cost approximation 
                f_cost = -0.2
                v_cost = -pp.enery_consumption()/(10e3) # Variable cost (heat)
                cost = f_cost + v_cost # Total cost

        # ----------------------------------------- HEX -----------------------------------------
        elif action in range(2, 5):
            self.hex_count += 1
            self.actions_list.append(f"HX{self.hex_count}")
            T_hex = self.heaters[action]

            hex = Heater(f"HX{self.hex_count}", T_hex, sin)
            sout = hex.heat()
            self.sim.EngineRun()
            
            if self.sim.Convergence():
                self.info[f"HX{self.hex_count}"] = self.get_outputs(sout)

                # Costs --> normalized cost approximation 
                f_cost = -0.2
                v_cost = -hex.enery_consumption()/(10e3) # Variable cost (heat)
                cost = f_cost + v_cost # Total cost
                
        
        # ----------------------------------------- Cooler -----------------------------------------
        elif action in range(5, 8):
            self.cooler_count += 1
            self.actions_list.append(f"C{self.cooler_count}")
            T_cooler = self.coolers[action]

            cool = Cooler(f"C{self.cooler_count}", T_cooler, sin)

            sout = cool.cool()
            self.sim.EngineRun()
            
            if self.sim.Convergence():
                self.info[f"C{self.cooler_count}"] = self.get_outputs(sout)

                # Costs --> normalized cost approximation 
                f_cost = -0.2
                v_cost = -cool.enery_consumption()/(10e3) # Variable cost (heat)
                cost = f_cost + v_cost # Total cost
        

        # ----------------------------------------- PFR -----------------------------------------
        elif action in range(8, 17):
            self.reac_count += 1
            self.actions_list.append(f"R{self.reac_count}")

            D, L = self.pfrs[action]

            pfr = PFR(f"R{self.reac_count}", D, L, sin)
    
            sout = pfr.react()
            self.sim.EngineRun()
            
            if self.sim.Convergence():
                self.info[f"R{self.reac_count}"] = [D,L, self.get_outputs(sout)]

                # Costs --> normalized cost approximation
                f_cost = -0.4*(1 + self.fixed_cost_reactor(D, L))
                v_cost = -pfr.enery_consumption()/(10e3) # Variable cost (heat)
                cost = f_cost + v_cost # Total cost
        
        # ----------------------------------------- Adiabatic PFR -----------------------------------------
        elif action in range(17, 26):
            self.reac_count += 1
            self.actions_list.append(f"AR{self.reac_count}")

            D, L = self.apfrs[action]

            pfr_a = PFR_A(f"AR{self.reac_count}", D, L, sin)
    
            sout = pfr_a.react()
            self.sim.EngineRun()
            
            if self.sim.Convergence():
                self.info[f"AR{self.reac_count}"] = [D,L, self.get_outputs(sout)]
                
                # Costs --> normalized cost approximation
                f_cost = -0.4*(1 + self.fixed_cost_reactor(D, L))
                v_cost = -0 # Variable cost (heat)
                cost = f_cost + v_cost # Total cost
        
        # ----------------------------------------- Column -----------------------------------------
        elif action in range(26, 35):
            self.column_count += 1
            self.actions_list.append(f"DC{self.column_count}")

            press = sin.get_press()/2.5
            nstages, dist_rate = self.columns[action]

            if sin.get_press() > 5 or self.water_pure:
                distillation_rate = dist_rate
            else:
                if dist_rate == 80:
                    distillation_rate = 20
                elif dist_rate == 85:
                    distillation_rate = 40
                elif dist_rate == 95:
                    distillation_rate = 60

            col = Column(f"DC{self.column_count}", nstages, distillation_rate, 2.5, press, sin)
            
            d, sout = col.distill()
            self.sim.EngineRun()
            
            if self.sim.Convergence():
                self.info[f"DC{self.column_count}"] = [
                    nstages, distillation_rate, self.get_outputs(d), 
                    self.get_outputs(sout)]
                
                if self.get_outputs(d)[-1] > 10:
                    self.dme_out = d

                # Costs --> normalized cost approximation
                Diam, Height = col.sizing()
                f_cost = -0.48*(1 + self.fixed_cost_column(Diam, Height))
                v_cost = -col.enery_consumption()/(10e3) # Variable cost (heat)
                cost = f_cost + v_cost # Total cost
        

        # ----------------------------------------- Column with recycle -----------------------------------------
        elif action in range(35,62):
            self.column_count += 1
            self.actions_list.append(f"DCR{self.column_count}")

            nstages, mid_rate, rr = self.columns_r[action]

            press = sin.get_press()/2.5

            col = Column(f"DCR{self.column_count}", nstages, mid_rate, 2.5, press, sin)
            
            d, sout = col.distill()
            splitter = Splitter(f"S{self.column_count}", rr, d)
            rec, purge = splitter.recycle()

            for i, uo in enumerate(self.actions_list):
                if "M" in uo:
                    self.sim.StreamConnect(self.actions_list[i], rec.name, "F(IN)")
                    break

            self.sim.EngineRun()


            if self.sim.Convergence():
                self.info[f"DCR{self.column_count}"] = [
                    nstages, mid_rate, rr, self.get_outputs(purge), 
                    self.get_outputs(sout)]
                self.actions_list.clear()

                # Costs --> normalized cost approximation
                Diam, Height = col.sizing()
                f_cost = -0.48*(1 + self.fixed_cost_column(Diam, Height))
                v_cost = -col.enery_consumption()*rr/(10e3) # Variable cost (heat)
                cost = f_cost + v_cost # Total cost
               

        # ----------------------------------------- TriColumn -----------------------------------------
        elif action in range(62,89):
            self.column_count += 1
            self.actions_list.append(f"TC{self.column_count}")

            nstages, dist_rate, mid_rate = self.tricolumns[action]

            press = sin.get_press()/2.5

            col = TriColumn(f"TC{self.column_count}", nstages, dist_rate, 2.5, press, mid_rate, sin)
            d, mid, sout = col.distill()
            self.sim.EngineRun()
            
            if self.sim.Convergence():
                self.info[f"TC{self.column_count}"] = [
                    nstages, dist_rate, mid_rate, self.get_outputs(d),
                    self.get_outputs(mid), self.get_outputs(sout)]
                
                if self.get_outputs(d)[-1] > 10:
                    self.dme_out = d

                # Costs --> normalized cost approximation
                Diam, Height = col.sizing()
                f_cost = -0.48*(1 + self.fixed_cost_column(Diam, Height))
                v_cost = -col.enery_consumption()/(10e3) # Variable cost (heat)
                cost = f_cost + v_cost # Total cost
        
        # ----------------------------------------- TriColumn with recycle -----------------------------------------
        elif action in range(89,170):
            self.column_count += 1
            self.actions_list.append(f"TCR{self.column_count}")

            nstages, dist_rate, mid_rate, rr = self.tricolumns_r[action]

            press = sin.get_press()/2.5

            col = TriColumn(f"TCR{self.column_count}", nstages, dist_rate, 2.5, press, mid_rate, sin)
            d, mid, sout = col.distill()
            splitter = Splitter(f"S{self.column_count}", rr, mid)
            rec, purge = splitter.recycle()

            
            for i, uo in enumerate(self.actions_list):
                if "M" in uo:
                    self.sim.StreamConnect(self.actions_list[i], rec.name, "F(IN)")
                    break

            self.sim.EngineRun()
            
            if self.sim.Convergence():
                self.actions_list.clear()


                self.info[f"TCR{self.column_count}"] = [
                    nstages, dist_rate, mid_rate, rr, self.get_outputs(d), 
                    self.get_outputs(purge), self.get_outputs(sout)]
                
                if self.get_outputs(d)[-1] > 10:
                    self.dme_out = d

                # Costs --> normalized cost approximation
                Diam, Height = col.sizing()
                f_cost = -0.48*(1 + self.fixed_cost_column(Diam, Height))
                v_cost = -col.enery_consumption()*rr/(10e3) # Variable cost (heat)
                cost = f_cost + v_cost # Total cost

        # ---------------------------------- Constraints and rewards ----------------------------------     
        if self.sim.Convergence():

            # Constraints

            # Cons 1: (Temperature inside of reactor no greater than 400°C)
            if action in range(8,26) and sout.get_temp() < 400:
                bonus_T = 0.4
            else:
                bonus_T = 0.
            
        
            # Driving force (reduction of the amount of MeOH)
            m_frac_prev = sin.get_molar_flow("METHANOL")/sin.get_total_molar_flow()
            m_frac = sout.get_molar_flow("METHANOL")/sout.get_total_molar_flow()
            bonus = m_frac_prev - m_frac

            # Cons 2. Output purities
            if not self.water_pure:
                w_frac = sout.get_molar_flow("WATER")/sout.get_total_molar_flow()
                self.water_pure = w_frac >= self.pure
            
            if self.dme_out != 0:
                self.dme_pure = self.dme_out.get_molar_flow("DME")/self.dme_out.get_total_molar_flow() >= self.pure
            
            penalty = 0
            reward_flow = 0
            dme_extra = 0
            
            if self.iter >= self.max_iter:
                self.done = True
                
                if not self.water_pure or not self.dme_pure:
                    penalty -= 15*(self.pure - w_frac)
            else:
                if self.water_pure and self.dme_pure:
                    self.done = True
                    reward_flow += 0.2*(self.max_iter - self.iter)

            if self.water_pure and not self.dme_pure:
                sout = self.dme_out
            
            # Reward for more DME flow
            if self.dme_pure and not self.dme_extra_added:
                dme_extra = self.dme_out.get_molar_flow("DME") / (self.Cao / 2)
                self.dme_extra_added = True  # Set the flag to True to indicate that dme_extra has been added           
       
            reward = cost + bonus + bonus_T + penalty + reward_flow + dme_extra

            self.state = np.array([
                sout.get_temp()/400,
                sout.get_press()/10,
                sout.get_molar_flow("METHANOL")/sout.get_total_molar_flow(),
                sout.get_molar_flow("WATER")/sout.get_total_molar_flow(),
                sout.get_molar_flow("DME")/sout.get_total_molar_flow(),
                self.iter/self.max_iter])        
        
        
        else:
            self.done = True
            reward = -8

        
        # Return step information
        return self.state, reward, self.done, self.info, sout
        

    def fixed_cost_reactor(self, D, H):
        M_S = 1638.2  # Marshall & Swift equipment index 2018 (1638.2, fixed)
        f_cost = (M_S)/280 * 101.9 * D**1.066 * H**0.802 * (2.18 + 1.15)
        max_cost = (M_S)/280 * 101.9 * 3.5**1.066 * 12**0.802 * (2.18 + 1.15)
        norm_cost = f_cost/max_cost
        return norm_cost
    
    def fixed_cost_column(self, D, H):
        M_S = 1638.2  # Marshall & Swift equipment index 2018 (1638.2, fixed)
        max_H = 1.2*0.61*(25 - 2)
        # Internal costs
        int_cost = (M_S)/280 * D**1.55 * H
        max_int_cost = (M_S)/280 * 2.5**1.55 * max_H
        norm_cost1 = int_cost/max_int_cost

        # External costs
        f_cost = (M_S)/280 * 101.9 * D**1.066 * H**0.802 * (2.18 + 1.15)
        max_cost = (M_S)/280 * 101.9 * 2.5**1.066 * max_H**0.802 * (2.18 + 1.15)
        norm_cost2 = f_cost/max_cost

        return norm_cost1 + norm_cost2

    

    def action_masks(self, sin, inlet=None):
        self.masking(sin, inlet)
        v1 = np.ones((self.d_actions,), dtype=np.int32)*self.avail_actions
        mask_vec = np.where(v1 > 0, 1, 0)
        mask_vec = np.array(mask_vec, dtype=bool)
        return mask_vec

    
    def render(self):
        for i in self.info:
            print(f"{i}: {self.info[i]}")


    def reset(self):
        # Reset all instances
        self.iter = 0
        self.sim.Reinitialize()
        T, P, compounds = self.inlet_specs
        Fm, Fw, Fdme = compounds["METHANOL"], compounds["WATER"], compounds["DME"]
        tot_flow = Fw + Fm + Fdme
        sin  = Stream("IN", self.inlet_specs)

        self.state = np.array([
                T/400,
                P/10, 
                Fm/tot_flow,
                Fw/tot_flow,
                Fdme/tot_flow,
                self.iter/self.max_iter])

        self.info.clear()
        self.actions_list.clear()
        self.done = False
        self.avail_actions = np.zeros((self.d_actions), dtype=np.int32)
        self.avail_actions[0] = 1
        self.avail_actions[1] = 1
        self.value_step = "pre"
        
        self.mixer_count = 0
        self.hex_count = 0
        self.cooler_count = 0
        self.pump_count = 0
        self.reac_count = 0
        self.column_count = 0

        self.water_pure = False
        self.dme_pure = False
        self.dme_extra_added = False

        self.dme_out = 0
        
        return self.state, sin
    

    def masking(self, sin, inlet):

        if inlet:
            T, P, _ = self.inlet_specs
            meoh_flow = self.Cao
            conv = 0

        else:
            T = sin.get_temp()
            P = sin.get_press()
            meoh_flow = sin.get_molar_flow("METHANOL")
            conv = (self.Cao - meoh_flow)/self.Cao       
        
        
        if self.water_pure:
            self.value_step = "pure"
        # Preprocess 
        elif T >= 200 and P >= 1 and conv < 0.1:
            self.value_step = "reac"
        elif conv >= 0.75 and self.value_step == "reac":
            self.value_step = "cool"
        elif self.value_step == "cool" or self.value_step == "distill":
            self.value_step = "distill"
        


        # Preparation step
        if self.value_step == "pre":
            # Pump deactivation and heater activation (otherwise error in simulation)
            if P > 1:
                self.avail_actions[1] = 0
                self.avail_actions[2:5] = 1
        
        elif self.value_step == "reac":
            self.avail_actions = np.zeros((self.d_actions), dtype=np.int32)
            self.avail_actions[8:26] = 1
        
        elif self.value_step == "cool":
            self.avail_actions = np.zeros((self.d_actions), dtype=np.int32)
            self.avail_actions[5:8] = 1
        
        elif self.value_step == "distill":
            self.avail_actions = np.zeros((self.d_actions), dtype=np.int32)
            self.avail_actions[26:35] = 1

            if any("M" in action for action in self.actions_list):
                if any("DC" in action for action in self.actions_list):
                    self.avail_actions[26:35] = 0
                    self.avail_actions[35:62] = 1
                else:
                    self.avail_actions[89:170] = 1
            else:
                self.avail_actions[62:89] = 1
        
        elif self.value_step == "pure":
            self.avail_actions[26:35] = 1



        return self.avail_actions