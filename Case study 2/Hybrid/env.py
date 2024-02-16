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
        self.d_actions = 10
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
        self.avail_actions = np.array(
            [1, # Mixer
             0, # Heater
             1, # Pump
             0, # Cooler
             0, # PFR
             0, # PFR adiabatic
             0, # Column
             0, # Column with recycle
             0, # Tri-Column
             0, # Tri-Column with recycle
             ], dtype=np.int32)
        
        
        self.mixer_count = 0
        self.hex_count = 0
        self.cooler_count = 0
        self.pump_count = 0
        self.reac_count = 0
        self.column_count = 0
        

        # Action declaration
        self.action_space = Dict({
            "discrete": Discrete(self.d_actions), 
            "continuous": Box(low=np.zeros(19,), high=np.ones(19,), dtype=np.float32)})


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

        d_action = action["discrete"]
        c_action = action["continuous"]
        c_action = self.interpolation(np.array(c_action))
        T_hex, T_cooler, D1, L1, D2, L2,\
            nstages_c, dist_rate_c, mid_rate_c,\
            nstages_cr, mid_rate_cr, rr_cr,\
            nstages_tc, dist_rate_tc, mid_rate_tc,\
            nstages_tcr, dist_rate_tcr, mid_rate_tcr, rr_tcr = c_action
      
        
        # ----------------------------------------- Mixer -----------------------------------------
        if d_action == 0:
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
              
        # ----------------------------------------- HEX -----------------------------------------
        elif d_action == 1:
            self.hex_count += 1
            self.actions_list.append(f"HX{self.hex_count}")

            hex = Heater(f"HX{self.hex_count}", T_hex, sin)
            sout = hex.heat()
            self.sim.EngineRun()
            
            if self.sim.Convergence():
                self.info[f"HX{self.hex_count}"] = [T_hex, self.get_outputs(sout)]
                
                # Costs --> normalized cost approximation 
                f_cost = -0.2
                v_cost = -hex.enery_consumption()/(10e3) # Variable cost (heat)
                cost = f_cost + v_cost # Total cost
        

        # ----------------------------------------- Pump -----------------------------------------
        elif d_action == 2:
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
        
        
        # ----------------------------------------- Cooler -----------------------------------------
        elif d_action == 3:
            self.cooler_count += 1
            self.actions_list.append(f"C{self.cooler_count}")

            cool = Cooler(f"C{self.cooler_count}", T_cooler, sin)

            sout = cool.cool()
            self.sim.EngineRun()
            
            if self.sim.Convergence():
                self.info[f"C{self.cooler_count}"] = [T_cooler, self.get_outputs(sout)]

                # Costs --> normalized cost approximation 
                f_cost = -0.2
                v_cost = -cool.enery_consumption()/(10e3) # Variable cost (heat)
                cost = f_cost + v_cost # Total cost
        

        # ----------------------------------------- PFR -----------------------------------------
        elif d_action == 4:
            self.reac_count += 1
            self.actions_list.append(f"R{self.reac_count}")

            pfr = PFR(f"R{self.reac_count}", D1, L1, sin)
    
            sout = pfr.react()
            self.sim.EngineRun()
            
            if self.sim.Convergence():
                self.info[f"R{self.reac_count}"] = [D1,L1, self.get_outputs(sout)]

                # Costs --> normalized cost approximation
                f_cost = -0.4*(1 + self.fixed_cost_reactor(D1, L1))
                v_cost = -pfr.enery_consumption()/(10e3) # Variable cost (heat)
                cost = f_cost + v_cost # Total cost
        
        # ----------------------------------------- Adiabatic PFR -----------------------------------------
        elif d_action == 5:
            self.reac_count += 1
            self.actions_list.append(f"AR{self.reac_count}")

            pfr_a = PFR_A(f"AR{self.reac_count}", D2, L2, sin)
    
            sout = pfr_a.react()
            self.sim.EngineRun()
            
            if self.sim.Convergence():
                self.info[f"AR{self.reac_count}"] = [D2,L2, self.get_outputs(sout)]

                # Costs --> normalized cost approximation
                f_cost = -0.4*(1 + self.fixed_cost_reactor(D2, L2))
                v_cost = -0 # Variable cost (heat)
                cost = f_cost + v_cost # Total cost
        
        # ----------------------------------------- Column -----------------------------------------
        elif d_action == 6:
            self.column_count += 1
            self.actions_list.append(f"DC{self.column_count}")

            press = sin.get_press()/2.5

            if sin.get_press() > 5 or self.water_pure:
                distillation_rate = dist_rate_c
            else:
                distillation_rate = mid_rate_c

            col = Column(f"DC{self.column_count}", nstages_c, distillation_rate, 2.5, press, sin)
            
            d, sout = col.distill()
            self.sim.EngineRun()
            
            if self.sim.Convergence():
                self.info[f"DC{self.column_count}"] = [
                    nstages_c, distillation_rate, self.get_outputs(d), 
                    self.get_outputs(sout)]
                
                if self.get_outputs(d)[-1] > 10:
                    self.dme_out = d
        

                # Costs --> normalized cost approximation
                Diam, Height = col.sizing()
                f_cost = -0.48*(1 + self.fixed_cost_column(Diam, Height))
                v_cost = -col.enery_consumption()/(10e3) # Variable cost (heat)
                cost = f_cost + v_cost # Total cost
        
        # ----------------------------------------- Column with recycle -----------------------------------------
        elif d_action == 7:
            self.column_count += 1
            self.actions_list.append(f"DCR{self.column_count}")

            press = sin.get_press()/2.5

            col = Column(f"DCR{self.column_count}", nstages_cr, mid_rate_cr, 2.5, press, sin)
            
            d, sout = col.distill()
            splitter = Splitter(f"S{self.column_count}", rr_cr, d)
            rec, purge = splitter.recycle()

            for i, uo in enumerate(self.actions_list):
                if "M" in uo:
                    self.sim.StreamConnect(self.actions_list[i], rec.name, "F(IN)")
                    break

            self.sim.EngineRun()


            if self.sim.Convergence():
                self.info[f"DCR{self.column_count}"] = [
                    nstages_cr, mid_rate_cr, rr_cr, self.get_outputs(purge), 
                    self.get_outputs(sout)]
                self.actions_list.clear()

                # Costs --> normalized cost approximation
                Diam, Height = col.sizing()
                f_cost = -0.48*(1 + self.fixed_cost_column(Diam, Height))
                v_cost = -col.enery_consumption()*rr_cr/(10e3) # Variable cost (heat)
                cost = f_cost + v_cost # Total cost
        

        # ----------------------------------------- TriColumn -----------------------------------------
        elif d_action == 8:
            self.column_count += 1
            self.actions_list.append(f"TC{self.column_count}")

            press = sin.get_press()/2.5

            col = TriColumn(f"TC{self.column_count}", nstages_tc, dist_rate_tc, 2.5, press, mid_rate_tc, sin)
            d, mid, sout = col.distill()
            self.sim.EngineRun()
            
            if self.sim.Convergence():
                self.info[f"TC{self.column_count}"] = [
                    nstages_tc, dist_rate_tc, mid_rate_tc, self.get_outputs(d),
                    self.get_outputs(mid), self.get_outputs(sout)]
                
                if self.get_outputs(d)[-1] > 10:
                    self.dme_out = d
                
                # Costs --> normalized cost approximation
                Diam, Height = col.sizing()
                f_cost = -0.48*(1 + self.fixed_cost_column(Diam, Height))
                v_cost = -col.enery_consumption()/(10e3) # Variable cost (heat)
                cost = f_cost + v_cost # Total cost
        
        # ----------------------------------------- TriColumn with recycle -----------------------------------------
        elif d_action == 9:
            self.column_count += 1
            self.actions_list.append(f"TCR{self.column_count}")

            press = sin.get_press()/2.5

            col = TriColumn(f"TCR{self.column_count}", nstages_tcr, dist_rate_tcr, 2.5, press, mid_rate_tcr, sin)
            d, mid, sout = col.distill()
            splitter = Splitter(f"S{self.column_count}", rr_tcr, mid)
            rec, purge = splitter.recycle()

            
            for i, uo in enumerate(self.actions_list):
                if "M" in uo:
                    self.sim.StreamConnect(self.actions_list[i], rec.name, "F(IN)")
                    break

            self.sim.EngineRun()
            
            if self.sim.Convergence():
                self.actions_list.clear()


                self.info[f"TCR{self.column_count}"] = [
                    nstages_tcr, dist_rate_tcr, mid_rate_tcr, rr_tcr, self.get_outputs(d), 
                    self.get_outputs(purge), self.get_outputs(sout)]
                
                if self.get_outputs(d)[-1] > 10:
                    self.dme_out = d

                # Costs --> normalized cost approximation
                Diam, Height = col.sizing()
                f_cost = -0.48*(1 + self.fixed_cost_column(Diam, Height))
                v_cost = -col.enery_consumption()*rr_tcr/(10e3) # Variable cost (heat)
                cost = f_cost + v_cost # Total cost
        

        # ---------------------------------- Constraints and rewards ----------------------------------     
        if self.sim.Convergence():

            # Constraints

            # Cons 1: (Temperature inside of reactor no greater than 400°C)
            if d_action in (4, 5) and sout.get_temp() <= 400:
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


    def interpolation(self, c_action):
        T_hex, T_cooler, D1, L1, D2, L2,\
            nstages_c, dist_rate_c, mid_rate_c,\
            nstages_cr, mid_rate_cr, rr_cr,\
            nstages_tc, dist_rate_tc, mid_rate_tc,\
            nstages_tcr, dist_rate_tcr, mid_rate_tcr, rr_tcr = c_action

        T_hex = np.interp(T_hex, [0, 1], (150, 400))
        T_cooler = np.interp(T_cooler, [0, 1], (5, 50))
        D1 = np.interp(D1, [0, 1], (0.5, 3.5))
        L1 = np.interp(L1, [0, 1], (6.5, 12.0))
        D2 = np.interp(D2, [0, 1], (0.5, 3.5))
        L2 = np.interp(L2, [0, 1], (6.5, 12.0))
        nstages_c = round(np.interp(nstages_c, [0, 1], [5, 25]) + 0.5)
        dist_rate_c = np.interp(dist_rate_c, [0, 1], (70.0, 110.0))
        mid_rate_c = np.interp(mid_rate_c, [0, 1], (20.0, 60.0))
        nstages_cr = round(np.interp(nstages_cr, [0, 1], [5, 25]) + 0.5)
        mid_rate_cr = np.interp(mid_rate_cr, [0, 1], (20.0, 60.0))
        rr_cr = np.interp(rr_cr, [0, 1], (0.5, 0.95))
        nstages_tc = round(np.interp(nstages_tc, [0, 1], [5, 25]) + 0.5)
        dist_rate_tc = np.interp(dist_rate_tc, [0, 1], (70.0, 110.0))
        mid_rate_tc = np.interp(mid_rate_tc, [0, 1], (20.0, 60.0))
        nstages_tcr = round(np.interp(nstages_tcr, [0, 1], [5, 25]) + 0.5)
        dist_rate_tcr = np.interp(dist_rate_tcr, [0, 1], (70.0, 110.0))
        mid_rate_tcr = np.interp(mid_rate_tcr, [0, 1], (20.0, 60.0))
        rr_tcr = np.interp(rr_tcr, [0, 1], (0.5, 0.95))


        y = T_hex, T_cooler, D1, L1, D2, L2,\
            nstages_c, dist_rate_c, mid_rate_c,\
            nstages_cr, mid_rate_cr, rr_cr,\
            nstages_tc, dist_rate_tc, mid_rate_tc,\
            nstages_tcr, dist_rate_tcr, mid_rate_tcr, rr_tcr
        
        return y


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
        self.avail_actions = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
        
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
                self.avail_actions[2] = 0
                self.avail_actions[1] = 1
            
        
        elif self.value_step == "reac":
            self.avail_actions = np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0], dtype=np.int32)
        
        elif self.value_step == "cool":
            self.avail_actions = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.int32)
        
        elif self.value_step == "distill":
            self.avail_actions = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=np.int32)

            if any("M" in action for action in self.actions_list):
                if any("DC" in action for action in self.actions_list):
                    self.avail_actions[6] = 0
                    self.avail_actions[7] = 1
                else: 
                    self.avail_actions[9] = 1
            else:
                self.avail_actions[8] = 1
        
        elif self.value_step == "pure":
            self.avail_actions = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=np.int32)

        return self.avail_actions