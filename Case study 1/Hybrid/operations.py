import numpy as np
from scipy.optimize import minimize, fsolve, curve_fit
from scipy import optimize as opt

class CSTR:
    def __init__(self, inlet, D, H, ns=80):
        # Parameters
        Cao, To, Fo = inlet
        self.dH = -3000
        self.U = 300
        self.E_R = 15075
        self.k0 = 4.08e10
        self.Cp = 0.75
        self.Cpj = 1.0
        self.rho = 50
        self.rhoj = 62.3
        self.Fo = Fo 
        self.Tjo = 530

        self.Cao = Cao
        self.To = To
        self.D = D
        self.H = H
        self.ns = ns
        self.Fj = 30
        self.tf = 20

    def model(self, vars, t, Fj, To):
        Ca = vars[0]
        T = vars[1]
        Tj = vars[2]

        k = self.k0*np.exp(-self.E_R*(1/T))
        V = np.pi/4 * self.D**2 * self.H
        A = np.pi*self.D*self.H 
        Vj = A/3 

        f1 = (self.Fo/V)*(self.Cao - Ca) - k*Ca
        f2 = (self.Fo/V)*(To - T) + (-self.dH*k*Ca)/(self.Cp*self.rho) -\
            ((self.U*A)/(V*self.Cp*self.rho))*(T - Tj)
        f3 = (Fj/Vj)*(self.Tjo - Tj) + ((self.U*A)/(Vj*self.Cpj*self.rhoj))*(T - Tj)
        
        return [f1, f2, f3]
            
    def steady_state(self):
        x_ss = fsolve(self.model, [self.Cao, self.To, self.Tjo], args=(None, self.Fj, self.To))
        Ca, T, Tj = x_ss
        return Ca, T,self.Fo


class Mixer:
    def __init__(self, inlet, recycle=None):
        self.inlet = inlet
        self.recycle = recycle

    def mix(self):
        if self.recycle is None:
            Ca, T, F = self.inlet
        else:
            Ca_r, T_r, V = self.recycle
            Ca, T, F = self.inlet

            Ca = (Ca*F + Ca_r*V)/(V + F)
            T = (T*F + T_r*V)/(V + F)
            F += V

        return Ca, T, F


class Flash_recycle:
    def __init__(self, q, inlet, flowsheet):
        self.inlet = inlet
        self.q = q
        self.flowsheet = flowsheet

    def flash(self, input_vals):
        Ca, T, F = input_vals
        self.Ca = Ca
        self.T = T
        self.F = F
                
        def fun(vars, z):
            global V, L
            
            x = vars[0]
            y = vars[1]
            
            F = self.F
            alpha = 4.5
            
            L = F*self.q
            V = F - L
            f1 = y - ((alpha*x)/(1 + x*(alpha - 1)))
            f2 = y - ((self.q/(self.q-1))*x + z/(1-self.q))

            return [f1, f2]
        
        # Initial guesses
        x0 = (0,0)
        sol = fsolve(fun, x0=x0, args=(self.Ca,))
        x, y = sol[0], sol[1]

        return x, y, V, L
        

    def recycle(self):
        # Looping in the flowsheet
        start_key = 'M'
        found_start_key = False
        uo_list = []

        for ind, key in enumerate(self.flowsheet):
            if key == start_key:
                found_start_key = True
            
            if found_start_key:
                uo_list.append(key) 

        if found_start_key is False:
            Ca, Ca_v, V, L = self.flash(self.inlet)
            T = self.inlet[1]
            return Ca, T, L

        
        Ca, Ca_v, V, L = self.flash(self.inlet)
        T = self.inlet[1]
        while True:
            for uo in uo_list:
                if "M" in uo:
                    mixer = Mixer(self.flowsheet[uo], [Ca_v, T, V])
                    Ca, T, F = mixer.mix()
                elif "C" in uo:
                    D, H = self.flowsheet[uo]
                    cstr = CSTR([Ca, T, F], D, H, 80)
                    Ca, T, _ = cstr.steady_state()
                
            Ca, Ca_v, V, L = self.flash([Ca, T, F])

            if abs(self.inlet[-1]-L) <= 1e-3:
                break

        return Ca, T, L