# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 13:54:18 2025

@author: Iffy_z
"""

import numpy as np
from cobaya.theory import Theory
from .background_cosmodels import LIDE, FlatLCDM
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.constants as const


class MyCobayaLCDM(Theory):
#    params = {
#        "H0":{"prior": {"min": 40, "max": 100}, "ref": 70.0},
#        "Omega_m":{"prior": {"min": 0.0, "max": 1.0}, "ref": 0.3},
#        "Ob0h2":{"prior": {"min": 0.020, "max": 0.024}, "ref": 0.0224},
#        "rs_drag": {"derived": True}
#        }
    params = {
        "H0": None,
        "Omega_m": None,
        "Ob0h2": None,
#        "rs_drag": {"derived": True}
        "rdrag": {"derived": True},
##        "rs_zstar" : {"derived": True}
        }
        
    def initialize_with_provider(self, provider):
        self.provider = provider

    def get_can_provide(self):
        return ["Hubble" ,"H_of_z", "comoving_distance", "luminosity_distance", "angular_diameter_distance"]

    def calculate(self, state, want_derived=True, **params_values_dict):
        #Params
        #H0 = params_values_dict.get("H0", self.provider.get_param("H0"))
        #Om = params_values_dict.get("Omega_m", self.provider.get_param("Omega_m"))
        #Ob0h2 = params_values_dict.get("Ob0h2", self.provider.get_param("Ob0h2"))
        H0 = params_values_dict["H0"];
        Om = params_values_dict["Omega_m"];
        Ob0h2 = params_values_dict["Ob0h2"];
                
        h = H0/100.0;
        Or = 9.27E-5; #From Planck
        Ob = Ob0h2/h/h;
        
        #MyModel
        backend = FlatLCDM(H_0=H0, O_m0=Om, O_b0=Ob, O_r0= Or);

        z_min = 0.0; z_mid = 10.0; z_max = 2.0; #z_max is index
        #Z~1000 need considering OmegaR
        z1 = np.linspace(z_min, z_mid, 1001);
        z2 = np.logspace(np.log10(z_mid + 0.001), z_max, 800)
        zgrid = np.unique(np.concatenate([z1,z2]))
        agrid = 1.0/(1.0+zgrid);
        
        Hz_grid = np.sqrt(backend.Hsquare(agrid));
        Dc_grid = backend.D_c(zgrid);

        #rd:
        r_d = backend.r_d(); 
        #rs_zstar:
##        rs_zstar = backend.rs_star();
            
        #state: cache of cobaya
        state["z_grid"] = zgrid;
        state["Hz_grid"] = Hz_grid;
        state["Dc_grid"] = Dc_grid;
#        state["derived"] = {"rs_drag": r_d};
        state["derived"] = {"rdrag" : r_d};
##        state["derived"] = {
##            "rdrag" : r_d,
##            "rs_zstar" : rs_zstar };
       
        #interpolation
        H_spline = InterpolatedUnivariateSpline(zgrid, Hz_grid, k=3)
        Dc_spline = InterpolatedUnivariateSpline(zgrid, Dc_grid, k=3)
        
        state["H_spline"] = H_spline
        state["Dc_spline"] = Dc_spline

    #For get_X method
    def get_H_of_z(self, z):
        z = np.atleast_1d(z)
        Hs = self.current_state["H_spline"](z)
        return np.asarray(Hs)

    def get_comoving_distance(self, z):
        z = np.atleast_1d(z)
        Dcs = self.current_state["Dc_spline"](z)
        return np.asarray(Dcs)

    def get_luminosity_distance(self, z):
        Dc = self.get_comoving_distance(z)
        Dl = (1.0 + z) * Dc
        return Dl
    
    def get_angular_diameter_distance(self, z):
        Dc = self.get_comoving_distance(z)
        Da = Dc / (1.0 + z)
        return Da
    
    #For BAO part:
    def get_Hubble(self, z, units="km/s/Mpc"):
        H = self.get_H_of_z(z)
        if units == "km/s/Mpc":
            return H
        elif units == "1/Mpc":
            return H / (const.c / 1000.0)
        else:
            raise ValueError("Unknown units")   
        
    
class MyCobayaLIDE(Theory):
#    params = {
#        "H0": {"prior": {"min": 40, "max": 100}, "ref": 70.0},
#        "Omega_m": {"prior": {"min": 0.0, "max": 1.0}, "ref": 0.3},
#        "omega": {"prior": {"min": -2.0, "max": 2.0}, "ref": -1.0},
#        "k1": {"prior": {"min": -0.1, "max": 0.5}, "ref": 0.004},
#        "k2": {"prior": {"min": -0.5, "max": 0.1}, "ref": -0.0015},
#        "Ob0h2":{"prior": {"min": 0.020, "max": 0.024}, "ref": 0.0224},
#        "rs_drag": {"derived": True}
#    }
    params = {
        "H0": None,
        "Omega_m": None,
        "omega": None,
        "k1": None,
        "k2": None,
        "Ob0h2": None,
#        "rs_drag": {"derived": True}
        "rdrag": {"derived": True},
#        "rs_zstar" : {"derived": True}
        }

    def initialize_with_provider(self, provider):
        self.provider = provider

    def get_can_provide(self):
        return ["H_of_z", "comoving_distance", "luminosity_distance", "angular_diameter_distance"]
    def calculate(self, state, want_derived=True, **params_values_dict):
        #Params
        #H0 = params_values_dict.get("H0", self.provider.get_param("H0"))
        #Om = params_values_dict.get("Omega_m", self.provider.get_param("Omega_m"))
        #w = params_values_dict.get("omega", self.provider.get_param("omega"))
        #k1 = params_values_dict.get("k1", self.provider.get_param("k1"))
        #k2 = params_values_dict.get("k2", self.provider.get_param("k2"))
        
        #Ob0h2 = params_values_dict.get("Ob0h2", self.provider.get_param("Ob0h2"))
        H0 = params_values_dict["H0"];
        Om = params_values_dict["Omega_m"];
        w = params_values_dict["omega"];
        k1 = params_values_dict["k1"];
        k2 = params_values_dict["k2"];
        
        Ob0h2 = params_values_dict["Ob0h2"];
        
        
        h = H0/100.0;
        Or = 9.27E-5; #From Planck
        Ob = Ob0h2/h/h;
        
        #MyModel
        backend = LIDE(H_0=H0, O_r0=Or, O_b0=Ob, O_m0=Om, Ksi_1=k1, Ksi_2=k2, omega=w);

        z_min = 0.0; z_mid = 10.0; z_max = 2.0;
        z1 = np.linspace(z_min, z_mid, 1001);
        z2 = np.logspace(np.log10(z_mid + 0.001), z_max, 800)
        zgrid = np.unique(np.concatenate([z1,z2]))
        agrid = 1.0/(1.0+zgrid);
        
        Hz_grid = np.sqrt(backend.Hsquare(agrid));
        Dc_grid = backend.D_c(zgrid);
        
        #rd:
        r_d = backend.r_d(); 
        #rs_zstar:

        #state: cache of cobaya
        state["z_grid"] = zgrid
        state["Hz_grid"] = Hz_grid
        state["Dc_grid"] = Dc_grid
        #state["derived"] = {"rs_drag": r_d};
        state["derived"] = {"rdrag": r_d}
        
        #interpolation
        H_spline = InterpolatedUnivariateSpline(zgrid, Hz_grid, k=3)
        Dc_spline = InterpolatedUnivariateSpline(zgrid, Dc_grid, k=3)
        
        state["H_spline"] = H_spline
        state["Dc_spline"] = Dc_spline

    #For get_X method
    def get_H_of_z(self, z):
        z = np.atleast_1d(z)
        Hs = self.current_state["H_spline"](z)
        return np.asarray(Hs)

    def get_comoving_distance(self, z):
        z = np.atleast_1d(z)
        Dcs = self.current_state["Dc_spline"](z)
        return np.asarray(Dcs)

    def get_luminosity_distance(self, z):
        Dc = self.get_comoving_distance(z)
        Dl = (1.0 + z) * Dc
        return Dl
    
    def get_angular_diameter_distance(self, z):
        Dc = self.get_comoving_distance(z)
        Da = Dc / (1.0 + z)
        return Da
    
    #For BAO part:
    def get_Hubble(self, z, units="km/s/Mpc"):
        H = self.get_H_of_z(z)
        if units == "km/s/Mpc":
            return H
        elif units == "1/Mpc":
            return H / (const.c / 1000.0)
        else:
            raise ValueError("Unknown units")
        
