# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 14:55:43 2026

@author: Iffy_z
"""

import numpy as np
from cobaya.likelihood import Likelihood
from .background_cosmodels import LIDE,FlatLCDM;
from .constants import C


class My_calculator:
    def __init__(self, H0=70, Ob0h2=0.02, Omega_m=0.3, k1=0.0,k2=0.0,w=-1.0):
        self.H0 = H0;
        self.Ob0h2 = Ob0h2;
        self.Omega_m = Omega_m;
        self.Or = 9.27E-5; #From Planck
        self.c = C;
        self.k1 = k1;
        self.k2 = k2;
        self.w = w;
        # Bib: 1808.05724
        # Planck 2018 distance priors
        # Order: [R, l_A, omega_b h^2]
        self.data = np.array([1.7502, 301.471, 0.02236])
        self.model = np.array([0.0,0.0,0.0])

        # R, l_A, obh2
        self.invcov = np.array([
            [94392.3971,   -1360.4913,    1664517.2916],
            [-1360.4913,   161.4349,     3671.6180],
            [1664517.2916,  3671.6180,    79719182.5162]
        ])
        
    def z_star(self):
        
        Ob0h2 = self.Ob0h2;
        H0 = self.H0;
        h = H0 / 100.0;
        Om0 = self.Omega_m;
        om0h2 = Om0 * h**2;

        g1 = (0.0738 * Ob0h2**(-0.238)) / (1.0 + 39.5 * Ob0h2**0.763);
        g2 = 0.560 / (1.0 + 21.1 * Ob0h2**1.81);

        z_star = 1048.0 * (1.0 + 0.00124 * Ob0h2**(-0.738)) * (1.0 + g1 * om0h2**g2);

        return z_star
    
    def logp(self):
        #print("==============test===================");
        
        z_star = self.z_star();
        h = self.H0/ 100.0;
        Ob0 = self.Ob0h2/h**2;
        
        #print("z_star=",z_star);
        #print("h=",h);
        #print("Ob0=",Ob0);
        #print("invcov=",self.invcov);
        #For test LCDM
        #Cosmology = FlatLCDM(H_0=self.H0,O_r0=self.Or,O_b0=Ob0,O_m0=self.Omega_m);
        #For LIDE
        #print("model's prams: H0=",self.H0,", O_r0=",self.Or,", O_b0=",Ob0,", O_m0=",self.Omega_m,", k1=",self.k1,", k2=",self.k2,", w=",self.w);
        Cosmology = LIDE(H_0=self.H0,O_r0=self.Or,O_b0=Ob0,O_m0=self.Omega_m,Ksi_1 = self.k1, Ksi_2 = self.k2, omega = self.w);       
        #print("=====cos_model loaded==============");
        
        DA = Cosmology.D_A(z_star);                                 # Mpc
        H0 = np.sqrt(Cosmology.Hsquare(1.0));                       # km/s/Mpc
        rs = Cosmology.r_s(z_star);                                 # Mpc
        Om = self.Omega_m;
        obh2 = self.Ob0h2;

        #print("DA=",DA);
        #print("H0model=",H0);
        #print("rs=",rs);
    
        # Calculator
        DM = (1 + z_star) * DA
        lA = np.pi * DM / rs
        
        R = np.sqrt(Om * (H0**2)) * DM / self.c 

        model = np.array([R, lA, obh2])
        self.model = model;
    
        # ---- chi2 ----
        diff = model - self.data
        chi2 = diff @ self.invcov @ diff
        
        #print("model's value: R , lA and obh2=",self.model);
        #print("chi2=",chi2);
        #print("=============test===================");

        return -0.5 * chi2
    


class CMB_Distance(Likelihood):

    # Cobaya_params requirements
    def get_requirements(self):
        return {
            "Ob0h2": None,
            "Omega_m": None,
            "H0": None,
            "k1": None,
            "k2": None,
            "omega": None
        }

    def logp(self, **params_values):
        try:
            # Load params
            Ob0h2 = params_values["Ob0h2"]
            Omega_m = params_values["Omega_m"]
            H0 = params_values["H0"]
            k1 = params_values["k1"]
            k2 = params_values["k2"]
            w = params_values["omega"]

            # external calculator
            My_calc = My_calculator(H0=H0,Ob0h2=Ob0h2,Omega_m=Omega_m,k1=k1,k2=k2,w=w);
            
            result = My_calc.logp();

            return result

        except Exception as e:
            print("Likelihood error:", e)
            return -1e30
        
"""
class CMB_Distance(Likelihood):
    
    def initialize(self):
        self.c = C;
        
        # Bib: 1808.05724
        # Planck 2018 distance priors
        # Order: [R, l_A, omega_b h^2]
        self.data = np.array([1.7502, 301.471, 0.02236])

        # R, l_A, obh2
        self.invcov = np.array([
            [94392.3971,   -1360.4913,    1664517.2916],
            [-1360.4913,   161.4349,     3671.6180],
            [1664517.2916,  3671.6180,    79719182.5162]
        ])
        
        
#    def get_requirements(self):
#        return {
#            "angular_diameter_distance": {"z": [1100]},
#            "Hubble": {"z": [0]},
#            "sound_horizon": {"z": [1100]},
#        }        
    
    def z_star(self):
       
        
        Ob0h2 = self.provider.get_param("Ob0h2")
        H0 = self.provider.get_H_of_z(0)
        h = H0 / 100.0

        Om0 = self.provider.get_param("Omega_m")
        om0h2 = Om0 * h**2

        g1 = (0.0738 * Ob0h2**(-0.238)) / (1.0 + 39.5 * Ob0h2**0.763)
        g2 = 0.560 / (1.0 + 21.1 * Ob0h2**1.81)

        z_star = 1048.0 * (1.0 + 0.00124 * Ob0h2**(-0.738)) * (1.0 + g1 * om0h2**g2)

        return z_star
    
    def logp(self, **params):
        
        z_star = self.z_star()

        DA = self.provider.get_angular_diameter_distance(z_star)   # Mpc
        H0 = self.provider.get_H_of_z(0)                           # km/s/Mpc
        rs = self.provider.get_param("rs_zstar")                   # Mpc
        Om = self.provider.get_param("Omega_m")
        obh2 = self.provider.get_param("ombh2")

        # Calculator
        DM = (1 + z_star) * DA
        lA = np.pi * DM / rs
        
        R = np.sqrt(Om * (H0**2)) * DM / self.c 

        model = np.array([R, lA, obh2])

        # ---- chi2 ----
        diff = model - self.data
        chi2 = diff @ self.icov @ diff

        return -0.5 * chi2
"""