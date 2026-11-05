# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 14:24:42 2025

@author: Iffy_z
"""

from .constants import PI, E, H_0, Mpc, C, G
from .utils import vectorize_integral
from scipy.integrate import quad
import numpy as np
import math


class _BasicModel:                                                             #flat
    def __init__(self, H_0 = H_0, O_r0 = 5E-5, O_b0 = 0.05):
        self.H_0 = H_0;
        self.h = self.H_0 / 100.0;                                             #1 demensionless hubble constant
        self.t_H = Mpc * 1.0 /self.H_0;                                        #s hubble time
        self.D_H = C / self.H_0                                                #Mpc hubble distance
        self.rho_crit = Mpc * 3.0*self.H_0**2 / (8.0*PI*G);                    #kg Mpc^-3 rho_c
        self.Omega_b0 = O_b0;                                                  #normal matter
        self.Omega_r0 = O_r0;                                                  #radiation
        
        self.flagvalid = True;
        #==valid_check======
        
        
    def _Z(self,a):
        return (1.0 / a) - 1.0;
    
    def _A(self,z):
        return 1.0/(1.0 + z);
    
    def _Func_01(self, z):
        ans = 1.0/np.sqrt(self.Esquare(self._A(z)));
        return ans;
        #for comoving distance calculation (1/E)
        
    def _Func_02(self, z):
        ans = 1.0/((1.0 + z)*np.sqrt(self.Esquare(self._A(z))))
        return ans;
        #for lookback time calc
    
    def Esquare(self, a):
        ans = self.Omega_r0 * a**-4 + self.Omega_b0 * a**-3; 
        return ans;
        #only normal matters
    
    def Hsquare(self, a):
        ans = np.full_like(a,1E8);
        if self.flagvalid:
            ans = self.H_0**2*self.Esquare(a);
            
        if np.any(ans <= 0.0):
            ans = np.full_like(a,1E8);
            self.flagvalid = False;
        
        return ans;
    
    @vectorize_integral
    def D_c(self, z):
        ans = 1E8;
        if self.flagvalid:
             rslt , err = quad(self._Func_01, 0, z);
             ans = self.D_H * rslt;
        return ans;
        #comoving distance Mpc
        
    def D_M(self, z):
        return self.D_c(z);
        #comoving transverse distance for flat universe
        
    def D_A(self, z):
        return self.D_M(z) / (1.0 + z);
        #angular distance
        
    def D_L(self, z):
        return self.D_M(z) * (1.0 + z);
        #luminosity distance
        
        
    @vectorize_integral   
    def t_L(self, z):
        rslt, err = quad(self._Func_02, 0 ,z);
        ans = self.t_H * rslt;
        return ans;
        #lookback time second
    
class _NoflatBasic(_BasicModel):
    def __init__(self, H_0 = H_0, O_r0 = 5E-5, O_b0 = 0.05):
        self.H_0 = H_0;
        self.h = self.H_0 / 100.0;
        self.Omega_b0 = O_b0;
        self.Omega_r0 = O_r0;
        #Finish them in the future

 
class FlatLCDM(_BasicModel):
    def __init__(self, H_0 = H_0, O_r0 = 5E-5, O_b0 = 0.05, O_m0 = 0.3):
        super().__init__(H_0=H_0, O_r0 = O_r0, O_b0 = O_b0);
        self.Omega_m0 = O_m0;
        self.Omega_dm = O_m0 - O_b0;
        self.Omega_L = 1.0 - O_r0 - O_m0;
        self.om0h2 = self.Omega_m0 * self.h**2.0;
        self.ob0h2 = self.Omega_b0 * self.h**2.0;
        self.Omega_gamma0 = 2.47E-5; #From Planck;
        
        self._init_params();
        
    def _init_params(self):
        if self.Omega_m0 <= 0.0:
            self.flagvalid = False;
            
        if self.Omega_dm <= 0.0:
            self.flagvalid = False;

#estimate r_s            
    def _Func_03(self, a):
        ans = 1.0/(a**2.0 * np.sqrt(self.Hsquare(a)) * np.sqrt(1.0 + 31500.0 *self.ob0h2*(2.7255/2.7)**(-4.0)*a));
        return ans;
        #For r_s

#    def _Func_03(self, a):
#        R = (3.0 * self.Omega_b0)/(4.0 * self.Omega_r0);
#        ans = 1.0/(a**2.0 * np.sqrt(self.Hsquare(a))*np.sqrt(1.0 + R*a));
#        return ans;
        
    def EsOfDarksector(self, a):
        ans = self.Omega_dm*a**-3 + self.Omega_L
        return ans;
    
    def Esquare(self, a):
        ans = self.Omega_r0 * a**-4 + self.Omega_b0 * a**-3 + self.EsOfDarksector(a);
        return ans;
    
    #======================Estimate z_d & r_d==================================
    
    def z_d(self):
        b1 = 0.313 * self.om0h2**(-0.419) * (1.0 + 0.607 * self.om0h2**0.674);
        b2 = 0.238 * self.om0h2**0.223;
        ans = ((1291*self.om0h2**0.251) / (1.0 + 0.659*self.om0h2**(0.828)))*(1.0 + b1 * self.ob0h2**b2);
        return ans;
    
    @vectorize_integral
    def r_s(self, z):
        rslt , err = quad(self._Func_03, 0, 1.0/(1.0+z));
        ans = C/np.sqrt(3.0) * rslt;
        return ans;
    
    def r_d(self):
        z_d = self.z_d();
        ans = self.r_s(z_d);
        return ans;
    
    #====================For CMB 2018 distance=================================
    
    def z_star(self):
        #W. Hu and N. Sugiyama, Astrophys. J. 471, 542 (1996) [astro-ph/9510117].
        g1 = (0.0738 * self.ob0h2**(-0.238)) / (1.0 + 39.5 * self.ob0h2**0.763)
        g2 = 0.560 / (1.0 + 21.1 * self.ob0h2**1.81)

        z_star = 1048.0 * (1.0 + 0.00124 * self.ob0h2**(-0.738)) * (1.0 + g1 * self.om0h2**g2)
        return z_star;
    
    def rs_star(self):
        z_star = self.z_star();
        ans = self.r_s(z_star);
        return ans;
    
        
#=================Linear Intereaction Model(flat)==============================
class LIDE(FlatLCDM):
    def __init__(self, H_0 = H_0, O_r0 = 5E-5, O_b0 = 0.05, O_m0 = 0.3, Ksi_1 = 0.001, Ksi_2 = -0.0005, omega = -1.0):
        super().__init__(H_0 = H_0, O_r0 = O_r0, O_b0 = O_b0, O_m0 = O_m0);
        self.Omega_dm0 = self.Omega_dm;
        self.Omega_L0 = self.Omega_L;
        self.k1 = Ksi_1;
        self.k2 = Ksi_2;
        self.w = omega;
        self.rhodm0 = self.Omega_dm0*self.rho_crit;
        self.rhol0 = self.Omega_L0*self.rho_crit;
        self.tot = self.k1+self.k2+self.w;
        self.r0 = self.rhodm0/self.rhol0;
        #=====================
        self._D1 = None;
        self._D2 = None;
        self._C1 = None;
        self._C2 = None;
        
        self.Delta = None;
        self.DeltaSym = None;
        self.Lambda = None;
        self.SqrtDeltaAbs = None;
        self.Lambda1 = None;
        self.Lambda2 = None;
   
        
        #=intermediates=======
        self._initialize_parameters();
        #=initialize==========
    
    def _initialize_parameters(self):
        # All branches follow from the same evolution matrix.  In particular,
        # xi_2 = 0 is regular and must not be selected through a separate
        # eigenvector formula containing 1/xi_2.
        self.Delta, self.DeltaSym = self._initialize_Delta();
        self.Lambda, self.SqrtDeltaAbs = self._initialize_Lambda();

        if self.DeltaSym > 0:
            self.Lambda1 = self.Lambda - 0.5*self.SqrtDeltaAbs;
            self.Lambda2 = self.Lambda + 0.5*self.SqrtDeltaAbs;
            self._D1, self._D2, self._C1, self._C2 = self._initialize_coeffs();
        elif self.DeltaSym == 0:
            self.Lambda1 = self.Lambda2 = self.Lambda;
            self._D1, self._D2, self._C1, self._C2 = self._initialize_coeffs_zero();
        else:
            self._D1, self._D2, self._C1, self._C2 = self._initialize_coeffs_negaL();
                      
    #===========================initialization=================================
        
    def _initialize_Delta(self):
        Delta = self.k1**2.0 + self.k2**2.0 + self.w**2.0 -2.0*self.k1*self.k2 +2.0*self.k1*self.w +2.0*self.k2*self.w;

        # Only values numerically indistinguishable from zero use the Jordan
        # branch.  The previous 1e-8 band replaced physically distinct modes
        # by a different solution and introduced a discontinuity.
        tolerance = 64.0*np.finfo(float).eps*max(1.0, abs(self.tot)**2, abs(4.0*self.k1*self.k2));
        if abs(Delta) <= tolerance:
            Delta = 0.0;
            Sym = 0;
        elif Delta > 0.0:
            Sym = 1;
        else:
            Sym = -1;
        
        return Delta, Sym
        
    def _initialize_Lambda(self):
        Lambda = 0.5 *(- self.k1 + self.k2 - self.w - 2.0);
        SqrtD = (abs(self.Delta))**0.5;
        return Lambda, SqrtD
    
    def _initialize_coeffs(self):
        D1 = -(self.tot-self.SqrtDeltaAbs)/(2.0*self.SqrtDeltaAbs)*self.rhodm0 - (self.k1/self.SqrtDeltaAbs) *self.rhol0;
        D2 = (self.tot+self.SqrtDeltaAbs)/(2.0*self.SqrtDeltaAbs)*self.rhodm0 + (self.k1/self.SqrtDeltaAbs) *self.rhol0;
        C1 = self.k2/self.SqrtDeltaAbs*self.rhodm0 + (self.tot+self.SqrtDeltaAbs)/(2.0*self.SqrtDeltaAbs)*self.rhol0;
        C2 = -self.k2/self.SqrtDeltaAbs*self.rhodm0 - (self.tot-self.SqrtDeltaAbs)/(2.0*self.SqrtDeltaAbs)*self.rhol0;
        return D1,D2,C1,C2    
    
    def _initialize_coeffs_zero(self):
        # rho = a^(3 lambda) (constant + logarithmic_coefficient ln a)
        # obtained directly from 3(A-lambda I) rho_0.  This expression is
        # finite for xi_2=0 and has the correct physical dimensions.
        D1 = self.rhodm0;
        D2 = 1.5*(self.tot*self.rhodm0 + 2.0*self.k1*self.rhol0);
        C1 = self.rhol0;
        C2 = 1.5*(-2.0*self.k2*self.rhodm0 - self.tot*self.rhol0);
        return D1,D2,C1,C2
      
    def _initialize_coeffs_negaL(self):
        # Coefficients of cos(theta) and sin(theta), fixed directly by rho_0.
        # This avoids the old eigenvector normalization by xi_2.
        q = self.SqrtDeltaAbs;
        D1 = self.rhodm0;
        D2 = (self.tot*self.rhodm0 + 2.0*self.k1*self.rhol0)/q;
        C1 = self.rhol0;
        C2 = (-2.0*self.k2*self.rhodm0 - self.tot*self.rhol0)/q;
        return D1,D2,C1,C2
    
    #=====================initialization(Delta<0)==============================
    def _rho_dm_case1(self,a): #positive delta
        ans = self._D1*a**(3.0*self.Lambda1) + self._D2*a**(3.0*self.Lambda2);
        return ans
    
    def _rho_dm_case2(self,a): #zero delta
        ans = self._D1*a**(3.0*self.Lambda) + self._D2*a**(3.0*self.Lambda)*np.log(a);
        return ans    
    
    def _rho_dm_case3(self,a): #negative delta
        Theta = 1.5*self.SqrtDeltaAbs*np.log(a);
        ans = a**(3.0*self.Lambda)*(self._D1*np.cos(Theta) + self._D2*np.sin(Theta));
        return ans
        

    def _rho_L_case1(self,a): #positive delta
        ans = self._C1*a**(3.0*self.Lambda1) + self._C2*a**(3.0*self.Lambda2);
        return ans
        
    def _rho_L_case2(self,a): #zero delta
        ans = self._C1*a**(3.0*self.Lambda) + self._C2*a**(3.0*self.Lambda)*np.log(a);
        return ans
        
    def _rho_L_case3(self,a): #negative delta
        Theta = 1.5*self.SqrtDeltaAbs*np.log(a);
        ans = a**(3.0*self.Lambda)*(self._C1*np.cos(Theta) + self._C2*np.sin(Theta));
        return ans
    #========================background evolution==============================
    
    def EsOfDarksector(self, a):
        ans = 1.0/self.rho_crit*(self.Rho_dm(a) + self.Rho_L(a));
        return ans;  
    
    def Rho_dm(self,a):
        if self.DeltaSym > 0:
            ans = self._rho_dm_case1(a);
        elif self.DeltaSym < 0:
            ans = self._rho_dm_case3(a);
        else:
            ans = self._rho_dm_case2(a);
        return ans
        
    def Rho_L(self,a):    
        if self.DeltaSym > 0:
            ans = self._rho_L_case1(a);
        elif self.DeltaSym < 0:
            ans = self._rho_L_case3(a);
        else:
            ans = self._rho_L_case2(a);
        return ans
    
    #==========================================================================
    def Q(self,a):
        ans = 3.0*np.sqrt(self.Hsquare(a))*(self.k1*self.Rho_L(a) + self.k2*self.Rho_dm(a));
        return ans    
    #=============================misc=========================================  

#===============================CPL============================================
#===============================w = w0 + wa(1-a)===============================
class CPL(FlatLCDM):
    def __init__(self, H_0 = H_0, O_r0 = 5E-5, O_b0 = 0.05, O_m0 = 0.3, w0 = -1.0, wa = 0.0):
        super().__init__(H_0 = H_0, O_r0 = O_r0, O_b0 = O_b0, O_m0 = O_m0);
        self.w0 = w0;
        self.wa = wa;
        
    def EsOfDarksector(self, a):
        ans = self.Omega_dm*a**-3 + self.Omega_L*a**(-3.0*(1.0+self.w0+self.wa))*np.exp(-3.0*self.wa*(1.0-a));
        return ans
    
#===========================Another Intereaction===============================

    
        
    
    
        

