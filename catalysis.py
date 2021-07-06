#
# Module Thermodynamics
#
# Developer   : Min Jong Noh
# Last update : 2020/10/01
# E-mail      : starnmj@kaist.ac.kr

import os, sys, glob, math
from . atoms import *

class Modeling:

    def __init__(self, atoms):
        self.atoms = atoms

    def get_zmax_index(self):
        atoms = self.atoms
        z_axis = []
        for atom in atoms:
            x, y, z = atom.get_position()
            z_axis.append(z)
        maxindex = z_axis.index(max(z_axis))
        return maxindex

    def HER_transition_gen(self, active=None, dist=1.5):
        """
        specify hydrogen atomic position on the catalyst

        Parameters
        ----------
        active = list
        dist   = 1.5
            assign the position [A] of adsorbed hydrogen [x, y, z]
            Ex. active = [1.2, 1.35, 23.75]
            if active is None, the atomic position is automatically assigned
            along to highest z-axis position (ontop site) with specific distance (dist=1.5)
        """
        atoms = self.atoms

        if active is None:
           top_index = self.get_zmax_index()
           top_x, top_y, top_z = atoms[top_index][0], atoms[top_index][1], atoms[top_index][2]
           atomsH = atoms + Atom('H', (top_x, top_y, top_z+dist))
        else:
           H_x, H_y, H_z = active[0], active[1], active[2]
           atomsH = atoms + Atom('H', (H_x, H_y, H_z))

        return atomsH

    def four_electron_transition_gen(self, active=None, dist=1.5, mode='ORR'):
        """
        specify initial atomic position on the catalyst

        Parameters
        ----------
        active = list
        dist   = 1.5
            assign the position [A] of adsorbed hydrogen [x, y, z]
            Ex. active = [1.2, 1.35, 23.75]
            if active is None, the atomic position is automatically assigned
            along to highest z-axis position (ontop site) with specific distance (dist=1.5)
        
        mode = 'ORR' or 'OER'
        active = list
            same as HER_transition_gen
        """
        atoms = self.atoms
        
        if active is None:
           top_index = self.get_zmax_index()
           top_x, top_y, top_z = atoms[top_index][0], atoms[top_index][1], atoms[top_index][2]
           active_position = (top_x, top_y, top_z+dist)
        else:
           active_position = (active[0], active[1], active[2])

        atomsO2  = atoms \
                   + Atom('O', (active_position[0], active_position[1], active_position[2])) \
                   + Atom('O', (active_position[0]-1.000, active_position[1]+0.400, active_position[2]+0.600)) 

        atomsOOH = atoms \
                   + Atom('O', (active_position[0], active_position[1], active_position[2])) \
                   + Atom('O', (active_position[0]-0.403, active_position[1]+1.054, active_position[2]+0.733)) \
                   + Atom('H', (active_position[0]-1.350, active_position[1]+1.196, active_position[2]+0.480))
        atomsO   = atoms \
                   + Atom('O', (active_position[0], active_position[1], active_position[2])) 

        atomsOH  = atoms \
                   + Atom('O', (active_position[0], active_position[1], active_position[2])) \
                   + Atom('H', (active_position[0], active_position[1], active_position[2]+0.971))

        if mode == 'ORR':
            return atomsO2, atomsOOH, atomsO, atomsOH 
        
        elif mode == 'OER':
            return atomsOH, atomsO, atomsOOH
        
    
    
class Calculation:
    
    def free_energies(temp=298.15, pH=0, p=0.035, sol=0):

        E_H2O    = -14.236         # Total energy for H2O(g) in VASP
        if sol == 1:
            E_H2O = E_H2O - 0.310
        else:
            pass
        ZPE_H2O  =  0.560          # Norskov parameter
        S_H2O    =  0.00223333333  # Norskov parameter 0.67 eV at 300 K
        G_H2O_g = E_H2O + ZPE_H2O - temp*S_H2O
        
        E_H2   =  -6.760           # Total energy for H2(g) in VASP
        if sol == 1:
            E_H2 = E_H2 + 0.056
        else:
            pass
        ZPE_H2   = 0.270           # Norskov parameter
        TS_H2   =  0.001366666666  # Norskov parameter 0.41 eV at 300 K
        G_H2_g  = E_H2  + ZPE_H2  - temp*TS_H2
        
        R      = 0.0000861733254056734 # eV / K
        kB     = 0.0000861733254056734 # eV / K
        p0     = 1
        E_pH   = kB * temp * np.log(10) * pH
        
        # H2O
        G_H2O_aq = G_H2O_g + R*temp*np.log(p/p0) # only available for around p=0.035, T=298.15K

        # O2
        O2_g     = 2 * G_H2O_aq - 2 * G_H2_g + 4.92 # O2(g) from O2 + 2H2 -> 2H2O
        
        # H+
        H_ion    = 0.5*G_H2_g - E_pH
        
        # OH-
        OH_ion   = G_H2O_aq - H_ion
        
        return G_H2O_aq, O2_g, H_ion, OH_ion

    def Gibbs_HER(Sys, SysH, ZPE=None, TS=None):
        E_H2   =  -6.760;  ZPE_H2   = 0.270; TS_H2   = 0.410 
        
        n_component = len(Sys)

        if ZPE is None or TS is None:
            ZPE = []
            for i in range(n_component):
                ZPE_default = 0.04 + 0.5*ZPE_H2  # Cu111 + H (Norskov) J. Electro. Soc. 152(3), J23 (2005)
                ZPE.append(ZPE_default)
        else:
            ZPE = ZPE

        if TS is None:
            TS = []
            for i in range(n_component):
                TS_default  = 0.20 + 0.5*TS_H2   # Cu111 + H (Norskov)
                TS.append(TS_default)
        else:
            TS = TS

        if len(SysH) + len(ZPE) + len(TS) - 3*n_component == 0:
            pass
        else:
            print("The input components do not match")
            print("Check the input list")
            print("E_System:", len(Sys))
            print("E_System+H:", len(SysH))
            print("E_ZPE:", len(ZPE))
            print("E_TS:", len(TS))
       
        Gibbs_H = []

        for i in range(n_component):
            delta_E    = SysH[i] - (Sys[i] + 0.5*E_H2)
            delta_ZPE  = ZPE[i] - 0.5*ZPE_H2
            delta_TS   = TS[i]  - 0.5*TS_H2
            delta_G    = delta_E + delta_ZPE + delta_TS
            Gibbs_H.append(delta_G)

        return Gibbs_H
  
    def Gibbs_ORR_4e_acid(TE, ZPE=None, TS=None, temp=298.15, pH=0, p=0.035, sol=0):
        
        n_component = len(TE)  
        # TE (DFT total energy) must contain a series of energies
        # TE = [Sys, Sys+O2, Sys+OOH, Sys+O, Sys+OH]
                                                                                                                               
        if ZPE is None:
            ZPE = [0.000, 0.169, 0.511, 0.117, 0.430]
            # reference values from Nat.commun. 8, 15938 (2017) STable 7
        else:
            ZPE = ZPE
        
        if TS is None:
            TS = [0.000, 0.097, 0.087, 0.020, 0.030]
            # reference values from Nat.commun. 8, 15938 (2017) STable 7
        else:
            TS = TS

        if len(TE) + len(ZPE) + len(TS) - 3*n_component == 0:
            pass
        else:
            print("The input components do not match")
            print("Check the input list")
            print("TE:", len(TE))
            print("ZPE:", len(ZPE))
            print("TS:", len(TS))
         
        G_H2O_aq, O2_g, H_ion, OH_ion = Calculation.free_energies(temp=temp, pH=pH, p=p, sol=sol)
        G_Sys     = TE[0] + ZPE[0] - TS[0] + 4 * H_ion + 0 * G_H2O_aq + 1 * O2_g
        G_SysO2   = TE[1] + ZPE[1] - TS[1] + 4 * H_ion + 0 * G_H2O_aq
        G_SysOOH  = TE[2] + ZPE[2] - TS[2] + 3 * H_ion + 0 * G_H2O_aq
        G_SysO    = TE[3] + ZPE[3] - TS[3] + 2 * H_ion + 1 * G_H2O_aq
        G_SysOH   = TE[4] + ZPE[4] - TS[4] + 1 * H_ion + 1 * G_H2O_aq                         
        G_Sys_end = TE[0] + ZPE[0] - TS[0] + 0 * H_ion + 2 * G_H2O_aq 
        Gibbs_E   = [G_Sys, G_SysO2, G_SysOOH, G_SysO, G_SysOH, G_Sys_end]

        return Gibbs_E

    def Gibbs_ORR_4e_alkaline(TE, ZPE=None, TS=None, temp=298.15, pH=14, p=0.035, sol=0):
        
        n_component = len(TE)  
        # TE (DFT total energy) must contain a series of energies
        # TE = [Sys, Sys+O2, Sys+OOH, Sys+O, Sys+OH]
                                                                                                                               
        if ZPE is None:
            ZPE = [0.000, 0.169, 0.511, 0.117, 0.430]
            # reference values from Nat.commun. 8, 15938 (2017) STable 7
        else:
            ZPE = ZPE
        
        if TS is None:
            TS = [0.000, 0.097, 0.087, 0.020, 0.030]
            # reference values from Nat.commun. 8, 15938 (2017) STable 7
        else:
            TS=TS
                                                                                                                       
        if len(TE) + len(ZPE) + len(TS) - 3*n_component == 0:
            pass
        else:
            print("The input components do not match")
            print("Check the input list")
            print("TE:", len(TE))
            print("ZPE:", len(ZPE))
            print("TS:", len(TS))
        
        G_H2O_aq, O2_g, H_ion, OH_ion = Calculation.free_energies(temp=temp, pH=pH, p=p, sol=sol)
        G_Sys     = TE[0] + ZPE[0] - TS[0] + 0 * OH_ion + 2 * G_H2O_aq + 1 * O2_g
        G_SysO2   = TE[1] + ZPE[1] - TS[1] + 0 * OH_ion + 2 * G_H2O_aq
        G_SysOOH  = TE[2] + ZPE[2] - TS[2] + 1 * OH_ion + 1 * G_H2O_aq
        G_SysO    = TE[3] + ZPE[3] - TS[3] + 2 * OH_ion + 1 * G_H2O_aq
        G_SysOH   = TE[4] + ZPE[4] - TS[4] + 3 * OH_ion + 0 * G_H2O_aq                         
        G_Sys_end = TE[0] + ZPE[0] - TS[0] + 4 * OH_ion + 0 * G_H2O_aq 
        Gibbs_E   = [G_Sys, G_SysO2, G_SysOOH, G_SysO, G_SysOH, G_Sys_end]
        
        return Gibbs_E

    def Gibbs_OER_4e_acid(TE, ZPE=None, TS=None, temp=298.15, pH=0, p=0.035):
        
        n_component = len(TE)  
        # TE (DFT total energy) must contain a series of energies
        # TE = [Sys, Sys+O2, Sys+OOH, Sys+O, Sys+OH]
                                                                                                                               
        if ZPE is None:
            ZPE = [0.000, 0.430, 0.117, 0.511]
            # reference values from Nat.commun. 8, 15938 (2017) STable 7
        else:
            ZPE = ZPE
        
        if TS is None:
            TS = [0.000, 0.030, 0.020, 0.087]
            # reference values from Nat.commun. 8, 15938 (2017) STable 7
        else:
            TS = TS
                                                                                                                                
        if len(TE) + len(ZPE) + len(TS) - 3*n_component == 0:
            pass
        else:
            print("The input components do not match")
            print("Check the input list")
            print("TE:", len(TE))
            print("ZPE:", len(ZPE))
            print("TS:", len(TS))
         
        G_H2O_aq, O2_g, H_ion, OH_ion = Calculation.free_energies(temp=temp, pH=pH, p=p)
        G_Sys     = TE[0] + ZPE[0] - TS[0] + 0 * H_ion + 2 * G_H2O_aq 
        G_SysOH   = TE[1] + ZPE[1] - TS[1] + 1 * H_ion + 1 * G_H2O_aq
        G_SysO    = TE[2] + ZPE[2] - TS[2] + 2 * H_ion + 1 * G_H2O_aq
        G_SysOOH  = TE[3] + ZPE[3] - TS[3] + 3 * H_ion + 0 * G_H2O_aq
        G_Sys_end = TE[0] + ZPE[0] - TS[0] + 4 * H_ion + 0 * G_H2O_aq + 1 * O2_g 
        Gibbs_E   = [G_Sys, G_SysOH, G_SysO, G_SysOOH, G_Sys_end]
                                                                                                                                
        return Gibbs_E

class Performance:

    def plot_HER(Gibbs_H, legend=None, y_lower=-1, y_upper=1, label="HER", y_ticks=[-1.0, -0.5, 0, 0.5, 1.0], dpi=600):
        """
        plot HER profile with python-matplotlib
        this module is independent, so if you can use only hand-shaved values
        
        Parameters
        ----------
        Gibbs_H: list
           Gibbs free energy which to be directly plotted.
           Ex. [-0.32, -0.11, 0.12]
           tips: module run_HER returns Gibbs_H value, therefore, append the value on the list.
        
        legend:
           the legend when matplotlib will visualize.
           this is only available for number of dataset is matched between Gibbs_H and legend.
           otherwise, the legend is automatically assigend just [1, 2, 3 ...]
        """
        e_Gibbs_H = Gibbs_H

        X = [0, 1, 2]
        Z = []

        if legend is None:
            for i in range(len(e_Gibbs_H)):
                Z.append(i+1)
        elif len(Gibbs_H) == len(legend):
            Z = legend
        else:
            raise ValueError("The number of data between Gibbs energies and their legends is not matched")

        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 8.5))
        plt.rcParams['axes.linewidth'] = 2.0
        X_domain    = [-0.5,  0.0,  2.5]
        upper_limit = [ 0.2,  0.2,  0.2]
        lower_limit = [-0.2, -0.2, -0.2]

        hori1 = plt.plot(X_domain, upper_limit, linestyle='dotted', linewidth='0.5', color='k')
        hori2 = plt.plot(X_domain, lower_limit, linestyle='dotted', linewidth='0.5', color='k')
        plt.fill_between(X_domain, upper_limit, 0, color = 'whitesmoke')
        plt.fill_between(X_domain, lower_limit, 0, color = 'whitesmoke')

        for i in range(len(e_Gibbs_H)):
            Y = [0, e_Gibbs_H[i], 0]
            plt.plot(X, Y, marker='_', ms='50', mew='5', linestyle='dashed', linewidth='1', label='%s' % str(Z[i]))
        plt.axis([-0.4, 2.4, y_lower, y_upper])
        plt.xticks([0, 1, 2], [r' H$^{+}$+e$^{-}$ ', r' H$^{*}$ ', r' 1/2H$_{2}$ '], fontsize=20)
        plt.yticks(y_ticks, fontsize=20)
        plt.tick_params(axis="y", direction="in", length=10, width=2)
        plt.tick_params(axis="x", direction="in", length=10, width=2)
        plt.xlabel('Reaction Coordinate', fontsize=25)
        plt.ylabel('Free Energy [eV]', fontsize=25)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={"size":15}, frameon=False)
        plt.savefig('%s.png' % label, format='png', dpi=dpi, bbox_inches = 'tight')

    def plot_ORR_4e_acid(Gibbs_E, U=None, legend=None, y_lower=-2, y_upper=6, label="ORR_4e_acid", y_ticks=[-2, 0, 2, 4, 6], dpi=600, G_cal=0):
        
        X        = [-1, 0, 1, 2, 3, 4, 5, 6]
        
        G_eq  = Gibbs_E 
        
        # Gibbs free energy with applied potential
        
        applied_U = [1.23]
        
        elec_conv = -1
        X         = [-1, 0, 1, 2, 3, 4, 5, 6]
        G_eU_max  = [G_eq[0]-G_eq[5]+4*elec_conv*applied_U[0], # dummy for first step plot
                     G_eq[0]-G_eq[5]+4*elec_conv*applied_U[0], # G_System+O2   + 4*e-
                     G_eq[1]-G_eq[5]+4*elec_conv*applied_U[0], # G_System-O2*  + 4*e-
                     G_eq[2]-G_eq[5]+3*elec_conv*applied_U[0], # G_System-OOH* + 3*e-
                     G_eq[3]-G_eq[5]+2*elec_conv*applied_U[0], # G_System+O    + 2*e-
                     G_eq[4]-G_eq[5]+1*elec_conv*applied_U[0], # G_System+OH   + 1*e-
                     G_eq[5]-G_eq[5]+0*elec_conv*applied_U[0], # G_System      + 0*e-
                     G_eq[5]-G_eq[5]+0*elec_conv*applied_U[0], # dummy for last step plot
                    ]

        G_rel     = []
      
        for i in range(len(G_eU_max)-1):      
            rel_G = G_eU_max[i+1] - G_eU_max[i]   
            G_rel.append(rel_G)           
                                        
        if U is None:                    
            U_theory = 1.23 - max(G_rel) 
            if U_theory < 0:
                U = 0
            else:
                U = U_theory                 
        else:                            
            U = U                        
        
        if legend is None:
            legend = ['U=1.23V', 'U=%3.2fV' % U, 'U=0.00V']
        else:
            legend = legend
        
        applied_U = [1.23, U, 0]
        
        G_eU      = [G_eq[0]-G_eq[5]+4*elec_conv*applied_U[1], # dummy for first step plot
                     G_eq[0]-G_eq[5]+4*elec_conv*applied_U[1], # G_System+O2    + 4*e-
                     G_eq[1]-G_eq[5]+4*elec_conv*applied_U[1], # G_System-O2*   + 4*e-
                     G_eq[2]-G_eq[5]+3*elec_conv*applied_U[1], # G_System-OOH*  + 3*e-
                     G_eq[3]-G_eq[5]+2*elec_conv*applied_U[1], # G_System+O     + 2*e-
                     G_eq[4]-G_eq[5]+1*elec_conv*applied_U[1], # G_System+OH    + 1*e-
                     G_eq[5]-G_eq[5]+0*elec_conv*applied_U[1], # G_System       + 0*e-
                     G_eq[5]-G_eq[5]+0*elec_conv*applied_U[1], # dummy for last step plot
                    ]
        
        G_eU_min  = [G_eq[0]-G_eq[5]+4*elec_conv*applied_U[2], # dummy for first step plot
                     G_eq[0]-G_eq[5]+4*elec_conv*applied_U[2], # G_System+O2    + 4*e-
                     G_eq[1]-G_eq[5]+4*elec_conv*applied_U[2], # G_System-O2*   + 4*e-
                     G_eq[2]-G_eq[5]+3*elec_conv*applied_U[2], # G_System-OOH*  + 3*e-
                     G_eq[3]-G_eq[5]+2*elec_conv*applied_U[2], # G_System+O     + 2*e-
                     G_eq[4]-G_eq[5]+1*elec_conv*applied_U[2], # G_System+OH    + 1*e-
                     G_eq[5]-G_eq[5]+0*elec_conv*applied_U[2], # G_System       + 0*e-
                     G_eq[5]-G_eq[5]+0*elec_conv*applied_U[2], # dummy for last step plot
                   ]
        
        if G_cal:
            return G_eU     
        else:
            pass
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8.5, 6.5))
        plt.rcParams['axes.linewidth'] = 2.0
        
        X_dom       = [-2,  0, 4, 10]
        Y_hori      = [ 0,  0, 0,  0]
        plt.scatter(X, G_eU_max, color='black', marker='_', s=6000, linewidth=4, label='%s' % legend[0])
        for i in range(len(X)-1):
            plt.vlines((X[i+1]+X[i])/2, G_eU_max[i], G_eU_max[i+1], linestyle=':', color='black', linewidth=2)
        plt.scatter(X, G_eU, color='red', marker='_', s=6000, linewidth=4, label='%s' % legend[1])
        for i in range(len(X)-1):
            plt.vlines((X[i+1]+X[i])/2, G_eU[i], G_eU[i+1], linestyle=':', color='red', linewidth=2)
        plt.scatter(X, G_eU_min, color='blue', marker='_', s=6000, linewidth=4, label='%s' % legend[2])
        for i in range(len(X)-1):
            plt.vlines((X[i+1]+X[i])/2, G_eU_min[i], G_eU_min[i+1], linestyle=':', color='blue', linewidth=2)           

        plt.axis([-0.5, 5.5, y_lower, y_upper])
        plt.xticks([0, 1, 2, 3, 4, 5], ['O$_{2}$', 'O$_{2}^{*}$', 'OOH$^{*}$', 'O$^{*}$', 'OH$^{*}$', 'H$_{2}$O'], fontsize=20)
        plt.yticks(y_ticks, fontsize=20)
        plt.tick_params(axis="y", direction="in", length=10, width=2)
        plt.tick_params(axis="x", direction="in", length=10, width=2)
        plt.xlabel('Reaction Coordinate', fontsize=25)
        plt.ylabel('Free Energy [eV]', fontsize=25)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5), prop={"size":25}, frameon=False)
        plt.savefig('%s.png' % label, format='png', dpi=dpi, bbox_inches = 'tight')

    def plot_ORR_4e_alkaline(Gibbs_E, U=None, legend=None, y_lower=-6, y_upper=2, label="ORR_4e_alkaline", y_ticks=[-6,-4,-2,0,2],dpi=600, G_cal=0):
        
        X        = [-1, 0, 1, 2, 3, 4, 5, 6]
        
        G_eq  = Gibbs_E 
        
        # Gibbs free energy with applied potential
        
        applied_U = [0.40, U, -0.83]
        elec_conv = -1
        X         = [-1, 0, 1, 2, 3, 4, 5, 6]
        G_eU_max_ref = G_eq[0]-G_eq[5]+(4)*elec_conv*applied_U[0] 
        G_eU_max  = [G_eq[0]-G_eq[5]+(4)*elec_conv*applied_U[0]-G_eU_max_ref, # dummy for first step plot
                     G_eq[0]-G_eq[5]+(4)*elec_conv*applied_U[0]-G_eU_max_ref, # G_System+O2   + 4*e-
                     G_eq[1]-G_eq[5]+(4)*elec_conv*applied_U[0]-G_eU_max_ref, # G_System-O2*  + 4*e-
                     G_eq[2]-G_eq[5]+(3)*elec_conv*applied_U[0]-G_eU_max_ref, # G_System-OOH* + 3*e-
                     G_eq[3]-G_eq[5]+(2)*elec_conv*applied_U[0]-G_eU_max_ref, # G_System+O    + 2*e-
                     G_eq[4]-G_eq[5]+(1)*elec_conv*applied_U[0]-G_eU_max_ref, # G_System+OH   + 1*e-
                     G_eq[5]-G_eq[5]+(0)*elec_conv*applied_U[0]-G_eU_max_ref, # G_System      + 0*e-
                     G_eq[5]-G_eq[5]+(0)*elec_conv*applied_U[0]-G_eU_max_ref, # dummy for last step plot
                    ]
                                                                                                                                
        G_rel     = []
                                                            
        for i in range(len(G_eU_max)-1):      
            rel_G = G_eU_max[i+1] - G_eU_max[i]   
            G_rel.append(rel_G)           
                                        
        if U is None:                    
            U_theory = 0.40 - max(G_rel) 
            if U_theory < -0.83:
                U = -0.83
            else:
                U = U_theory                 
        else:                            
            U = U                        
        
        if legend is None:
            legend = ['U=0.40V', 'U=%3.2fV' % U, 'U=-0.83V']
        else:
            legend = legend
        
        applied_U = [0.40, U, -0.83]
        G_eU_ref  = G_eq[0]-G_eq[5]+(4)*elec_conv*applied_U[1]
        G_eU      = [G_eq[0]-G_eq[5]+(4)*elec_conv*applied_U[1]-G_eU_ref, # dummy for first step plot
                     G_eq[0]-G_eq[5]+(4)*elec_conv*applied_U[1]-G_eU_ref, # G_System+O2    + 4*e-
                     G_eq[1]-G_eq[5]+(4)*elec_conv*applied_U[1]-G_eU_ref, # G_System-O2*   + 4*e-
                     G_eq[2]-G_eq[5]+(3)*elec_conv*applied_U[1]-G_eU_ref, # G_System-OOH*  + 3*e-
                     G_eq[3]-G_eq[5]+(2)*elec_conv*applied_U[1]-G_eU_ref, # G_System+O     + 2*e-
                     G_eq[4]-G_eq[5]+(1)*elec_conv*applied_U[1]-G_eU_ref, # G_System+OH    + 1*e-
                     G_eq[5]-G_eq[5]+(0)*elec_conv*applied_U[1]-G_eU_ref, # G_System       + 0*e-
                     G_eq[5]-G_eq[5]+(0)*elec_conv*applied_U[1]-G_eU_ref, # dummy for last step plot
                    ]
        G_eU_min_ref = G_eq[0]-G_eq[5]+(4)*elec_conv*applied_U[2]
        G_eU_min  = [G_eq[0]-G_eq[5]+(4)*elec_conv*applied_U[2]-G_eU_min_ref, # dummy for first step plot
                     G_eq[0]-G_eq[5]+(4)*elec_conv*applied_U[2]-G_eU_min_ref, # G_System+O2    + 4*e-
                     G_eq[1]-G_eq[5]+(4)*elec_conv*applied_U[2]-G_eU_min_ref, # G_System-O2*   + 4*e-
                     G_eq[2]-G_eq[5]+(3)*elec_conv*applied_U[2]-G_eU_min_ref, # G_System-OOH*  + 3*e-
                     G_eq[3]-G_eq[5]+(2)*elec_conv*applied_U[2]-G_eU_min_ref, # G_System+O     + 2*e-
                     G_eq[4]-G_eq[5]+(1)*elec_conv*applied_U[2]-G_eU_min_ref, # G_System+OH    + 1*e-
                     G_eq[5]-G_eq[5]+(0)*elec_conv*applied_U[2]-G_eU_min_ref, # G_System       + 0*e-
                     G_eq[5]-G_eq[5]+(0)*elec_conv*applied_U[2]-G_eU_min_ref, # dummy for last step plot
                   ]
        
        if G_cal:
            return G_eU    
        else:
            pass

        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8.5, 6.5))
        plt.rcParams['axes.linewidth'] = 2.0
        
        X_dom       = [-2,  0, 4, 10]
        Y_hori      = [ 0,  0, 0,  0]
        plt.scatter(X, G_eU_max, color='black', marker='_', s=6000, linewidth=4, label='%s' % legend[0])                 
        for i in range(len(X)-1):
            plt.vlines((X[i+1]+X[i])/2, G_eU_max[i], G_eU_max[i+1], linestyle=':', color='black', linewidth=2)
        plt.scatter(X, G_eU, color='red', marker='_', s=6000, linewidth=4, label='%s' % legend[1])
        for i in range(len(X)-1):
            plt.vlines((X[i+1]+X[i])/2, G_eU[i], G_eU[i+1], linestyle=':', color='red', linewidth=2)
        plt.scatter(X, G_eU_min, color='blue', marker='_', s=6000, linewidth=4, label='%s' % legend[2])
        for i in range(len(X)-1):
            plt.vlines((X[i+1]+X[i])/2, G_eU_min[i], G_eU_min[i+1], linestyle=':', color='blue', linewidth=2)           
                                                                                                                                
        plt.axis([-0.5, 5.5, y_lower, y_upper])
        plt.xticks([0, 1, 2, 3, 4, 5], ['O$_{2}$', 'O$_{2}^{*}$', 'OOH$^{*}$', 'O$^{*}$', 'OH$^{*}$', 'OH$^{-}$'], fontsize=20)
        plt.yticks(y_ticks, fontsize=20)
        plt.tick_params(axis="y", direction="in", length=10, width=2)
        plt.tick_params(axis="x", direction="in", length=10, width=2)
        plt.xlabel('Reaction Coordinate', fontsize=25)
        plt.ylabel('Free Energy [eV]', fontsize=25)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5), prop={"size":25}, frameon=False)
        plt.savefig('%s.png' % label, format='png', dpi=dpi, bbox_inches = 'tight')

    def plot_OER_4e_acid(Gibbs_E, U=None, legend=None, y_lower=-6, y_upper=6, label="OER_4e_acid", y_ticks=[-6, -4, -2, 0, 2, 4, 6], dpi=600, G_cal=0):
        
        X        = [-1, 0, 1, 2, 3, 4, 5]
        
        G_eq  = Gibbs_E 
        
        # Gibbs free energy with applied potential
        
        applied_U = [1.23]
        
        elec_conv = -1
        X         = [-1, 0, 1, 2, 3, 4, 5]
        G_eU_max_ref = G_eq[0]-G_eq[4]+0*elec_conv*applied_U[0]
        G_eU_max  = [G_eq[0]-G_eq[4]+0*elec_conv*applied_U[0]-G_eU_max_ref, # dummy for first step plot
                     G_eq[0]-G_eq[4]+0*elec_conv*applied_U[0]-G_eU_max_ref, # G_System+2H2O + 0*e-
                     G_eq[1]-G_eq[4]+1*elec_conv*applied_U[0]-G_eU_max_ref, # G_System-OH*  + 1*e-
                     G_eq[2]-G_eq[4]+2*elec_conv*applied_U[0]-G_eU_max_ref, # G_System-O*   + 2*e-
                     G_eq[3]-G_eq[4]+3*elec_conv*applied_U[0]-G_eU_max_ref, # G_System+OOH* + 3*e-
                     G_eq[4]-G_eq[4]+4*elec_conv*applied_U[0]-G_eU_max_ref, # G_System+O2   + 4*e-
                     G_eq[4]-G_eq[4]+4*elec_conv*applied_U[0]-G_eU_max_ref, # dummy for last step plot 
                    ]
                                                                                                                                                
        G_rel     = []
      
        for i in range(len(G_eU_max)-1):      
            rel_G = G_eU_max[i+1] - G_eU_max[i]   
            G_rel.append(rel_G)           
                                        
        if U is None:                    
            U_theory = 1.23 + max(G_rel) 
            if U_theory > 2.46:
                U = 2.46
            else:
                U = U_theory                 
        else:                            
            U = U                        
        
        if legend is None:
            legend = ['U=1.23V', 'U=%3.2fV' % U, 'U=0.00V']
        else:
            legend = legend
        
        applied_U = [1.23, U, 0]
        G_eU_ref  =  G_eq[0]-G_eq[4]+0*elec_conv*applied_U[1],
        G_eU      = [G_eq[0]-G_eq[4]+0*elec_conv*applied_U[1]-G_eU_ref, # dummy for first step plot
                     G_eq[0]-G_eq[4]+0*elec_conv*applied_U[1]-G_eU_ref, # G_System+2H2O  + 0*e-
                     G_eq[1]-G_eq[4]+1*elec_conv*applied_U[1]-G_eU_ref, # G_System-OH*   + 1*e-
                     G_eq[2]-G_eq[4]+2*elec_conv*applied_U[1]-G_eU_ref, # G_System-O*    + 2*e-
                     G_eq[3]-G_eq[4]+3*elec_conv*applied_U[1]-G_eU_ref, # G_System+OOH*  + 3*e-
                     G_eq[4]-G_eq[4]+4*elec_conv*applied_U[1]-G_eU_ref, # G_System+O2    + 4*e-
                     G_eq[4]-G_eq[4]+4*elec_conv*applied_U[1]-G_eU_ref, # dummy for last step plot
                    ]
        
        G_eU_min_ref = G_eq[0]-G_eq[4]+0*elec_conv*applied_U[2],
        G_eU_min  = [G_eq[0]-G_eq[4]+0*elec_conv*applied_U[2]-G_eU_min_ref, # dummy for first step plot
                     G_eq[0]-G_eq[4]+0*elec_conv*applied_U[2]-G_eU_min_ref, # G_System+2H2O  + 0*e-
                     G_eq[1]-G_eq[4]+1*elec_conv*applied_U[2]-G_eU_min_ref, # G_System-OH*   + 1*e-
                     G_eq[2]-G_eq[4]+2*elec_conv*applied_U[2]-G_eU_min_ref, # G_System-O*    + 2*e-
                     G_eq[3]-G_eq[4]+3*elec_conv*applied_U[2]-G_eU_min_ref, # G_System+OOH*  + 3*e-
                     G_eq[4]-G_eq[4]+4*elec_conv*applied_U[2]-G_eU_min_ref, # G_System+O2    + 4*e-
                     G_eq[4]-G_eq[4]+4*elec_conv*applied_U[2]-G_eU_min_ref, # dummy for last step plot
                   ]
        
        if G_cal:
            return G_eU     
        else:
            pass
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8.5, 6.5))
        plt.rcParams['axes.linewidth'] = 2.0
        
        X_dom       = [-2,  0, 4, 10]
        Y_hori      = [ 0,  0, 0,  0]
        plt.scatter(X, G_eU_max, color='black', marker='_', s=8000, linewidth=4, label='%s' % legend[0])
        for i in range(len(X)-1):
            plt.vlines((X[i+1]+X[i])/2, G_eU_max[i], G_eU_max[i+1], linestyle=':', color='black', linewidth=2)
        plt.scatter(X, G_eU, color='red', marker='_', s=8000, linewidth=4, label='%s' % legend[1])
        for i in range(len(X)-1):
            plt.vlines((X[i+1]+X[i])/2, G_eU[i], G_eU[i+1], linestyle=':', color='red', linewidth=2)
        plt.scatter(X, G_eU_min, color='blue', marker='_', s=8000, linewidth=4, label='%s' % legend[2])
        for i in range(len(X)-1):
            plt.vlines((X[i+1]+X[i])/2, G_eU_min[i], G_eU_min[i+1], linestyle=':', color='blue', linewidth=2)           
                                                                                                                                                
        plt.axis([-0.50, 4.50, y_lower, y_upper])
        plt.xticks([0, 1, 2, 3, 4], ['H$_{2}$O', 'OH$^{*}$', 'O$^{*}$', 'OOH$^{*}$', 'O$_{2}$'], fontsize=20)
        plt.yticks(y_ticks, fontsize=20)
        plt.tick_params(axis="y", direction="in", length=10, width=2)
        plt.tick_params(axis="x", direction="in", length=10, width=2)
        plt.xlabel('Reaction Coordinate', fontsize=25)
        plt.ylabel('Free Energy [eV]', fontsize=25)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5), prop={"size":25}, frameon=False)
        plt.savefig('%s.png' % label, format='png', dpi=dpi, bbox_inches = 'tight')
