from . atoms import *
from . import io
from . io import cleansymb, get_unique_symbs, convert_xyz2abc, ang2bohr
from . units import ang2bohr
from glob import glob
import os, math
import numpy as np

#
# VASP Simulation Object
#

class Vasp(object):

    """
    Vasp(atoms)

    Class for management of VASP simulation

    Parameters
    ----------
    symbol  : AtomsSystem
        Class instance of AtomsSystem

    Optional parameters
    -------------------
    """


    def __init__(self, atoms):
        self.atoms = atoms 
        self._params = {
              #1. Name and basic options       
               'SYSTEM'      :     'vasp',       # text, system name
               'NPAR'        :          1,       # integer, number of bands
               'IBRION'      :          2,       # 2=CG/Default, 5=Hessian 
               'LWAVE'       :        'F',       # boolean, write WAVECAR
               'LCHARG'      :        'F',       # boolean, write CHGCAR
               'NSW'         :          0,       # integer, optimization step
               'PREC'        : 'Accurate',       # precision (Low, Normal, Accurate)
               'ALGO'        :     'FAST',       # Algorithm, GGA/LDA=Normal, Fast
               'ISTART'      :          0,       # 0:new, 1:WAVECAR, 2:samecutoff
               'ICHARG'      :          2,       # charge from 0:WAVECAR, 1:file, 2:atomic, 11:keep CHARGE
               'ISIF'        :          2,       # 2=Constant cell, 3=relax cell
              #2. SCF/kgrid/functional parameters          
               'ENCUT'       :        400,       # float, plane wave basis energy cutoff
               'ISMEAR'      :          0,       # integer, smearing 0=Gauss, 1=Metal
               'SIGMA'       :       0.05,       # positive float
               'NSIM'        :          1,       # integer, bands optimized in RMM-DIIS
               'NELMIN'      :          4,       # integer, min SCF steps
               'NELM'        :        500,       # integer, max SCF steps
               'EDIFF'       :     0.0001,       # float, tolerance of ground state
               'KPOINTS'     :  [1, 1, 1],       # list, 3-vector
               'XC'          :      'GGA',       # GGA, LDA
               'XCAUTHOR'    :      'PE' ,       # PE=PBE, 91=PW91, RP=Revised PBE 
              #3. Optional parameters 
               'POTIM'       :        0.3,       # displacement  
               'EDIFFG'      :      -0.05,       # float, stopping relaxation loop
               'IVDW'        :         12,       # 11: D3 zero damping 12: D3 BJ damping
               'LDIPOL'      :        'F',       # dipole correction
               'IDIPOL'      :          3,       # 1: x, 2: y, 3: z, 4: all
               'LPLANE'      :        'T',       # data distribution over Nodes
               'ADDGRID'     :        'T',       # add grid for charge augmentation
               'LREAL'       :     'Auto',       # for real space projection
               'ISYM'        :         -1,       # -1 = symmetry off completely
               'LASPH'       :        'T',       # non-spherical contribtuion
               'LMAXMIX'     :          4,       # Density Mixer handles quantumNumber upto (4: d-elements, 6: f-elements)
               'ISPIN'       :          2,       # 1 = Spin-restricted, 2 = spin-unrestricted
              }

    def get_options(self):
        """
        print the list of available options and their default values
       
        Parameters
        ----------

        Optional parameters
        -------------------

        Example
        -------
        >>> sim.get_options()
        """
        return self._params.items()

    def file_read(fname):
        lineinfo = []
        wordinfo = []
        with open(fname) as f:
            for i, l in enumerate(f):
                line = l
                word = line.split()
                lineinfo.append(line)
                wordinfo.append(word)

        return lineinfo, wordinfo
    
    def set_option(self, key, value):
        
        """
        change the options

        available key and default values
        --------------------------------

        Parameters
        ----------
        key: str
            option name
        value: (various)
            option name

        Optional parameters
        -------------------

        Example
        -------
        >>> sim.set_options('KPOINTS', [5, 5, 1])
        """
        if key not in self._params.keys():
            raise ValueError("Invalid option," + key)
        else:
            self._params[key] = value

    def write_POSCAR(self, file_name='POSCAR', mode='cartesian', fix=None):
        components = self.atoms.get_contents().items()
        message  = ' '
        for i in components:
            message = message + str(i[0]) + '   '
        cell1    = self.atoms.get_cell()[0]
        cell2    = self.atoms.get_cell()[1]
        cell3    = self.atoms.get_cell()[2]

        #-------------POSCAR--------------------
        POSCAR = open(file_name, 'w')
        POSCAR.write("%s\n" % message)
        POSCAR.write("1.000  # fixed lattice parameter unit\n")
        POSCAR.write("%15.9f %15.9f %15.9f\n" % tuple(cell1))
        POSCAR.write("%15.9f %15.9f %15.9f\n" % tuple(cell2))
        POSCAR.write("%15.9f %15.9f %15.9f\n" % tuple(cell3))
        atm_line = ''; len_line = ''
        lines = []
        for sym, num in components:
            self.atoms.select_elements(sym)
            atoms1 = self.atoms.copy_atoms()
            atm_line = atm_line + sym      + '   '
            len_line = len_line + str(num) + '   ' 
            for atom in atoms1:
                x = 0. ; y = 0.; z = 0.
                if mode == 'cartesian':
                   x, y, z = Vector(atom.get_position())
                elif mode == 'direct':
                   x, y, z = Vector(atom.get_position())
                   x = x/(cell1[0] + cell1[1] + cell1[2])
                   y = y/(cell2[0] + cell2[1] + cell2[2])
                   z = z/(cell3[0] + cell3[1] + cell3[2])
                lines.append("%15.9f %15.9f %15.9f" % (x, y, z))
        atm_line += '\n'; len_line += '\n'
        POSCAR.write(atm_line)
        POSCAR.write(len_line)
        POSCAR.write("Selective Dynamics # constraints enabled\n")

        if mode == "cartesian":
            POSCAR.write("Cartesian \n")
        elif mode == "direct":
            POSCAR.write("Direct \n")
        
        for i in range(len(lines)):
            idx = i+1
            if fix == None:
                POSCAR.write(str(lines[i]) + "   T   T   T \n")
            elif fix is not None:
                if idx in fix:
                    POSCAR.write(str(lines[i]) + "   F   F   F \n")
                else:
                    POSCAR.write(str(lines[i]) + "   T   T   T \n")
        POSCAR.close()
    
    def write_KPOINTS(self):
        #-------------KPOINTS-------------------        
        KPOINTS = open('KPOINTS', 'w')
        KPOINTS.write("k-points\n")
        KPOINTS.write("0\n")
        KPOINTS.write("G\n")
        KPOINTS.write("%i %i %i\n" % (self._params['KPOINTS'][0], self._params['KPOINTS'][1], self._params['KPOINTS'][2]))
        KPOINTS.write("0 0 0 \n")
        KPOINTS.close()

    def write_POTCAR(self, xc='PBE'):
        #-------------POTCAR--------------------         
        from NanoCore.env import vasp_POTCAR_LDA  as LDA_path
        from NanoCore.env import vasp_POTCAR_PBE  as PBE_path
        from NanoCore.env import vasp_POTCAR_PW91 as PW91_path
        if xc == 'PBE':
            POTCAR_PATH = PBE_path
        elif xc == 'LDA':
            POTCAR_PATH = LDA_path
        elif xc == 'PW91':
            POTCAR_PATH = PW91_path
        else:
            print("select type of XC in PBE, LDA, PW91")

        components = self.atoms.get_contents().items()
        element = []; element_refine = []
        for sym, num in components:
            element.append(sym)
        
        sv_list = ["Li", "K", "Rb", "Cs", "Sr", "Ba", "Sc", "Y", "Zr"]
        pv_list = ["Na", "Ca", "Ti", "V", "Cr", "Mn", "Nb", "Mo", "Tc", "Hf", "Ta", "W", "Os"]
        d_list  = ["Ga", "Ge", "In", "Sn", "Tl", "Pb", "Bi", "Po", "At"]

        for name in element:
            if name in sv_list:
                element_refine.append(name+'_sv')
            elif name in pv_list:
                element_refine.append(name+'_pv')
            elif name in d_list:
                element_refine.append(name+'_d')
            else:
                element_refine.append(name)
        
        cmd = 'cat'
        os.system('rm -rf POTCAR')
        for element in element_refine:
            addcmd = ' ' + '%s/%s/POTCAR' % (POTCAR_PATH, element)
            cmd = cmd + addcmd
        cmd = cmd + ' > POTCAR'
        os.system('%s' % cmd)

    def write_INCAR(self):
        #-------------INCAR---------------------
        INCAR = open('INCAR', 'w')
        INCAR.write("# VASP general descriptors \n\n")
        INCAR.write("SYSTEM        =   %s\n" % self._params['SYSTEM'])
        INCAR.write("NPAR          =   %i\n" % int(self._params['NPAR']))
        INCAR.write("IBRION        =   %i\n" % int(self._params['IBRION']))
        INCAR.write("LWAVE         =   %s\n" % self._params['LWAVE']) 
        INCAR.write("LCHARG        =   %s\n" % self._params['LCHARG']) 
        INCAR.write("NSW           =   %i\n" % self._params['NSW']) 
        INCAR.write("PREC          =   %s\n" % self._params['PREC']) 
        INCAR.write("ALGO          =   %s\n" % self._params['ALGO'])
        INCAR.write("ISTART        =   %i\n" % self._params['ISTART']) 
        INCAR.write("ICHARG        =   %i\n" % self._params['ICHARG']) 
        INCAR.write("ISIF          =   %i\n\n" % self._params['ISIF']) 
        INCAR.write("# VASP convergence parameters \n\n")
        INCAR.write("ENCUT         =   %f\n" % float(self._params['ENCUT']))
        INCAR.write("ISMEAR        =   %i\n" % self._params['ISMEAR'])
        INCAR.write("SIGMA         =   %f\n" % self._params['SIGMA'])
        INCAR.write("NSIM          =   %i\n" % self._params['NSIM'])
        INCAR.write("NELMIN        =   %i\n" % self._params['NELMIN'])
        INCAR.write("NELM          =   %i\n" % self._params['NELM'])
        INCAR.write("EDIFF         =   %f\n" % float(self._params['EDIFF']))
        INCAR.write("EDIFFG        =   %f\n" % float(self._params['EDIFFG']))
        INCAR.write("%s           =   %s\n\n" % (self._params['XC'], self._params['XCAUTHOR']))
        INCAR.write("# VASP optional parameters \n\n")
        INCAR.write("POTIM         =   %f\n" % float(self._params['POTIM']))
        INCAR.write("IVDW          =   %i\n" % int(self._params['IVDW']))
        INCAR.write("LDIPOL        =   %s\n" % self._params['LDIPOL'])
        INCAR.write("IDIPOL        =   %i\n" % int(self._params['IDIPOL']))
        INCAR.write("LPLANE        =   %s\n" % self._params['LPLANE'])
        INCAR.write("ADDGRID       =   %s\n" % self._params['ADDGRID'])
        INCAR.write("LREAL         =   %s\n" % self._params['LREAL'])
        INCAR.write("ISYM          =   %i\n" % self._params['ISYM'])
        INCAR.write("LASPH         =   %s\n" % self._params['LASPH'])
        INCAR.write("LMAXMIX       =   %i\n" % self._params['LMAXMIX'])
        INCAR.write("ISPIN         =   %i\n\n" % self._params['ISPIN'])
        INCAR.close()

    def run_VASP(self, mode='single', nproc=1, npar=1, encut=400, kpoints=[1,1,1], 
                 ediff = 0.0001, ediffg = -0.05,  fix=None):
        """ 
        Example:
        --------

        from NanoCore import vasp2
        from NanoCore import vasp_old
        at = vasp_old.read_poscar('POSCAR')
        at2 = vasp2.Vasp(at)
        at2.run_VASP(nproc=8, npar=2, kpoints=[2,2,1])
        
        """
        from NanoCore.env import vasp_calculator as executable
        
        self._params['NPAR']       = npar
        self._params['KPOINTS']    = kpoints
        self._params['ENCUT']      = encut
        self._params['EDIFF']      = ediff
        self._params['EDIFFG']     = ediffg
        
        if mode == 'opt':
            self._params['IBRION'] = 2
            self._params['POTIM']  = 0.300
            self._params['NSW']    = 500
        
        if mode == 'single':
            self._params['IBRION'] = 2
            self._params['POTIM']  = 0.300
            self._params['NSW']    =   0

        
        if mode == 'vib':
            self._params['IBRION'] = 5
            self._params['POTIM']  = 0.015
            self._params['NSW']    = 1

        # run_simulation
        cmd = 'mpirun -np %i %s > stdout.txt' % (nproc, executable)

        self.write_POSCAR(fix=fix)
        self.write_KPOINTS()
        self.write_INCAR()
        self.write_POTCAR()
        
        os.system(cmd)

    def get_total_energy(self, output_name='OUTCAR'):
        
        line_info, word_info = Vasp.file_read(output_name)
        
        VASP_E = []
        for i in range(len(line_info)):
            if 'y  w' in line_info[i]:
                TE = float(word_info[i][-1])
                VASP_E.append(TE)
            else:
                pass
        
        min_E = min(VASP_E)

        return min_E 

    def get_vibration_energy(self, output_name='OUTCAR', Temp=298.15):
        """
        Example:
        --------
        from NanoCore import vasp    
        ZPE, TS = vasp.get_vibration_energy(Temp=300)
        """

        line_info, word_info = Vasp.file_read(output_name)

        ZPE = 0.0; TS = 0.0
        kB  = 0.0000861733576020577 # eV K-1
        kT  = kB * Temp

        VASP_Vib = []
        for i in range(len(line_info)):
            if 'THz' in line_info[i]:
                freq_E = 0.001*float(word_info[i][-2])
                VASP_Vib.append(freq_E)   # for eV
            else:
                pass
        
        for i in range(len(VASP_Vib)):
            energy = VASP_Vib[i]
            x      = energy / kT
            v1     = x / (math.exp(x) - 1)
            v2     = 1 - math.exp(-x)
            E_TS   = v1 - math.log(v2)
                  
            ZPE = ZPE + 0.5*energy
            TS  = TS  + kT*E_TS
        
        return ZPE, TS

    def get_vibration_spectrum(output_name='OUTCAR', start=0, end=6000, npts=None, width=20.0, matplot=1):               
        """
        Example:
        --------
        from NanoCore import vasp_old       
        at = vasp_old.read_poscar('POSCAR')
        at2 = vasp2.Vasp(at)
        at2.get_vibration_specctrum(output_name='OUTCAR_imag', matplot=1, start=-2000, end=6000)
        """
                                                                                                                          
        def file_read(fname):
            lineinfo = []
            wordinfo = []
            with open(fname) as f:
                for i, l in enumerate(f):
                    line = l
                    word = line.split()
                    lineinfo.append(line)
                    wordinfo.append(word)
         
            return lineinfo, wordinfo 
        
        line_info, word_info = file_read(output_name) 
                                                                                                                          
        vib_cm = []
        for i in range(len(line_info)):
            if 'THz' in line_info[i]:
                a = str(word_info[i][1].strip())
                if a[-1] == "=":
                    meV = float(word_info[i][-2])
                    convert = meV * -8.06554
                    vib_cm.append(convert)
                else:
                    meV = float(word_info[i][-2])
                    convert = meV * 8.06554
                    vib_cm.append(convert)
                                                                                                                          
        def fold(frequencies, intensities, start=0, end=6000, npts=None, width=20.0):
            if not npts:
                npts = int((end - start) / width * 10 + 1)
            prefactor = 1; sigma = width / 2. / np.sqrt(2. * np.log(2.))
            spectrum = np.empty(npts)
            energies = np.linspace(start, end, npts)
            for i, energy in enumerate(energies):
                energies[i] = energy
                spectrum[i] = (intensities * 0.5 * width / np.pi / ((frequencies - energy) ** 2 + 0.25 * width**2)).sum()
                                                                                                                          
            return [energies, prefactor * spectrum]
                                                                                                                          
        frequencies = vib_cm
        intensities = np.ones(len(frequencies))
        energies, spectrum = fold(frequencies, intensities, start=start, end=end, width=50.0)
        outdata = np.empty([len(energies), 2])
        outdata.T[0] = energies
        outdata.T[1] = spectrum
                                                                                                                          
        VDOS = open('VDOS.dat', 'w')
        for row in outdata:
            VDOS.write('%.3f %15.5e\n' % (row[0], row[1]))
        VDOS.close()

        if matplot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(7,5))
            plt.rcParams['axes.linewidth'] = 2
                                                                                
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
                                                                                
            line = plt.plot(energies, spectrum, linewidth='2', color='k')
            plt.xlabel('Frequency [cm$^{-1}$]', fontsize=20)
            plt.ylabel('VDOS [1/cm$^{-1}$]', fontsize=20)
            plt.savefig('VDOS.png', format='png', dpi=600, bbox_inches='tight')
        else:
            pass
 
    def run_series_HER(self, mode='opt', nproc=1, npar=1, encut=400, kpoints=[1,1,1], 
                       ediff = 0.0001, ediffg = -0.05, fix=None, active=None, vib=1, label='test'):

        from NanoCore.catalysis import Modeling
        
        atoms = self.atoms
        n_atoms = len(atoms)
        atoms_HER = Vasp(atoms)
        atoms_HER.run_VASP(mode=mode, nproc=nproc, npar=npar, encut=encut, kpoints=kpoints, \
                                ediff=ediff, ediffg=ediffg, fix=fix)
        
        os.system('mv OUTCAR OUTCAR_%s_Sys' % label)
        os.system('mv XDATCAR XDATCAR_%s_Sys' % label)

        TE_Sys = atoms_HER.get_total_energy(output_name='OUTCAR_%s_Sys' % label)

        from NanoCore.vasp_old import read_poscar

        atoms_opt = read_poscar('CONTCAR')
        atoms2 = Modeling(atoms_opt) 
        atomsH = atoms2.HER_transition_gen(active=active)
        
        atomsH_HER = Vasp(atomsH)
        atomsH_HER.run_VASP(mode=mode, nproc=nproc, npar=npar, encut=encut, kpoints=kpoints, \
                                 ediff=ediff, ediffg=ediffg, fix=fix)
        
        os.system('mv OUTCAR OUTCAR_%s_SysH' % label)
        os.system('mv XDATCAR XDATCAR_%s_SysH' % label)  

        TE_SysH = atomsH_HER.get_total_energy(output_name='OUTCAR_%s_SysH' % label)

        if vib:
            atomsH_opt = read_poscar('CONTCAR')
            
            fix_vib = []
            for i in range(n_atoms):
                idx = i+1
                fix_vib.append(idx)

            atomsH_Vib = Vasp(atomsH_opt)
            atomsH_Vib.run_VASP(mode='vib', nproc=nproc, npar=npar, encut=encut, kpoints=kpoints, \
                                     ediff=ediff, ediffg=ediffg, fix=fix_vib)

            os.system('mv OUTCAR OUTCAR_%s_SysH_Vib' % label)
            
            ZPE, TS = atomsH_Vib.get_vibration_energy(output_name='OUTCAR_%s_SysH_Vib' % label)

        if vib:
            return float(TE_Sys), float(TE_SysH), float(ZPE), float(TS)
        else:
            return float(TE_Sys), float(TE_SysH)

    def run_series_ORR(self, mode='opt', nproc=1, npar=1, encut=400, kpoints=[1,1,1],                
                    ediff = 0.0001, ediffg = -0.05, fix=None, active=None, vib=1, label='test'):
                                                                                                 
        from NanoCore.catalysis import Modeling
     
        System       = self.atoms
        n_atoms      = len(System)
        ORR_Sys      = Vasp(System)
        ORR_Sys.run_VASP(mode=mode, nproc=nproc, npar=npar, encut=encut, kpoints=kpoints, \
                             ediff=ediff, ediffg=ediffg, fix=fix)
     
        os.system('mv OUTCAR OUTCAR_%s_Sys' % label)
        os.system('mv XDATCAR XDATCAR_%s_Sys' % label)
        TE_ORR_Sys   = ORR_Sys.get_total_energy(output_name='OUTCAR_%s_Sys' % label)
                                                                                                 
        from NanoCore.vasp_old import read_poscar
                                                                                                 
        System_opt   = read_poscar('CONTCAR')
        ORR_Sys_opt  = Modeling(System_opt)
        
        ORR_SysO2, ORR_SysOOH, ORR_SysO, ORR_SysOH = ORR_Sys_opt.four_electron_transition_gen(mode='ORR', active=active)
        
        #####

        cal_target   = [ORR_SysO2, ORR_SysOOH, ORR_SysO, ORR_SysOH]
        cal_name     = ['O2', 'OOH', 'O', 'OH']  
        
        TE           = [TE_ORR_Sys]
        E_ZPE        = [float(0.000)]
        E_TS         = [float(0.000)]
        
        fix_vib      = []
        for j in range(n_atoms):
            idx = j+1
            fix_vib.append(idx)

        for i in range(len(cal_target)):
            cal = Vasp(cal_target[i])
            cal.run_VASP(mode=mode, nproc=nproc, npar=npar, encut=encut, kpoints=kpoints, \
                         ediff=ediff, ediffg=ediffg, fix=fix)
            os.system('mv OUTCAR OUTCAR_%s_Sys%s'   % (label, cal_name[i]))
            os.system('mv XDATCAR XDATCAR_%s_Sys%s' % (label, cal_name[i]))  
            E = cal.get_total_energy(output_name='OUTCAR_%s_Sys%s' % (label, cal_name[i]))
            TE.append(E)

            if vib:
                cal_opt = read_poscar('CONTCAR')
                cal_vib = Vasp(cal_opt)
                cal_vib.run_VASP(mode='vib', nproc=nproc, npar=npar, encut=encut, kpoints=kpoints, \
                                   ediff=ediff, ediffg=ediffg, fix=fix_vib)
                os.system('mv OUTCAR OUTCAR_%s_Sys%s_Vib' % (label, cal_name[i]))
                ZPE, TS = cal_vib.get_vibration_energy(output_name='OUTCAR_%s_Sys%s_Vib' % (label, cal_name[i]))
                E_ZPE.append(ZPE)
                E_TS.append(TS)
                                                                                                 
        if vib:
            return TE, E_ZPE, E_TS
        else:
            return TE

