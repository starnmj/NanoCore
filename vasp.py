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
        p = self._params
        KPOINTS = open('KPOINTS', 'w')
        KPOINTS.write("k-points\n")
        KPOINTS.write("0\n")
        KPOINTS.write("G\n")
        KPOINTS.write("%i %i %i\n" % (p['KPOINTS'][0], p['KPOINTS'][1], p['KPOINTS'][2]))
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
        p = self._params
        INCAR = open('INCAR', 'w')
        INCAR.write("# VASP general descriptors \n\n")
        INCAR.write("SYSTEM        =   %s\n" % p['SYSTEM'])
        INCAR.write("NPAR          =   %i\n" % int(p['NPAR']))
        INCAR.write("IBRION        =   %i\n" % int(p['IBRION']))
        INCAR.write("LWAVE         =   %s\n" % p['LWAVE']) 
        INCAR.write("LCHARG        =   %s\n" % p['LCHARG']) 
        INCAR.write("NSW           =   %i\n" % p['NSW']) 
        INCAR.write("PREC          =   %s\n" % p['PREC']) 
        INCAR.write("ALGO          =   %s\n" % p['ALGO'])
        INCAR.write("ISTART        =   %i\n" % p['ISTART']) 
        INCAR.write("ICHARG        =   %i\n" % p['ICHARG']) 
        INCAR.write("ISIF          =   %i\n\n" % p['ISIF']) 
        INCAR.write("# VASP convergence parameters \n\n")
        INCAR.write("ENCUT         =   %f\n" % float(p['ENCUT']))
        INCAR.write("ISMEAR        =   %i\n" % p['ISMEAR'])
        INCAR.write("SIGMA         =   %f\n" % p['SIGMA'])
        INCAR.write("NSIM          =   %i\n" % p['NSIM'])
        INCAR.write("NELMIN        =   %i\n" % p['NELMIN'])
        INCAR.write("NELM          =   %i\n" % p['NELM'])
        INCAR.write("EDIFF         =   %f\n" % float(p['EDIFF']))
        INCAR.write("EDIFFG        =   %f\n" % float(p['EDIFFG']))
        INCAR.write("%s           =   %s\n\n" % (p['XC'], p['XCAUTHOR']))
        INCAR.write("# VASP optional parameters \n\n")
        INCAR.write("POTIM         =   %f\n" % float(p['POTIM']))
        INCAR.write("IVDW          =   %i\n" % int(p['IVDW']))
        INCAR.write("LDIPOL        =   %s\n" % p['LDIPOL'])
        INCAR.write("IDIPOL        =   %i\n" % int(p['IDIPOL']))
        INCAR.write("LPLANE        =   %s\n" % p['LPLANE'])
        INCAR.write("ADDGRID       =   %s\n" % p['ADDGRID'])
        INCAR.write("LREAL         =   %s\n" % p['LREAL'])
        INCAR.write("ISYM          =   %i\n" % p['ISYM'])
        INCAR.write("LASPH         =   %s\n" % p['LASPH'])
        INCAR.write("LMAXMIX       =   %i\n" % p['LMAXMIX'])
        INCAR.write("ISPIN         =   %i\n\n" % p['ISPIN'])
        INCAR.close()

    def run_VASP(self, mode='single', nproc=1, npar=1, encut=400, kpoints=[1,1,1], 
                 ediff = 0.0001, ediffg = -0.05,  fix=None):
        """ 
        Example:
        --------

        from NanoCore import vasp2
        from NanoCore import io 
        at = io.read_poscar('POSCAR')
        at2 = vasp2.Vasp(at)
        at2.run_VASP(nproc=8, npar=2, kpoints=[2,2,1])
        
        """
        from NanoCore.env import vasp_calculator as executable

        p = self._params
        
        p['NPAR']       = npar
        p['KPOINTS']    = kpoints
        p['ENCUT']      = encut
        p['EDIFF']      = ediff
        p['EDIFFG']     = ediffg
        
        if mode == 'opt':
            p['IBRION'] = 2
            p['POTIM']  = 0.300
            p['NSW']    = 500
        
        if mode == 'single':
            p['IBRION'] = 2
            p['POTIM']  = 0.300
            p['NSW']    =   0

        
        if mode == 'vib':
            p['IBRION'] = 5
            p['POTIM']  = 0.015
            p['NSW']    = 1

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
        from NanoCore import io       
        at = io.read_poscar('POSCAR')
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
 
def run_series_HER(atoms, mode='opt', nproc=1, npar=1, encut=400, kpoints=[1,1,1], 
                   ediff = 0.0001, ediffg = -0.05, fix=None, active=None, vib=1, label='test'):

    from NanoCore.catalysis import Modeling
    
    n_atoms = len(atoms)
    atoms_HER = Vasp(atoms)
    atoms_HER.run_VASP(mode=mode, nproc=nproc, npar=npar, encut=encut, kpoints=kpoints, \
                            ediff=ediff, ediffg=ediffg, fix=fix)
    
    os.system('mv OUTCAR OUTCAR_%s_Sys' % label)
    os.system('mv XDATCAR XDATCAR_%s_Sys' % label)

    TE_Sys = atoms_HER.get_total_energy(output_name='OUTCAR_%s_Sys' % label)

    from NanoCore.io import read_poscar

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

def run_series_ORR(atoms, mode='opt', nproc=1, npar=1, encut=400, kpoints=[1,1,1],                
                ediff = 0.0001, ediffg = -0.05, fix=None, active=None, vib=1, label='test'):
                                                                                             
    from NanoCore.catalysis import Modeling
 
    n_atoms      = len(atoms)
    ORR_Sys      = Vasp(atoms)
    ORR_Sys.run_VASP(mode=mode, nproc=nproc, npar=npar, encut=encut, kpoints=kpoints, \
                         ediff=ediff, ediffg=ediffg, fix=fix)
 
    os.system('mv OUTCAR OUTCAR_%s_Sys' % label)
    os.system('mv XDATCAR XDATCAR_%s_Sys' % label)
    TE_ORR_Sys   = ORR_Sys.get_total_energy(output_name='OUTCAR_%s_Sys' % label)
                                                                                             
    from NanoCore.io import read_poscar
                                                                                             
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

def pdos_split_sum(sum_list=[1]):
    
    line_info, word_info = Vasp.file_read('DOSCAR')
    n_points = int(word_info[5][2])
    E_fermi  = float(word_info[5][3])

    ### Data list (s, p, d + spin) ###
    s_up     = []   ;    s_dn   = []
    py_up    = []   ;    py_dn  = []
    pz_up    = []   ;    pz_dn  = []
    px_up    = []   ;    px_dn  = []
    dxy_up   = []   ;    dxy_dn = []
    dyz_up   = []   ;    dyz_dn = []
    dz2_up   = []   ;    dz2_dn = []
    dxz_up   = []   ;    dxz_dn = []
    dx2_up   = []   ;    dx2_dn = []
    s_up_sum = []   ;  s_dn_sum = []
    p_up_sum = []   ;  p_dn_sum = []
    d_up_sum = []   ;  d_dn_sum = []
    ###

    E_modi = []
    for i in range(len(word_info)):
        if len(word_info[i]) == 19: # only for the s,p,d + spin case
            item = word_info[i]
            E_modi.append(float(item[0])-E_fermi)
            s_up.append(float(item[1]))
            s_dn.append(-float(item[2]))
            py_up.append(float(item[3]))
            py_dn.append(-float(item[4]))
            pz_up.append(float(item[5]))
            pz_dn.append(-float(item[6]))
            px_up.append(float(item[7]))
            px_dn.append(-float(item[8]))
            dxy_up.append(float(item[9]))
            dxy_dn.append(-float(item[10]))
            dyz_up.append(float(item[11]))
            dyz_dn.append(-float(item[12]))
            dz2_up.append(float(item[13]))
            dz2_dn.append(-float(item[14]))
            dxz_up.append(float(item[15]))
            dxz_dn.append(-float(item[16]))
            dx2_up.append(float(item[17]))
            dx2_dn.append(-float(item[18]))

    N_atoms = len(E_modi) / n_points # double check

    # write arranged data

    for i in range(int(N_atoms)):
        dataDOS = open('%03d_ATOM.dat' % int(i+1), 'w')
        for j in range(n_points):
            char = "%15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f" % (E_modi[i*n_points+j],s_up[i*n_points+j]+py_up[i*n_points+j]+pz_up[i*n_points+j]+px_up[i*n_points+j]+dxy_up[i*n_points+j]+dyz_up[i*n_points+j]+dz2_up[i*n_points+j]+dxz_up[i*n_points+j]+dx2_up[i*n_points+j],s_dn[i*n_points+j]+py_dn[i*n_points+j]+pz_dn[i*n_points+j]+px_dn[i*n_points+j]+dxy_dn[i*n_points+j]+dyz_dn[i*n_points+j]+dz2_dn[i*n_points+j]+dxz_dn[i*n_points+j]+dx2_dn[i*n_points+j],s_up[i*n_points+j],s_dn[i*n_points+j],py_up[i*n_points+j]+pz_up[i*n_points+j]+px_up[i*n_points+j],py_dn[i*n_points+j]+pz_dn[i*n_points+j]+px_dn[i*n_points+j],dxy_up[i*n_points+j]+dyz_up[i*n_points+j]+dz2_up[i*n_points+j]+dxz_up[i*n_points+j]+dx2_up[i*n_points+j],dxy_dn[i*n_points+j]+dyz_dn[i*n_points+j]+dz2_dn[i*n_points+j]+dxz_dn[i*n_points+j]+dx2_dn[i*n_points+j],s_up[i*n_points+j],s_dn[i*n_points+j],py_up[i*n_points+j],py_dn[i*n_points+j],pz_up[i*n_points+j],pz_dn[i*n_points+j],px_up[i*n_points+j],px_dn[i*n_points+j],dxy_up[i*n_points+j],dxy_dn[i*n_points+j],dyz_up[i*n_points+j],dyz_dn[i*n_points+j],dz2_up[i*n_points+j],dz2_dn[i*n_points+j],dxz_up[i*n_points+j],dxz_dn[i*n_points+j],dx2_up[i*n_points+j],dx2_dn[i*n_points+j])    
            dataDOS.write(char)
            dataDOS.write("\n")
        dataDOS.close()
    
    os.system('mkdir PDOS')
    os.system('mv *_ATOM.dat PDOS')
    
    ### Sum data list (s, p, d + spin) ###   
    sum_s_up     = []   ;    sum_s_dn     = []     
    sum_py_up    = []   ;    sum_py_dn    = []
    sum_pz_up    = []   ;    sum_pz_dn    = []
    sum_px_up    = []   ;    sum_px_dn    = []
    sum_dxy_up   = []   ;    sum_dxy_dn   = []
    sum_dyz_up   = []   ;    sum_dyz_dn   = []
    sum_dz2_up   = []   ;    sum_dz2_dn   = []
    sum_dxz_up   = []   ;    sum_dxz_dn   = []
    sum_dx2_up   = []   ;    sum_dx2_dn   = []
    sum_s_up_sum = []   ;    sum_s_dn_sum = []
    sum_p_up_sum = []   ;    sum_p_dn_sum = []
    sum_d_up_sum = []   ;    sum_d_dn_sum = []
    ###
    
    for j in range(n_points):
        num_s_up   = 0   ;   num_s_dn   = 0
        num_py_up  = 0   ;   num_py_dn  = 0
        num_pz_up  = 0   ;   num_pz_dn  = 0
        num_px_up  = 0   ;   num_px_dn  = 0
        num_dxy_up = 0   ;   num_dxy_dn = 0
        num_dyz_up = 0   ;   num_dyz_dn = 0
        num_dz2_up = 0   ;   num_dz2_dn = 0
        num_dxz_up = 0   ;   num_dxz_dn = 0
        num_dx2_up = 0   ;   num_dx2_dn = 0
        num_s_up_sum = 0 ;   num_s_dn_sum = 0
        num_p_up_sum = 0 ;   num_p_dn_sum = 0
        num_d_up_sum = 0 ;   num_d_dn_sum = 0
        for i in sum_list:
            # each orbitals
            num_s_up += s_up[(i-1)*n_points+j]       ;  num_s_dn += s_dn[(i-1)*n_points+j]
            num_py_up += py_up[(i-1)*n_points+j]     ;  num_py_dn += py_dn[(i-1)*n_points+j]
            num_pz_up += pz_up[(i-1)*n_points+j]     ;  num_pz_dn += pz_dn[(i-1)*n_points+j]
            num_px_up += px_up[(i-1)*n_points+j]     ;  num_px_dn += px_dn[(i-1)*n_points+j]
            num_dxy_up += dxy_up[(i-1)*n_points+j]   ;  num_dxy_dn += dxy_dn[(i-1)*n_points+j]
            num_dyz_up += dyz_up[(i-1)*n_points+j]   ;  num_dyz_dn += dyz_dn[(i-1)*n_points+j]
            num_dz2_up += dz2_up[(i-1)*n_points+j]   ;  num_dz2_dn += dz2_dn[(i-1)*n_points+j]
            num_dxz_up += dxz_up[(i-1)*n_points+j]   ;  num_dxz_dn += dxz_dn[(i-1)*n_points+j]
            num_dx2_up += dx2_up[(i-1)*n_points+j]   ;  num_dx2_dn += dx2_dn[(i-1)*n_points+j]
            # each summed orbitals
            num_s_up_sum += s_up[(i-1)*n_points+j]   
            num_s_dn_sum += s_dn[(i-1)*n_points+j]
            num_p_up_sum += py_up[(i-1)*n_points+j] + pz_up[(i-1)*n_points+j] + px_up[(i-1)*n_points+j]
            num_p_dn_sum += py_dn[(i-1)*n_points+j] + pz_dn[(i-1)*n_points+j] + px_dn[(i-1)*n_points+j]
            num_d_up_sum += dxy_up[(i-1)*n_points+j] + dyz_up[(i-1)*n_points+j] + dz2_up[(i-1)*n_points+j] + dxz_up[(i-1)*n_points+j] + dx2_up[(i-1)*n_points+j]
            num_d_dn_sum += dxy_dn[(i-1)*n_points+j] + dyz_dn[(i-1)*n_points+j] + dz2_dn[(i-1)*n_points+j] + dxz_dn[(i-1)*n_points+j] + dx2_dn[(i-1)*n_points+j]
        # summation values
        sum_s_up.append(num_s_up)         ;   sum_s_dn.append(num_s_dn)
        sum_py_up.append(num_py_up)       ;   sum_py_dn.append(num_py_dn)
        sum_pz_up.append(num_pz_up)       ;   sum_pz_dn.append(num_pz_dn)
        sum_px_up.append(num_px_up)       ;   sum_px_dn.append(num_px_dn)
        sum_dxy_up.append(num_dxy_up)     ;   sum_dxy_dn.append(num_dxy_dn)
        sum_dyz_up.append(num_dyz_up)     ;   sum_dyz_dn.append(num_dyz_dn)
        sum_dz2_up.append(num_dz2_up)     ;   sum_dz2_dn.append(num_dz2_dn)
        sum_dxz_up.append(num_dxz_up)     ;   sum_dxz_dn.append(num_dxz_dn)
        sum_dx2_up.append(num_dx2_up)     ;   sum_dx2_dn.append(num_dx2_dn)
        sum_s_up_sum.append(num_s_up_sum) ;   sum_s_dn_sum.append(num_s_dn_sum)
        sum_p_up_sum.append(num_p_up_sum) ;   sum_p_dn_sum.append(num_p_dn_sum)
        sum_d_up_sum.append(num_d_up_sum) ;   sum_d_dn_sum.append(num_d_dn_sum)
    dataSUMDOS = open('SUM_ATOM.dat', 'w')
    for i in range(n_points):
        char = "%15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f" % (E_modi[i],sum_s_up_sum[i]+sum_p_up_sum[i]+sum_d_up_sum[i],sum_s_dn_sum[i]+sum_p_dn_sum[i]+sum_d_dn_sum[i],sum_s_up_sum[i],sum_s_dn_sum[i],sum_p_up_sum[i],sum_p_dn_sum[i],sum_d_up_sum[i],sum_d_dn_sum[i],sum_s_up[i],sum_s_dn[i],sum_py_up[i],sum_py_dn[i],sum_pz_up[i],sum_pz_dn[i],sum_px_up[i],sum_px_dn[i],sum_dxy_up[i],sum_dxy_dn[i],sum_dyz_up[i],sum_dyz_dn[i],sum_dz2_up[i],sum_dz2_dn[i],sum_dxz_up[i],sum_dxz_dn[i],sum_dx2_up[i],sum_dx2_dn[i])
        dataSUMDOS.write(char)
        dataSUMDOS.write("\n")
    dataSUMDOS.close()
    os.system('mv SUM_ATOM.dat PDOS')

def pdos_orbital_analysis(fname='SUM_ATOM.dat', orbitals=1):
    '''
    d-band center theory : Nature, 376, 238 (1995)
    Fermi abundance      : J. Phys. Chem. C 121, 1530 (2017)
    highest peak         : Nat. Energy, 1, 16130 (2016)
    '''
    import os, sys, math
    import numpy as np
    from scipy.misc import derivative
 
    def fermi_dirac(E, fermi=0, T=298.15):
        return 1 / (1 + np.exp((E - fermi) / 0.4))

    def d_fd(E):
        deri_fd = derivative(fermi_dirac, E, dx=1e-6)
        return deri_fd

    def find_index(data, target):
        res = []
        lis = data
        while True:
            try:
                res.append(lis.index(target) + (res[-1]+1 if len(res)!=0 else 0))
                lis = data[res[-1]+1:]
            except:
                break
        return res
    
    line_info, word_info = Vasp.file_read(fname)
    
    E = [] ; Orb = []
    
    for i in range(len(word_info)):
        E.append(float(word_info[i][0]))
        Orb.append(float(word_info[i][orbitals]))

    for i in range(len(E)-1):
        if E[i+1] * E[i] < 0 or E[i+1] * E[i] == 0:
            Fermi_idx = i
        else:
            pass
    
    # find parameteres
    dE = E[1] - E[0]
    n_Orb = 0 ; E_n_Orb = 0
    df_n_Orb = 0 ; df_E_n_Orb = 0
    
    Orb_Ef = Orb[:Fermi_idx]
    if sum(Orb_Ef) >= 0:
        Orb_max = max(Orb_Ef) 
    elif sum(Orb_Ef) < 0:
        Orb_max = min(Orb_Ef)
    else:
        print("The data is not consistent for spin up or dn")
    
    E_Orb_max = find_index(Orb_Ef, Orb_max)

    if len(E_Orb_max) == 1:
        E_max = E[E_Orb_max[0]]
    else:
        print("Same max values exist")

    # calculation 
    for i in range(Fermi_idx):
        n_Orb       = n_Orb      + dE * Orb_Ef[i]
        E_n_Orb     = E_n_Orb    + dE * Orb_Ef[i] * E[i]
        df_n_Orb    = df_n_Orb   + dE * Orb_Ef[i] * d_fd(E[i])
        df_E_n_Orb  = df_E_n_Orb + dE * Orb_Ef[i] * E[i] * d_fd(E[i])

    Orb_cent = E_n_Orb / n_Orb
    Ef_abund = df_E_n_Orb / df_n_Orb
    print('orbital center', Orb_cent)
    print('fermi-abudnace', Ef_abund)
    print('highest peak', E_max,'at', E_Orb_max)
    
    return Orb_cent, Ef_abund, (E_max, Orb_Ef[E_Orb_max[0]])

