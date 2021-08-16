####                                       ####
#   SEISTA Oribtal Projected Module by RONG   #
####                                       ####

# Ver. 1.0. Date : 2020/6/17

#--------- Medules list ---------#

import siestaio as io
import collections
import numpy as np
import string
import struct
import scipy
import glob
import math
import timeit
from scipy.interpolate import interp1d
from mpi4py import MPI


#------- Parameters -------#

pi = np.pi
bohr2ang = 0.529177249
smearing = 0.150
overlap_tolerance = 1.0e-10
weight_tolerance = 1.0e-4
phi_tolerance = 1.0e-10

class OrbitalProjection:
    
    def __init__(self):

        #---- Input options ------#
        self._readInput() 

        #---- initiate MPI -------# 

        if self.MPI == 'true':
            self.comm = MPI.COMM_WORLD
            self.size = self.comm.Get_size()
            self.rank = self.comm.Get_rank()
            print('MPI: rank %d'%self.rank)
        
        #---- Read SIESTA files ----#
        self._readWFSX()
        self._readHSX()
        self._readIon()
        self._readStruct()
        self._supercell_vectors()

        #---- Initiate variable ----#
        self._target = []  

        #---- Interfaces ----#
        self.selectMode()        

    #------- Methods : Read and Write (IO) -------#

    def _readInput(self):

        input_file = open('PyProjection.fdf', 'r')
        input_lines = input_file.readlines()

        for line in input_lines:
            words = line.split()
            if len(words)>0:
                if words[0].strip() == 'SystemLabel':
                    self._system = str(words[1])
                elif words[0].strip() == 'PyProjection.MPI':
                    self.MPI = str(words[1])
                elif words[0].strip() == 'PyProjection.TypeOfRun':
                    self._mode = str(words[1])
                elif words[0].strip() == 'PyProjection.MinE':
                    self.minE = float(words[1])
                elif words[0].strip() == 'PyProjection.MaxE':
                    self.maxE = float(words[1])
                elif words[0].strip() == 'PyProjection.NumE':
                    self.nptE = int(words[1])
                elif words[0].strip() == 'PyProjection.NumA':
                    self.na = int(words[1])
                elif words[0].strip() == 'PyProjection.NumB':
                    self.nb = int(words[1])
                elif words[0].strip() == 'PyProjection.NumC':
                    self.nc = int(words[1])
                elif words[0].strip() == 'PyProjection.TargetOrbital':
                    self.target_orbital = str(words[1]) 

        
    def _readWFSX(self):
        
        system_label = self._system

        if self._mode == 'pdos' or self._mode == 'pldos':
            mode_label = '.fullBZ'
        elif self._mode == 'fat':
            mode_label = '.bands'
        else:
            print('Invalid input mode\n')

        file_name = system_label + mode_label + '.WFSX'
        results = io.readWFSX(file_name)

        self.gamma = results[0]
        self.kpoints = results[1]
        self.kweight = results[2]
        self.wavefunction = results[3]
        self.eigenvalue = results[4]
        
        self._atom_index = results[5]
        self._atom_symbol = results[6]
        self._orbital_index = results[7]
        self._orbital_n = results[8]
        self._orbital_symbol = results[9]

    def _readHSX(self):
        
        system_label = self._system
        file_name = system_label + '.HSX'
        results = io.readHSX(file_name)    

        self.numh = results[0]
        self.listhptr = results[1]
        self.listh = results[2]
        self.indxuo = results[3]
        
        self.Hamilt = results[4]
        self.Sover = results[5]
        self.xij = results[6]
        
        self.atom_index = results[7]
        self.atom_species = results[8]
        self.orbital_n = results[9]
        self.orbital_l = results[10]
        self.orbital_ml = results[11]   # m + l + 1
        self.orbital_zeta = results[12]

    def _readIon(self):
        
        self.ions = {}
        
        total_atomic_species = self._atom_symbol
        atomic_species = dict(collections.Counter(total_atomic_species)).keys()

        for ia in atomic_species:
            
            file_name = ia + '.ion'
            ion = io.readIon(file_name)
            self.ions.update({ia:ion})

    def _readStruct(self):
        
        results = io.readStruct()
        
        self.cell = results[0]
        self.atoms = results[1]
        self.species = results[2]

    def _writeCube(self, fileName, cube_data):

        cube = open(fileName,'w')

        cell = self.cell
        atoms =self.atoms
        species = self.species
        number_of_atoms = len(atoms)   
        
        na = self.na
        nb = self.nb
        nc = self.nc
        ua = self._ua
        ub = self._ub
        uc = self._uc
        
        # file names
        cube.write(' '+fileName+'\n')
        cube.write(' '+fileName+'\n')
        # atoms and grid information
        cube.write('{0:>5d}{1:>12.6f}{2:>12.6f}{3:>12.6f}\n'.format(number_of_atoms, 0, 0, 0))
        cube.write('{0:>5d}{1:>12.6f}{2:>12.6f}{3:>12.6f}\n'.format(na, ua[0], ua[1], ua[2]))  
        cube.write('{0:>5d}{1:>12.6f}{2:>12.6f}{3:>12.6f}\n'.format(nb, ub[0], ub[1], ub[2]))
        cube.write('{0:>5d}{1:>12.6f}{2:>12.6f}{3:>12.6f}\n'.format(nc, uc[0], uc[1], uc[2]))
        for ia in range(number_of_atoms):
            cube.write('{0:>5d}{1:>12.6f}{2:>12.6f}{3:>12.6f}{4:>12.6f}\n'.format(species[ia], 0,
                                                                        atoms[ia][0],
                                                                        atoms[ia][1],
                                                                        atoms[ia][2]
                                                                        ))
        for ia in range(na):
            for ib in range(nb):
                line_counter = 0
                for ic in range(nc):
                    cube.write('{0:>13.5E}'.format(cube_data[ia][ib][ic]))
                    if (ic+1)%10 == 0:
                        cube.write('\n')
                    else:
                        if ic == nc-1:
                            cube.write('\n')
        cube.close
        
    def _writeBand(self, filename, fat_data, eigenvalues, kpath):

        band = open(filename, 'w')

        print(eigenvalues.shape)
        nbands = eigenvalues.shape[0]
        nspin = eigenvalues.shape[1]
        nkpoint = eigenvalues.shape[2]
 
        band.write('{0:>7d}{1:>7d}{2:>7d}\n'.format(nkpoint, nspin, nbands))

        for ikp in range(nkpoint):
            for isp in range(nspin):
                for iw in range(nbands):
                    band.write('{0:>10.5f}{1:>17.12f}{2:>17.12f}\n'.format(kpath[ikp],
                                                                 fat_data[ikp][isp][iw],
                                                                 eigenvalues[iw][isp][ikp]))
                                                        
        band.close

    def _writePDOS(self, filename, pdos_data, energys):

        pdos = open(filename, 'w')
        nenergy = len(energys)
        pdos.write('{0:7d}\n'.format(nenergy))
        for ie in range(nenergy):
            pdos.write('{0:>17.12f}{1:17.12f}\n'.format(energys[ie],
                                                     pdos_data[ie]))
        pdos.close

    #------- Methods : Calculation -------#
            
    def Rnl(self, symbol, n, l, zeta, r):
        
        pao_basis = self.ions[symbol][n][l][zeta]

        cutoff_radius = pao_basis['cutoff']
                    
        if r < cutoff_radius:
            R = pao_basis['r']
            Phi = pao_basis['phi']
            f_phi = interp1d(R,Phi)
            Phi_r = f_phi(r)
        else:
            Phi_r = 0
    
        return Phi_r * (r**l)
                
    def Yml(self, vector, m, l):
        
        '''
            Real spherical harmonics
        ''' 

        x, y, z = vector
         
        r2 = vector**2    
        r = np.sqrt(sum(r2))

        phi = np.arccos(z/r)
        theta = np.arctan2(y,x)
        if theta <= 0:
            theta = theta + 2 * pi

        cY = scipy.special.sph_harm(abs(m), l, theta, phi)

        if m > 0:
            rY = cY.real * np.sqrt(2)
        elif m == 0:
            rY = cY.real
        elif m < 0:
            rY = cY.imag * np.sqrt(2)
            
        return rY

    def numerical_Yml(self, vector, mmax, lmax):

        maxlp = lmax+1
        rly = np.zeros((maxlp**2+1), dtype = float)
        P = np.zeros((maxlp+1,maxlp+1), dtype = float)
        C= np.zeros((maxlp**2+1), dtype = float)
        RL = np.zeros((lmax+2), dtype = float)
        r = np.sqrt(sum(vector**2))
        rx, ry, rz = vector / r
        rxy = np.sqrt(rx**2+ry**2)

        # Evaluate normalization constants
        for l in range(0, lmax+1):
            ilmo = l*l + l
            for m in range(0, l+1):
                factorial = (2*l+1)/(4*pi)
                for i in range(l-m+1, l+m+1):
                    factorial = factorial/i
                C[ilmo+m] = np.sqrt(factorial)
                if (m != 0): C[ilmo+m] = C[ilmo+m]*np.sqrt(2)
                C[ilmo-m] = C[ilmo+m]

        # Find associated Legendre polynomials based on 'Numberical Recipes'
        for m in reversed(range(0,lmax+1)):
            P[m][m] = 1
            factorial = 1
            for i in range(1,m+1):
                P[m][m] = -(P[m][m]*factorial*rxy)
                factorial = factorial + 2
            P[m+1][m] = rz * (2*m+1) * P[m][m]
            for l in range(m+2,lmax+1):
                P[l][m] = (rz*(2*l-1) * P[l-1][m] - (l+m-1)*P[l-2][m])/(l-m)
            
        # Find spherical harmonics
        cosphi = rx/rxy
        sinphi = ry/rxy
        cosm = 1
        sinm = 0
        for m in range(0,lmax+1):
            for l in range(m, lmax+1):
                for ms in [-1,1]:
                    if ms == -1:
                        ilm = l*l + l - m
                        yy = C[ilm] * P[l][m] * sinm
                    else:
                        ilm = l*l + l + m
                        yy = C[ilm] * P[l][m] * cosm
                    rly[ilm] = yy # r**l is not included !! (different from SIESTA)
            previous_cosm = cosm
            previous_sinm = sinm
            cosm = previous_cosm * cosphi - previous_sinm * sinphi
            sinm = previous_cosm * sinphi + previous_sinm * cosphi  

        return rly[lmax*lmax + lmax + mmax]

 
    def delta(self, x):

        if abs(x) > 8*smearing:
            return 0
        else:
            result = np.exp(-(x/smearing)**2)/(smearing*np.sqrt(pi))
            return result
    
    def length(self, vector):
        
        square = 0
        for ix in vector:
            square += ix**2
        return np.sqrt(square)
    
    
    #------- Methods : Orbital Mask -------#
    
    def orbital_mask(self, select): # label, n, l, m, zeta

        '''
        Usage : ['X_n_l_m_zeta'] or ['X_n_l_m'] or ['X_n_l'] or ... all possible 
            
                X : 'C' or '1'  Atomic symbol or Atom's serial number
        
        '''
 #       label, za, zn, zl, zx, zz, 
        
        label = self._atom_symbol
        za = self.atom_index
        zn = self.orbital_n
        zl = self.orbital_l
        zx = self.orbital_ml
        zz = self.orbital_zeta
        
        for index in select:
            
            indexes = index.split('_')
            islabel= 0
            
            try:
                atom_index = int(indexes[0])
            except ValueError:
                atom_label = indexes[0]
                islabel = 1
                
            if (islabel):
                ibuff = np.where(label == atom_label)[0]
            else:
                ibuff = np.where(za == atom_index)[0]
            
            if len(indexes) >= 2:
                buff = []
                atom_n = int(indexes[1])
                for i in ibuff:
                    if zn[i] == atom_n:
                        buff.append(i)
                ibuff = buff
                
                if len(indexes) >= 3:
                    buff = []
                    atom_l = int(indexes[2])
                    for i in ibuff:
                        if zl[i] == atom_l:
                            buff.append(i)
                    ibuff = buff
                    
                    if len(indexes) >= 4:
                        buff = []
                        atom_m = int(indexes[3])
                        for i in ibuff:
                            if zx[i] == atom_m + zl[i] + 1:
                                buff.append(i)
                        ibuff = buff
                        
                        if len(indexes) == 5:
                            buff = []
                            atom_zeta = int(indexes[4])
                            for i in ibuff:
                                if zz[i] == atom_zeta:
                                    buff.append(i)
                            ibuff = buff
                            
            m_iao = ibuff
            nao = len(m_iao)
            m_ia = np.zeros((nao), dtype = int)
            m_izn = np.zeros((nao), dtype = int)
            m_izl = np.zeros((nao), dtype = int)
            m_izm = np.zeros((nao), dtype = int)
            m_izz = np.zeros((nao), dtype = int)
    
            for i in range(nao):
                index = m_iao[i]
                m_ia[i] = za[index]
                m_izn[i] = zn[index]
                m_izl[i] = zl[index]
                m_izm[i] = zx[index] - zl[index] - 1
                m_izz[i] = zz[index]
            
            buff_info = {
                    'number_of_components' : nao,
                    'atomic_index' : m_ia,
                    'orbital_index' : m_iao,
                    'principle_quantum_number' : m_izn,
                    'angular_quantum_number' : m_izl,
                    'magnetic_quantum_nuber' : m_izm,
                    'zeta' : m_izz
                }
            
            self._target.append(buff_info)
            

    def mask_to_pointer(self):

        target = self._target
        
        numh= self.numh
        listhptr = self.listhptr
        listh = self.listh
        indxuo = self.indxuo
        Sover = self.Sover
        
        list_ptr = []
        list_io = []
    
        for i1 in target:
            list_ptr.append([])
            list_io.append([])
            iao_list = i1['orbital_index']
            for j1 in iao_list:
                list_io[-1].append([])
                for k1 in range(numh[j1]): 
                    ind = listhptr[j1] + k1
                    io = indxuo[listh[ind]-1] # python <-> fortran index
                    if abs(Sover[ind]) >= overlap_tolerance:
                        list_ptr[-1].append(ind)
                        list_io[-1][-1].append(io-1) # python <-> fortran index
        
        list_ptr = np.array(list_ptr)
        list_io = np.array(list_io)        
        number_of_projections = len(list_ptr[0])

        if self.MPI == 'true':
            if self.rank == 0:
                print('Total number of  prjections : %d'%number_of_projections)
        else: print('Total number of  prjections : %d'%number_of_projections)
        self._projection_ptr = list_ptr
        self._projection_io = list_io

    #------- Methods : Real space grid -------#

    def unit_cell_grid(self):
        
        cell = self.cell
        na = self.na
        nb = self.nb
        nc = self.nc
        
        if na == 1: ua = cell[0]
        else:       ua = cell[0] / (na-1)
        if nb == 1: ub = cell[1]
        else:       ub = cell[1] / (nb-1)
        if nc == 1: uc = cell[2]
        else:       uc = cell[2] / (nc-1)
        
        xgrid = np.zeros((na,nb,nc), dtype = float)
        ygrid = np.zeros((na,nb,nc), dtype = float)
        zgrid = np.zeros((na,nb,nc), dtype = float)
        
        for ia in range(na):
            for ib in range(nb):
                for ic in range(nc):
                    position_vector = ia * ua + ib * ub + ic * uc
                    xgrid[ia,ib,ic] = position_vector[0]
                    ygrid[ia,ib,ic] = position_vector[1]
                    zgrid[ia,ib,ic] = position_vector[2]

        self._ua = ua
        self._ub = ub
        self._uc = uc
        self._xgrid = xgrid
        self._ygrid = ygrid
        self._zgrid = zgrid
        
    def _supercell_vectors(self):
        
        cell = self.cell
        basis = self.ions
                
        max_cutoff = 0
        for species in basis:
            for qn in basis[species]:
                for ql in basis[species][qn]:
                    for qlm in basis[species][qn][ql]:
                        cutoff = basis[species][qn][ql][qlm]['cutoff']
                        if cutoff >= max_cutoff:
                            max_cutoff = cutoff
            
        length_a = self.length(cell[0])
        length_b = self.length(cell[1])
        length_c = self.length(cell[2])
        
        na = int(math.ceil(cutoff/length_a))
        nb = int(math.ceil(cutoff/length_b))      
        nc = int(math.ceil(cutoff/length_c))
        
        nsc = (2*na+1) * (2*nb+1) * (2*nc+1)
        vectors = np.zeros((nsc,3), dtype = float)
        
        index = 0
        for ia in range(2*na+1):
            for ib in range(2*nb+1):
                for ic in range(2*nc+1):
                    vectors[index] += (ia-na) * cell[0]
                    vectors[index] += (ib-nb) * cell[1]
                    vectors[index] += (ic-nc) * cell[2]
                    
                    index += 1
                    
        self._supercell_vectors = vectors

    def kpath(self, kpoint):
    
        nkpoints = len(kpoint)
        kpath = np.zeros((nkpoints), dtype = float)
        kdir = 0
        for ikp in range(1,nkpoints):
            kdir += np.sqrt(((kpoint[ikp] - kpoint[ikp-1])**2).sum()) 
            kpath[ikp] = kdir
        return kpath
    
    #------ MPI ------#

    def mpi_spread(self, size, rank, njob):
        
        div = njob // size

        if (rank+1) != size:
            job = range(rank*div, (rank+1)*div)
        else:
            job = range(rank*div, njob)    
        return job


    #------- Methods : Interface -------#
        
    def selectMode(self):
        
        mode = self._mode
        if mode == 'fat':
            self.orbital_projected_bandstructure()
        elif mode == 'pdos':
            self.orbital_projected_denstiy_of_state()
        elif mode == 'pldos':
            self.orbital_projected_local_density_of_state()
        else:
            print('Invalid input mode\n')
    
    def get_energy_range(self):
        
        return np.linspace(self.minE, self.maxE, self.nptE, dtype = float)

    #------- Methods : Oribital projeccted bandstructure -------#

    def orbital_projected_bandstructure(self):
        
        # Select target orbitals
        select = [self.target_orbital]
        self.orbital_mask(select)
        self.mask_to_pointer()
        
        gamma = self.gamma
        
        # WFSX information
        wf = self.wavefunction
        kpt = self.kpoints
        eig = self.eigenvalue

        # Selected orbitals
        target = self._target
        list_io = self._projection_io
        list_ptr = self._projection_ptr

        # HSX information
        Sover = self.Sover
        xij = self.xij
        
        nwavefunctions = eig.shape[0]
        nspin = eig.shape[1]
        nkpoints = eig.shape[2]
        ntarget = len(target)

        # MPI
        if self.MPI == 'true':
            comm = self.comm
            rank = self.rank
            size = self.size
            node_kpt = self.mpi_spread(size, rank, nkpoints)
        else: node_kpt = range(nkpoints)

        if self.MPI == 'true':
            if rank == 0:
                fat = np.empty((ntarget, nkpoints, nspin, nwavefunctions), dtype='f')
                strtime = timeit.default_timer() 
        else:
            fat = np.empty((ntarget, nkpoints, nspin, nwavefunctions), dtype='f')
            strtime = timeit.default_timer()
        fat_buff = np.zeros((ntarget, nkpoints, nspin, nwavefunctions), dtype = float)

        for itar in range(ntarget):
            tar = target[itar]
            for ik in node_kpt: # loop over k-points
                for isp in range(nspin): # loop over spin
                    for iw in range(nwavefunctions): # loop over wavefunctions
                        nprojection = len(list_ptr[itar])
                        buff = np.zeros((nprojection), dtype = float)
                        list_target_io = tar['orbital_index']
                        iio1 = 0
                        iio2 = 0
                        for io1 in list_target_io: # list of target orbitals
                            for io2 in list_io[itar][iio1]: # list of projection orbitals
                                ind = list_ptr[itar][iio2]
                                if (gamma==-1):
                                    qcos = wf[0][io1][iw][isp][ik]*wf[0][io2][iw][isp][ik]
                                    qsin = 0
                                elif (gamma==0):
                                    qcos = wf[0][io1][iw][isp][ik] * wf[0][io2][iw][isp][ik] + \
                                           wf[1][io1][iw][isp][ik] * wf[1][io2][iw][isp][ik]
                                    qsin = wf[0][io1][iw][isp][ik] * wf[1][io2][iw][isp][ik] - \
                                           wf[1][io1][iw][isp][ik] * wf[0][io2][iw][isp][ik]
                                phase = (kpt[ik] * xij[ind]).sum() # phase fator
                                factor = qcos * np.cos(phase) - qsin * np.sin(phase)
                                buff[iio2] = Sover[ind] * factor                            
                                iio2 += 1
                            iio1 += 1
                                
                        fat_buff[itar][ik][isp][iw] = buff.sum() # sum over projection orbitals

        if self.MPI == 'true':
            fat = comm.reduce(fat_buff, op= MPI.SUM)
            if rank == 0:
                self.fat = fat
                fintime = timeit.default_timer()
                print('Delay times (min): %f'%((fintime-strtime)/60))
        else:
            fat = fat_buff
            self.fat = fat
            fintime = timeit.default_timer()
            print('Delay times (min): %f'%((fintime-strtime)/60))
        kpath = self.kpath(kpt)
        # write fat files
        if self.MPI == 'true':
            if rank == 0: 
                for itar in range(len(select)):
                    target_orbital_index = select[itar]
                    file_name = target_orbital_index + '.fat'
                    data = fat[itar]
                    self._writeBand(file_name, data, eig, kpath)

        else:
            for itar in range(len(select)):
                target_orbital_index = select[itar]
                file_name = target_orbital_index + '.fat'
                data = fat[itar]
                self._writeBand(file_name, data, eig, kpath)

    #------- Methods : Oribital projeccted density of state -------#

    def orbital_projected_denstiy_of_state(self):

        # Select range of energy
        energys = self.get_energy_range()
        nenergy = self.nptE
        
        # Select target orbitals
        select = [self.target_orbital]
        self.orbital_mask(select)
        self.mask_to_pointer()
        
        gamma = self.gamma
        
        # WFSX information
        wf = self.wavefunction
        wk = self.kweight
        kpt = self.kpoints
        eig = self.eigenvalue

        # Selected orbitals
        target = self._target
        list_io = self._projection_io
        list_ptr = self._projection_ptr

        # HSX information
        Sover = self.Sover
        xij = self.xij
        
        nwavefunctions = eig.shape[0]
        nspin = eig.shape[1]
        nkpoints = eig.shape[2]
        ntarget = len(target)

        # MPI
        if self.MPI == 'true':
            comm = self.comm
            rank = self.rank
            size = self.size
            node_kpt = self.mpi_spread(size, rank, nkpoints)
        else: node_kpt = range(nkpoints)

        if self.MPI == 'true':
            if rank == 0:
                pdos = np.empty((ntarget, nspin, nenergy), dtype='f')
                strtime = timeit.default_timer()
        else:
            pdos = np.empty((ntarget, nspin, nenergy), dtype='f')
            strtime = timeit.default_timer()

        pdos_buff = np.zeros((ntarget, nspin, nenergy), dtype = float)        
        for itar in range(ntarget):
            tar = target[itar]
            buff4 = np.zeros((isp, nenergy, nkpoints), dtype = float)
            for ik in node_kpt: # loop over k-points
                buff3 = np.zeros((isp, nenergy, nwavefunctions), dtype = float)
                for isp in range(nspin): # loop over spin
                    buff2 = np.zeros((nenergy, nwavefunctions), dtype = float)
                    for iw in range(nwavefunctions): # loop over bands
                        nprojection = len(list_ptr[itar])
                        buff1 = np.zeros((nprojection), dtype = float)
                        list_target_io = tar['orbital_index']
                        iio1 = 0
                        iio2 = 0
                        for io1 in list_target_io: # list of target orbitals
                            for io2 in list_io[itar][iio1]: # list of projection orbitals
                                ind = list_ptr[itar][iio2]
                                if (gamma==-1):
                                    qcos = wf[0][io1][iw][isp][ik]*wf[0][io2][iw][isp][ik]
                                    qsin = 0
                                elif (gamma==0):
                                    qcos = wf[0][io1][iw][isp][ik] * wf[0][io2][iw][isp][ik] + \
                                           wf[1][io1][iw][isp][ik] * wf[1][io2][iw][isp][ik]
                                    qsin = wf[0][io1][iw][isp][ik] * wf[1][io2][iw][isp][ik] - \
                                           wf[1][io1][iw][isp][ik] * wf[0][io2][iw][isp][ik]
                                alfa = (kpt[ik] * xij[ind]).sum() # phase fator
                                factor = qcos * np.cos(alfa) - qsin * np.sin(alfa)
                                buff1[iio2] = Sover[ind] * factor
                                iio2 += 1
                            iio1 += 1
                        buff2[:,iw] = buff1.sum() # sum over projection orbitals
                        # wavefunction loop
                        eigenvalue = eig[iw][isp][ik] 
                        for ie in range(nenergy):
                            factor = self.delta(energys[ie]-eigenvalue)
                            buff2[ie][iw] = factor * buff2[ie][iw]
                    buff3[isp,:,:] = buff2
                for isp in range(nspin):
                    buff4[isp,:,ik] = [wk[ik] * buff3[isp,ie].sum() for ie in range(nenergy)]
            # first loop
            for isp in range(nspin):
                pdos_buff[itar,isp,:] = [buff4[isp][ie].sum() for ie in range(nenergy)]

            if self.MPI == 'true':
                comm.Barrier()

        # normalization
        if self.MPI == 'true':
            pdos = comm.reduce(pdos_buff, op= MPI.SUM)
            if rank == 0:
                wksum = 0
                for ik in range(nkpoints):
                    wksum += wk[ik]

                pdos = pdos / wksum
                self.pdos = pdos
                fintime = timeit.default_timer()
                print('Delay times: %f'%(fintime-strime))

        else:
            pdos = pdos_buff
            wksum = 0 
            for ik in range(nkpoints):
                wksum += wk[ik]

            pdos = pdos / wksum
            self.pdos = pdos
            fintime = timeit.default_timer()
            print('Delay times: %f'%(fintime-strtime))

        # write pdos files
        if self.MPI == 'true':
            if rank == 0:
                for itar in range(len(select)):
                    target_orbital_index = select[itar]
                    file_name = target_orbital_index + '.pdos'
                    data = pdos[itar]
                    target_energy = energys
                    self._writePDOS(file_name, data, target_energy)
        else:
             for itar in range(len(select)):
                 target_orbital_index = select[itar]
                 file_name = target_orbital_index + '.pdos'
                 data = pdos[itar]
                 target_energy = energys
                 self._writePDOS(file_name, data, target_energy)


    #------- Methods : Oribital projeccted local density of states -------#

    #@profile
    def orbital_projected_local_density_of_state(self):
        
        # Select range of energy
        energys = self.get_energy_range()
        nenergy = self.nptE
    
        # Select target orbitals
        select = [self.target_orbital]
        self.orbital_mask(select)
        self.mask_to_pointer()
        
        # Define number of grid points
        self.unit_cell_grid()
        
        # WFSX information
        gamma = self.gamma
        wf = self.wavefunction
        wk = self.kweight
        kpt = self.kpoints
        eig = self.eigenvalue

        index = self.atom_index
        symbol = self._atom_symbol
        print(index)
        
        # Selected orbitals
        target = self._target
        list_io = self._projection_io
        list_ptr = self._projection_ptr

        # HSX information
        Sover = self.Sover
        xij = self.xij
        
        # Struct information
        cell = self.cell
        atoms = self.atoms
        origin = atoms[0] # first atom
        supercell_vectors = self._supercell_vectors
        nvectors = len(supercell_vectors)
        
        # Grid information
        na = self.na
        nb = self.nb
        nc = self.nc
        xgrid = self._xgrid
        ygrid = self._ygrid
        zgrid = self._zgrid        

        # Parallelization ( Developing .. )

        nwavefunctions = eig.shape[0]
        nspin = eig.shape[1]
        nkpoints = eig.shape[2]
        ntarget = len(target)

        spatial = np.zeros((2, nvectors), dtype= float)
        pldos = np.zeros((ntarget, nenergy, na, nb, nc), dtype = float)
        
        for itar in range(ntarget):
            tar = target[itar]
            
            for ix in range(na):
                for iy in range(nb):
                    for iz in range(nc):
                        print('[ %d %d %d grid ]\n' %(ix+1,iy+1,iz+1))
                        position_vector = np.zeros((3), dtype = float)
                        position_vector[0] = xgrid[ix][iy][iz]
                        position_vector[1] = ygrid[ix][iy][iz]
                        position_vector[2] = zgrid[ix][iy][iz]                    
                        list_target_io = tar['orbital_index']
                        number_of_target = len(list_target_io)
                        buff0 = np.zeros((nenergy, number_of_target), dtype = float)
                        iio1 = 0
                        iio2 = 0
                        for io1 in list_target_io: # list of target orbitals

                            iio2_str = iio2
                            atom_symbol = symbol[io1]
                            atom_index = index[io1] - 1
                            target_position = atoms[atom_index]
                            target_vector = target_position - position_vector # grid point to target atom        
                            target_n = tar['principle_quantum_number'][iio1]
                            target_l = tar['angular_quantum_number'][iio1]
                            target_m = tar['magnetic_quantum_nuber'][iio1]
                            target_z = tar['zeta'][iio1]
                            
                            buff1 = np.zeros((nenergy, 2, nkpoints, nspin, nwavefunctions), dtype = float)
                            buff = np.zeros((nenergy, nkpoints), dtype = float)
                            for ik in range(nkpoints): # loop over k-points
                                #print('Loop over kpoints : %d / %d'%(ik+1, nkpoints))
                                for isp in range(nspin): # loop over spin
                                    for iw in range(nwavefunctions): # loop over wavefunctions
                                        eigenvalue = eig[iw][isp][ik]
                                        nprojection = len(list_ptr[itar])
                                        list_target_io = tar['orbital_index']
                                        buff2 = np.zeros((2, nprojection), dtype = float)
                                        iio2 = iio2_str
                                        for io2 in list_io[itar][iio1]: # list of projection orbitals
                                            ind = list_ptr[itar][iio2]
                                            if (gamma==-1):
                                                qcos = wf[0][io1][iw][isp][ik]*wf[0][io2][iw][isp][ik]
                                                qsin = 0
                                            elif (gamma==0):
                                                qcos = wf[0][io1][iw][isp][ik] * wf[0][io2][iw][isp][ik] + \
                                                    wf[1][io1][iw][isp][ik] * wf[1][io2][iw][isp][ik]
                                                qsin = wf[0][io1][iw][isp][ik] * wf[1][io2][iw][isp][ik] - \
                                                    wf[1][io1][iw][isp][ik] * wf[0][io2][iw][isp][ik]

                                            alfa = np.inner(kpt[ik], xij[ind]) # phase fator
                                            buff2[0][iio2] = qcos * math.cos(alfa) - qsin * math.sin(alfa)
                                            buff2[1][iio2] = qcos * math.sin(alfa) + qsin * math.sin(alfa)
                                            buff2 *=  Sover[ind]
                                            iio2 += 1
                                                                                    
                                        rval = (buff2[0]).sum()
                                        ival = (buff2[1]).sum()
                                        for ie in range(nenergy):
                                            factor = self.delta(energys[ie]-eigenvalue)
                                            buff1[ie][0][ik][isp][iw] = rval * factor
                                            buff1[ie][1][ik][isp][iw] = ival * factor
                                
                                # k loop

                                buff1 = wk[ik] * buff1

                                for iv in range(nvectors):
                                    xji = -(target_vector + supercell_vectors[iv])
                                    phase = np.inner(kpt[ik], xji/bohr2ang)
                                    r = np.sqrt(xji.dot(xji))
                                    phir = self.Rnl(atom_symbol, target_n, target_l, target_z, r)
                                    
                                    if phir < phi_tolerance:
                                        factor = 0
                                    else:
                                       spherical = self.Yml(xji, target_m, target_l) # complex
                                       factor = phir * spherical 
                                        
                                    spatial[0][iv] = factor**2 * math.cos(phase)
                                    spatial[1][iv] = factor**2 * math.sin(phase)
                                rspatial = spatial[0].sum()
                                ispatial = spatial[1].sum()
                                
                                for ie in range(nenergy):
                                    rphi_orbital = buff1[ie][0][ik].sum()
                                    iphi_orbital = buff1[ie][1][ik].sum()
                                    buff[ie][ik] = rphi_orbital * rspatial - iphi_orbital * ispatial

                            # io1 loop        
                            for ie in range(nenergy):
                                buff0[ie][iio1] = buff[ie].sum()
                            iio1 += 1
                            
                        # xyz loop
                        for ie in range(nenergy):
                            pldos[itar][ie][ix][iy][iz] = buff0[ie].sum()

        wksum = 0
        for ik in range(nkpoints):
            wksum += wk[ik]

        pldos = pldos / wksum
        self.pldos = pldos

        # write cube files
        for itar in range(len(select)):
            for ie in range(nenergy):
                target_orbital_index = select[itar]
                target_energy = energys[ie]
                file_name = target_orbital_index + '_%2.5f_eV.cube' % target_energy
                data = pldos[itar][ie]
                self._writeCube(file_name, data)


if __name__ == '__main__':        
    PyProjection = OrbitalProjection()



