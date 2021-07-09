#
# Templet for generating carbon-based 1-D, 2-D structures
# Translated from old XYZ made by Jung Hoon Park
# Modified by Hu Sung Kim : 2011. 3. 23
# Modified by Hu Sung Kim and Minkyu Park : 2011. 7. 11
#
# 110711 modified grp_zigzag and grp_armchair (odd numbers can be allowed.)
# 110709 added grp_zigzag and grp_armchair 
# 120305 debugging grp_zigzag and grp_armchair with Hydrogen passivation
# 190311 
#


from __future__ import print_function
from . atoms import *
import numpy as np


def dmax(a,b):
    """
    Return the G.C.D of A and B
    """
    while a:
        a,b = b % a,a
    return b

def grp(n, m, bl=1.415, vacuum=10.0):
    """
    Generate Graphene  
    """
    from math import sqrt
    r3 = sqrt(3)
    a = bl*r3
    basis = [('C',0,0,0),('C',r3*bl/2,bl/2,0)]
    cell = [[    (n+1)*sqrt(3)*bl,               0,    0],
            [(m+1)*(sqrt(3)/2)*bl,(m+1)*(3./2.)*bl,    0],
            [                 0.0,             0.0, vacuum]]
    grp = []; i = 0
    while i <= n:
        temp = AtomsSystem(basis); temp.select_all()
        temp.translate(a*i, 0, 0)
        for atom in temp: grp.append(atom.copy())
        i += 1
    grp2 = []; i = 0
    while i <= m:
        temp = AtomsSystem(grp); temp.select_all()
        temp.translate(a/2*i, r3*a/2*i, 0)
        for atom in temp: grp2.append(atom.copy())
        i += 1
    return AtomsSystem(grp2, cell=cell)

def SC_2d(n, m, basis, uc):
    sc_basis = []
    sc_cell = []
    a1 = np.array(uc[0])
    a2 = np.array(uc[1])
    ### make SC basis
    for atom in basis:
        for i in range(n):
            for j in range(m):
                sc_atom_xy  = np.array(atom[1:3])
                sc_atom_xy  = sc_atom_xy + i*np.array(uc[0][:2])
                sc_atom_xy  = sc_atom_xy + j*np.array(uc[1][:2])
                sc_atom_basis=[atom[0]]
                sc_atom_basis.extend(list(sc_atom_xy))
                sc_atom_basis.append(atom[3])
                #print(f"add atoms in sc {sc_atom_basis}")
                sc_basis.append(tuple(sc_atom_basis))
    ### make SC
    cell0 = np.array(uc[0]) * n
    cell1 = np.array(uc[1]) * m
    sc_cell = [list(cell0), list(cell1), uc[2]]
    return sc_basis, sc_cell
        
grp_hexa = grp

def grp_rect(n, m, bl=1.42, vacuum=10.0, center=0.4):
    """
    Generate Graphene as Rectangular shape 
    bl      C-C bond length ase 1.420
    center  location of surface between 0 and 1
    """
    from math import sqrt, asin, pi
    a1 = 3.*bl
    r3 = sqrt(3)
    a2 = bl*r3  # length of a1,a2 vector

    if not center:
        z = 0
    elif center:
        z = vacuum * center

    basis = [('C',0,0,z),('C',bl,0,z),('C',3./2.*bl,r3*bl/2,z),('C',5./2.*bl,r3*bl/2,z)]

    cell = [[  a1, 0.0, 0],
            [ 0.0,  a2, 0],
            [ 0.0, 0.0, vacuum]]
    #print(basis, cell)    
    if n*m != 1:
        basis, cell = SC_2d(n, m, basis, cell)

    return AtomsSystem(basis, cell)

def cnt(n, m, bl=1.415):
    """
    Generate Carbon nanotube
    """
    from math import sqrt, sin, cos, pi
    r3 = sqrt(3)
    a = bl*r3
    r = a*sqrt(n**2+n*m+m**2)/(2*pi)
    print ("r :", r)
    c = a*sqrt(n**2+n*m+m**2)
    print ("c :", c)
    unit= grp_nr(n, m, bl)
    unit1 = []
    for atom in unit:
        x,y,z = atom.get_position()
        x1 = r*cos(x*2*pi/c)
        z1 = r*sin(x*2*pi/c)
        atom.set_position((x1,z1,y))
        unit1.append(atom.copy())
    cnt_unit = AtomsSystem(unit1, cell=unit.get_cell())
    nr_cell = cnt_unit.get_cell()
    c = nr_cell[0][0]
    t = nr_cell[1][1]
    cell = [[c+10,0,0], [0,c+10,0], [0,0,t]]
    cnt_unit.set_cell(cell)
    return cnt_unit

def gnr_ac(n, m, bl=1.42, bl2 = 1.0927, passivation=0, symmetric=0, BN=0, repeat=1, vacuum=10.0, center=0.4):
    '''
    bl  C-C distance
    bl2 H-C distance
    n   number of chains along with x-axis which direct to z-axis
    m   length of chain along with z-axis
    '''
    from math import sqrt

    ### obtain grephene rectangle
    image = grp_rect(n, m, bl=bl, vacuum=vacuum, center=center)

    ### add last half chain if symmetric
    if symmetric:
        pass
    if passivation:
        pass

    cell = image.get_cell()
    cell[1][1] = cell[1][1] + vacuum
    image.set_cell(cell)
    image.select_all()
    image.translate(dx=0, dy=vacuum/2., dz=0)
    
    '''
    basis1 = Atom('C', [ 0.0,       y,    0.0   ])
    basis2 = Atom('C', [ 0.0,       y,    bl    ])
    basis3 = Atom('C', [ r3*bl/2,   y,    1.5*bl])
    basis4 = Atom('C', [ r3*bl/2,   y,    2.5*bl])
    basis  = [basis1, basis2, basis3, basis4]
    if n != 1:
        

    if n*m != 1:
        basis, _ = SC_2d(n, m, basis, uc)

    if BN:
        basis1 = Atom('B', [sqrt(3)*bl/2, 0.0,    0.0])
        basis2 = Atom('N', [         0.0, 0.0,   bl/2])
        basis3 = Atom('B', [         0.0, 0.0, 1.5*bl])
        basis4 = Atom('N', [sqrt(3)*bl/2, 0.0,   2*bl])
        basis  = AtomsSystem([basis1, basis2, basis3, basis4])

   # It's for Passivation 
    if passivation:
        #bl2 = 1.0927
        Hbasis1 = Atom('H', [-sqrt(3)*bl2/2,              0.0,   bl/2-bl2/2])
        Hbasis2 = Atom('H', [-sqrt(3)*bl2/2,              0.0, 1.5*bl+bl2/2])
        Hbasis3 = Atom('H', [ sqrt(3)*bl/2+sqrt(3)*bl2/2, 0.0,        bl2/2])
        Hbasis4 = Atom('H', [ sqrt(3)*bl/2+sqrt(3)*bl2/2, 0.0,   2*bl-bl2/2])
        Hbasis  = AtomsSystem([Hbasis1, Hbasis2])
        Hbasis_even = AtomsSystem([Hbasis3, Hbasis4])

    atoms = []
    i_n = 0
    
    if passivation:
        Htmp = Hbasis.copy()
        Htmp.select_all()
        atoms.append(Htmp[0].copy())
        atoms.append(Htmp[1].copy())    

        Htmp3 = Hbasis_even.copy()
        Htmp3.select_all()
        Htmp3.translate(sqrt(3)*bl*(i_n-1), 0.0, 0.0)
        atoms.append(Htmp3[0].copy())
        atoms.append(Htmp3[1].copy())

    cell = [ [ ax,   0.0,    0.0],
             [ 0.0,   ay,    0.0],
             [ 0.0,  0.0,     az]]
    '''           
    return image


def gnr_zigzag(n,m,bl=1.42, bl2=1.0974, passivation=1, BN=0, repeat=1, vacuum=10.0):
    from math import sqrt
    basis1 = Atom('C', [   0.0, 0.0, sqrt(3)*bl/2])
    basis2 = Atom('C', [  bl/2, 0.0,          0.0])
    basis3 = Atom('C', [1.5*bl, 0.0,          0.0])
    basis4 = Atom('C', [  2*bl, 0.0, sqrt(3)*bl/2])
    basis  = AtomsSystem([basis1, basis2, basis3, basis4])

    if BN:
        basis1 = Atom('B', [   0.0, 0.0, sqrt(3)*bl/2])
        basis2 = Atom('N', [  bl/2, 0.0,          0.0])
        basis3 = Atom('B', [1.5*bl, 0.0,          0.0])
        basis4 = Atom('N', [  2*bl, 0.0, sqrt(3)*bl/2])
        basis  = AtomsSystem([basis1, basis2, basis3, basis4])
    
    if passivation:
        Hbasis1 = Atom('H', [-bl2, 0.0, sqrt(3)*bl/2])
        Hbasis = AtomsSystem([Hbasis1])

        Hbasis2=  Atom('H', [bl/2, 0.0, 0.0])
        Hbasis_odd = AtomsSystem([Hbasis2])

    atoms = []
    nn, nm = divmod(n,2)
    i_n = 0
    while i_n < nn:
        tmp = basis.copy()
        tmp.select_all()
        tmp.translate(3*bl*i_n, 0.0, 0.0)
        for atm in tmp:
                atoms.append(atm.copy())
        i_n += 1
    if passivation:
        Htmp3 = Hbasis.copy()
        Htmp3.select_all()
        atoms.append(Htmp3[0].copy())

# If n == odd numbers
    if nm:
        tmp = AtomsSystem([basis1,basis2])
        tmp.select_all()
        tmp.translate(3*bl*i_n, 0.0, 0.0)
        atoms.append(tmp[0].copy())
        atoms.append(tmp[1].copy())

        if passivation:
            Htmp2 = Hbasis_odd.copy()
            Htmp2.select_all()
            Htmp2.translate(nn*(2*(bl+0.5*bl))+bl2, 0.0, 0.0)
            atoms.append(Htmp2[0].copy())

# If n == even numbers
    elif nm==0:
        if passivation:
            Htmp3 = Hbasis.copy()
            Htmp3.select_all()
            Htmp3.translate(2*bl*i_n+(i_n-1)*bl+2*bl2, 0.0, 0.0)
            atoms.append(Htmp3[0].copy())

    atoms2 = []
    i_m = 0
    while i_m < m:
        tmp = AtomsSystem(atoms)
        tmp.select_all()
        tmp.translate(0.0, 0.0, i_m*(sqrt(3)*bl))
        for atm in tmp:
                atoms2.append(atm.copy())
        i_m += 1

    cell = [ [20.0,  0.0,          0.0],
             [ 0.0, 20.0,          0.0],
             [ 0.0,  0.0, m*sqrt(3)*bl]]
    
    atoms2 = AtomsSystem(atoms2, cell=cell) * (1, 1, repeat)
    atoms2.select_all()
    atoms2.sort('z')
    atoms2.set_serials(1)
    return atoms2


def chain(repeat, bl=1.285, bla=0.0):
    from math import sqrt
    basis1 = Atom('C', [0.0, 0.0, 0.0])
    basis2 = Atom('C', [0.0, 0.0, bl+bla])
    cell = [[10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 2*bl]]
    return AtomsSystem([basis1, basis2], cell=cell)*(1,1,repeat)

