from NanoCore import *

# modeling tool
atoms = carbonlab.grp(0,0)
print atoms

# simulation object
sim = s2.Siesta(atoms)

# set simulation options
sim.set_option('kgrid', [10,10,1])      # set #kpoints
sim.set_option('kshift', [0.5,0.5,0.0]) # set k shift from gamma
sim.set_option('MixingWt', 0.10)        # adjust mixing weight (density)
sim.set_option('BasisSize', 'SZ')       # adjust basis size

# run siesta: 1st run
sim.run()
e1 = s2.get_total_energy()
print "With 10 kpts, Etot is %12.6f." % e1

# increase #kpoints
sim.set_option('kgrid', [100,100,1])    # set #kpoints

# run siesta: 2nd run
sim.run()
e2 = s2.get_total_energy()
print "With 100 kpts, Etot is %12.6f." % e2
