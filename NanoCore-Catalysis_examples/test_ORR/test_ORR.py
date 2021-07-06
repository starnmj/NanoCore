from NanoCore import vasp, vasp_old
from NanoCore import catalysis     

at = vasp_old.read_poscar('CONTCAR_Pt-SAC')
at2 = vasp.Vasp(at)

TE, ZPE, TS = vasp.Vasp.run_series_ORR(at2, nproc=40, npar=8, mode='opt', kpoints=[5,5,1], vib=1, label='test')
print(TE)
print(ZPE)
print(TS)

G_ORR      = catalysis.Calculation.Gibbs_ORR_4e_acid(TE=TE, pH=0)
print(G_ORR)
G_ORR_vib  = catalysis.Calculation.Gibbs_ORR_4e_acid(TE=TE, ZPE=ZPE, TS=TS, pH=0)
print(G_ORR_vib)
catalysis.Performance.plot_ORR_4e_acid(G_ORR_vib, U=0.7, legend=['U=1.23V', 'U=0.70V', 'U=0.00V'])

