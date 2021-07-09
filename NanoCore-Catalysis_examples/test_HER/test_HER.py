from NanoCore import vasp
from NanoCore import catalysis
from NanoCore import surflab

at = surflab.fccsurfaces('Pt', '111', (2,2,4), vac=15)
at2 = vasp.Vasp(at)

TE_Sys, TE_SysH, ZPE, TS = vasp.run_series_HER(at2, nproc=40, npar=8, kpoints=[4,4,1],ediffg = -0.04)

Gibbs_noVib = catalysis.Calculation.Gibbs_HER([TE_Sys], [TE_SysH])
Gibbs_Vib   = catalysis.Calculation.Gibbs_HER([TE_Sys], [TE_SysH], [ZPE], [TS])

G_H_legend = ['noVib', 'Vib']

G_H        = Gibbs_noVib + Gibbs_Vib

catalysis.Performance.plot_HER(G_H, G_H_legend)



