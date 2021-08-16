#
# VASP
#

vaspini = '/TGM/Apps/VASP/POTCAR/'

vasp_calculator = '/TGM/Apps/VASP/bin/5.4.4/O3/NORMAL/vasp.5.4.4.pl2.O3.NORMAL.std.x' 
vasp_POTCAR_LDA  = vaspini + '1.POTPAW.LDA.54.RECOMMEND'
vasp_POTCAR_PBE  = vaspini + '2.POTPAW.PBE.54.RECOMMEND'
vasp_POTCAR_PW91 = vaspini + '2.POTPAW.PBE.54.RECOMMEND'        # connect to correct directory

#
# SIESTA
#

# Siesta calculator location
siesta_calculator = '[siesta-installed-location]/siesta-4.1-b3/Obj/siesta'

# Pseudopentiential files location
siesta_psf_location = ''

# Pyprojection files location
import os
temporary_location  = os.getcwd()
siesta_pyprojection = temporary_location + '/pyprojection/pyprojection.py'

# Siesta utilities location
siesta_util_location = '[siesta-installed-location]/siesta-4.1-b3/Util'
siesta_util_band = 'Bands/gnubands'
siesta_util_dos  = 'Eig2DOS/Eig2DOS'
siesta_util_pdos = 'Contrib/APostnikov/fmpdos'
siesta_util_rho  = 'Contrib/APostnikov/rho2xsf'
siesta_util_vh   = 'Macroave/Src/macroave'

