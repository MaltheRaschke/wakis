import numpy as np
import matplotlib.pyplot as plt
import os, sys
import pyvista as pv
import h5py
from tqdm import tqdm
from scipy.constants import c as c_light

#sys.path.append('../')

from wakis import SolverFIT3D
from wakis import GridFIT3D
from wakis import WakeSolver

# ---------- Domain setup ---------
# Number of mesh cells
Nx = 27
Ny = 27
Nz = 111
#dt = 5.707829241e-12 # CST

# STL geometry from CST studio
stl_file_cavity = '/home/malthera/cernbox/Documents/CAVITY_pillbox_rounded_vaccum.stl'
stl_file_shell = '/home/malthera/cernbox/Documents/CAVITY_pillbox_rounded_solid.stl' 
surf = pv.read(stl_file_shell) + pv.read(stl_file_cavity)
stl_scale = 1e-2
#surf.plot()

stl_scale = {'cavity': [1e-2, 1e-2, 1e-2], 'shell': [1e-2, 1e-2, 1e-2]} # scale factor

stl_solids = {'cavity': stl_file_cavity, 'shell': stl_file_shell}
stl_materials = {'cavity': 'vacuum',
                 'shell': [100.0, 1.0, 100.0]}

# Domain bounds

xmin, xmax, ymin, ymax, zmin, zmax = surf.bounds
print(surf.bounds)
#Lx, Ly, Lz = (xmax-xmin), (ymax-ymin), (zmax-zmin)

# set grid and geometry
grid = GridFIT3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz, 
                stl_solids=stl_solids, 
                stl_materials=stl_materials)

#cpw = 40
    
# ------------ Beam source ----------------
# Beam parameters
sigmaz = 0.1       #[m] -> 2 GHz
q = 1e-9            #[C]
beta = 1.0          # beam beta TODO
xs = 0.             # x source position [m]
ys = 0.             # y source position [m]
xt = 0.             # x test position [m]
yt = 0.             # y test position [m]
# [DEFAULT] tinj = 8.53*sigmaz/c_light  # injection time offset [s] 

# Simualtion
wakelength = 100 #[m]
add_space = 1   # no. cells

wake = WakeSolver(q=q, sigmaz=sigmaz, beta=beta,
            xsource=xs, ysource=ys, xtest=xt, ytest=yt,
            add_space=add_space, save=True, logfile=False)

# ----------- Solver & Simulation ----------
# boundary conditions``
bc_low=['pec', 'pec', 'pec']
bc_high=['pec', 'pec', 'pec']

solver = SolverFIT3D(grid, wake,
                     bc_low=bc_low, bc_high=bc_high, 
                     use_stl=True, bg='pec')

#solver.ieps.inspect()

# Plot settings
if not os.path.exists('img/'): os.mkdir('img/')
plotkw = {'title':'img/Ez', 
            'add_patch':'cavity', 'patch_alpha':0.3,
            'vmin':-1e4, 'vmax':1e4,
            'plane': [int(Nx/2), slice(0, Ny), slice(add_space, Nz-add_space)]}

# Run wakefield time-domain simulation
run = True
if run:
    solver.wakesolve(wakelength=wakelength, add_space=add_space,
                    plot=True, plot_every=30, plot_until=30e3, save_J=False,
                    **plotkw)

plot = True
if plot:
    fig, ax = plt.subplots(1,2, figsize=[12,4], dpi=150)
    ax[0].plot(wake.s*1e3, wake.WP, c='r', lw=1.5)
    ax[0].set_xlabel('s [mm]')
    ax[0].set_ylabel('Longitudinal wake potential [V/pC]', color='r')

    ax[1].plot(wake.f*1e-9, np.abs(wake.Z), c='b', lw=1.5)
    ax[1].set_xlabel('f [GHz]')
    ax[1].set_ylabel('Longitudinal impedance [Abs][$\Omega$]', color='b')

    fig.tight_layout()
    fig.savefig('/home/malthera/Documents/WAKIS/wakis/results/longitudinal.png')

    plt.show()