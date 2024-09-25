import numpy as np
import matplotlib.pyplot as plt
import os, sys
import pyvista as pv
import h5py
from tqdm import tqdm
from scipy.constants import c as c_light

sys.path.append('../')

from wakis import SolverFIT3D
from wakis import GridFIT3D
from wakis import WakeSolver

# ---------- Domain setup ---------
# Number of mesh cells
Nx = 50
Ny = 50
Nz = 150
dt = 5.707829241e-12 # CST

# STL geometry from CST studio
stl_file_cavity = '/home/malthera/cernbox/Documents/CAVITY_pillbox_rounded_vaccum.stl'
stl_file_shell = '/home/malthera/cernbox/Documents/CAVITY_pillbox_rounded_solid.stl' 
surf = pv.read(stl_file_shell)

stl_solids = {'cavity': stl_file_cavity, 'shell': stl_file_shell}
stl_materials = {'cavity': 'vacuum',
                 'shell': 'metal'}

# Domain bounds
xmin, xmax, ymin, ymax, zmin, zmax = surf.bounds
Lx, Ly, Lz = (xmax-xmin), (ymax-ymin), (zmax-zmin)

# set grid and geometry
grid = GridFIT3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz, 
                stl_solids=stl_solids, 
                stl_materials=stl_materials)
    
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
wakelength = 120 #[m]
add_space = 30   # no. cells

wake = WakeSolver(q=q, sigmaz=sigmaz, beta=beta,
            xsource=xs, ysource=ys, xtest=xt, ytest=yt,
            add_space=add_space, save=True, logfile=True)

# ----------- Solver & Simulation ----------
# boundary conditions``
bc_low=['pec', 'pec', 'pec']
bc_high=['pec', 'pec', 'pec']

solver = SolverFIT3D(grid, wake, dt=dt,
                     bc_low=bc_low, bc_high=bc_high, 
                     use_stl=True, bg='pec')
# Plot settings
if not os.path.exists('img/'): os.mkdir('img/')
plotkw = {'title':'img/Ez', 
            'add_patch':'cavity', 'patch_alpha':0.3,
            'vmin':-1e4, 'vmax':1e4,
            'plane': [int(Nx/2), slice(0, Ny), slice(add_space, -add_space)]}

# Run wakefield time-domain simulation
run = True
if run:
    solver.wakesolve(wakelength=wakelength, add_space=add_space,
                    plot=False, plot_every=30, save_J=False,
                    use_etd=False,
                    **plotkw)

# Run only electromagnetic time-domain simulation
runEM = False
if runEM:
    from wakis.sources import Beam
    beam = Beam(q=q, sigmaz=sigmaz, beta=beta,
                xsource=xs, ysource=ys)

    solver.emsolve(Nt=500, source=beam, add_space=add_space,
                    plot=False, plot_every=30, save_J=False,
                    use_etd=True, **plotkw)

# Compute beam-coupling impedance and wake potential
wake.calc_long_WP()
wake.calc_long_Z(samples=1001)

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


"""
# Optional: plot the geometry imported using PyVista
# geometry = pv.read(stl_file_shell) + pv.read(stl_file_cavity)
# geometry.plot() # add pyvista **kwargs to make a fancier plot


# Domain bounds
xmin, xmax, ymin, ymax, zmin, zmax = surf.bounds
Lx, Ly, Lz = (xmax-xmin), (ymax-ymin), (zmax-zmin)

# set grid and geometry
grid = GridFIT3D(xmin, xmax, ymin, ymax, zmin, zmax,
                Nx, Ny, Nz, 
                stl_solids=stl_solids, 
                stl_materials=stl_materials)

# ------------ Beam source ----------------
# Beam parameters
sigmaz = 0.1    #[m]
q = 1e-9            #[C]
beta = 1.0          # beam beta
xs = 0.             # x source position [m]
ys = 0.             # y source position [m]
xt = 0.             # x test position [m]
yt = 0.             # y test position [m]
#DEFAULT> tinj = 8.53*sigmaz/c_light  # injection time offset [s]

# Simulation define wakelength

wakelength = 10 #[m]
add_space = 30 # no. cells

# Initialize wake solver with beam information
wake = WakeSolver(q=q,
                  sigmaz=sigmaz,
                  beta=beta,
                  xsource=xs, 
                  ysource=ys,
                  xtest=xt, 
                  ytest=yt,
                  add_space=add_space,
                  save=True, 
                  logfile=True,
                  results_folder='/home/malthera/Documents/WAKIS/wakis/results',
                  Ez_file='/home/malthera/Documents/WAKIS/wakis/results/Ez.h5')

# ----------- Solver & Simulation ----------
# boundary conditions
bc_low=['pec', 'pec', 'pec']
bc_high=['pec', 'pec', 'pec']


solver = SolverFIT3D(grid, 
                     bc_low=bc_low,
                     bc_high=bc_high, 
                     use_stl=True,
                     bg='pec')

# Plot settings
if not os.path.exists('img/'): os.mkdir('img/')
plotkw = {'title':'img/Ez', 
            'add_patch':'cavity', 'patch_alpha':0.3,
            'vmin':-1e4, 'vmax':1e4,
            'plane': [int(Nx/2), slice(0, Ny), slice(add_space, -add_space)]}
    
# Run wakefield time-domain simulation
run = True
if run:
    solver.wakesolve(wakelength=wakelength, add_space=add_space,
                    plot=False, plot_every=30, save_J=False,
                    use_etd=False,
                    **plotkw)

# Run only electromagnetic time-domain simulation
runEM = False
if runEM:
    from sources import Beam
    beam = Beam(q=q, sigmaz=sigmaz, beta=beta,
                xsource=xs, ysource=ys)

    solver.emsolve(Nt=500, source=beam, add_space=add_space,
                    plot=False, plot_every=30, save_J=False,
                    use_etd=True, **plotkw)

x, y, z = solver.x, solver.y, solver.z
ixs, iys = np.abs(x-xs).argmin(), np.abs(y-ys).argmin()
ixt, iyt = np.abs(x-xt).argmin(), np.abs(y-yt).argmin()

def beam(solver, t):
    # Define gaussian
    s0 = z.min()-c_light*tinj
    s = z-c_light*t
    profile = 1/np.sqrt(2*np.pi*sigmaz**2)*np.exp(-(s-s0)**2/(2*sigmaz**2))

    # Update current
    current = q*c_light*profile/solver.dx/solver.dy
    solver.J[ixs,iys,:,'z'] = current

# ------------ Time loop ----------------
# Define wake length


# Obtain simulation time
tmax = (wakelength + tinj*c_light + (zmax-zmin))/c_light #[s]
Nt = int(tmax/solver.dt)
t = np.arange(0, Nt*solver.dt, solver.dt)
ninj = int(tinj/solver.dt)

# Prepare save files
save = True
plot = False

if save:
    hf = h5py.File('/home/malthera/Documents/WAKIS/wakis/results/Ez.h5', 'w')
    hf2 = h5py.File('/home/malthera/Documents/WAKIS/wakis/results/Jz.h5', 'w')
    hf['x'], hf['y'], hf['z'], hf['t'] = x, y, z, t

for n in tqdm(range(Nt)):

    # Initial condition
    beam(solver, n*solver.dt)

    # Advance
    solver.one_step()

    # Plot 2D on-the-fly
    if n>ninj and n%10 == 0 and plot:
        solver.plot2D(field='E', component='y', plane='ZY', pos=0.5, norm=None, 
                      vmin=-5e5, vmax=5e5, cmap='rainbow', patch_alpha=0.5, 
                      add_patch='Cavity', patch_reverse=True, title='imgCav/Ey', off_screen=True,  
                      n=n, interpolation='spline36')
    
    # Save in hdf5 format (needed for Impedance)
    if save:
        hf['#'+str(n).zfill(5)]=solver.E[ixt, iyt, :, 'z'] 
        #hf2['#'+str(n).zfill(5)]=solver.J[ixt, iyt, :, 'z'] 

# Close save file
if save:
    hf.close()
    hf2.close()

# Plot 3D built-in
plot3d = False
if plot3d:
    solver.plot3D(field='E', component='Abs', clim=None, add_stl='Cavity', stl_opacity=0.0, 
                stl_colors='white', field_on_stl=True, nan_opacity=1.0, field_opacity=1.0,
                cmap='rainbow', off_screen=False, zoom=1.0, n=n)

# ------------ Beam-coupling Impedance ----------------


# Compute beam-coupling impedance and wake potential
wake.calc_long_WP()
wake.calc_long_Z(samples=1001)

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

    """
