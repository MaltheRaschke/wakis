import os, sys
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.constants import c as c_light
from tqdm import tqdm

sys.path.append('../../')

from wakis import SolverFIT3D
from wakis import GridFIT3D 
from wakis import WakeSolver

# ---------- Domain setup ---------
# Number of mesh cells
Nx = 80
Ny = 80
Nz = 141
#dt = 2.187760221e-12 # CST

# Embedded boundaries
stl_cavity = 'cav.stl' 
stl_shell = 'shell.stl'
surf = pv.read(stl_shell)
#surf.plot()

stl_solids = {'cavity': stl_cavity, 'shell': stl_shell}
stl_materials = {'cavity': 'vacuum', 'shell': [30, 1.0, 30]}

# Domain bounds
xmin, xmax, ymin, ymax, zmin, zmax = surf.bounds
Lx, Ly, Lz = (xmax-xmin), (ymax-ymin), (zmax-zmin)

# set grid and geometry
grid = GridFIT3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz, 
                stl_solids=stl_solids, 
                stl_materials=stl_materials,
                stl_scale=1.0)
    
# ------------ Beam source ----------------
# Beam parameters
sigmaz = 10e-2      #[m] -> 2 GHz
q = 1e-9            #[C]
beta = 1.0          # beam beta 
xs = 0.             # x source position [m]
ys = 0.             # y source position [m]
xt = 0.             # x test position [m]
yt = 0.             # y test position [m]
# [DEFAULT] tinj = 8.53*sigmaz/c_light  # injection time offset [s] 

# Simualtion
wakelength = 10. #[m]
add_space = 10   # no. cells

wake = WakeSolver(q=q, sigmaz=sigmaz, beta=beta,
            xsource=xs, ysource=ys, xtest=xt, ytest=yt,
            add_space=add_space, save=True, logfile=True)

# ----------- Solver & Simulation ----------
# boundary conditions``
bc_low=['pec', 'pec', 'pec']
bc_high=['pec', 'pec', 'pec']

solver = SolverFIT3D(grid, wake, #dt=dt,
                     bc_low=bc_low, bc_high=bc_high, 
                     use_stl=True, bg='pec')
# Plot settings
if not os.path.exists('img/'): os.mkdir('img/')
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('name', plt.cm.jet(np.linspace(0.1, 0.9))) # CST's colormap

plotkw2D = {'title':'img/Ez', 
            'add_patch':['cavity'], 'patch_alpha':1.0,
            'patch_reverse' : True, 
            'vmin':-1e3, 'vmax':1e3,
            'cmap': cmap,
            'plane': [int(Nx/2), slice(0, Ny), slice(add_space, -add_space)]}

plotkw3D = {'title':'img/Ez3d', 
            'add_stl':'shell',
            'field_opacity':0.5,}

# For plotting
run = True
if run:
    from wakis.sources import Beam
    beam = Beam(q=q, sigmaz=sigmaz, beta=beta,
                xsource=xs, ysource=ys)
    Nt = 3000            
    for n in tqdm(range(Nt)):

        beam.update(solver, n*solver.dt)

        if n%30 == 0 and n>600:
            solver.plot3DonSTL('E', component='Abs', cmap=cmap, clim=[0, 500],
                stl_with_field='cavity', field_opacity=1.0,
                stl_transparent='shell', stl_opacity=0.1, stl_colors='white',
                clip_plane=True, clip_normal='-y', clip_origin=[0,0,0],
                off_screen=True, zoom=1.2, n=n, title='img/Ez')
            
        solver.one_step()

# Run wakefield time-domain simulation
run = False
if run:
    solver.wakesolve(wakelength=wakelength, add_space=add_space,
                    plot=True, plot_every=30, plot_until=3000,
                    save_J=False, **plotkw2D)

# Run only electromagnetic time-domain simulation
runEM = False
if runEM:
    from wakis.sources import Beam
    beam = Beam(q=q, sigmaz=sigmaz, beta=beta,
                xsource=xs, ysource=ys)

    solver.emsolve(Nt=8000, source=beam,
                   plot=True, plot_every=100, 
                   **plotkw2D)

# Or, load previous results
run = False
if run:
    wake.solve()
    #wake.load_results(folder='results/')

#-------------- Compare with CST -------------

#--- Longitudinal wake and impedance ---
plot = False
if plot:
    # CST wake
    cstWP = wake.read_txt('cst/WP_wl10000.txt')
    cstZ = wake.read_txt('cst/Z_wl10000.txt')
    wake.f = np.abs(wake.f)

    fig, ax = plt.subplots(1,2, figsize=[12,4], dpi=150)
    ax[0].plot(wake.s*1e2, wake.WP, c='r', lw=1.5, label='FIT+Wakis')
    ax[0].plot(cstWP[0], cstWP[1], c='k', ls='--', lw=1.5, label='CST')
    ax[0].set_xlabel('s [cm]')
    ax[0].set_ylabel('Longitudinal wake potential [V/pC]', color='r')
    ax[0].legend()
    ax[0].set_xlim(xmax=wakelength*1e2)

    ax[1].plot(wake.f*1e-9, np.abs(wake.Z), c='b', lw=1.5, label='FIT+Wakis')
    ax[1].plot(cstZ[0], cstZ[1], c='k', ls='--', lw=1.5, label='CST')
    ax[1].set_xlabel('f [GHz]')
    ax[1].set_ylabel('Longitudinal impedance [Abs][$\Omega$]', color='b')
    ax[1].legend()

    fig.suptitle('Benchmark with CST Wakefield Solver')
    fig.tight_layout()
    fig.savefig('results/benchmark.png')

    plt.show()

plot = False
if plot:
    wake.f = np.abs(wake.f)
    zmax = solver.z.max() + solver.add_space*solver.dz
    zmin = solver.z.min() - solver.add_space*solver.dz
    end = 3000*solver.dt*wake.v - (zmax-zmin) + wake.ti*wake.v

    for n in range(3000,35500,500):

        part = n*solver.dt*wake.v - (zmax-zmin) + wake.ti*wake.v
        end = part

        fig, ax = plt.subplots(1,1, figsize=[8,4], dpi=150)
        ax.plot(wake.s[wake.s < part]*1e2, wake.WP[wake.s < part], c='r', lw=1.5, label='Wakis')
        ax.set_xlabel('s [cm]')
        ax.set_ylabel('Longitudinal wake potential [V/pC]', color='r')
        ax.set_xlim(xmin=wake.s.min()*1e2, xmax=end*1.05*1e2)
        ax.set_ylim(ymin=-wake.WP.max(), ymax=wake.WP.max())

        axx = ax.twinx()
        axx.plot(wake.s[wake.s < part]*1e2, wake.lambdas[wake.s < part], c='darkorange')
        axx.set_ylabel('Beam current density [C/m]', color='darkorange')
        axx.set_ylim(ymin=-wake.lambdas.max(), ymax=wake.lambdas.max())

        fig.tight_layout()
        fig.savefig(f'img/WP_{str(n).zfill(5)}.png')
        plt.close()

        #plt.show()

#--- 1d Ez field ---
plot = False
if plot:
    # E field
    d = wake.read_Ez('results_sigma5/Ez.h5',return_value=True)
    dd = wake.read_Ez('results_pec/Ez.h5',return_value=True)
    t, z = np.array(d['t']), np.array(d['z'])    
    dt = t[1]-t[0]
    steps = list(d.keys())

    # Beam J
    current = wake.read_Ez('Jz.h5',return_value=True)

    for n, step in enumerate(steps[:1740:20]):
        fig, ax = plt.subplots(1,1, figsize=[6,4], dpi=150)
        axx = ax.twinx()  

        ax.plot(z, d[step][1,1,:], c='g', lw=1.5, label='Ez(0,0,z) FIT | $\sigma$ = 5 S/m')
        ax.plot(z, dd[step][1,1,:], c='grey', lw=1.5, label='Ez(0,0,z) FIT | PEC')
        ax.set_xlabel('z [m]')
        ax.set_ylabel('$E_z$ field amplitude [V/m]', color='g')
        ax.set_ylim(-3e3, 3e3)
        ax.set_xlim(z.min(), z.max())
        
        # CST E field
        try:    
            cstfiles = sorted(os.listdir('cst/1d/'))
            cst = wake.read_txt('cst/1d/'+cstfiles[n])
            ax.plot(cst[0]*1e-2, cst[1], c='k', lw=1.5, ls='--', label='Ez(0,0,z) CST | $\sigma$ = 10 S/m')
        except:
            pass

        ax.legend(loc=1)

        # charge distribution
        axx.plot(z, current[step][1,1,:], c='r', lw=1.0, label='lambda λ(z)')
        axx.set_ylabel('$J_z$ beam current [C/m]', color='r')
        axx.set_ylim(-8e4, 8e4)

        fig.suptitle('timestep='+str(n*20))
        fig.tight_layout()
        fig.savefig('img/Ez1d_'+str(n*20).zfill(6)+'.png')

        plt.clf()
        plt.close(fig)

#-------------- Compare result files -------------

#--- Longitudinal wake and impedance ---

# compare sigma
plot = False
if plot:
    # CST wake
    cstWP = wake.read_txt('cst/WP.txt')
    cstZ = wake.read_txt('cst/Z.txt')

    fig, ax = plt.subplots(1,2, figsize=[12,4], dpi=150)

    # Wakis wake
    keys = ['pec', 'sigma1000', 'sigma100', 'sigma10', 'sigma5']
    res = {}
    for k in keys:
        res[k] = wake.copy()
        res[k].load_results(f'results_{k}/')

    ax[0].plot(res[keys[0]].s, res[keys[0]].WP, c='grey', lw=1.5, alpha=0.4, label='PEC')
    ax[0].plot(res[keys[1]].s, res[keys[1]].WP, c='r', lw=1.5, alpha=0.2, label='$\sigma$ = 1000 S/m')
    ax[0].plot(res[keys[2]].s, res[keys[1]].WP, c='r', lw=1.5, alpha=0.4, label='$\sigma$ = 100 S/m')
    ax[0].plot(res[keys[3]].s, res[keys[2]].WP, c='r', lw=1.5, alpha=0.6, label='$\sigma$ = 10 S/m')
    ax[0].plot(res[keys[4]].s, res[keys[3]].WP, c='r', lw=1.5, label='$\sigma$ = 5 S/m')

    ax[0].plot(cstWP[0]*1e-2, cstWP[1], c='k', ls='--', lw=1.5, label='CST $\sigma$ = 10 S/m')
    ax[0].set_xlabel('s [m]')
    ax[0].set_ylabel('Longitudinal wake potential [V/pC]', color='r')
    ax[0].legend()

    ax[1].plot(res[keys[0]].f*1e-9, np.abs(res[keys[0]].Z), c='grey', lw=1.5, alpha=0.4, label='PEC')
    ax[1].plot(res[keys[1]].f*1e-9, np.abs(res[keys[1]].Z), c='b', lw=1.5, alpha=0.2, label='$\sigma$ = 1000 S/m')
    ax[1].plot(res[keys[2]].f*1e-9, np.abs(res[keys[2]].Z), c='b', lw=1.5, alpha=0.4, label='$\sigma$ = 100 S/m')
    ax[1].plot(res[keys[3]].f*1e-9, np.abs(res[keys[3]].Z), c='b', lw=1.5, alpha=0.6, label='$\sigma$ = 10 S/m')
    ax[1].plot(res[keys[4]].f*1e-9, np.abs(res[keys[4]].Z), c='b', lw=1.5, label='$\sigma$ = 5 S/m')

    ax[1].plot(cstZ[0], cstZ[1], c='k', ls='--', lw=1.5, label='CST $\sigma$ = 10 S/m')
    ax[1].set_xlabel('f [GHz]')
    ax[1].set_ylabel('Longitudinal impedance [Abs][$\Omega$]', color='b')
    ax[1].legend()

    fig.suptitle('Benchmark with CST Wakefield Solver')
    fig.tight_layout()
    fig.savefig('benchmark_sigma.png')

    plt.show()

# compare Nz
plot = False
if plot:

    # CST wake
    cstWP = wake.read_txt('cst/WP.txt')
    cstZ = wake.read_txt('cst/Z.txt')

    fig, ax = plt.subplots(1,2, figsize=[12,4], dpi=150)

    # Wakis wake
    keys = ['sigma5', 'sigma5Nz129', 'sigma5Nz149']
    res = {}
    for k in keys:
        res[k] = wake.copy()
        res[k].load_results(f'results_{k}/')

    ax[0].plot(res[keys[0]].s, res[keys[0]].WP, c='r', lw=1.5, alpha=0.4, label='Nz = 109')
    ax[0].plot(res[keys[0]].s, res[keys[0]].WP, c='r', lw=1.5, alpha=0.6, label='Nz = 129')
    ax[0].plot(res[keys[0]].s, res[keys[0]].WP, c='r', lw=1.5, label='Nz = 149')

    ax[0].plot(cstWP[0]*1e-2, cstWP[1], c='k', ls='--', lw=1.5, label='CST Nz = 109')
    ax[0].set_xlabel('s [m]')
    ax[0].set_ylabel('Longitudinal wake potential [V/pC]', color='r')
    ax[0].legend()

    ax[1].plot(res[keys[0]].f*1e-9, np.abs(res[keys[0]].Z), c='b', lw=1.5, alpha=0.4, label='Nz = 109')
    ax[1].plot(res[keys[1]].f*1e-9, np.abs(res[keys[1]].Z), c='b', lw=1.5, alpha=0.6, label='Nz = 129')
    ax[1].plot(res[keys[2]].f*1e-9, np.abs(res[keys[2]].Z), c='b', lw=1.5, label='Nz = 149')

    ax[1].plot(cstZ[0], cstZ[1], c='k', ls='--', lw=1.5, label='CST Nz = 109')
    ax[1].set_xlabel('f [GHz]')
    ax[1].set_ylabel('Longitudinal impedance [Abs][$\Omega$]', color='b')
    ax[1].legend()

    fig.suptitle('Benchmark with CST Wakefield Solver')
    fig.tight_layout()
    fig.savefig('benchmark_Nz.png')

    plt.show()

# compare add_space
plot = False
if plot:
    # CST wake
    cstWP = wake.read_txt('cst/WP.txt')
    cstZ = wake.read_txt('cst/Z.txt')

    fig, ax = plt.subplots(1,2, figsize=[12,4], dpi=150)

    # Wakis wake
    keys = ['addspace0', 'sigma5', 'addspace20', 'addspace30']
    res = {}
    for k in keys:
        res[k] = wake.copy()
        res[k].load_results(f'results_{k}/')

    ax[0].plot(res[keys[0]].s, res[keys[0]].WP, c='grey', lw=1.5, alpha=0.4, label='add_space = 0')
    ax[0].plot(res[keys[1]].s, res[keys[1]].WP, c='r', lw=1.5, alpha=0.4, label='add_space = 10')
    ax[0].plot(res[keys[2]].s, res[keys[2]].WP, c='r', lw=1.5, alpha=0.6, label='add_space = 20')
    ax[0].plot(res[keys[3]].s, res[keys[3]].WP, c='r', lw=1.5, label='add_space = 30')

    ax[0].plot(cstWP[0]*1e-2, cstWP[1], c='k', ls='--', lw=1.5, label='CST')
    ax[0].set_xlabel('s [m]')
    ax[0].set_ylabel('Longitudinal wake potential [V/pC]', color='r')
    ax[0].legend()

    ax[1].plot(res[keys[0]].f*1e-9, np.abs(res[keys[0]].Z), c='grey', lw=1.5, alpha=0.4, label='add_space = 0')
    ax[1].plot(res[keys[1]].f*1e-9, np.abs(res[keys[1]].Z), c='b', lw=1.5, alpha=0.4, label='add_space = 10')
    ax[1].plot(res[keys[2]].f*1e-9, np.abs(res[keys[2]].Z), c='b', lw=1.5, alpha=0.6, label='add_space = 20')
    ax[1].plot(res[keys[3]].f*1e-9, np.abs(res[keys[3]].Z), c='b', lw=1.5, label='add_space = 30')

    ax[1].plot(cstZ[0], cstZ[1], c='k', ls='--', lw=1.5, label='CST')
    ax[1].set_xlabel('f [GHz]')
    ax[1].set_ylabel('Longitudinal impedance [Abs][$\Omega$]', color='b')
    ax[1].legend()

    fig.suptitle('Benchmark with CST Wakefield Solver')
    fig.tight_layout()
    fig.savefig('benchmark_addspace.png')

    plt.show()

# compare NxNy
plot = False
if plot:
    # CST wake
    cstWP = wake.read_txt('cst/WP.txt')
    cstZ = wake.read_txt('cst/Z.txt')

    fig, ax = plt.subplots(1,2, figsize=[12,4], dpi=150)

    # Wakis wake
    keys = ['sigma5', 'NxNy67', 'NxNy77', 'NxNyNz+20']
    res = {}
    for k in keys:
        res[k] = wake.copy()
        res[k].load_results(f'results_{k}/')

    ax[0].plot(res[keys[0]].s, res[keys[0]].WP, c='grey', lw=1.5, alpha=0.4, label='Nx, Ny = 57')
    ax[0].plot(res[keys[1]].s, res[keys[1]].WP, c='r', lw=1.5, alpha=0.4, label='Nx, Ny = 67')
    ax[0].plot(res[keys[2]].s, res[keys[2]].WP, c='r', lw=1.5, alpha=0.6, label='Nx, Ny = 77')
    ax[0].plot(res[keys[3]].s, res[keys[3]].WP, c='r', lw=1.5, label='Nx, Ny, Nz += 20')

    ax[0].plot(cstWP[0]*1e-2, cstWP[1], c='k', ls='--', lw=1.5, label='CST')
    ax[0].set_xlabel('s [m]')
    ax[0].set_ylabel('Longitudinal wake potential [V/pC]', color='r')
    ax[0].legend()

    ax[1].plot(res[keys[0]].f*1e-9, np.abs(res[keys[0]].Z), c='grey', lw=1.5, alpha=0.4, label='Nx, Ny = 57')
    ax[1].plot(res[keys[1]].f*1e-9, np.abs(res[keys[1]].Z), c='b', lw=1.5, alpha=0.4, label='Nx, Ny = 67')
    ax[1].plot(res[keys[2]].f*1e-9, np.abs(res[keys[2]].Z), c='b', lw=1.5, alpha=0.6, label='Nx, Ny = 77')
    ax[1].plot(res[keys[3]].f*1e-9, np.abs(res[keys[3]].Z), c='b', lw=1.5, label='Nx, Ny, Nz += 20')

    ax[1].plot(cstZ[0], cstZ[1], c='k', ls='--', lw=1.5, label='CST')
    ax[1].set_xlabel('f [GHz]')
    ax[1].set_ylabel('Longitudinal impedance [Abs][$\Omega$]', color='b')
    ax[1].legend()

    fig.suptitle('Benchmark with CST Wakefield Solver')
    fig.tight_layout()
    fig.savefig('benchmark_NxNy.png')

    plt.show()


#-------------- Compare .h5 files -------------
plot = False
if plot:
    # E field
    d1 = wake.read_Ez('EzABC.h5',return_value=True)
    d2 = wake.read_Ez('EzPEC.h5',return_value=True)
    d3 = wake.read_Ez('EzPECadd.h5',return_value=True)

    t = np.array(d1['t'])   
    dt = t[1]-t[0]
    steps = list(d1.keys())

    # Beam J
    dd = wake.read_Ez('Jz.h5',return_value=True)

    for n, step in enumerate(steps[:3750:30]):
        fig, ax = plt.subplots(1,1, figsize=[6,4], dpi=150)
        axx = ax.twinx()  

        # Beam current
        axx.plot(np.array(d1['z']), dd[step][1,1,:], c='r', lw=1.0, label='lambda λ(z)')
        axx.set_ylabel('$J_z$ beam current [C/m]', color='r')
        axx.set_ylim(-7e6, 7e6)

        # E field
        ax.plot(np.array(d1['z']) , d1[step][1,1,:], c='b', lw=1.5, label='Ez(0,0,z) ABC bc')
        ax.plot(np.array(d2['z']) , d2[step][1,1,:], c='g', lw=1.5, label='Ez(0,0,z) PEC bc')
        ax.plot(np.array(d3['z']) , d3[step][1,1,:], c='limegreen', lw=1.5, label='Ez(0,0,z) PEC+addspace 15')
        ax.set_xlabel('z [m]')
        ax.set_ylabel('$E_z$ field amplitude [V/m]', color='k')
        ax.set_ylim(-4e4, 4e4)
        ax.set_xlim(np.array(d1['z']).min(), np.array(d1['z']).max())
        ax.legend(loc=1)

        fig.suptitle('timestep='+str(n*30))
        fig.tight_layout()
        fig.savefig('img/Ez1d_'+str(n*30).zfill(6)+'.png')

        plt.clf()
        plt.close(fig)