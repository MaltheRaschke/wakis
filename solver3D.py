import numpy as np
from scipy.constants import c as c_light, epsilon_0 as eps_0, mu_0 as mu_0
from solver import EMSolver2D


def eq(a, b, tol=1e-8):
    return abs(a - b) < tol


def neq(a, b, tol=1e-8):
    return not eq(a, b, tol)


class EMSolver3D:
    def __init__(self, grid, sol_type, cfln, i_s, j_s, k_s):
        self.grid = grid
        self.type = type
        self.cfln = cfln

        self.dt = cfln / (c_light * np.sqrt(1 / self.grid.dx ** 2 + 1 / self.grid.dy ** 2 +
                                            1 / self.grid.dz ** 2))
        self.dx = self.grid.dx
        self.dy = self.grid.dy
        self.dz = self.grid.dz
        self.Nx = self.grid.nx
        self.Ny = self.grid.ny
        self.Nz = self.grid.nz
        self.sol_type = sol_type

        self.Ex = np.zeros((self.Nx + 1, self.Ny + 1, self.Nz + 1))
        self.Ey = np.zeros((self.Nx + 1, self.Ny + 1, self.Nz + 1))
        self.Ez = np.zeros((self.Nx + 1, self.Ny + 1, self.Nz + 1))
        self.Hx = np.zeros((self.Nx, self.Ny, self.Nz))
        self.Hy = np.zeros((self.Nx, self.Ny, self.Nz))
        self.Hz = np.zeros((self.Nx, self.Ny, self.Nz))
        self.Jx = np.zeros((self.Nx + 1, self.Ny + 1, self.Nz + 1))
        self.Jy = np.zeros((self.Nx + 1, self.Ny + 1, self.Nz + 1))
        self.Jz = np.zeros((self.Nx + 1, self.Ny + 1, self.Nz + 1))
        self.rho_xy = np.zeros((self.Nx, self.Ny, self.Nz))
        self.rho_yz = np.zeros((self.Nx, self.Ny, self.Nz))
        self.rho_zx = np.zeros((self.Nx, self.Ny, self.Nz))

        if (sol_type is not 'FDTD') and (sol_type is not 'DM') and (sol_type is not 'ECT'):
            raise ValueError("sol_type must be:\n" +
                             "\t'FDTD' for standard staircased FDTD\n" +
                             "\t'DM' for Dey-Mittra conformal FDTD\n" +
                             "\t'ECT' for Enlarged Cell Technique conformal FDTD")

        if sol_type is 'DM' or sol_type is 'ECT':
            self.Vxy = np.zeros((self.Nx + 1, self.Ny + 1, self.Nz + 1))
            self.Vyz = np.zeros((self.Nx + 1, self.Ny + 1, self.Nz + 1))
            self.Vzx = np.zeros((self.Nx + 1, self.Ny + 1, self.Nz + 1))
        if sol_type is 'ECT':
            self.Vxy_enl = np.zeros((self.Nx + 1, self.Ny + 1, self.Nz + 1))
            self.Vyz_enl = np.zeros((self.Nx + 1, self.Ny + 1, self.Nz + 1))
            self.Vzx_enl = np.zeros((self.Nx + 1, self.Ny + 1, self.Nz + 1))

        self.C1 = self.dt / (self.dx * mu_0)
        self.C2 = self.dt / (self.dy * mu_0)
        self.C7 = self.dt / (self.dz * mu_0)
        self.C4 = self.dt / (self.dy * eps_0)
        self.C5 = self.dt / (self.dx * eps_0)
        self.C8 = self.dt / (self.dz * eps_0)
        self.C3 = self.dt / eps_0
        self.C6 = self.dt / eps_0

        self.CN = self.dt/mu_0

        # indices for the source
        self.i_s = i_s
        self.j_s = j_s
        self.k_s = k_s

        self.time = 0

    def gauss(self, t):
        tau = 20 * self.dt
        if t < 6 * tau:
            return 100 * np.exp(-(t - 3 * tau) ** 2 / tau ** 2)
        else:
            return 0.

    def one_step(self):
        if self.sol_type == 'ECT':
            self.compute_v_and_rho()
            self.one_step_ect()
            self.advance_e_dm()
            self.time += self.dt
        if self.sol_type == 'FDTD':
            self.one_step_fdtd()
        if self.sol_type == 'DM':
            self.one_step_dm()

    def one_step_fdtd(self):
        Ex = self.Ex
        Ey = self.Ey
        Ez = self.Ez
        Hx = self.Hx
        Hy = self.Hy
        Hz = self.Hz
        # Compute cell voltages
        for ii in range(self.Nx):
            for jj in range(self.Ny):
                for kk in range(self.Nz):
                    if self.grid.flag_int_cell_yz[ii, jj, kk]:
                        Hx[ii, jj, kk] = (Hx[ii, jj, kk] -
                                          self.C2 * (Ez[ii, jj + 1, kk] - Ez[ii, jj, kk]) +
                                          self.C7 * (Ey[ii, jj, kk + 1] - Ey[ii, jj, kk]))

                    if self.grid.flag_int_cell_zx[ii, jj, kk]:
                        Hy[ii, jj, kk] = (Hy[ii, jj, kk] -
                                          self.C7 * (Ex[ii, jj, kk + 1] - Ex[ii, jj, kk]) +
                                          self.C1 * (Ez[ii + 1, jj, kk] - Ez[ii, jj, kk]))

                    if self.grid.flag_int_cell_xy[ii, jj, kk]:
                        Hz[ii, jj, kk] = (Hz[ii, jj, kk] -
                                          self.C1 * (Ey[ii + 1, jj, kk] - Ey[ii, jj, kk]) +
                                          self.C2 * (Ex[ii, jj + 1, kk] - Ex[ii, jj, kk]))

        for ii in range(self.Nx):
            for jj in range(self.Ny):
                for kk in range(self.Nz):
                    if self.grid.l_x[ii, jj, kk] > 0:
                        Ex[ii, jj, kk] = (Ex[ii, jj, kk] - self.C3 * self.Jx[ii, jj, kk] +
                                          self.C4 * (Hz[ii, jj, kk] - Hz[ii, jj - 1, kk]) -
                                          self.C8 * (Hy[ii, jj, kk] - Hy[ii, jj, kk - 1]))

                    if self.grid.l_y[ii, jj, kk] > 0:
                        Ey[ii, jj, kk] = (Ey[ii, jj, kk] - self.C3 * self.Jy[ii, jj, kk] +
                                          self.C8 * (Hx[ii, jj, kk] - Hx[ii, jj, kk - 1]) -
                                          self.C5 * (Hz[ii, jj, kk] - Hz[ii - 1, jj, kk]))

                    if self.grid.l_z[ii, jj, kk] > 0:
                        Ez[ii, jj, kk] = (Ez[ii, jj, kk] - self.C3 * self.Jz[ii, jj, kk] +
                                          self.C5 * (Hy[ii, jj, kk] - Hy[ii - 1, jj, kk]) -
                                          self.C4 * (Hx[ii, jj, kk] - Hx[ii, jj - 1, kk]))

        self.time += self.dt

    def one_step_dm(self):
        self.compute_v_and_rho()

        for i in range(self.Nx):
            for j in range(self.Ny):
                for k in range(self.Nz):
                    if self.grid.flag_int_cell_xy[i, j, k]:
                        self.Hz[i, j, k] = (self.Hz[i, j, k] -
                                            self.dt / (mu_0 * self.grid.Sxy[i, j, k]) *
                                            self.Vxy[i, j, k])

                    if self.grid.flag_int_cell_yz[i, j, k]:
                        self.Hx[i, j, k] = (self.Hx[i, j, k] -
                                            self.dt / (mu_0 * self.grid.Syz[i, j, k]) *
                                            self.Vyz[i, j, k])

                    if self.grid.flag_int_cell_zx[i, j, k]:
                        self.Hy[i, j, k] = (self.Hy[i, j, k] -
                                            self.dt / (mu_0 * self.grid.Szx[i, j, k]) *
                                            self.Vzx[i, j, k])

        self.advance_e_dm()

        self.time += self.dt

    def one_step_ect(self):
        for kk in range(self.Nz):
            EMSolver2D.one_step_ect(Nx=self.Nx, Ny=self.Ny, V_enl=self.Vxy_enl[:, :, kk],
                                    rho=self.rho_xy[:, :, kk], Hz=self.Hz[:, :, kk], C1=self.CN,
                                    flag_int_cell=self.grid.flag_int_cell_xy[:, :, kk],
                                    flag_unst_cell=self.grid.flag_unst_cell_xy[:, :, kk],
                                    S=self.grid.Sxy[:, :, kk],
                                    borrowing=self.grid.borrowing_xy[:, :, kk],
                                    S_enl=self.grid.Sxy_enl[:, :, kk],
                                    lending=self.grid.lending_xy[:, :, kk],
                                    S_red=self.grid.Sxy_red[:, :, kk])

        for ii in range(self.Nx):
            EMSolver2D.one_step_ect(Nx=self.Ny, Ny=self.Nz, V_enl=self.Vyz_enl[ii, :, :],
                                    rho=self.rho_yz[ii, :, :], Hz=self.Hx[ii, :, :], C1=self.CN,
                                    flag_int_cell=self.grid.flag_int_cell_yz[ii, :, :],
                                    flag_unst_cell=self.grid.flag_unst_cell_yz[ii, :, :],
                                    S=self.grid.Syz[ii, :, :],
                                    borrowing=self.grid.borrowing_yz[ii, :, :],
                                    S_enl=self.grid.Syz_enl[ii, :, :],
                                    lending=self.grid.lending_yz[ii, :, :],
                                    S_red=self.grid.Syz_red[ii, :, :])

        for jj in range(self.Ny):
            EMSolver2D.one_step_ect(Nx=self.Nx, Ny=self.Nz, V_enl=self.Vzx_enl[:, jj, :],
                                    rho=self.rho_zx[:, jj, :], Hz=self.Hy[:, jj, :], C1=self.CN,
                                    flag_int_cell=self.grid.flag_int_cell_zx[:, jj, :],
                                    flag_unst_cell=self.grid.flag_unst_cell_zx[:, jj, :],
                                    S=self.grid.Szx[:, jj, :],
                                    borrowing=self.grid.borrowing_zx[:, jj, :],
                                    S_enl=self.grid.Szx_enl[:, jj, :],
                                    lending=self.grid.lending_zx[:, jj, :],
                                    S_red=self.grid.Szx_red[:, jj, :])

        
    def compute_v_and_rho(self):
        l_x = self.grid.l_x
        l_y = self.grid.l_y
        l_z = self.grid.l_z

        for ii in range(self.Nx):
            for jj in range(self.Ny):
                for kk in range(self.Nz):
                    if self.grid.flag_int_cell_xy[ii, jj, kk]:
                        self.Vxy[ii, jj, kk] = (self.Ex[ii, jj, kk] * l_x[ii, jj, kk] -
                                                self.Ex[ii, jj + 1, kk] * l_x[ii, jj + 1, kk] +
                                                self.Ey[ii + 1, jj, kk] * l_y[ii + 1, jj, kk] -
                                                self.Ey[ii, jj, kk] * l_y[ii, jj, kk])

                        if self.sol_type != 'DM':
                            self.rho_xy[ii, jj, kk] = (self.Vxy[ii, jj, kk] /
                                                       self.grid.Sxy[ii, jj, kk])

                    if self.grid.flag_int_cell_yz[ii, jj, kk]:
                        self.Vyz[ii, jj, kk] = (self.Ey[ii, jj, kk] * l_y[ii, jj, kk] -
                                                self.Ey[ii, jj, kk + 1] * l_y[ii, jj, kk + 1] +
                                                self.Ez[ii, jj + 1, kk] * l_z[ii, jj + 1, kk] -
                                                self.Ez[ii, jj, kk] * l_z[ii, jj, kk])

                        if self.sol_type != 'DM':
                            self.rho_yz[ii, jj, kk] = (self.Vyz[ii, jj, kk] /
                                                       self.grid.Syz[ii, jj, kk])

                    if self.grid.flag_int_cell_zx[ii, jj, kk]:
                        self.Vzx[ii, jj, kk] = (self.Ez[ii, jj, kk] * l_z[ii, jj, kk] -
                                                self.Ez[ii + 1, jj, kk] * l_z[ii + 1, jj, kk] +
                                                self.Ex[ii, jj, kk + 1] * l_x[ii, jj, kk + 1] -
                                                self.Ex[ii, jj, kk] * l_x[ii, jj, kk])

                        if self.sol_type != 'DM':
                            self.rho_zx[ii, jj, kk] = (self.Vzx[ii, jj, kk] /
                                                       self.grid.Szx[ii, jj, kk])

    def advance_e_dm(self):
        Ex = self.Ex
        Ey = self.Ey
        Ez = self.Ez
        Hx = self.Hx
        Hy = self.Hy
        Hz = self.Hz
        for ii in range(self.Nx):
            x = self.grid.xmin + ii * self.grid.dx
            for jj in range(self.Ny):
                y = self.grid.ymin + jj * self.grid.dy
                for kk in range(self.Nz):
                    z = self.grid.zmin + kk * self.grid.dz
                    if self.grid.l_x[ii, jj, kk] > 0:
                        Ex[ii, jj, kk] = (Ex[ii, jj, kk] - self.C3 * self.Jx[ii, jj, kk] +
                                          self.C4 * (Hz[ii, jj, kk] - Hz[ii, jj - 1, kk]) -
                                          self.C8 * (Hy[ii, jj, kk] - Hy[ii, jj, kk - 1]))

                    if self.grid.l_y[ii, jj, kk] > 0:
                        Ey[ii, jj, kk] = (Ey[ii, jj, kk] - self.C3 * self.Jy[ii, jj, kk] +
                                          self.C8 * (Hx[ii, jj, kk] - Hx[ii, jj, kk - 1]) -
                                          self.C5 * (Hz[ii, jj, kk] - Hz[ii - 1, jj, kk]))

                    if self.grid.l_z[ii, jj, kk] > 0:
                        Ez[ii, jj, kk] = (Ez[ii, jj, kk] - self.C3 * self.Jz[ii, jj, kk] +
                                          self.C5 * (Hy[ii, jj, kk] - Hy[ii - 1, jj, kk]) -
                                          self.C4 * (Hx[ii, jj, kk] - Hx[ii, jj - 1, kk]))