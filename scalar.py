import numpy as np
np.seterr(over="raise")
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import h5py
from mpi4py import MPI
#import os
import sys
from scipy.special import gamma
from scipy.special import jv

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Input parameters
N = int(sys.argv[1])
sim_time = int(sys.argv[2])
eps = float(sys.argv[3])
kappa = float(sys.argv[4])
NR = float(sys.argv[5])
start_from_snapshot = int(sys.argv[6]) if len(sys.argv) > 7 else 0
simu_id = sys.argv[7] if len(sys.argv) > 7 else 0

if sim_time == 0:
    n_steps = 10000
elif sim_time == 1:
    n_steps = 200000
else:
    n_steps = 500000

# General parameters
mesh = None
dtype = np.float64
dealias = 3 / 2
seed = None

# Properties of domain
dim = 2
L = 2 * np.pi
dx = L / N
dVol = dx**dim
Vol = L**dim

#non-reciprocal cross-diffusion
NR_kappa = NR * kappa

# Parameters of forcing statistics
D_f_0 = 1 #amplitude of forcing correlation
LL_int = np.pi #scale of forcing
kk_int = 2 * np.pi / LL_int #inverse of scale of forcing
#LL_int = 2 * np.pi / kk_int

# Parametres of velocity statistics
D_u_0 = 4 * np.pi * eps #amplitude of veloicity correlation chosen in such a way that for eps -> 0 then D1 -> 1

kk_mu = 0 #2 * np.pi / L / 2 #beginning of inertial range
kk_eta = 2 * np.pi / L * 16 * N #end of inertial range

D1 = D_u_0 * gamma(1-eps/2) * (dim - 1) / 2**(dim-1+eps) / np.pi**(dim/2) / eps / (dim + eps) / gamma((dim + eps)/2)

# Definition of dedalus variables
coords = d3.CartesianCoordinates("x","y")
dist = d3.Distributor(coords, mesh=mesh, dtype=dtype)
xbasis = d3.RealFourier(coords[0], N, bounds=(0, L), dealias=dealias)
ybasis = d3.RealFourier(coords[1], N, bounds=(0, L), dealias=dealias)

ex, ey = coords.unit_vector_fields(dist)

# Fields
ThetaA = dist.Field(name="ThetaA", bases=(xbasis, ybasis))
ThetaB = dist.Field(name="ThetaB", bases=(xbasis, ybasis))

# Velocity fields
rand_u = dist.VectorField(coords, name="u", bases=(xbasis,ybasis)) #dummy variable to generate the random velocity 
noise_u = dist.VectorField(coords, name="u", bases=(xbasis,ybasis)) #noise in the equation for u
p = dist.Field(name="p", bases=(xbasis, ybasis)) #pressure
tau_p = dist.Field(name="tau_p")
u = dist.VectorField(coords, name="u", bases=(xbasis,ybasis))

# Forcing fields
FThetaA = dist.Field(name="FThetaA", bases=(xbasis, ybasis))
FThetaB = dist.Field(name="FThetaB", bases=(xbasis, ybasis))
psi = dist.Field(name="psi", bases=(xbasis, ybasis)) #dummy variable to generate the random forcing

# k basis
kx = xbasis.wavenumbers[dist.local_modes(xbasis)]
ky = ybasis.wavenumbers[dist.local_modes(ybasis)]
dkx = dky = 2 * np.pi / L

# Functions for forcing and velocity correlation spectrum
def forcing_spectrum(k2):
    kk = np.sqrt(k2)
    return (2 * np.pi * LL_int**2)**(dim/2) * np.exp(- k2 * LL_int**2/ 2) * (k2!=0)

def u_spectrum(k2):
    den2 = k2 + kk_mu**2
    P1 = LL_int**(-eps) * np.exp(- k2 / 2 / kk_eta**2) / ((den2 + (den2==0))**((dim + eps)/2)) * (den2!=0)
    return P1

# Functions generating random forcing and velocity fields
def random_forcing(f_amp_forcing):
    psi.fill_random("c", seed=None, distribution='normal', scale=1)
    return np.sqrt(f_amp_forcing) * psi["c"]

def random_u(f_amp_u):
    rand_u.fill_random("c", seed=None, distribution='normal', scale=1)
    return np.sqrt(f_amp_u) * rand_u["c"]

# Pre-computing spectrum amplitudes of forcing and velocity fields
k2 = kx**2 + ky**2
k = np.sqrt(k2)
f_amp_forcing = forcing_spectrum(k2) / 2**((kx == 0).astype(float) + (ky == 0).astype(float) - 2) / Vol
f_amp_u = u_spectrum(k2) / 2**((kx == 0).astype(float) + (ky == 0).astype(float) + (kx == 0).astype(float) * (ky == 0).astype(float) - 2) / Vol

# computing U_rms
u["c"] = random_u(f_amp_u)
u_ = u["g"][0]

n = u_.shape[0]
k_x_1, k_y_1 = np.indices((n, n))
center = (n / 2, n / 2)
k2_grid = (k_x_1 - center[0])**2 + (k_y_1 - center[1])**2

P1 = D_u_0 * (dim - 1) * u_spectrum(k2_grid)
A = np.sqrt(np.sum(P1) / Vol)

P1 = D_f_0 * forcing_spectrum(k2_grid)
C0 = np.sum(P1 / Vol)
del P1,k2_grid,k_x_1,k_y_1,center,n

# velocity relaxes on timescale tau_u
# I want:
# u_rms = A / sqrt(2 * tau_u + dt)
# dt = alpha dx / u_rms
# tau_u = beta dt
# then.
# of course, if u_rms = 0, we have to take tau_u = alpha dx^2 / kappa
alpha = 0.1
beta = 10

u_rms = A**2 / dx / alpha / (2 * beta + 1)
if kappa / dx > u_rms and NR < 1:
    delta_t = alpha * dx**2 / kappa
elif NR_kappa / dx > u_rms and NR > 1:
    delta_t = alpha * dx**2 / NR_kappa
else:
    delta_t = alpha * dx / u_rms
tau_u = beta * delta_t
u_rms = A / np.sqrt(2*tau_u+delta_t)

n_snapshots = 100
n_scalar = 1000
n_show = 1

stop_sim_time = n_steps * delta_t
snapshots_dt = stop_sim_time / n_snapshots
scalars_dt = stop_sim_time / n_scalar

if rank == 0:
    print(f"")
    print(f"Simulation will run for for {n_steps} iterations, with a time-step of delta_t = {delta_t}")
    print(f"and a total simulation time of {stop_sim_time:.2f}")
    print(f"")
    print(f"Energy injection scale = {LL_int}")
    print(f"Variance of energy injection xi(r=0) = {C0}")
    print(f"")
    print(f"Velocity field relaxes on timescales of tau_u = {tau_u}")
    print(f"Velocity decays with exponent = {eps}")
    print(f"The same-site variance is given to A = {A}")
    print(f"Leading to the following root mean square velocity u_rms = {u_rms}")
    print(f"")
    print(f"Diffusion and odd diffusion coefficents are:")
    print(f"kappa = {kappa}")
    print(f"kappa_odd = {NR_kappa}")
    print(f"ratio between diffusive timescale and advection timescale = {u_rms * dx / kappa}")
    print(f"An estimate of the diffusive scale is k_diff = {kk_int * (D1 / kappa / 2)**(1/eps)}")
    print(f"")
    print(f"start_from_snapshot = {start_from_snapshot}")
    print(f"Simulation ID = {simu_id}")
    print(f"")


# initial conditions
if start_from_snapshot:
    logger.info(f"Starting from snapshot {simu_id}")
    _path="/project/vitelli/scandolo/TURBO/Continue"
    file_name = f"{_path}/{simu_id}/snapshots/snapshots_s1.h5"

    f = h5py.File(file_name)
    for field in [ThetaA, ThetaB]:
        slices = dist.grid_layout.slices(field.domain,scales=1) # assuming the output is in grid space without dealiasing
        field['g'] = np.array(f['tasks/%s' %field.name][-1][slices]) # loading in the last snapshot

    slices = dist.grid_layout.slices(u.domain,scales=1) # assuming the output is in grid space without dealiasing
    for i in range(2):
        u['g'][i] = np.array(f['tasks/%s' %u.name][-1][i][slices])

# setup problem
problem = d3.IVP([ThetaA,ThetaB,u,p,tau_p], namespace=locals())#ThetaB
problem.add_equation("dt(ThetaA) - kappa * lap(ThetaA) + NR_kappa * lap(ThetaB) = - u@grad(ThetaA) + FThetaA")
problem.add_equation("dt(ThetaB) - kappa * lap(ThetaB) - NR_kappa * lap(ThetaA) = - u@grad(ThetaB) + FThetaB")
problem.add_equation("tau_u * dt(u) + grad(p) + u = noise_u")
problem.add_equation("div(u) + tau_p = 0")
problem.add_equation("integ(p) = 0")

timestepper = d3.RK443

solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

snapshots = solver.evaluator.add_file_handler("snapshots" , sim_dt=snapshots_dt, max_writes=10000, mode="overwrite")
snapshots.add_task(u, name="u")
snapshots.add_task(ThetaA, name="ThetaA")
snapshots.add_task(ThetaB, name="ThetaB")
snapshots.add_task(FThetaA, name="FThetaA")
snapshots.add_task(FThetaB, name="FThetaB")

scalars = solver.evaluator.add_file_handler("scalars", sim_dt=scalars_dt, mode="overwrite")
scalars.add_task(d3.Average(u**2/2), name="u2")
scalars.add_task(d3.Average(ThetaA**2), name="A2")
scalars.add_task(d3.Average(ThetaB**2), name="B2")

params = {"L": L, "N": N, "stop_sim_time": stop_sim_time, "dt_snap": snapshots_dt, "kappa": kappa, "NR": NR, "D_f_0": D_f_0, "k_int": kk_int, "C0": C0, "D_u_0": D_u_0, "eps": eps, "k_mu": kk_mu, "k_eta": kk_eta, "tau_u": tau_u, "u_rms": u_rms, "A": A, "delta_t": delta_t}
np.savez("params.npz", params=params)

#max_dt = 10 * delta_t
#CFL = d3.CFL(solver, initial_dt = 0.1 * max_dt, cadence=1, safety=alpha, max_change=1.5, min_change=0.5, max_dt=max_dt, threshold=0.05)
#CFL.add_velocity(u)

count = n_show

try:
    logger.info("Starting loop")
    while solver.proceed:
        timestep = delta_t #CFL.compute_timestep() #
        

        FThetaA["c"] = np.sqrt(D_f_0 / (timestep)) * random_forcing(f_amp_forcing)
        FThetaB["c"] = np.sqrt(D_f_0 / (timestep)) * random_forcing(f_amp_forcing)
        noise_u["c"] = np.sqrt(D_u_0 / (timestep)) * random_u(f_amp_u)
        
        if np.any(np.isnan(ThetaA["g"])):
            raise Exception("nan")
        if np.any(np.isnan(ThetaB["g"])):
            raise Exception("nan")
        if np.any(np.isnan(u["g"])):
            raise Exception("nan")

        solver.step(timestep)
        if (100 * solver.sim_time /stop_sim_time) > count:
            logger.info(f"Iteration={solver.iteration}, Time={solver.sim_time:.2g}, dt={timestep:.3g} ({100*solver.sim_time/stop_sim_time:.2f}%)")
            count += n_show

except:
    logger.error("Exception raised, triggering end of main loop.")
    raise
finally:
    solver.log_stats()
