import itertools
import fresnel
import math
import hoomd
import numpy 
import gsd.hoomd
from sim import render
from init import SOLUTE_main


def IC_SOLVENT():
    with gsd.hoomd.open("./Frames/Active_IC.gsd") as traj:
        snapshot = hoomd.Snapshot.from_gsd_frame(traj[0], hoomd.communicator.Communicator())
    
    box = hoomd.Box.from_box(snapshot.configuration.box)
    rng = numpy.random.default_rng(seed=42)
    sigma = 1
    kT =0.6

    assert box.Lx == box.Ly == box.Lz, "The simulation box is not cubic!"
    L = box.Lx  # Single side length of the cube

    snapshot.mpcd.types = ["A"]
    snapshot.mpcd.N = numpy.round((10/sigma**3) * box.volume).astype(int)

# Generate uniform positions inside a cube centered at (0,0,0) from -L/2 to L/2
    snapshot.mpcd.position[:] = rng.uniform(
        low=-L / 2, high=L / 2, size=(snapshot.mpcd.N, 3)
    )

# Velocities remain the same (Maxwell-Boltzmann distribution)
    vel = rng.normal(loc=0.0, scale=numpy.sqrt(kT), size=(snapshot.mpcd.N, 3))
    vel -= numpy.mean(vel, axis=0)
    snapshot.mpcd.velocity[:] = vel
    
    return snapshot,snapshot.particles.position,snapshot.particles.orientation,snapshot.mpcd.N,L


def IO_SOLVENT(pos,orient,L)->None:
    # 1. Get the actual number of solute particles from the pos array
    N_solute = len(pos)
    
    # 2. Build the frame using N_solute
    frame = gsd.hoomd.Frame()
    frame.particles.N = N_solute
    frame.particles.position = pos
    frame.particles.orientation = orient
    
    
    frame.particles.typeid = [0] * N_solute
    frame.particles.types = ["A"]
    frame.configuration.box = [L, L, L, 0, 0, 0]
    
   
    with gsd.hoomd.open(name="./Frames/Solute_IC.gsd", mode="w") as f:
        f.append(frame)
    
    print("Adding Solute particles...")
    image = render(pos, orient, L)


def SOLVENT_main():
    snap,pos,orient,N,L = IC_SOLVENT()
    IO_SOLVENT(pos,orient,L)
    
    
    SOLUTE_main()
    cpu = hoomd.device.CPU()
    simulation = hoomd.Simulation(device=cpu, seed=42)

    try:
        simulation.create_state_from_snapshot(snap)
        print("Simulation state for Solvent successfully initialized from GSD file.")
        
        integrator = hoomd.mpcd.Integrator(dt=0.005)
        simulation.operations.integrator = integrator

        cell = hoomd.md.nlist.Cell(buffer=0.4)
        wca = hoomd.md.pair.LJ(nlist=cell)
        wca.params[("A", "A")] = dict(epsilon=1.0, sigma=1.0)
        wca.r_cut[("A", "A")] = 2 ** (1.0 / 6.0)
        integrator.forces.append(wca)

        nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
        integrator.methods.append(nve)

        integrator.collision_method = hoomd.mpcd.collide.StochasticRotationDynamics(
            period=20, angle=130, kT=1.0, embedded_particles=hoomd.filter.All()
        )
        integrator.mpcd_particle_sorter = hoomd.mpcd.tune.ParticleSorter(
            trigger=integrator.collision_method.period * 20
        )
        integrator.streaming_method = hoomd.mpcd.stream.Bulk(
            period=integrator.collision_method.period
        )

        print("Solvent integrator setup complete")
        return simulation
    except FileNotFoundError:

        print("Error: The file './Frames/Active_IC.gsd' was not found.")
        print("Please make sure your initial condition generation script ran successfully__Solute_IC.gsd.")
    
    
    raise SystemExit(1)

    