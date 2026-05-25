
from init_solvent import SOLVENT_main
import hoomd
import sys
import importlib.metadata
import matplotlib.pyplot as plt
from sim import render,render_gsd_animation
from clustering import ClusterValidator
import fresnel

try:
    import freud
    import gsd
    import numpy
    
    print(f"freud version: {freud.__version__}")
    gsd_version = importlib.metadata.version("gsd")
    print(f"gsd version: {gsd_version}")
    print(f"gsd version: {numpy.__version__}")
    
    
except ModuleNotFoundError as e:
    
    print(f"Error: Required module '{e.name}' is not installed in this environment.")
    print("Please make sure you have added it via your package manager (e.g., 'pixi add').")
    
    
    raise SystemExit(1)


if __name__ == "__main__":
    sim2 = SOLVENT_main()
    T_str = input("Enter total production timesteps: ")
    T = int(T_str) 


    logger = hoomd.logging.Logger(categories=['scalar'])
    logger.add(sim2, quantities=['tps', 'timestep'])
    table = hoomd.write.Table(trigger=200, logger=logger)
    sim2.operations.writers.append(table)
    # ---------------------------------------------------------
    # PHASE 1: Randomization (Thermalization)
    # ---------------------------------------------------------
    print("Randomizing velocities...")
    target_kT = input("Enter target Kt for sim temperature: ")
    target_kT = float(target_kT)
    sim2.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=target_kT)
    sim2.operations.writers.remove(table)
    # ---------------------------------------------------------
    # PHASE 4: Production & Logging (Your original code)
    # ---------------------------------------------------------
    # --- Configure continuous data logging to GSD trajectory file ---
    gsd_writer = hoomd.write.GSD(
        trigger=1000,
        filename="sim_finish.gsd",
        dynamic=["particles/position", "particles/image", "particles/velocity"],
    )
    sim2.operations.writers.append(gsd_writer)
    
    validator_action = ClusterValidator()
    validator_updater = hoomd.update.CustomUpdater(
        action=validator_action, 
        trigger=1001,
    )
    sim2.operations.updaters.append(validator_updater)
    sim2.run(T)
    # --- Configure continuous data logging to GSD trajectory file ---

    # --- Generate and save a static 3D visualization image of the final frame ---
    print("simulation complete,opening writer")
    snapshot = sim2.state.get_snapshot()
    positions = snapshot.particles.position
    orientations = snapshot.particles.orientation
    box_length = snapshot.configuration.box
    image_fin = render(positions,orientations,box_length)
    image_fin.save("./Renders/final_image.png")

    print("simulation complete,deleting writer")
    sim2.operations.writers.remove(gsd_writer)
    del gsd_writer
    # --- Generate and save a static 3D visualization image of the final frame ---
    
    # --- Generate and save a static 3D visualization video  ---
    render_perm = False
    if render_perm:
        render_gsd_animation("sim_finish.gsd", "my_animation.gif")
    
    # --- Generate and save a static 3D visualization video  ---
    
    # --- Read back saved trajectory and unwrap coordinates for MSD calculation ---
    with gsd.hoomd.open("sim_finish.gsd") as traj:
        num_frames = len(traj)
        N = traj[0].particles.N
        positions = numpy.zeros((num_frames, N, 3), dtype=float)
        for i, snap in enumerate(traj):
            box = freud.box.Box.from_box(snap.configuration.box)
            positions[i] = box.unwrap(snap.particles.position, snap.particles.image)
    print("WRITE complete, plotting")
    # --- Read back saved trajectory and unwrap coordinates for MSD calculation --

    # --- Compute Mean Squared Displacement (MSD) and save plot ---
    msd = freud.msd.MSD(box, mode="window")
    msd.compute(positions)

    print("Plotting")
    ax = msd.plot()
    plt.savefig("./Renders/msd_plot.png", dpi=300, bbox_inches='tight')
    print("MSD Plot successfully saved to ./Renders/msd_plot.png")
