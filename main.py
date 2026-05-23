
from init_solvent import SOLVENT_main
import hoomd
import sys
import importlib.metadata
import matplotlib.pyplot as plt
from sim import render


try:
    import freud
    import gsd
    import numpy
    
    print(f"freud version: {freud.__version__}")
    gsd_version = importlib.metadata.version("gsd")
    print(f"gsd version: {gsd_version}")
    print(f"gsd version: {numpy.__version__}")
    
    
except ModuleNotFoundError as e:
    # e.name contains the exact name of the missing module (e.g., 'freud' or 'gsd')
    print(f"Error: Required module '{e.name}' is not installed in this environment.")
    print("Please make sure you have added it via your package manager (e.g., 'pixi add').")
    
    # Gracefully exit the script with an error code
    raise SystemExit(1)


if __name__ == "__main__":
    sim2 = SOLVENT_main()
    
    # --- Configure continuous data logging to GSD trajectory file ---
    gsd_writer = hoomd.write.GSD(
        trigger=2000,
        filename="sim_finish.gsd",
        dynamic=["particles/position", "particles/image"],
    )
    sim2.operations.writers.append(gsd_writer)
    sim2.run(200000)
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

    # --- Read back saved trajectory and unwrap coordinates for MSD calculation ---
    with gsd.hoomd.open("./Frames/sim_finish.gsd") as traj:
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
