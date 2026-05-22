# script1.py
import time
import gsd.hoomd
import os
import freud

def LOAD():
    trajectory_path = 'sim_finish.gsd'
    
    # Check physical file traits first
    print(f"File physical path check: {os.path.abspath(trajectory_path)}") ##if block exit here
    print(f"Does it exist? {os.path.exists(trajectory_path)}") ##if block here to exit
    if os.path.exists(trajectory_path):
        print(f"File Size: {os.path.getsize(trajectory_path)} bytes")

    try:
        with gsd.hoomd.open(name=trajectory_path, mode='r') as traj:
           ##checks to ensure proper loading
            print(f"Success opening file! Total frames found: {len(traj)}")
            
            if len(traj) == 0:
                print("Error: Trajectory is empty.")
                return 400
                
            frame = traj[-1]
            print("--- FRAME DATA ---")
            
            
            # Using getattr to safely check properties if it's an old schema version
            print("Particles count:", len(frame.particles.position))
            ##checks to ensure proper loading

            ##load sim data
            box = freud.box.Box.from_box(frame.configuration.box)
            positions = frame.particles.position
            print(f"Loaded {len(positions)} particles from the GSD file.")
            print(f"Simulation box dimensions: {box}")
            ##load sim data

            ##compute clustering
            cl = freud.cluster.Cluster()
            cl.compute(system=(box, positions), neighbors={'r_max': 1.2})
    
            print(f"Found {cl.num_clusters} clusters in this frame.")
            ##compute clustering
            return 200

    except Exception as e:
        # error type
        print(f"\n!!! CRITICAL ERROR DURING GSD LOAD !!!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        return 400
    
   

if __name__ == "__main__":
    if LOAD() == 400:
        print("Error Mod1")
    else:
        print("success Mod1")
