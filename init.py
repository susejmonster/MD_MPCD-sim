import itertools
import fresnel
import math
import hoomd
import numpy 
import gsd.hoomd
from sim import render
import sys
import traceback

def IC():
    #setup the particle box
    m = 4
    N_particles = 2 * m**3
    K = math.ceil(N_particles ** (1 / 3))
    #setting particles in sphere->cube in periodic bounding box->cube
    spacing = 1
    rho = 0.578 #BS CAN BE USED HERE
    
    for i in range(0,m):
        if rho == K**3/K**3*spacing**3: #sigma fixed
            break
        spacing+=0.01
    
    L = math.ceil(K * spacing)
    x = numpy.linspace(-L / 2, L / 2, K, endpoint=False)
    position = list(itertools.product(x, repeat=3))
    
    position = position[0:N_particles]
    orientation = [(1, 0, 0, 0)] * N_particles

    return position,orientation,N_particles,L

def IO(pos,orient,N,L)->None:
        #handle file IO
        frame = gsd.hoomd.Frame()
        frame.particles.N = N
        frame.particles.position = pos
        frame.particles.orientation = orient
        
        frame.particles.typeid = [0] * N
        frame.particles.types = ["A"]
        frame.configuration.box = [L, L, L, 0, 0, 0]
        
        with gsd.hoomd.open(name="./Frames/Active_IC.gsd", mode="w") as f:
            f.append(frame)
        print("Rendering scene...")
        image = render(pos, orient, L)
        image.save("./Renders/_Active_initial_condition.png")

def SOLUTE_main():
    try:
        # 1. Generate the initial condition positions, orientations, etc.
        pos, orient, N, L = IC()
        
        # 2. Run your file handling and rendering pipeline
        IO(pos, orient, N, L)
        
        print("Solute initialization completed successfully!")

    except Exception as e:
        print(f"\n[Error] SOLUTE_main failed during execution!")
        print(f"Reason: {e}")
        print("-" * 50)
        # This prints the exact line number and file where the crash happened
        traceback.print_exc()
        print("-" * 50)
        
        # Gracefully exit with a failure code
        sys.exit(1)

