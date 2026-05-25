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
    
    rho = 0.05 
    L = 64 
    N_particles = int(rho * (L**3)) 
    K = math.ceil(N_particles ** (1 / 3)) 
    spacing = L / K 
    x = numpy.linspace(-L / 2, L / 2, K, endpoint=False)
    position = list(itertools.product(x, repeat=3))
    

    position = position[0:N_particles]
    
    # --- Randomize ---
    random_quats = numpy.random.randn(N_particles, 4)
    random_quats /= numpy.linalg.norm(random_quats, axis=1)[:, numpy.newaxis]
    orientation = [tuple(q) for q in random_quats]

    return position, orientation, N_particles, L

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
       
        pos, orient, N, L = IC()
        IO(pos, orient, N, L)
        print("Solute initialization completed successfully!")

    except Exception as e:
        print(f"\n[Error] SOLUTE_main failed during execution!")
        print(f"Reason: {e}")
        print("-" * 50)
        
        traceback.print_exc()
        print("-" * 50)
        
        
        sys.exit(1)

