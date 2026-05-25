import itertools
import fresnel
import math
import hoomd
import numpy 
import gsd.hoomd
from PIL import Image

def render(positions, orientations, box_length):
    """
    Renders a 3D scene of particles inside a box using Fresnel.
    """
    # 1. Initialize a Fresnel scene
    scene = fresnel.Scene()

    # 2. Define particle geometry
    geometry = fresnel.geometry.Sphere(scene, N=len(positions))
    
    # 3. Assign positions and radii
    geometry.position[:] = positions
    geometry.radius[:] = [0.4] * len(positions)  # Slightly smaller radius so they don't overlap heavily
    
    # 4. Apply material/color formatting
    geometry.material = fresnel.material.Material(
        color=fresnel.color.linear([0.2, 0.5, 0.8]),
        roughness=0.3,
        specular=0.5
    )
    
    # 5. Add a subtle wireframe outline around the particles for depth
    geometry.outline_width = 0.0

    # 6. Setup Scene Camera automatically based on the geometry
    scene.camera = fresnel.camera.Orthographic.fit(scene)
    
    # 7. Apply standard directional lights striking the front face
    scene.lights = [
        fresnel.light.Light(direction=[1, 1, 1], color=[0.9, 0.9, 0.9], theta=0),
        fresnel.light.Light(direction=[-1, -1, 1], color=[0.3, 0.3, 0.3], theta=0)
    ]

    # 8. Render the scene (Switching to preview mode avoids the black screen issue)
    
    image = fresnel.preview(scene, w=600, h=600)
    
    # 9. Save or display the image using Pillow
    from PIL import Image
    # Extract RGB channels (ignoring alpha channel)
    img = Image.fromarray(image[:, :, 0:3], mode='RGB')
    return img

def render_gsd_animation(gsd_file, output_gif):
    """
    Reads a GSD trajectory and renders a 3D animated GIF using Fresnel.
    """
    # 1. Open the GSD trajectory
    traj = gsd.hoomd.open(gsd_file, "r")
    num_particles = traj[0].particles.N

    # 2. Initialize the Fresnel scene
    scene = fresnel.Scene()

    # 3. Define particle geometry (Set up ONCE for the whole animation)
    geometry = fresnel.geometry.Sphere(scene, N=num_particles)
    geometry.radius[:] = [0.4] * num_particles
    
    # 4. Apply material/color formatting
    geometry.material = fresnel.material.Material(
        color=fresnel.color.linear([0.2, 0.5, 0.8]),
        roughness=0.3,
        specular=0.5
    )
    geometry.outline_width = 0.0
    
    # 5. Apply standard directional lights
    scene.lights = [
        fresnel.light.Light(direction=[1, 1, 1], color=[0.9, 0.9, 0.9], theta=0),
        fresnel.light.Light(direction=[-1, -1, 1], color=[0.3, 0.3, 0.3], theta=0)
    ]

    # Create a list to store the rendered frames
    image_frames = []

    # 6. Loop through each frame in the trajectory
    print(f"Rendering {len(traj)} frames...")
    for i, frame in enumerate(traj):
        
        # Update the particle positions for the current frame
        geometry.position[:] = frame.particles.position
        
        # Fit the camera on the first frame so the zoom level stays consistent
        if i == 0:
            scene.camera = fresnel.camera.Orthographic.fit(scene)
        
        # Render the scene to a numpy array
        image_array = fresnel.preview(scene, w=600, h=600)
        
        # Extract RGB channels and convert to Pillow Image
        img = Image.fromarray(image_array[:, :, 0:3], mode='RGB')
        image_frames.append(img)

    # 7. Compile and save the animation as a GIF
    if image_frames:
        image_frames[0].save(
            output_gif,
            save_all=True,
            append_images=image_frames[1:],
            duration=100,  # Duration of each frame in milliseconds
            loop=0         # 0 means the GIF will loop infinitely
        )
        print(f"Animation successfully saved to {output_gif}")


m = 4
N_particles = 2 * m**3
spacing = 1.2
K = math.ceil(N_particles ** (1 / 3))
L = K * spacing

x = numpy.linspace(-L / 2, L / 2, K, endpoint=False)
position = list(itertools.product(x, repeat=3))
print(position[0:4])

position = position[0:N_particles]
orientation = [(1, 0, 0, 0)] * N_particles
render(position, orientation, L)