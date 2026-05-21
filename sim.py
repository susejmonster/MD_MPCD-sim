import itertools
import fresnel
import math
import hoomd
import numpy 

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
    geometry.outline_width = 0.05

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