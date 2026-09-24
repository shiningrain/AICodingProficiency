# Explain: The LLM (4o mini) is asked to use Three.js (and its python adapter), but does not use in the generation results.

# Importing the required libraries from Three.js and matplotlib for 3D rendering
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors as mcolors

# Function to create a sphere
def create_sphere(radius, resolution):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z

# Parameters
radius = 2
resolution = 32
wireframe_color = (0, 0, 0, 0.5)  # Black with 50% transparency
center = (0, 0, 0)

# Create sphere
x, y, z = create_sphere(radius, resolution)

# Setting up the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface of the sphere
ax.plot_surface(x + center[0], y + center[1], z + center[2], color='cyan', alpha=0.7)

# Plot the wireframe
ax.plot_wireframe(x + center[0], y + center[1], z + center[2], color=wireframe_color[:3], alpha=wireframe_color[3])

# Setting the limits and labels
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('3D Sphere with Wireframe Overlay')

# Show the plot
plt.show()