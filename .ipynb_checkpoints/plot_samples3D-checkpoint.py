### for visualizing single layer samplers ###


import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.random as random
from A3DPINN.samplers import (
    SeqCollocationSampler3D,
    BedTemperatureSampler3D,
    DepositionFrontSampler3D,
    NeumannBoundarySamplerX0,
    ConvectionBoundarySamplerYMax,
    ConvectionBoundarySamplerYMin,
    ConvectionBoundarySamplerZMax,
)

# Set the random seed for reproducibility
rng_key = random.PRNGKey(1234)

# Define example parameters for the 3D samplers in physical units
batch_size = 512
x_init = 0.0               
velocity = jnp.array([1.0, 0.0, 0.0])
bead_width = 1.0 
bead_height = 1.0 

# Instantiate the samplers with physical units
seq_collocation_sampler_3d = SeqCollocationSampler3D(
    batch_size, jnp.array([x_init]), velocity, bead_width, bead_height)
bed_temperature_sampler_3d = BedTemperatureSampler3D(
    batch_size, jnp.array([x_init]), velocity, bead_width)
deposition_front_sampler_3d = DepositionFrontSampler3D(
    batch_size, jnp.array([x_init]), velocity, bead_width, bead_height)
neumann_boundary_sampler_x0 = NeumannBoundarySamplerX0(
    batch_size, bead_width, bead_height)
convection_boundary_sampler_ymax = ConvectionBoundarySamplerYMax(
    batch_size, jnp.array([x_init]), velocity, bead_width, bead_height)
convection_boundary_sampler_ymin = ConvectionBoundarySamplerYMin(
    batch_size, jnp.array([x_init]), velocity, bead_width, bead_height)
convection_boundary_sampler_zmax = ConvectionBoundarySamplerZMax(
    batch_size, jnp.array([x_init]), velocity, bead_width, bead_height)

sampling_time_steps = [0.0, 1.0, 2.5]  # Original three time steps for 3D visualization

time_steps = jnp.linspace(0.1, 2.5, 50)  # 50 time steps from 0.1 to 2.5

# Define the sampler groups for each row
sampler_groups = [
    # Row 1: Convection boundary points (y_max, y_min, z_max)
    [
        ('Convection BC y=y_max', convection_boundary_sampler_ymax, 'orange', 'v'),
        ('Convection BC y=0', convection_boundary_sampler_ymin, 'brown', '<'),
        ('Convection BC z=z_max', convection_boundary_sampler_zmax, 'pink', '>'),
    ],
    # Row 2: Neumann boundary and deposition front points
    [
        ('Neumann BC x=0', neumann_boundary_sampler_x0, 'purple', 'D'),
        ('Deposition Front', deposition_front_sampler_3d, 'red', 's'),
    ],
    # Row 3: Bed temperature and general collocation points
    [
        ('Bed Temperature', bed_temperature_sampler_3d, 'blue', '^'),
        ('Collocation Points', seq_collocation_sampler_3d, 'green', 'o'),
    ]
]

# Function to visualize 3D samples
def plot_3d_samples(ax, x, y, z, label, color, marker):
    ax.scatter(x, y, z, s=10, color=color, alpha=0.6, label=label, marker=marker)

# Function to compute the volume of the domain
def compute_volume(t, x_init, bead_width, bead_height, velocity):
    return (x_init + velocity[0] * t) * bead_width * bead_height

# Function to compute the density of collocation points
def compute_density(num_points, volume):
    return num_points / volume

# Create lists to store volume and density values for each time step
volumes = []
densities = []
collocation_points = [batch_size] * len(time_steps)

# Compute volume and density at each time step
for time in time_steps:
    # Geometry evolves in the x-direction as time proceeds
    volume = compute_volume(time, x_init, bead_width, bead_height, velocity)
    density = compute_density(batch_size, volume)
    
    volumes.append(volume)
    densities.append(density)

# Create a 1x3 grid of plots for collocation points, volume, and density vs. time
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Collocation points vs. time (constant)
axes[0].plot(time_steps, collocation_points, '-o', label='Collocation Points', color='g')
axes[0].set_xlabel('Time', fontsize=12)
axes[0].set_ylabel('Collocation Points', fontsize=12)
axes[0].set_title('Collocation Points vs Time', fontsize=14)
axes[0].grid(True)

# Plot 2: Volume vs. time
axes[1].plot(time_steps, volumes, '-o', label='Volume', color='r')
axes[1].set_xlabel('Time', fontsize=12)
axes[1].set_ylabel('Volume', fontsize=12)
axes[1].set_title('Volume vs Time', fontsize=14)
axes[1].grid(True)

# Plot 3: Collocation point density vs. time
axes[2].plot(time_steps, densities, '-o', label='Collocation Point Density', color='b')
axes[2].set_xlabel('Time', fontsize=12)
axes[2].set_ylabel('Collocation Point Density', fontsize=12)
axes[2].set_title('Collocation Point Density vs Time', fontsize=14)
axes[2].grid(True)

# Adjust layout for better spacing and visibility
plt.tight_layout()
plt.show()

# Create a figure with subplots for each combination of row (collocation group) and column (original time steps)
fig = plt.figure(figsize=(18, 18))  # Adjusted figure size for clarity in 3x3 grid
for col_idx, time in enumerate(sampling_time_steps):  # Use original time steps for 3D visualization
    # Generate a new random key for each time step
    rng_key, subkey = random.split(rng_key)
    subkeys = random.split(subkey, sum(len(group) for group in sampler_groups))

    key_index = 0  # Initialize subkey index
    for row_idx, group in enumerate(sampler_groups):
        # Create a 3D subplot for each combination of row and column
        ax = fig.add_subplot(3, 3, row_idx * 3 + col_idx + 1, projection='3d')
        ax.set_xlim([0, x_init + velocity[0] * max(sampling_time_steps)])  # Updated x limit
        ax.set_ylim([0, bead_width])
        ax.set_zlim([0, bead_height])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        # if col_idx == 0:
        #     ax.set_title(f"Sampling Points: {group[0][0].split()[0]} Group", fontsize=10)
        if row_idx == 0:
            ax.set_title(f"t = {time}", fontsize=12)

        for name, sampler, color, marker in group:
            batch = sampler.data_generation(subkeys[key_index], time)
            key_index += 1  # Increment the subkey index
            x_batch = batch[:, 1]
            y_batch = batch[:, 2]
            z_batch = batch[:, 3]
            plot_3d_samples(ax, x_batch, y_batch, z_batch, name, color, marker)

        if col_idx == 2:
            ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.show()
