import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.random as random
from A3DPINN.samplers import SeqCollocationSampler3D

rng_key = random.PRNGKey(1234)
batch_size = 512
init_length = jnp.array([0.01])
velocity = jnp.array([1.0, 0.0, 0.0])
bead_width = 1.0
bead_height = 0.5

# Instantiate the collocation sampler
seq_collocation_sampler_3d = SeqCollocationSampler3D(batch_size, init_length, velocity, bead_width, bead_height, rng_key)

time_steps = jnp.linspace(0.0, 2.5, 50)  # Time steps from 0 to 2.5, 50 points

densities = []
volumes = []

for time in time_steps:
    collocation_batch = seq_collocation_sampler_3d.data_generation(rng_key, time)
    
    # Calculate the volume of the domain
    x_length = velocity[0] * time + init_length[0]  # Evolving x-dimension
    volume = x_length * bead_width * bead_height  # x * y * z
    
    # Calculate the density of points (points per unit volume)
    density = batch_size / volume  # Number of points / volume
    
    # Store the density and volume for plotting
    densities.append(density)
    volumes.append(volume)

# Collocation Point Density vs Time
plt.figure(figsize=(8, 6))
plt.plot(time_steps, densities, label='Collocation Point Density')
plt.xlabel('Time')
plt.ylabel('Density (points per unit volume)')
plt.title('Collocation Point Density vs Time')
plt.grid(True)
plt.legend()
plt.show()

# Plot Volume vs Time
plt.figure(figsize=(8, 6))
plt.plot(time_steps, volumes, label='Volume', color='orange')
plt.xlabel('Time')
plt.ylabel('Volume')
plt.title('Volume of Domain vs Time')
plt.grid(True)
plt.legend()
plt.show()
