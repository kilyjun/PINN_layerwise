### Script to visualize multilayer samplers ###


import matplotlib.pyplot as plt
import jax.random as random
import jax.numpy as jnp

# Import all multi-layer sampler classes
from A3DPINN.samplers import (
    SeqBaseSampler,
    MultiLayerCollocationSampler3D,
    MultiLayerBedTemperatureSampler3D,
    MultiLayerTopBoundarySampler3D,
    MultiLayerX0Sampler3D,
    MultiLayerXMaxSampler3D,
    MultiLayerY0Sampler3D,
    MultiLayerYMaxSampler3D,
    MultiLayerInitialConditionSampler3D,
    MultiLayerPrevInitialConditionSampler3D
)

# ---------------------------
# 1) Define problem parameters
# ---------------------------
batch_size = 512
t_L = 1.0  # new layer every 1 second
layer_height = 0.5
x_max = 2.0
y_max = 1.0

time_steps = [0.5, 1.5, 2.5]  # columns

# -----------------------------------------
# 2) Instantiate each multi-layer sampler
# -----------------------------------------
def create_sampler(sampler_class, seed):
    return sampler_class(
        batch_size=batch_size,
        t_L=t_L,
        layer_height=layer_height,
        x_max=x_max,
        y_max=y_max
    )

colloc_sampler = create_sampler(MultiLayerCollocationSampler3D, 0)
bed_sampler = create_sampler(MultiLayerBedTemperatureSampler3D, 1)
top_sampler = create_sampler(MultiLayerTopBoundarySampler3D, 2)
x0_sampler = create_sampler(MultiLayerX0Sampler3D, 3)
xmax_sampler = create_sampler(MultiLayerXMaxSampler3D, 4)
y0_sampler = create_sampler(MultiLayerY0Sampler3D, 5)
ymax_sampler = create_sampler(MultiLayerYMaxSampler3D, 6)
ic_new_sampler = create_sampler(MultiLayerInitialConditionSampler3D, 7)
ic_prev_sampler = create_sampler(MultiLayerPrevInitialConditionSampler3D, 8)

# ----------------------------------------------------------------
# 3) Define sampler "groups" for each row in the final figure
# ----------------------------------------------------------------
row1 = [(colloc_sampler, "Collocation (Interior)", 'red')]
row2 = [
    (bed_sampler, "Bed (z=0) BC", 'blue'),
    (top_sampler, "Top z=(i+1)*layer_height BC", 'green'),
]
row3 = [
    (x0_sampler, "x=0 BC", 'magenta'),
    (xmax_sampler, "x=x_max BC", 'cyan'),
    (y0_sampler, "y=0 BC", 'orange'),
    (ymax_sampler, "y=y_max BC", 'purple'),
]
row4 = [
    (ic_new_sampler, "IC (new)", 'brown'),
    (ic_prev_sampler, "IC (previous)", 'gray'),
]

sampler_rows = [row1, row2, row3, row4]

# ----------------------------------------------------
# 4) Create the figure with 4 rows x len(time_steps) columns
# ----------------------------------------------------
nrows = len(sampler_rows)
ncols = len(time_steps)
fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    subplot_kw={'projection': '3d'},
    figsize=(5 * ncols, 4 * nrows)
)

if nrows == 1:
    axes = axes[None, :]
if ncols == 1:
    axes = axes[:, None]

# ----------------------------------------------------------------
# 5) Fill each subplot: row => group of samplers, col => time step
# ----------------------------------------------------------------
for row_idx, group in enumerate(sampler_rows):
    for col_idx, t_val in enumerate(time_steps):
        ax = axes[row_idx, col_idx]

        for sampler, name, color in group:
            key = random.PRNGKey(row_idx * len(time_steps) + col_idx)
            subkeys = random.split(key, 3)  # Adjust this based on the sampler type
            batch = sampler.data_generation(subkeys, t_val)
            
            x, y, z = batch[:, 1], batch[:, 2], batch[:, 3]
            ax.scatter(x, y, z, alpha=0.7, s=10, c=color, label=name)

        ax.set_title(f"t = {t_val}", fontsize=10)
        ax.set_xlim([0, x_max])
        ax.set_ylim([0, y_max])
        ax.set_zlim([0, 4 * layer_height])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        if len(group) > 1:
            ax.legend(loc='upper right', fontsize=7)

fig.suptitle("Multi-Layer Samplers", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

