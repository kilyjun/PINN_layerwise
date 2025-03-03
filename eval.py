### single layer implementation ###

    

# import os
# import ml_collections
# from jax import vmap
# import jax.numpy as jnp
# import matplotlib.pyplot as plt
# from A3DPINN.utils import restore_checkpoint
# import models
# import glob
# from PIL import Image
# import contextlib
# import plotly.graph_objs as go
# from plotly.subplots import make_subplots


# def create_gif(folder_path, gif_name, size=(300, 300)):
#     images = []
#     for filename in sorted(os.listdir(folder_path), key=lambda x: int(x.split('_')[1].split('.')[0])):
#         if filename.endswith('.png'):
#             img = Image.open(os.path.join(folder_path, filename))
#             images.append(img)

#     # Save as GIF
#     images[0].save(gif_name, save_all=True, append_images=images[1:], duration=100, loop=0)

# def evaluate(config: ml_collections.ConfigDict, workdir: str):
#     time_array = jnp.linspace(0., config.dimensions.t_max, 100)

#     # Define slices in the y-direction (5 slices across the bead width)
#     y_slices = jnp.linspace(0., config.dimensions.y_max, 5)

#     # Define x and z coordinates as specified
#     x_coords = [0.0, 0.4, 0.8]
#     z_coords = [0.0, 0.000375, 0.00075, 0.001125]

#     # Restore model
#     model = models.A3DHeatTransfer(config)
#     ckpt_path = os.path.abspath(os.path.join(workdir, "ckpt", config.wandb.name))
#     state = restore_checkpoint(model.state, ckpt_path)
#     params = state['params']

#     # Save the figures
#     save_dir = os.path.join(workdir, "figures", config.wandb.name)
#     if not os.path.isdir(save_dir):
#         os.makedirs(save_dir)

#     # Create a figure for temperature vs. time with multiple y-slices
#     fig, axs = plt.subplots(nrows=len(z_coords), ncols=len(x_coords), figsize=(15, 12))
#     u_max = config.process_conditions.deposition_temperature

#     for i, z in enumerate(z_coords):  # Iterate over z-coordinates (rows)
#         for j, x in enumerate(x_coords):  # Iterate over x-coordinates (columns)
#             ax = axs[i, j]
#             for y in y_slices:  # Iterate over y slices
#                 xyz = jnp.array([x, y, z])

#                 # Normalize spatial coordinates
#                 x_scaled = xyz[0] / config.dimensions.x_max
#                 y_scaled = xyz[1] / config.dimensions.y_max
#                 z_scaled = xyz[2] / config.dimensions.z_max

#                 # Compute the time when the deposition front reaches position x
#                 v = config.process_conditions.print_speed  # Deposition speed in x-direction
#                 t_deposition = x / v

#                 # Compute temperature over time at the fixed point (x, y, z)
#                 temp = model.evalfn_(params, time_array / config.dimensions.t_max, x_scaled, y_scaled, z_scaled) * u_max

#                 # Adjust the temperature array
#                 temp_adjusted = temp.copy()

#                 # Before the deposition front reaches x, set temperature to deposition temperature
#                 temp_adjusted = temp_adjusted.at[time_array < t_deposition].set(u_max)

#                 # Plot the adjusted temperature vs. time for the current y-slice
#                 ax.plot(time_array, temp_adjusted, label=f'y={y:.6f}')

#             ax.set_ylim([303, u_max * 1.15])
#             ax.set_xlim([0., config.dimensions.t_max])
#             ax.set_xlabel("t")
#             ax.set_ylabel("u")
#             ax.set_title(f"x={x:.6f}, z={z:.6f}")

#             # Plot the vertical line at t_deposition
#             ax.axvline(x=t_deposition, color='black', linestyle='--', linewidth=3)

#             # Add a legend to the subplot
#             ax.legend(loc='upper left', fontsize=8)

#     plt.tight_layout()

#     # Save the figure
#     fig_path = os.path.join(save_dir, "temperature_vs_time_multislice.pdf")
#     fig.savefig(fig_path, bbox_inches="tight", dpi=300)
#     plt.close(fig)

#     # Rest of your evaluation code remains unchanged
#     save_evol_dir = os.path.join(save_dir, "evolution")
#     if not os.path.isdir(save_evol_dir):
#         os.makedirs(save_evol_dir)

#     # Time steps for evolution plots
#     time_array = jnp.array([0, 2, 4, 6, 8, 10, 12])
#     time_array_scaled = time_array / config.dimensions.t_max

#     # Create evolution plots at different time steps
#     for i in range(time_array.shape[0]):
#         u_pred, x_, z_ = model.evaluate_Uplot(params, time_array_scaled[i], num_points=500)

#         fig, ax = plt.subplots(figsize=(6, 5))
#         sc = ax.scatter(x_, z_, s=4, c=u_pred, vmin=303., vmax=u_max, cmap='jet')
#         ax.set_xlim([0., config.dimensions.x_max])
#         ax.set_ylim([0., config.dimensions.z_max])
#         ax.set_title(f"Time = {time_array[i]:.2f}s")
#         ax.set_xlabel('$x$', fontsize=14)
#         ax.set_ylabel('$z$', fontsize=14)
#         plt.colorbar(sc, ax=ax)
#         plt.tight_layout()

#         fig_path = os.path.join(save_evol_dir, f"evolution_time_{i+1}.pdf")
#         fig.savefig(fig_path, dpi=300)
#         plt.close(fig)

#     # Initial temperature distribution plot
#     fig, ax = plt.subplots(figsize=(6, 5))
#     u_pred, x_, z_ = model.evaluate_init_plot(params, num_points=500)
#     sc = ax.scatter(x_, z_, s=4, c=u_pred, vmin=0., vmax=u_max, cmap='jet')
#     ax.set_xlim([0., config.dimensions.x_max])
#     ax.set_ylim([0., config.dimensions.z_max])
#     ax.set_title("Initial Temperature Distribution")
#     ax.set_xlabel('$x$', fontsize=14)
#     ax.set_ylabel('$z$', fontsize=14)
#     plt.colorbar(sc, ax=ax)
#     plt.tight_layout()

#     fig_path = os.path.join(save_evol_dir, "initial_temperature.pdf")
#     fig.savefig(fig_path, dpi=300)
#     plt.close(fig)

#     interactive_3d_visualization(config, params, model, save_dir)

#     # Create a GIF of the temperature evolution over time
#     save_evol_dir_gif = os.path.join(save_dir, "Pictures")
#     if not os.path.isdir(save_evol_dir_gif):
#         os.makedirs(save_evol_dir_gif)

#     time_array = jnp.linspace(0., config.dimensions.t_max, 40)
#     time_array_scaled = time_array / config.dimensions.t_max

#     for i in range(time_array.shape[0]):
#         u_pred, x_, z_ = model.evaluate_Uplot(params, time_array_scaled[i], num_points=500)

#         fig, ax = plt.subplots(figsize=(12, 3))
#         sc = ax.scatter(x_, z_, s=4, c=u_pred, vmin=303, vmax=u_max, cmap='jet')
#         ax.set_xlim([0., config.dimensions.x_max])
#         ax.set_ylim([0., config.dimensions.z_max])
#         ax.set_title(f"Time = {time_array[i]:.2f}s")
#         ax.set_xlabel('$x$', fontsize=14)
#         ax.set_ylabel('$z$', fontsize=14)
#         plt.colorbar(sc, ax=ax)
#         plt.tight_layout()

#         fig_path = os.path.join(save_evol_dir_gif, f"pic_{i+1}.png")
#         fig.savefig(fig_path, dpi=300)
#         plt.close(fig)

#     # Create GIF from the saved images
#     directory = os.path.join(save_evol_dir_gif, "*.png")
#     gif_path = os.path.join(save_evol_dir, "temperature_evolution.gif")

#     with contextlib.ExitStack() as stack:
#         imgs = (stack.enter_context(Image.open(f)) for f in sorted(glob.glob(directory), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])))
#         img = next(imgs)
#         img.save(fp=gif_path, format='GIF', append_images=imgs, save_all=True, duration=100, loop=0)







# def interactive_3d_visualization(config, params, model, save_dir):
#     """
#     Create an interactive 3D visualization with a slider to observe the heat distribution over time,
#     showing material deposition along the x-direction within a fixed volume.
#     """
#     import plotly.graph_objs as go
#     import jax.numpy as jnp
#     import os

#     # # Define the time steps for the slider
#     # time_steps = jnp.linspace(0., config.dimensions.t_max, 25)  # Adjust the number of frames as needed
#     # time_steps_scaled = time_steps / config.dimensions.t_max

#     # Define the time steps for the slider
#     default_time_steps = jnp.linspace(0., config.dimensions.t_max, 25)  # Default 25 intervals
#     specific_time_steps = jnp.array([0.01, 0.5, 1.0, 2.0])  # Add specific time steps
#     time_steps = jnp.unique(jnp.concatenate((default_time_steps, specific_time_steps)))  # Combine and deduplicate
#     time_steps_scaled = time_steps / config.dimensions.t_max

#     # Prepare constants
#     num_points_face = 200  # Adjust for resolution of the faces
#     x_max = config.dimensions.x_max
#     y_max = config.dimensions.y_max
#     z_max = config.dimensions.z_max
#     u_max = config.process_conditions.deposition_temperature

#     # Fixed values for velocity and initial length (scaled)
#     velocity_scaled = model.print_speed_scaled * model.velocity_vector[0]  # Assuming movement only in x-direction
#     init_length_scaled = model.init_length_scaled[0]

#     # Create a list to hold the frames for the slider
#     frames = []

#     for idx, t_scaled in enumerate(time_steps_scaled):
#         # Compute length_updated for this time step in actual units
#         length_updated_scaled = velocity_scaled * t_scaled + init_length_scaled
#         length_updated_scaled = jnp.minimum(length_updated_scaled, 1.0)  # Ensure it doesn't exceed scaled x_max (1.0)
#         length_updated = length_updated_scaled * x_max

#         surfaces = []

#         # Deposition Surface (x = length_updated)
#         y = jnp.linspace(0., y_max, num_points_face)
#         z = jnp.linspace(0., z_max, num_points_face)
#         yy, zz = jnp.meshgrid(y, z, indexing='ij')
#         xx = jnp.full_like(yy, length_updated)

#         # Normalize coordinates
#         x_scaled_flat = xx.flatten() / x_max
#         y_scaled_flat = yy.flatten() / y_max
#         z_scaled_flat = zz.flatten() / z_max

#         # Compute temperature
#         temp_scaled = model.u_pred_fn(params, t_scaled, x_scaled_flat, y_scaled_flat, z_scaled_flat)
#         temp = temp_scaled * u_max
#         temp_surface = temp.reshape(yy.shape)

#         # Create go.Surface for the deposition surface
#         surface_deposition = go.Surface(
#             x=xx,
#             y=yy,
#             z=zz,
#             surfacecolor=temp_surface,
#             colorscale='Jet',
#             cmin=303,
#             cmax=u_max,
#             showscale=False,
#             opacity=1.0,
#             name='Deposition Surface'
#         )
#         surfaces.append(surface_deposition)

#         # Other faces (Back, Left, Right, Top, Bottom)
#         faces = ['back', 'left', 'right', 'top', 'bottom']
#         for face in faces:
#             if face == 'back':
#                 # Back face (x = 0)
#                 xx = jnp.zeros_like(yy)
#             elif face == 'left':
#                 # Left face (y = 0)
#                 x = jnp.linspace(0., x_max, num_points_face)
#                 z = jnp.linspace(0., z_max, num_points_face)
#                 xx, zz = jnp.meshgrid(x, z, indexing='ij')
#                 yy = jnp.zeros_like(xx)
#             elif face == 'right':
#                 # Right face (y = y_max)
#                 yy = jnp.full_like(xx, y_max)
#             elif face == 'top':
#                 # Top face (z = z_max)
#                 x = jnp.linspace(0., x_max, num_points_face)
#                 y = jnp.linspace(0., y_max, num_points_face)
#                 xx, yy = jnp.meshgrid(x, y, indexing='ij')
#                 zz = jnp.full_like(xx, z_max)
#             elif face == 'bottom':
#                 # Bottom face (z = 0)
#                 zz = jnp.zeros_like(xx)

#             # Normalize coordinates
#             x_scaled_flat = xx.flatten() / x_max
#             y_scaled_flat = yy.flatten() / y_max
#             z_scaled_flat = zz.flatten() / z_max

#             # Compute temperature
#             temp_scaled = model.u_pred_fn(params, t_scaled, x_scaled_flat, y_scaled_flat, z_scaled_flat)
#             temp = temp_scaled * u_max
#             temp_surface = temp.reshape(xx.shape)

#             # Create mask for regions beyond deposition front
#             # mask = xx >= length_updated

#             # Define a small tolerance
#             tolerance = 1e-2  # Adjust as needed based on your grid resolution
            
#             # Modify the mask condition
#             mask = xx > (length_updated + tolerance)
#             # Set z to NaN where mask is True to make those regions transparent
#             zz_masked = jnp.where(mask, jnp.nan, zz)
#             xx_masked = jnp.where(mask, xx, xx)  # No change, but for consistency
#             yy_masked = jnp.where(mask, yy, yy)  # No change, but for consistency
#             temp_surface_masked = jnp.where(mask, jnp.nan, temp_surface)

#             surface = go.Surface(
#                 x=xx_masked,
#                 y=yy_masked,
#                 z=zz_masked,
#                 surfacecolor=temp_surface_masked,
#                 colorscale='Jet',
#                 cmin=303,
#                 cmax=u_max,
#                 showscale=False,
#                 opacity=1.0,
#                 name=f'{face.capitalize()} Face'
#             )
#             surfaces.append(surface)

#         # Append the frame with all surfaces
#         frames.append(go.Frame(data=surfaces, name=f'frame{idx}'))

#     # Create the initial plot with the first frame's surfaces
#     initial_surfaces = frames[0].data

#     # Define a custom camera view
#     camera = dict(
#         eye=dict(x=1.5, y=-1.7, z=1.0),  # Position of the camera (adjust x, y, and z as needed)
#         center=dict(x=0., y=0., z=0.),  # Point the camera is looking at
#         up=dict(x=0., y=0., z=1.)       # Direction of the "up" vector
#     )

#     # Define layout with colorbar
#     layout = go.Layout(
#         title='3D Heat Distribution Over Time',
#         scene=dict(
#             xaxis_title='X (m)',
#             yaxis_title='Y (m)',
#             zaxis_title='Z (m)',
#             xaxis=dict(range=[0, x_max]),
#             yaxis=dict(range=[0, y_max]),
#             zaxis=dict(range=[0, z_max]),
#             aspectmode='data',
#             camera=camera,
#         ),
#         updatemenus=[{
#             'buttons': [
#                 {
#                     'args': [None, {'frame': {'duration': 100, 'redraw': True},
#                                     'fromcurrent': True, 'transition': {'duration': 0}}],
#                     'label': 'Play',
#                     'method': 'animate'
#                 },
#                 {
#                     'args': [[None], {'frame': {'duration': 0, 'redraw': False},
#                                       'mode': 'immediate',
#                                       'transition': {'duration': 0}}],
#                     'label': 'Pause',
#                     'method': 'animate'
#                 }
#             ],
#             'direction': 'left',
#             'pad': {'r': 10, 't': 70},
#             'showactive': False,
#             'type': 'buttons',
#             'x': 0.1,
#             'xanchor': 'right',
#             'y': 0,
#             'yanchor': 'top'
#         }],
#         sliders=[{
#             'steps': [
#                 {
#                     'args': [[f'frame{idx}'],
#                              {'frame': {'duration': 0, 'redraw': True},
#                               'mode': 'immediate',
#                               'transition': {'duration': 0}}],
#                     'label': f'{time_steps[idx]:.2f}s',
#                     'method': 'animate'
#                 }
#                 for idx in range(len(frames))
#             ],
#             'transition': {'duration': 0},
#             'x': 0,
#             'y': 0,
#             'currentvalue': {'font': {'size': 16}, 'prefix': 'Time: ', 'visible': True, 'xanchor': 'right'},
#             'len': 1.0
#         }]
#     )

#     # Add a colorbar manually
#     colorbar_trace = go.Scatter3d(
#         x=[None],
#         y=[None],
#         z=[None],
#         mode='markers',
#         marker=dict(
#             colorscale='Jet',
#             showscale=True,
#             cmin=303,
#             cmax=u_max,
#             colorbar=dict(
#                 title='Temperature (K)',
#                 titleside='right'
#             )
#         ),
#         hoverinfo='none'
#     )

#     fig = go.Figure(data=list(initial_surfaces) + [colorbar_trace], frames=frames, layout=layout)

#     # Save the interactive plot as an HTML file
#     interactive_plot_path = os.path.join(save_dir, 'interactive_3d_heat_distribution.html')
#     fig.write_html(interactive_plot_path)
#     print(f'Interactive 3D visualization saved at {interactive_plot_path}')


### End of single layer implementation ###
















################## Multilayer ##########################
import os
import ml_collections
from jax import vmap
import jax.numpy as jnp
import matplotlib.pyplot as plt
from A3DPINN.utils import restore_checkpoint
import models
import glob
from PIL import Image
import contextlib
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def create_gif(folder_path, gif_name, size=(300, 300)):
    """Create a GIF from a sequence of PNG images."""
    images = []
    for filename in sorted(os.listdir(folder_path), key=lambda x: int(x.split('_')[1].split('.')[0])):
        if filename.endswith('.png'):
            img = Image.open(os.path.join(folder_path, filename))
            images.append(img)
    images[0].save(gif_name, save_all=True, append_images=images[1:], duration=100, loop=0)

def evaluate(config: ml_collections.ConfigDict, workdir: str):
    """Evaluate the A3DHeatTransfer model and generate plots for multilayer simulation."""
    # Define time array and spatial coordinates
    time_array = jnp.linspace(0., config.dimensions.t_max, 100)  # For temperature vs. time
    y_slices = jnp.linspace(0., config.dimensions.y_max, 5)
    x_coords = [0.0, 0.025, 0.05]  # Adjusted for x_max = 0.05 m
    z_coords = [0.00130, 0.00149, 0.00150, 0.00151, 0.00160, 0.00280, 0.00299, 0.00300, 0.00301, 0.00310, 0.00430, 0.00449]  # Bed and top of each layer

    # Restore model
    model = models.A3DHeatTransfer(config)
    ckpt_path = os.path.abspath(os.path.join(workdir, "ckpt", config.wandb.name))
    state = restore_checkpoint(model.state, ckpt_path)
    params = state['params']

    # Set up save directory
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # **Temperature vs. Time Plots with Masking**
    fig, axs = plt.subplots(nrows=len(z_coords), ncols=len(x_coords), figsize=(15, 12))
    u_max = config.process_conditions.deposition_temperature

    for i, z in enumerate(z_coords):
        for j, x in enumerate(x_coords):
            ax = axs[i, j]
            for y in y_slices:
                xyz = jnp.array([x, y, z])
                x_scaled = xyz[0] / config.dimensions.x_max
                y_scaled = xyz[1] / config.dimensions.y_max
                z_scaled = xyz[2] / config.dimensions.z_max

                # Calculate deposition time based on layer index
                layer_index = int(jnp.floor(z / config.multi_layer.layer_height))
                t_deposition = layer_index * config.multi_layer.t_L

                # Compute temperature over time
                temp = model.evalfn_(params, time_array / config.dimensions.t_max, x_scaled, y_scaled, z_scaled) * u_max

                # Mask temperatures before deposition time
                temp_masked = jnp.where(time_array >= t_deposition, temp, jnp.nan)

                ax.plot(time_array, temp_masked, label=f'y={y:.6f}')

            ax.set_ylim([303, u_max * 1.15])
            ax.set_xlim([0., config.dimensions.t_max])
            ax.set_xlabel("t")
            ax.set_ylabel("u")
            ax.set_title(f"x={x:.6f}, z={z:.6f}")
            ax.legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(save_dir, "temperature_vs_time_multislice.pdf")
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    # **New Plot: Temperature vs. Time at Fixed (x, y) for Multiple z-levels**
    # This plot helps visualize conduction into older layers when a new layer is deposited.
    fig, ax = plt.subplots(figsize=(10, 6))
    x_fixed = 0.025  # m, middle of x-domain
    y_fixed = 0.003  # m, middle of y-domain
    x_scaled = x_fixed / config.dimensions.x_max
    y_scaled = y_fixed / config.dimensions.y_max

    for z in z_coords:
        z_scaled = z / config.dimensions.z_max
        # Determine deposition time for this z
        layer_index = int(jnp.floor(z / config.multi_layer.layer_height))
        t_deposition = layer_index * config.multi_layer.t_L
        # Compute temperature over time
        temp = model.evalfn_(params, time_array / config.dimensions.t_max, x_scaled, y_scaled, z_scaled) * u_max
        # Mask before deposition
        temp_masked = jnp.where(time_array >= t_deposition, temp, jnp.nan)
        ax.plot(time_array, temp_masked, label=f'z={z:.6f} m')

    # Add vertical lines for deposition times
    for t_dep in [0, 2, 4]:
        ax.axvline(x=t_dep, color='gray', linestyle='--', linewidth=1)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Temperature (K)')
    ax.set_title(f'Temperature vs. Time at x={x_fixed:.4f} m, y={y_fixed:.4f} m')
    ax.set_ylim([480, 625])
    ax.set_xlim([0, config.dimensions.t_max])
    ax.legend(loc='upper right')
    plt.tight_layout()
    fig_path = os.path.join(save_dir, "temperature_vs_time_fixed_xy.pdf")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    # **Evolution Plots with Masking**
    save_evol_dir = os.path.join(save_dir, "evolution")
    if not os.path.isdir(save_evol_dir):
        os.makedirs(save_evol_dir)

    time_array = jnp.array([0, 2, 4, 6, 8])  # Times within t_max = 8 s
    time_array_scaled = time_array / config.dimensions.t_max

    for i in range(time_array.shape[0]):
        u_pred, x_, z_ = model.evaluate_Uplot(params, time_array_scaled[i], num_points=500)
        if u_pred.size > 0:  # Only plot if there are active points
            fig, ax = plt.subplots(figsize=(6, 5))
            sc = ax.scatter(x_, z_, s=4, c=u_pred, vmin=303., vmax=u_max, cmap='jet')
            ax.set_xlim([0., config.dimensions.x_max])
            ax.set_ylim([0., config.dimensions.z_max])
            ax.set_title(f"Time = {time_array[i]:.2f}s")
            ax.set_xlabel('$x$', fontsize=14)
            ax.set_ylabel('$z$', fontsize=14)
            plt.colorbar(sc, ax=ax)
            plt.tight_layout()
            fig_path = os.path.join(save_evol_dir, f"evolution_time_{i+1}.pdf")
            fig.savefig(fig_path, dpi=300)
            plt.close(fig)
        else:
            print(f"No active points at time {time_array[i]:.2f}s")

    # **Initial Temperature Distribution Plot**
    fig, ax = plt.subplots(figsize=(6, 5))
    u_pred, x_, z_ = model.evaluate_init_plot(params, num_points=500)
    sc = ax.scatter(x_, z_, s=4, c=u_pred, vmin=0., vmax=u_max, cmap='jet')
    ax.set_xlim([0., config.dimensions.x_max])
    ax.set_ylim([0., config.dimensions.z_max])
    ax.set_title("Initial Temperature Distribution")
    ax.set_xlabel('$x$', fontsize=14)
    ax.set_ylabel('$z$', fontsize=14)
    plt.colorbar(sc, ax=ax)
    plt.tight_layout()
    fig_path = os.path.join(save_evol_dir, "initial_temperature.pdf")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    # **Interactive 3D Visualization**
    interactive_3d_visualization(config, params, model, save_dir)

    # **Create GIF of Temperature Evolution**
    save_evol_dir_gif = os.path.join(save_dir, "Pictures")
    if not os.path.isdir(save_evol_dir_gif):
        os.makedirs(save_evol_dir_gif)

    time_array = jnp.linspace(0., config.dimensions.t_max, 40)  # 40 frames from 0 to 8 s
    time_array_scaled = time_array / config.dimensions.t_max

    for i in range(time_array.shape[0]):
        u_pred, x_, z_ = model.evaluate_Uplot(params, time_array_scaled[i], num_points=500)
        if u_pred.size > 0:  # Only plot if there are active points
            fig, ax = plt.subplots(figsize=(12, 3))
            sc = ax.scatter(x_, z_, s=4, c=u_pred, vmin=303, vmax=u_max, cmap='jet')
            ax.set_xlim([0., config.dimensions.x_max])
            ax.set_ylim([0., config.dimensions.z_max])
            ax.set_title(f"Time = {time_array[i]:.2f}s")
            ax.set_xlabel('$x$', fontsize=14)
            ax.set_ylabel('$z$', fontsize=14)
            plt.colorbar(sc, ax=ax)
            plt.tight_layout()
            fig_path = os.path.join(save_evol_dir_gif, f"pic_{i+1}.png")
            fig.savefig(fig_path, dpi=300)
            plt.close(fig)

    # Create GIF from saved images
    directory = os.path.join(save_evol_dir_gif, "*.png")
    gif_path = os.path.join(save_evol_dir, "temperature_evolution.gif")
    with contextlib.ExitStack() as stack:
        imgs = (stack.enter_context(Image.open(f)) for f in sorted(glob.glob(directory), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])))
        img = next(imgs)
        img.save(fp=gif_path, format='GIF', append_images=imgs, save_all=True, duration=100, loop=0)

def interactive_3d_visualization(config, params, model, save_dir):
    """
    Create an interactive 3D visualization with a slider to observe heat distribution over time,
    showing material deposition in the z-direction within a fixed volume.
    """
    # Define time steps for the slider
    default_time_steps = jnp.linspace(0., config.dimensions.t_max, 25)
    specific_time_steps = jnp.array([0.01, 0.5, 1.0, 2.0])
    time_steps = jnp.unique(jnp.concatenate((default_time_steps, specific_time_steps)))
    time_steps_scaled = time_steps / config.dimensions.t_max

    # Constants
    num_points_face = 200
    x_max = config.dimensions.x_max
    y_max = config.dimensions.y_max
    z_max = config.dimensions.z_max
    u_max = config.process_conditions.deposition_temperature
    t_L = config.multi_layer.t_L
    layer_height = config.multi_layer.layer_height

    frames = []

    for idx, t_scaled in enumerate(time_steps_scaled):
        t = t_scaled * config.dimensions.t_max
        i = int(jnp.floor(t / t_L))  # Current layer index
        current_z_top = min((i + 1) * layer_height, z_max)  # Cap at z_max

        surfaces = []

        # **Top Surface (z = current_z_top)**
        x = jnp.linspace(0., x_max, num_points_face)
        y = jnp.linspace(0., y_max, num_points_face)
        xx, yy = jnp.meshgrid(x, y, indexing='ij')
        zz = jnp.full_like(xx, current_z_top)
        x_scaled = xx.flatten() / x_max
        y_scaled = yy.flatten() / y_max
        z_scaled = zz.flatten() / z_max
        temp_scaled = model.u_pred_fn(params, t_scaled, x_scaled, y_scaled, z_scaled)
        temp = temp_scaled * u_max
        temp_surface = temp.reshape(xx.shape)
        surface_top = go.Surface(
            x=xx, y=yy, z=zz, surfacecolor=temp_surface,
            colorscale='Jet', cmin=303, cmax=u_max, showscale=False, opacity=1.0, name='Top Surface'
        )
        surfaces.append(surface_top)

        # **Other Faces (Back, Front, Left, Right, Bottom)**
        faces = ['back', 'front', 'left', 'right', 'bottom']
        for face in faces:
            if face == 'back':
                y = jnp.linspace(0., y_max, num_points_face)
                z = jnp.linspace(0., z_max, num_points_face)
                yy, zz = jnp.meshgrid(y, z, indexing='ij')
                xx = jnp.zeros_like(yy)
            elif face == 'front':
                yy, zz = jnp.meshgrid(y, z, indexing='ij')
                xx = jnp.full_like(yy, x_max)
            elif face == 'left':
                x = jnp.linspace(0., x_max, num_points_face)
                z = jnp.linspace(0., z_max, num_points_face)
                xx, zz = jnp.meshgrid(x, z, indexing='ij')
                yy = jnp.zeros_like(xx)
            elif face == 'right':
                yy = jnp.full_like(xx, y_max)
            elif face == 'bottom':
                x = jnp.linspace(0., x_max, num_points_face)
                y = jnp.linspace(0., y_max, num_points_face)
                xx, yy = jnp.meshgrid(x, y, indexing='ij')
                zz = jnp.zeros_like(xx)

            x_scaled = xx.flatten() / x_max
            y_scaled = yy.flatten() / y_max
            z_scaled = zz.flatten() / z_max
            temp_scaled = model.u_pred_fn(params, t_scaled, x_scaled, y_scaled, z_scaled)
            temp = temp_scaled * u_max
            temp_surface = temp.reshape(xx.shape)

            # Mask regions above current_z_top
            if face in ['back', 'front', 'left', 'right']:
                mask = zz > current_z_top
                zz_masked = jnp.where(mask, jnp.nan, zz)
                temp_surface_masked = jnp.where(mask, jnp.nan, temp_surface)
            else:
                zz_masked = zz
                temp_surface_masked = temp_surface

            surface = go.Surface(
                x=xx, y=yy, z=zz_masked, surfacecolor=temp_surface_masked,
                colorscale='Jet', cmin=303, cmax=u_max, showscale=False, opacity=1.0, name=f'{face.capitalize()} Face'
            )
            surfaces.append(surface)

        frames.append(go.Frame(data=surfaces, name=f'frame{idx}'))

    # Initial surfaces
    initial_surfaces = frames[0].data

    # Fixed camera view
    camera = dict(eye=dict(x=1.5, y=-1.7, z=1.0), center=dict(x=0., y=0., z=0.), up=dict(x=0., y=0., z=1.))

    # Layout with fixed aspect ratio and colorbar
    layout = go.Layout(
        title='3D Heat Distribution Over Time',
        scene=dict(
            xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
            xaxis=dict(range=[0, x_max]), yaxis=dict(range=[0, y_max]), zaxis=dict(range=[0, z_max]),
            aspectmode='cube',  # Ensures fixed volume
            camera=camera
        ),
        updatemenus=[{
            'buttons': [
                {'args': [None, {'frame': {'duration': 100, 'redraw': True}, 'fromcurrent': True, 'transition': {'duration': 0}}], 'label': 'Play', 'method': 'animate'},
                {'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 0}}], 'label': 'Pause', 'method': 'animate'}
            ],
            'direction': 'left', 'pad': {'r': 10, 't': 70}, 'showactive': False, 'type': 'buttons', 'x': 0.1, 'xanchor': 'right', 'y': 0, 'yanchor': 'top'
        }],
        sliders=[{
            'steps': [{'args': [[f'frame{idx}'], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}], 'label': f'{time_steps[idx]:.2f}s', 'method': 'animate'} for idx in range(len(frames))],
            'transition': {'duration': 0}, 'x': 0, 'y': 0, 'currentvalue': {'font': {'size': 16}, 'prefix': 'Time: ', 'visible': True, 'xanchor': 'right'}, 'len': 1.0
        }]
    )

    colorbar_trace = go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', marker=dict(colorscale='Jet', showscale=True, cmin=303, cmax=u_max, colorbar=dict(title='Temperature (K)', titleside='right')), hoverinfo='none')
    fig = go.Figure(data=list(initial_surfaces) + [colorbar_trace], frames=frames, layout=layout)
    interactive_plot_path = os.path.join(save_dir, 'interactive_3d_heat_distribution.html')
    fig.write_html(interactive_plot_path)
    print(f'Interactive 3D visualization saved at {interactive_plot_path}')



