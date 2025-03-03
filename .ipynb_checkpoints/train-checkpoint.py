# y (bead width direction)
# ^
# |
# |
# |
# |-------------->x deposition direction
# 0

import os
import time
import shutil

import jax
import jax.numpy as jnp

os.environ['ENABLE_PJRT_COMPATIBILITY'] = '1'

print("Available devices:", jax.devices())

from jax.tree_util import tree_map

import ml_collections
from absl import logging
import wandb

# Import the correct samplers for 3D
from A3DPINN.samplers import UniformSampler, StepIndexSampler
from A3DPINN.logging import Logger
from A3DPINN.utils import save_checkpoint
import orbax
import models
from utils import get_dataset

def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Initialize W&B
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    # Initialize logger
    logger = Logger()

    # Define residual sampler for time
    time_sampler = iter(UniformSampler(jnp.array([[0.0, 1.0]]), config.training.time_batch_size_per_device))

    # Compute scaled initial length, bead width, and bead height
    init_len = config.process_conditions.init_length / config.dimensions.x_max
    bead_width_scaled = config.process_conditions.bead_width / config.dimensions.y_max
    bead_height_scaled = config.process_conditions.bead_height / config.dimensions.z_max

    # Define initial condition sampler in 3D
    initial_sampler = iter(UniformSampler(
        jnp.array([
            [0.0, 0.0],           # t from 0 to 0 (initial time)
            [0.0, init_len[0]],   # x from 0 to initial length
            [0.0, bead_width_scaled],  # y from 0 to bead width
            [0.0, bead_height_scaled]  # z from 0 to bead height
        ]),
        config.training.batch_size_per_device
    ))

    # Define step sampler for sequential sampling
    step_sampler = iter(StepIndexSampler(100000, config.training.time_batch_size_per_device))

    # Initialize model
    model = models.A3DHeatTransfer(config)

    # Set up checkpoint path
    path = os.path.abspath(os.path.join(workdir, "ckpt", config.wandb.name))
    if os.path.exists(path):
        shutil.rmtree(path)

    # Initialize evaluator
    evaluator = models.A3DHeatTransferEvaluator(config, model)

    # Set up checkpoint manager
    mgr_options = orbax.checkpoint.CheckpointManagerOptions(save_interval_steps=1, max_to_keep=3)
    ckpt_mgr = orbax.checkpoint.CheckpointManager(
        path,
        orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler()),
        mgr_options
    )

    print("Waiting for JIT...")

    for step in range(config.training.max_steps):
        start_time = time.time()

        # Get batches from samplers
        time_batch = next(time_sampler)
        step_batch = next(step_sampler)
        batch_initial = next(initial_sampler)

        # Perform a training step
        model.state = model.step(model.state, time_batch, batch_initial, step_batch)

        # Update weights if using certain weighting schemes
        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, time_batch, batch_initial, step_batch)

        # Log training metrics, only use host 0 to record results
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                # Get the first replica of the state and batches
                state = jax.device_get(tree_map(lambda x: x[0], model.state))
                time_batch = jax.device_get(tree_map(lambda x: x[0], time_batch))
                batch_initial = jax.device_get(tree_map(lambda x: x[0], batch_initial))
                step_batch = jax.device_get(tree_map(lambda x: x[0], step_batch))

                # Evaluate and log metrics
                log_dict = evaluator(state, time_batch, batch_initial, step_batch)
                wandb.log(log_dict, step)

                end_time = time.time()

                logger.log_iter(step, start_time, end_time, log_dict)

        # Save checkpoints
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (step + 1) == config.training.max_steps:
                save_checkpoint(model.state, path, ckpt_mgr)

    return model


















# ################### multilayer #########################
# import os
# import time
# import shutil

# import jax
# import jax.numpy as jnp

# os.environ['ENABLE_PJRT_COMPATIBILITY'] = '1'

# print("Available devices:", jax.devices())

# from jax.tree_util import tree_map
# from jax import random

# import ml_collections
# from absl import logging
# import wandb

# # Import the correct samplers for 3D
# from A3DPINN.samplers import UniformSampler, StepIndexSampler
# from A3DPINN.logging import Logger
# from A3DPINN.utils import save_checkpoint
# import orbax
# import models
# from utils import get_dataset

# def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
#     # Initialize W&B
#     wandb_config = config.wandb
#     wandb.init(project=wandb_config.project, name=wandb_config.name)

#     # Initialize logger
#     logger = Logger()

#     # Define residual sampler for time
#     time_sampler = iter(UniformSampler(jnp.array([[0.0, 1.0]]), config.training.time_batch_size_per_device))

#     # Compute scaled initial length, bead width, and bead height
#     init_len = config.process_conditions.init_length / config.dimensions.x_max
#     bead_width_scaled = config.process_conditions.bead_width / config.dimensions.y_max
#     bead_height_scaled = config.process_conditions.bead_height / config.dimensions.z_max

#     # Define initial condition sampler in 3D
#     initial_sampler = iter(UniformSampler(
#         jnp.array([
#             [0.0, 0.0],           # t from 0 to 0 (initial time)
#             [0.0, init_len[0]],   # x from 0 to initial length
#             [0.0, bead_width_scaled],  # y from 0 to bead width
#             [0.0, bead_height_scaled]  # z from 0 to bead height
#         ]),
#         config.training.batch_size_per_device
#     ))

#     # Define step sampler for sequential sampling
#     step_sampler = iter(StepIndexSampler(100000, config.training.time_batch_size_per_device))

#     # Compute scaled t_L and define IC times
#     t_L_scaled = config.multi_layer.t_L / config.dimensions.t_max
#     num_layers = config.multi_layer.num_layers
#     ic_times_scaled = jnp.array([i * t_L_scaled for i in range(num_layers)])

#     # Define custom time batch function
#     def get_time_batch(step, config, ic_times_scaled):
#         n_devices = jax.local_device_count()
#         time_batch_size_per_device = config.training.time_batch_size_per_device
#         uniform_keys = random.split(random.PRNGKey(step), n_devices)
#         uniform_times = random.uniform(
#             uniform_keys,
#             shape=(n_devices, time_batch_size_per_device - 1),
#             minval=0.0,
#             maxval=1.0
#         )
#         ic_index = step % len(ic_times_scaled)
#         ic_time = ic_times_scaled[ic_index]
#         ic_time_array = jnp.full((n_devices, 1), ic_time)
#         time_batch = jnp.concatenate([uniform_times, ic_time_array], axis=1)
#         return time_batch

#     # Ensure weights for IC losses are defined
#     if "ic_new_layer" not in config.weighting.init_weights:
#         config.weighting.init_weights["ic_new_layer"] = 1.0
#     if "ic_prev_layers" not in config.weighting.init_weights:
#         config.weighting.init_weights["ic_prev_layers"] = 1.0

#     # Initialize model
#     model = models.A3DHeatTransfer(config)

#     # Set up checkpoint path
#     path = os.path.abspath(os.path.join(workdir, "ckpt", config.wandb.name))
#     if os.path.exists(path):
#         shutil.rmtree(path)

#     # Initialize evaluator
#     evaluator = models.A3DHeatTransferEvaluator(config, model)

#     # Set up checkpoint manager
#     mgr_options = orbax.checkpoint.CheckpointManagerOptions(save_interval_steps=1, max_to_keep=3)
#     ckpt_mgr = orbax.checkpoint.CheckpointManager(
#         path,
#         orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler()),
#         mgr_options
#     )

#     print("Waiting for JIT...")

#     for step in range(config.training.max_steps):
#         start_time = time.time()

#         # Get batches from samplers
#         step_batch = next(step_sampler)
#         batch_initial = next(initial_sampler)
#         time_batch = get_time_batch(step, config, ic_times_scaled)

#         # Perform a training step
#         model.state = model.step(model.state, time_batch, batch_initial, step_batch)

#         # Update weights if using certain weighting schemes
#         if config.weighting.scheme in ["grad_norm", "ntk"]:
#             if step % config.weighting.update_every_steps == 0:
#                 model.state = model.update_weights(model.state, time_batch, batch_initial, step_batch)

#         # Log training metrics, only use host 0 to record results
#         if jax.process_index() == 0:
#             if step % config.logging.log_every_steps == 0:
#                 state = jax.device_get(tree_map(lambda x: x[0], model.state))
#                 time_batch = jax.device_get(tree_map(lambda x: x[0], time_batch))
#                 batch_initial = jax.device_get(tree_map(lambda x: x[0], batch_initial))
#                 step_batch = jax.device_get(tree_map(lambda x: x[0], step_batch))

#                 log_dict = evaluator(state, time_batch, batch_initial, step_batch)
#                 wandb.log(log_dict, step)

#                 end_time = time.time()

#                 logger.log_iter(step, start_time, end_time, log_dict)

#         # Save checkpoints
#         if config.saving.save_every_steps is not None:
#             if (step + 1) % config.saving.save_every_steps == 0 or (step + 1) == config.training.max_steps:
#                 save_checkpoint(model.state, path, ckpt_mgr)

#     return model
