from abc import ABC, abstractmethod
from functools import partial

import jax.numpy as jnp
from jax import random, pmap, local_device_count

from torch.utils.data import Dataset


import jax.numpy as jnp
import jax.random as random
from jax import device_get, random

class BaseSampler(Dataset):
    def __init__(self, batch_size, rng_key=random.PRNGKey(1234)):
        self.batch_size = batch_size
        self.key = rng_key
        self.num_devices = local_device_count()

    def __getitem__(self, index):
        "Generate one batch of data"
        self.key, subkey = random.split(self.key)
        keys = random.split(subkey, self.num_devices)
        batch = self.data_generation(keys)
        return batch

    def data_generation(self, key):
        raise NotImplementedError("Subclasses should implement this!")

# Base class for all samplers
class SeqBaseSampler:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __call__(self, step, time):
        keys = random.split(random.PRNGKey(step), 1)  # Adjusted for single device
        batch = self.data_generation(keys, time)
        return batch

    def data_generation(self, key, time):
        raise NotImplementedError("Subclasses should implement this!")


class StepIndexSampler(BaseSampler):
    def __init__(self, n_test:int, batch_size, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size, rng_key)
        self.n_test=n_test
        self.dim = 1

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key):
        batch = random.categorical(key,
                                   logits=jnp.ones(int(self.n_test)),
                                   axis=0,
                                   shape=(self.batch_size, 1))
        return batch


class  UniformSampler(BaseSampler):
    def __init__(self, dom, batch_size, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size, rng_key)
        self.dom = dom
        self.dim = dom.shape[0]

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key):
        "Generates data containing batch_size samples"
        batch = random.uniform(
            key,
            shape=(self.batch_size, self.dim),
            minval=self.dom[:, 0],
            maxval=self.dom[:, 1],
        )

        return batch

# 3D extension of SeqCollocationSampler
class SeqCollocationSampler3D(SeqBaseSampler):
    def __init__(self, batch_size, init_length, velocity, bead_width, bead_height, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size)
        self.velocity = velocity  # Velocity in the x-direction
        self.init_length = init_length  # Initial length in x
        self.bead_width = bead_width  # Width in y
        self.bead_height = bead_height  # Height in z
        self.dim = 3

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key, time):
        # Evolving x dimension based on the time step and velocity
        length_updated = self.velocity[0] * time + self.init_length[0]

        # Sampling x from [0, length_updated]
        x_batch = random.uniform(key,
                                 shape=(self.batch_size, 1),
                                 minval=jnp.array([0.]),
                                 maxval=jnp.array([length_updated]))

        # Sampling y from [0, bead_width] (constant)
        y_batch = random.uniform(key + 1,
                                 shape=(self.batch_size, 1),
                                 minval=jnp.array([0.]),
                                 maxval=jnp.array([self.bead_width]))

        # Sampling z from [0, bead_height] (constant)
        z_batch = random.uniform(key + 2,
                                 shape=(self.batch_size, 1),
                                 minval=jnp.array([0.]),
                                 maxval=jnp.array([self.bead_height]))

        # Concatenating the time and spatial coordinates (x, y, z)
        volume_batch = jnp.concatenate((x_batch, y_batch, z_batch), axis=1)
        batch = jnp.concatenate((time * jnp.ones((self.batch_size, 1)), volume_batch), axis=1)[0]
        return batch

# 3D extension of SeqNeumanCollocationSampler_B1
class SeqNeumanCollocationSampler_B13D(SeqBaseSampler):
    def __init__(self, batch_size, init_length, velocity, bead_width, bead_height, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size)
        self.velocity = velocity
        self.init_length = init_length
        self.bead_width = bead_width
        self.bead_height = bead_height
        self.dim = 3

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key, time):
        length_updated = self.velocity[0] * time + self.init_length[0]

        # Sampling x from [0, length_updated]
        x_batch = random.uniform(key,
                                 shape=(self.batch_size, 1),
                                 minval=jnp.array([0.]),
                                 maxval=jnp.array([length_updated]))

        # Sampling y from [0, bead_width] (constant)
        y_batch = random.uniform(key + 1,
                                 shape=(self.batch_size, 1),
                                 minval=jnp.array([0.]),
                                 maxval=jnp.array([self.bead_width]))

        # Neumann boundary condition at z = bead_height (fixed)
        z_batch = jnp.ones((self.batch_size, 1)) * self.bead_height

        volume_batch = jnp.concatenate((x_batch, y_batch, z_batch), axis=1)
        batch = jnp.concatenate((time * jnp.ones((self.batch_size, 1)), volume_batch), axis=1)[0]
        return batch

# 3D extension of SeqNeumanCollocationSampler_B2
class SeqNeumanCollocationSampler_B23D(SeqBaseSampler):
    def __init__(self, batch_size, init_length, velocity, bead_width, bead_height, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size)
        self.velocity = velocity
        self.init_length = init_length
        self.bead_width = bead_width
        self.bead_height = bead_height
        self.dim = 3

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key, time):
        length_updated = self.velocity[0] * time + self.init_length[0]

        # Sampling x from [0, length_updated]
        x_batch = random.uniform(key,
                                 shape=(self.batch_size, 1),
                                 minval=jnp.array([0.]),
                                 maxval=jnp.array([length_updated]))

        # Sampling y from [0, bead_width] (constant)
        y_batch = random.uniform(key + 1,
                                 shape=(self.batch_size, 1),
                                 minval=jnp.array([0.]),
                                 maxval=jnp.array([self.bead_width]))

        # Neumann boundary condition at z = 0 (fixed)
        z_batch = jnp.zeros((self.batch_size, 1))

        volume_batch = jnp.concatenate((x_batch, y_batch, z_batch), axis=1)
        batch = jnp.concatenate((time * jnp.ones((self.batch_size, 1)), volume_batch), axis=1)[0]
        return batch

class DepositionFrontSampler3D(SeqBaseSampler):
    def __init__(self, batch_size, init_length, velocity, bead_width, bead_height, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size)
        self.velocity = velocity
        self.init_length = init_length
        self.bead_width = bead_width
        self.bead_height = bead_height
        self.dim = 3

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key, time):
        length_updated = self.velocity[0] * time + self.init_length[0]

        # x is fixed at the evolving front
        x_batch = jnp.ones((self.batch_size, 1)) * length_updated

        # y varies from 0 to bead_width
        y_batch = random.uniform(key + 1,
                                 shape=(self.batch_size, 1),
                                 minval=jnp.array([0.]),
                                 maxval=jnp.array([self.bead_width]))

        # z varies from 0 to bead_height
        z_batch = random.uniform(key + 2,
                                 shape=(self.batch_size, 1),
                                 minval=jnp.array([0.]),
                                 maxval=jnp.array([self.bead_height]))

        volume_batch = jnp.concatenate((x_batch, y_batch, z_batch), axis=1)
        batch = jnp.concatenate((time * jnp.ones((self.batch_size, 1)), volume_batch), axis=1)[0]
        return batch


class BedTemperatureSampler3D(SeqBaseSampler):
    def __init__(self, batch_size, init_length, velocity, bead_width, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size)
        self.velocity = velocity
        self.init_length = init_length
        self.bead_width = bead_width
        self.dim = 3

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key, time):
        length_updated = self.velocity[0] * time + self.init_length[0]

        # x varies from 0 to length_updated
        x_batch = random.uniform(key,
                                 shape=(self.batch_size, 1),
                                 minval=jnp.array([0.]),
                                 maxval=jnp.array([length_updated]))

        # y varies from 0 to bead_width
        y_batch = random.uniform(key + 1,
                                 shape=(self.batch_size, 1),
                                 minval=jnp.array([0.]),
                                 maxval=jnp.array([self.bead_width]))

        # z is fixed at 0 (bed surface)
        z_batch = jnp.zeros((self.batch_size, 1))

        volume_batch = jnp.concatenate((x_batch, y_batch, z_batch), axis=1)
        batch = jnp.concatenate((time * jnp.ones((self.batch_size, 1)), volume_batch), axis=1)[0]
        return batch

# Enforcing du/dx = 0 at x = 0
class NeumannBoundarySamplerX0(SeqBaseSampler):
    def __init__(self, batch_size, bead_width, bead_height, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size)
        self.bead_width = bead_width
        self.bead_height = bead_height

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key, time):
        # x is fixed at 0
        x_batch = jnp.zeros((self.batch_size, 1))

        # y varies from 0 to bead_width
        y_batch = random.uniform(key + 1,
                                 shape=(self.batch_size, 1),
                                 minval=jnp.array([0.]),
                                 maxval=jnp.array([self.bead_width]))

        # z varies from 0 to bead_height
        z_batch = random.uniform(key + 2,
                                 shape=(self.batch_size, 1),
                                 minval=jnp.array([0.]),
                                 maxval=jnp.array([self.bead_height]))

        volume_batch = jnp.concatenate((x_batch, y_batch, z_batch), axis=1)
        batch = jnp.concatenate((time * jnp.ones((self.batch_size, 1)), volume_batch), axis=1)[0]
        return batch

class BoundarySamplerX0(SeqBaseSampler):
    def __init__(self, batch_size, bead_width, bead_height, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size)
        self.bead_width = bead_width
        self.bead_height = bead_height

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key, time):
        # x is fixed at 0
        x_batch = jnp.zeros((self.batch_size, 1))

        # y varies from 0 to bead_width
        y_batch = random.uniform(key + 1,
                                 shape=(self.batch_size, 1),
                                 minval=jnp.array([0.]),
                                 maxval=jnp.array([self.bead_width]))

        # z varies from 0 to bead_height
        z_batch = random.uniform(key + 2,
                                 shape=(self.batch_size, 1),
                                 minval=jnp.array([0.]),
                                 maxval=jnp.array([self.bead_height]))

        volume_batch = jnp.concatenate((x_batch, y_batch, z_batch), axis=1)
        batch = jnp.concatenate((time * jnp.ones((self.batch_size, 1)), volume_batch), axis=1)[0]
        return batch

class ConvectionBoundarySamplerYMin(SeqBaseSampler):
    def __init__(self, batch_size, init_length, velocity, bead_width, bead_height, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size)
        self.velocity = velocity
        self.init_length = init_length
        self.bead_width = bead_width
        self.bead_height = bead_height
        self.dim = 3

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key, time):
        length_updated = self.velocity[0] * time + self.init_length[0]

        # x varies from 0 to length_updated
        x_batch = random.uniform(key,
                                 shape=(self.batch_size, 1),
                                 minval=jnp.array([0.]),
                                 maxval=jnp.array([length_updated]))
        # y is fixed at 0
        y_batch = jnp.zeros((self.batch_size, 1))

        # z varies from 0 to bead_height
        z_batch = random.uniform(key + 2,
                                 shape=(self.batch_size, 1),
                                 minval=jnp.array([0.]),
                                 maxval=jnp.array([self.bead_height]))

        volume_batch = jnp.concatenate((x_batch, y_batch, z_batch), axis=1)
        batch = jnp.concatenate((time * jnp.ones((self.batch_size, 1)), volume_batch), axis=1)[0]
        return batch

# Convection boundary condition
class ConvectionBoundarySamplerYMax(SeqBaseSampler):
    def __init__(self, batch_size, init_length, velocity, bead_width, bead_height, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size)
        self.velocity = velocity
        self.init_length = init_length
        self.bead_width = bead_width
        self.bead_height = bead_height
        self.dim = 3

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key, time):
        length_updated = self.velocity[0] * time + self.init_length[0]

        # x varies from 0 to length_updated
        x_batch = random.uniform(key,
                                 shape=(self.batch_size, 1),
                                 minval=jnp.array([0.]),
                                 maxval=jnp.array([length_updated]))

        # y is fixed at y_max
        y_batch = jnp.ones((self.batch_size, 1)) * self.bead_width

        # z varies from 0 to bead_height
        z_batch = random.uniform(key + 2,
                                 shape=(self.batch_size, 1),
                                 minval=jnp.array([0.]),
                                 maxval=jnp.array([self.bead_height]))

        volume_batch = jnp.concatenate((x_batch, y_batch, z_batch), axis=1)
        batch = jnp.concatenate((time * jnp.ones((self.batch_size, 1)), volume_batch), axis=1)[0]
        return batch

class ConvectionBoundarySamplerZMax(SeqBaseSampler):
    def __init__(self, batch_size, init_length, velocity, bead_width, bead_height, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size)
        self.velocity = velocity
        self.init_length = init_length
        self.bead_width = bead_width
        self.bead_height = bead_height
        self.dim = 3

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key, time):
        length_updated = self.velocity[0] * time + self.init_length[0]

        # x varies from 0 to length_updated
        x_batch = random.uniform(key,
                                 shape=(self.batch_size, 1),
                                 minval=jnp.array([0.]),
                                 maxval=jnp.array([length_updated]))

        # y varies from 0 to bead_width
        y_batch = random.uniform(key + 1,
                                 shape=(self.batch_size, 1),
                                 minval=jnp.array([0.]),
                                 maxval=jnp.array([self.bead_width]))

        # z is fixed at bead_height
        z_batch = jnp.ones((self.batch_size, 1)) * self.bead_height

        volume_batch = jnp.concatenate((x_batch, y_batch, z_batch), axis=1)
        batch = jnp.concatenate((time * jnp.ones((self.batch_size, 1)), volume_batch), axis=1)[0]
        return batch

class NeumannBoundarySamplerZMax(SeqBaseSampler):
    def __init__(self, batch_size, init_length, velocity, bead_width, bead_height, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size)
        self.velocity = velocity
        self.init_length = init_length
        self.bead_width = bead_width
        self.bead_height = bead_height
        self.dim = 3

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key, time):
        length_updated = self.velocity[0] * time + self.init_length[0]

        # x varies from 0 to length_updated
        x_batch = random.uniform(key,
                                 shape=(self.batch_size, 1),
                                 minval=jnp.array([0.]),
                                 maxval=jnp.array([length_updated]))

        # y varies from 0 to bead_width
        y_batch = random.uniform(key + 1,
                                 shape=(self.batch_size, 1),
                                 minval=jnp.array([0.]),
                                 maxval=jnp.array([self.bead_width]))

        # z is fixed at bead_height
        z_batch = jnp.ones((self.batch_size, 1)) * self.bead_height

        volume_batch = jnp.concatenate((x_batch, y_batch, z_batch), axis=1)
        batch = jnp.concatenate((time * jnp.ones((self.batch_size, 1)), volume_batch), axis=1)[0]
        return batch

# 3D extension of SeqInitialBoundarySampler
class SeqInitialBoundarySampler3D(SeqBaseSampler):
    def __init__(self, batch_size, init_length, velocity, bead_width, bead_height, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size)
        self.velocity = velocity
        self.init_length = init_length
        self.bead_width = bead_width
        self.bead_height = bead_height
        self.dim = 3

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key, time):
        length_updated = self.velocity[0] * time + self.init_length[0]

        # Sampling x from [0, length_updated]
        x_batch = random.uniform(key,
                                 shape=(self.batch_size, 1),
                                 minval=jnp.array([length_updated]),
                                 maxval=jnp.array([length_updated]))

        # Sampling y from [0, bead_width] and z from [0, bead_height]
        y_batch = random.uniform(key + 1,
                                 shape=(self.batch_size, 1),
                                 minval=jnp.array([0.0]),
                                 maxval=jnp.array([self.bead_width]))

        z_batch = random.uniform(key + 2,
                                 shape=(self.batch_size, 1),
                                 minval=jnp.array([0.]),
                                 maxval=jnp.array([self.bead_height]))

        volume_batch = jnp.concatenate((x_batch, y_batch, z_batch), axis=1)
        batch = jnp.concatenate((time * jnp.ones((self.batch_size, 1)), volume_batch), axis=1)[0]
        return batch


############################################### Layerwise samplers #####################################################


class MultiLayerCollocationSampler3D(SeqBaseSampler):
    def __init__(
        self,
        batch_size,
        t_L,                # time interval per layer
        layer_height,       # thickness of each layer
        x_max,
        y_max,
    ):
        super().__init__(batch_size)
        self.t_L = t_L
        self.layer_height = layer_height
        self.x_max = x_max
        self.y_max = y_max

    def __call__(self, step, time):
        key = random.PRNGKey(step)  # Generate key from step
        subkeys = random.split(key, 3)  # Split into 3 subkeys for x, y, z
        batch = self.data_generation(subkeys, time)
        return batch

    def data_generation(self, subkeys, time):
        i = jnp.floor(time / self.t_L).astype(int)
        z_top = (i + 1) * self.layer_height
        x_batch = random.uniform(subkeys[0], shape=(self.batch_size, 1), minval=0.0, maxval=self.x_max)
        y_batch = random.uniform(subkeys[1], shape=(self.batch_size, 1), minval=0.0, maxval=self.y_max)
        z_batch = random.uniform(subkeys[2], shape=(self.batch_size, 1), minval=0.0, maxval=z_top)
        t_col = jnp.ones_like(x_batch) * time
        return jnp.concatenate([t_col, x_batch, y_batch, z_batch], axis=1)


class MultiLayerBedTemperatureSampler3D(SeqBaseSampler):
    def __init__(self, batch_size, t_L, layer_height, x_max, y_max):
        super().__init__(batch_size)
        self.t_L = t_L
        self.layer_height = layer_height
        self.x_max = x_max
        self.y_max = y_max

    def __call__(self, step, time):
        key = random.PRNGKey(step)  # Generate key from step
        subkeys = random.split(key, 2)  # Split into 2 subkeys for x, y
        batch = self.data_generation(subkeys, time)
        return batch

    def data_generation(self, subkeys, time):
        i = jnp.floor(time / self.t_L).astype(int)
        z_top = (i + 1) * self.layer_height
        x_batch = random.uniform(subkeys[0], shape=(self.batch_size, 1), minval=0.0, maxval=self.x_max)
        y_batch =	random.uniform(subkeys[1], shape=(self.batch_size, 1), minval=0.0, maxval=self.y_max)
        z_batch = jnp.zeros((self.batch_size, 1))  # Fixed at z=0
        t_col = jnp.ones_like(x_batch) * time
        return jnp.concatenate([t_col, x_batch, y_batch, z_batch], axis=1)


class MultiLayerTopBoundarySampler3D(SeqBaseSampler):
    """
    Sampler for the "top" boundary z = (i+1)*layer_height, 
    valid for t in [ i*t_L, (i+1)*t_L ).
    """
    def __init__(
        self, 
        batch_size, 
        t_L, 
        layer_height, 
        x_max, 
        y_max
    ):
        super().__init__(batch_size)
        self.t_L = t_L
        self.layer_height = layer_height
        self.x_max = x_max
        self.y_max = y_max

    def __call__(self, step, time):
        key = random.PRNGKey(step)  # Generate key from step
        subkeys = random.split(key, 2)  # Split into 2 subkeys for x and y
        batch = self.data_generation(subkeys, time)
        return batch

    def data_generation(self, subkeys, time):
        i = jnp.floor(time / self.t_L).astype(int)
        z_top = (i + 1) * self.layer_height
        x_batch = random.uniform(subkeys[0], shape=(self.batch_size, 1), minval=0.0, maxval=self.x_max)
        y_batch = random.uniform(subkeys[1], shape=(self.batch_size, 1), minval=0.0, maxval=self.y_max)
        z_batch = jnp.ones((self.batch_size, 1)) * z_top  # Fixed at z = z_top
        t_col = jnp.ones_like(x_batch) * time
        return jnp.concatenate([t_col, x_batch, y_batch, z_batch], axis=1)

class MultiLayerX0Sampler3D(SeqBaseSampler):
    """
    Sampler for boundary x=0, 
    with y in [0, y_max], z in [0, (i+1)*layer_height].
    """
    def __init__(
        self, 
        batch_size,
        t_L,
        layer_height,
        x_max,
        y_max
    ):
        super().__init__(batch_size)
        self.t_L = t_L
        self.layer_height = layer_height
        self.x_max = x_max
        self.y_max = y_max

    def __call__(self, step, time):
        key = random.PRNGKey(step)  # Generate key from step
        subkeys = random.split(key, 2)  # Split into 2 subkeys for y and z
        batch = self.data_generation(subkeys, time)
        return batch

    def data_generation(self, subkeys, time):
        i = jnp.floor(time / self.t_L).astype(int)
        z_top = (i + 1) * self.layer_height
        x_batch = jnp.zeros((self.batch_size, 1))  # Fixed at x=0
        y_batch = random.uniform(subkeys[0], shape=(self.batch_size, 1), minval=0.0, maxval=self.y_max)
        z_batch = random.uniform(subkeys[1], shape=(self.batch_size, 1), minval=0.0, maxval=z_top)
        t_col = jnp.ones_like(x_batch) * time
        return jnp.concatenate([t_col, x_batch, y_batch, z_batch], axis=1)


class MultiLayerXMaxSampler3D(SeqBaseSampler):
    """
    Sampler for boundary x=x_max, 
    with y in [0, y_max], z in [0, (i+1)*layer_height].
    """
    def __init__(
        self,
        batch_size,
        t_L,
        layer_height,
        x_max,
        y_max
    ):
        super().__init__(batch_size)
        self.t_L = t_L
        self.layer_height = layer_height
        self.x_max = x_max
        self.y_max = y_max

    def __call__(self, step, time):
        key = random.PRNGKey(step)  # Generate key from step
        subkeys = random.split(key, 2)  # Split into 2 subkeys for y and z
        batch = self.data_generation(subkeys, time)
        return batch

    def data_generation(self, subkeys, time):
        i = jnp.floor(time / self.t_L).astype(int)
        z_top = (i + 1) * self.layer_height
        x_batch = jnp.ones((self.batch_size, 1)) * self.x_max  # Fixed at x=x_max
        y_batch = random.uniform(subkeys[0], shape=(self.batch_size, 1), minval=0.0, maxval=self.y_max)
        z_batch = random.uniform(subkeys[1], shape=(self.batch_size, 1), minval=0.0, maxval=z_top)
        t_col = jnp.ones_like(x_batch) * time
        return jnp.concatenate([t_col, x_batch, y_batch, z_batch], axis=1)


class MultiLayerY0Sampler3D(SeqBaseSampler):
    """
    y=0 boundary, with x in [0, x_max], z in [0, (i+1)*layer_height].
    """
    def __init__(
        self,
        batch_size,
        t_L,
        layer_height,
        x_max,
        y_max
    ):
        super().__init__(batch_size)
        self.t_L = t_L
        self.layer_height = layer_height
        self.x_max = x_max
        self.y_max = y_max

    def __call__(self, step, time):
        key = random.PRNGKey(step)  # Generate key from step
        subkeys = random.split(key, 2)  # Split into 2 subkeys for x and z
        batch = self.data_generation(subkeys, time)
        return batch

    def data_generation(self, subkeys, time):
        i = jnp.floor(time / self.t_L).astype(int)
        z_top = (i + 1) * self.layer_height
        x_batch = random.uniform(subkeys[0], shape=(self.batch_size, 1), minval=0.0, maxval=self.x_max)
        y_batch = jnp.zeros((self.batch_size, 1))  # Fixed at y=0
        z_batch = random.uniform(subkeys[1], shape=(self.batch_size, 1), minval=0.0, maxval=z_top)
        t_col = jnp.ones_like(x_batch) * time
        return jnp.concatenate([t_col, x_batch, y_batch, z_batch], axis=1)

class MultiLayerYMaxSampler3D(SeqBaseSampler):
    """
    y=y_max boundary, with x in [0, x_max], z in [0, (i+1)*layer_height].
    """
    def __init__(
        self,
        batch_size,
        t_L,
        layer_height,
        x_max,
        y_max
    ):
        super().__init__(batch_size)
        self.t_L = t_L
        self.layer_height = layer_height
        self.x_max = x_max
        self.y_max = y_max

    def __call__(self, step, time):
        key = random.PRNGKey(step)  # Generate key from step
        subkeys = random.split(key, 2)  # Split into 2 subkeys for x and z
        batch = self.data_generation(subkeys, time)
        return batch

    def data_generation(self, subkeys, time):
        i = jnp.floor(time / self.t_L).astype(int)
        z_top = (i + 1) * self.layer_height
        x_batch = random.uniform(subkeys[0], shape=(self.batch_size, 1), minval=0.0, maxval=self.x_max)
        y_batch = jnp.ones((self.batch_size, 1)) * self.y_max  # Fixed at y=y_max
        z_batch = random.uniform(subkeys[1], shape=(self.batch_size, 1), minval=0.0, maxval=z_top)
        t_col = jnp.ones_like(x_batch) * time
        return jnp.concatenate([t_col, x_batch, y_batch, z_batch], axis=1)

class MultiLayerInitialConditionSampler3D(SeqBaseSampler):
    def __init__(
        self,
        batch_size,
        t_L,
        layer_height,
        x_max,
        y_max
    ):
        super().__init__(batch_size)
        self.t_L = t_L
        self.layer_height = layer_height
        self.x_max = x_max
        self.y_max = y_max

    def __call__(self, step, time):
        key = random.PRNGKey(step)  # Generate key from step
        subkeys = random.split(key, 3)  # Split into 3 subkeys for x, y, z
        batch = self.data_generation(subkeys, time)
        return batch

    def data_generation(self, subkeys, time):
        i = jnp.floor(time / self.t_L).astype(int)
        z_min = i * self.layer_height
        z_max = (i + 1) * self.layer_height
        x_batch = random.uniform(subkeys[0], shape=(self.batch_size, 1), minval=0.0, maxval=self.x_max)
        y_batch = random.uniform(subkeys[1], shape=(self.batch_size, 1), minval=0.0, maxval=self.y_max)
        z_batch = random.uniform(subkeys[2], shape=(self.batch_size, 1), minval=z_min, maxval=z_max)
        t_col = jnp.ones_like(x_batch) * (i * self.t_L)
        print(f"Step: {i}, t: {t_col[0,0]}, z_range: [{z_min}, {z_max}]")
        return jnp.concatenate([t_col, x_batch, y_batch, z_batch], axis=1)

class MultiLayerPrevInitialConditionSampler3D(SeqBaseSampler):
    def __init__(
        self,
        batch_size,
        t_L,
        layer_height,
        x_max,
        y_max
    ):
        super().__init__(batch_size)
        self.t_L = t_L
        self.layer_height = layer_height
        self.x_max = x_max
        self.y_max = y_max

    def __call__(self, step, time):
        key = random.PRNGKey(step)  # Generate key from step
        subkeys = random.split(key, 3)  # Split into 3 subkeys for x, y, z
        batch = self.data_generation(subkeys, time)
        return batch

    def data_generation(self, subkeys, time):
        i = jnp.floor(time / self.t_L).astype(int)
        z_min = 0.0
        z_max = i * self.layer_height
        x_batch = random.uniform(subkeys[0], shape=(self.batch_size, 1), minval=0.0, maxval=self.x_max)
        y_batch = random.uniform(subkeys[1], shape=(self.batch_size, 1), minval=0.0, maxval=self.y_max)
        z_batch = random.uniform(subkeys[2], shape=(self.batch_size, 1), minval=z_min, maxval=z_max)
        t_col = jnp.ones_like(x_batch) * time
        return jnp.concatenate([t_col, x_batch, y_batch, z_batch], axis=1)



        



















## Single layer version without pmap (for visualization)

# from abc import ABC, abstractmethod
# from functools import partial

# import jax.numpy as jnp
# from jax import random, pmap, local_device_count

# from torch.utils.data import Dataset


# import jax.numpy as jnp
# import jax.random as random

# class BaseSampler(Dataset):
#     def __init__(self, batch_size, rng_key=random.PRNGKey(1234)):
#         self.batch_size = batch_size
#         self.key = rng_key
#         self.num_devices = local_device_count()

#     def __getitem__(self, index):
#         "Generate one batch of data"
#         self.key, subkey = random.split(self.key)
#         keys = random.split(subkey, self.num_devices)
#         batch = self.data_generation(keys)
#         return batch

#     def data_generation(self, key):
#         raise NotImplementedError("Subclasses should implement this!")

# # Base class for all samplers
# class SeqBaseSampler:
#     def __init__(self, batch_size):
#         self.batch_size = batch_size

#     def __call__(self, step, time):
#         keys = random.split(random.PRNGKey(step), 1)  # Adjusted for single device
#         batch = self.data_generation(keys, time)
#         return batch

#     def data_generation(self, key, time):
#         raise NotImplementedError("Subclasses should implement this!")

# class StepIndexSampler(BaseSampler):
#     def __init__(self, n_test:int, batch_size, rng_key=random.PRNGKey(1234)):
#         super().__init__(batch_size, rng_key)
#         self.n_test=n_test
#         self.dim = 1

#     @partial(pmap, static_broadcasted_argnums=(0,))
#     def data_generation(self, key):
#         batch = random.categorical(key,
#                                    logits=jnp.ones(int(self.n_test)),
#                                    axis=0,
#                                    shape=(self.batch_size, 1))
#         return batch


# class  UniformSampler(BaseSampler):
#     def __init__(self, dom, batch_size, rng_key=random.PRNGKey(1234)):
#         super().__init__(batch_size, rng_key)
#         self.dom = dom
#         self.dim = dom.shape[0]

#     @partial(pmap, static_broadcasted_argnums=(0,))
#     def data_generation(self, key):
#         "Generates data containing batch_size samples"
#         batch = random.uniform(
#             key,
#             shape=(self.batch_size, self.dim),
#             minval=self.dom[:, 0],
#             maxval=self.dom[:, 1],
#         )

#         return batch

# # 3D extension of SeqCollocationSampler
# class SeqCollocationSampler3D(SeqBaseSampler):
#     def __init__(self, batch_size, init_length, velocity, bead_width, bead_height, rng_key=random.PRNGKey(1234)):
#         super().__init__(batch_size)
#         self.velocity = velocity  # Velocity in the x-direction
#         self.init_length = init_length  # Initial length in x
#         self.bead_width = bead_width  # Width in y
#         self.bead_height = bead_height  # Height in z
#         self.dim = 3  # Now we have three spatial dimensions (x, y, z)

#     def data_generation(self, key, time):
#         # Evolving x dimension based on the time step and velocity
#         length_updated = self.velocity[0] * time + self.init_length[0]

#         # Sampling x from [0, length_updated]
#         x_batch = random.uniform(key,
#                                  shape=(self.batch_size, 1),
#                                  minval=jnp.array([0.]),
#                                  maxval=jnp.array([length_updated]))

#         # Sampling y from [0, bead_width] (constant)
#         y_batch = random.uniform(key + 1,
#                                  shape=(self.batch_size, 1),
#                                  minval=jnp.array([0.]),
#                                  maxval=jnp.array([self.bead_width]))

#         # Sampling z from [0, bead_height] (constant)
#         z_batch = random.uniform(key + 2,
#                                  shape=(self.batch_size, 1),
#                                  minval=jnp.array([0.]),
#                                  maxval=jnp.array([self.bead_height]))

#         # Concatenating the time and spatial coordinates (x, y, z)
#         volume_batch = jnp.concatenate((x_batch, y_batch, z_batch), axis=1)
#         batch = jnp.concatenate((time * jnp.ones((self.batch_size, 1)), volume_batch), axis=1)
#         return batch

# # 3D extension of SeqNeumanCollocationSampler_B1
# class SeqNeumanCollocationSampler_B13D(SeqBaseSampler):
#     def __init__(self, batch_size, init_length, velocity, bead_width, bead_height, rng_key=random.PRNGKey(1234)):
#         super().__init__(batch_size)
#         self.velocity = velocity
#         self.init_length = init_length
#         self.bead_width = bead_width
#         self.bead_height = bead_height
#         self.dim = 3

#     def data_generation(self, key, time):
#         length_updated = self.velocity[0] * time + self.init_length[0]

#         # Sampling x from [0, length_updated]
#         x_batch = random.uniform(key,
#                                  shape=(self.batch_size, 1),
#                                  minval=jnp.array([0.]),
#                                  maxval=jnp.array([length_updated]))

#         # Sampling y from [0, bead_width] (constant)
#         y_batch = random.uniform(key + 1,
#                                  shape=(self.batch_size, 1),
#                                  minval=jnp.array([0.]),
#                                  maxval=jnp.array([self.bead_width]))

#         # Neumann boundary condition at z = bead_height (fixed)
#         z_batch = jnp.ones((self.batch_size, 1)) * self.bead_height

#         volume_batch = jnp.concatenate((x_batch, y_batch, z_batch), axis=1)
#         batch = jnp.concatenate((time * jnp.ones((self.batch_size, 1)), volume_batch), axis=1)
#         return batch

# # 3D extension of SeqNeumanCollocationSampler_B2
# class SeqNeumanCollocationSampler_B23D(SeqBaseSampler):
#     def __init__(self, batch_size, init_length, velocity, bead_width, bead_height, rng_key=random.PRNGKey(1234)):
#         super().__init__(batch_size)
#         self.velocity = velocity
#         self.init_length = init_length
#         self.bead_width = bead_width
#         self.bead_height = bead_height
#         self.dim = 3

#     def data_generation(self, key, time):
#         length_updated = self.velocity[0] * time + self.init_length[0]

#         # Sampling x from [0, length_updated]
#         x_batch = random.uniform(key,
#                                  shape=(self.batch_size, 1),
#                                  minval=jnp.array([0.]),
#                                  maxval=jnp.array([length_updated]))

#         # Sampling y from [0, bead_width] (constant)
#         y_batch = random.uniform(key + 1,
#                                  shape=(self.batch_size, 1),
#                                  minval=jnp.array([0.]),
#                                  maxval=jnp.array([self.bead_width]))

#         # Neumann boundary condition at z = 0 (fixed)
#         z_batch = jnp.zeros((self.batch_size, 1))

#         volume_batch = jnp.concatenate((x_batch, y_batch, z_batch), axis=1)
#         batch = jnp.concatenate((time * jnp.ones((self.batch_size, 1)), volume_batch), axis=1)
#         return batch

# class DepositionFrontSampler3D(SeqBaseSampler):
#     def __init__(self, batch_size, init_length, velocity, bead_width, bead_height, rng_key=random.PRNGKey(1234)):
#         super().__init__(batch_size)
#         self.velocity = velocity
#         self.init_length = init_length
#         self.bead_width = bead_width
#         self.bead_height = bead_height
#         self.dim = 3

#     def data_generation(self, key, time):
#         length_updated = self.velocity[0] * time + self.init_length[0]

#         # x is fixed at the evolving front
#         x_batch = jnp.ones((self.batch_size, 1)) * length_updated

#         # y varies from 0 to bead_width
#         y_batch = random.uniform(key + 1,
#                                  shape=(self.batch_size, 1),
#                                  minval=jnp.array([0.]),
#                                  maxval=jnp.array([self.bead_width]))

#         # z varies from 0 to bead_height
#         z_batch = random.uniform(key + 2,
#                                  shape=(self.batch_size, 1),
#                                  minval=jnp.array([0.]),
#                                  maxval=jnp.array([self.bead_height]))

#         volume_batch = jnp.concatenate((x_batch, y_batch, z_batch), axis=1)
#         batch = jnp.concatenate((time * jnp.ones((self.batch_size, 1)), volume_batch), axis=1)
#         return batch


# class BedTemperatureSampler3D(SeqBaseSampler):
#     def __init__(self, batch_size, init_length, velocity, bead_width, rng_key=random.PRNGKey(1234)):
#         super().__init__(batch_size)
#         self.velocity = velocity
#         self.init_length = init_length
#         self.bead_width = bead_width
#         self.dim = 3

#     def data_generation(self, key, time):
#         length_updated = self.velocity[0] * time + self.init_length[0]

#         # x varies from 0 to length_updated
#         x_batch = random.uniform(key,
#                                  shape=(self.batch_size, 1),
#                                  minval=jnp.array([0.]),
#                                  maxval=jnp.array([length_updated]))

#         # y varies from 0 to bead_width
#         y_batch = random.uniform(key + 1,
#                                  shape=(self.batch_size, 1),
#                                  minval=jnp.array([0.]),
#                                  maxval=jnp.array([self.bead_width]))

#         # z is fixed at 0 (bed surface)
#         z_batch = jnp.zeros((self.batch_size, 1))

#         volume_batch = jnp.concatenate((x_batch, y_batch, z_batch), axis=1)
#         batch = jnp.concatenate((time * jnp.ones((self.batch_size, 1)), volume_batch), axis=1)
#         return batch

# # Enforcing du/dx = 0 at x = 0
# class NeumannBoundarySamplerX0(SeqBaseSampler):
#     def __init__(self, batch_size, bead_width, bead_height, rng_key=random.PRNGKey(1234)):
#         super().__init__(batch_size)
#         self.bead_width = bead_width
#         self.bead_height = bead_height

#     def data_generation(self, key, time):
#         # x is fixed at 0
#         x_batch = jnp.zeros((self.batch_size, 1))

#         # y varies from 0 to bead_width
#         y_batch = random.uniform(key + 1,
#                                  shape=(self.batch_size, 1),
#                                  minval=jnp.array([0.]),
#                                  maxval=jnp.array([self.bead_width]))

#         # z varies from 0 to bead_height
#         z_batch = random.uniform(key + 2,
#                                  shape=(self.batch_size, 1),
#                                  minval=jnp.array([0.]),
#                                  maxval=jnp.array([self.bead_height]))

#         volume_batch = jnp.concatenate((x_batch, y_batch, z_batch), axis=1)
#         batch = jnp.concatenate((time * jnp.ones((self.batch_size, 1)), volume_batch), axis=1)
#         return batch

# class ConvectionBoundarySamplerYMin(SeqBaseSampler):
#     def __init__(self, batch_size, init_length, velocity, bead_width, bead_height, rng_key=random.PRNGKey(1234)):
#         super().__init__(batch_size)
#         self.velocity = velocity
#         self.init_length = init_length
#         self.bead_width = bead_width
#         self.bead_height = bead_height
#         self.dim = 3

#     def data_generation(self, key, time):
#         length_updated = self.velocity[0] * time + self.init_length[0]

#         # x varies from 0 to length_updated
#         x_batch = random.uniform(key,
#                                  shape=(self.batch_size, 1),
#                                  minval=jnp.array([0.]),
#                                  maxval=jnp.array([length_updated]))
#         # y is fixed at 0
#         y_batch = jnp.zeros((self.batch_size, 1))

#         # z varies from 0 to bead_height
#         z_batch = random.uniform(key + 2,
#                                  shape=(self.batch_size, 1),
#                                  minval=jnp.array([0.]),
#                                  maxval=jnp.array([self.bead_height]))

#         volume_batch = jnp.concatenate((x_batch, y_batch, z_batch), axis=1)
#         batch = jnp.concatenate((time * jnp.ones((self.batch_size, 1)), volume_batch), axis=1)
#         return batch

# # Convection boundary condition
# class ConvectionBoundarySamplerYMax(SeqBaseSampler):
#     def __init__(self, batch_size, init_length, velocity, bead_width, bead_height, rng_key=random.PRNGKey(1234)):
#         super().__init__(batch_size)
#         self.velocity = velocity
#         self.init_length = init_length
#         self.bead_width = bead_width
#         self.bead_height = bead_height
#         self.dim = 3

#     def data_generation(self, key, time):
#         length_updated = self.velocity[0] * time + self.init_length[0]

#         # x varies from 0 to length_updated
#         x_batch = random.uniform(key,
#                                  shape=(self.batch_size, 1),
#                                  minval=jnp.array([0.]),
#                                  maxval=jnp.array([length_updated]))

#         # y is fixed at y_max
#         y_batch = jnp.ones((self.batch_size, 1)) * self.bead_width

#         # z varies from 0 to bead_height
#         z_batch = random.uniform(key + 2,
#                                  shape=(self.batch_size, 1),
#                                  minval=jnp.array([0.]),
#                                  maxval=jnp.array([self.bead_height]))

#         volume_batch = jnp.concatenate((x_batch, y_batch, z_batch), axis=1)
#         batch = jnp.concatenate((time * jnp.ones((self.batch_size, 1)), volume_batch), axis=1)
#         return batch

# class ConvectionBoundarySamplerZMax(SeqBaseSampler):
#     def __init__(self, batch_size, init_length, velocity, bead_width, bead_height, rng_key=random.PRNGKey(1234)):
#         super().__init__(batch_size)
#         self.velocity = velocity
#         self.init_length = init_length
#         self.bead_width = bead_width
#         self.bead_height = bead_height
#         self.dim = 3

#     def data_generation(self, key, time):
#         length_updated = self.velocity[0] * time + self.init_length[0]

#         # x varies from 0 to length_updated
#         x_batch = random.uniform(key,
#                                  shape=(self.batch_size, 1),
#                                  minval=jnp.array([0.]),
#                                  maxval=jnp.array([length_updated]))

#         # y varies from 0 to bead_width
#         y_batch = random.uniform(key + 1,
#                                  shape=(self.batch_size, 1),
#                                  minval=jnp.array([0.]),
#                                  maxval=jnp.array([self.bead_width]))

#         # z is fixed at bead_height
#         z_batch = jnp.ones((self.batch_size, 1)) * self.bead_height

#         volume_batch = jnp.concatenate((x_batch, y_batch, z_batch), axis=1)
#         batch = jnp.concatenate((time * jnp.ones((self.batch_size, 1)), volume_batch), axis=1)
#         return batch

# class NeumannBoundarySamplerZMax(SeqBaseSampler):
#     def __init__(self, batch_size, init_length, velocity, bead_width, bead_height, rng_key=random.PRNGKey(1234)):
#         super().__init__(batch_size)
#         self.velocity = velocity
#         self.init_length = init_length
#         self.bead_width = bead_width
#         self.bead_height = bead_height
#         self.dim = 3

#     def data_generation(self, key, time):
#         length_updated = self.velocity[0] * time + self.init_length[0]

#         # x varies from 0 to length_updated
#         x_batch = random.uniform(key,
#                                  shape=(self.batch_size, 1),
#                                  minval=jnp.array([0.]),
#                                  maxval=jnp.array([length_updated]))

#         # y varies from 0 to bead_width
#         y_batch = random.uniform(key + 1,
#                                  shape=(self.batch_size, 1),
#                                  minval=jnp.array([0.]),
#                                  maxval=jnp.array([self.bead_width]))

#         # z is fixed at bead_height
#         z_batch = jnp.ones((self.batch_size, 1)) * self.bead_height

#         volume_batch = jnp.concatenate((x_batch, y_batch, z_batch), axis=1)
#         batch = jnp.concatenate((time * jnp.ones((self.batch_size, 1)), volume_batch), axis=1)
#         return batch

# # 3D extension of SeqInitialBoundarySampler
# class SeqInitialBoundarySampler3D(SeqBaseSampler):
#     def __init__(self, batch_size, init_length, velocity, bead_width, bead_height, rng_key=random.PRNGKey(1234)):
#         super().__init__(batch_size)
#         self.velocity = velocity
#         self.init_length = init_length
#         self.bead_width = bead_width
#         self.bead_height = bead_height
#         self.dim = 3

#     def data_generation(self, key, time):
#         length_updated = self.velocity[0] * time + self.init_length[0]

#         # Sampling x from [0, length_updated]
#         x_batch = random.uniform(key,
#                                  shape=(self.batch_size, 1),
#                                  minval=jnp.array([length_updated]),
#                                  maxval=jnp.array([length_updated]))

#         # Sampling y from [0, bead_width] and z from [0, bead_height]
#         y_batch = random.uniform(key + 1,
#                                  shape=(self.batch_size, 1),
#                                  minval=jnp.array([0.0]),
#                                  maxval=jnp.array([self.bead_width]))

#         z_batch = random.uniform(key + 2,
#                                  shape=(self.batch_size, 1),
#                                  minval=jnp.array([0.]),
#                                  maxval=jnp.array([self.bead_height]))

#         volume_batch = jnp.concatenate((x_batch, y_batch, z_batch), axis=1)
#         batch = jnp.concatenate((time * jnp.ones((self.batch_size, 1)), volume_batch), axis=1)
#         return batch