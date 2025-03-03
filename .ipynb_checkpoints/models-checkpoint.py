
# ### Single layer implementation ###

# from functools import partial
# import jax.numpy as jnp
# from jax import jit, grad, vmap
# from A3DPINN.models import ForwardIVP
# from A3DPINN.evaluator import BaseEvaluator
# from A3DPINN.samplers import (
#     SeqCollocationSampler3D,
#     SeqNeumanCollocationSampler_B13D,
#     SeqNeumanCollocationSampler_B23D,
#     DepositionFrontSampler3D,
#     BedTemperatureSampler3D,
#     NeumannBoundarySamplerX0,
#     BoundarySamplerX0,
#     SeqInitialBoundarySampler3D,
#     ConvectionBoundarySamplerYMax,
#     ConvectionBoundarySamplerZMax,
#     ConvectionBoundarySamplerYMin,
#     NeumannBoundarySamplerZMax
# )
# from matplotlib import pyplot as plt

# class A3DHeatTransfer(ForwardIVP):
#     def __init__(self, config):
#         super().__init__(config)
#         self.config = config
#         self.u_max = config.process_conditions.deposition_temperature
#         self.t_max = config.dimensions.t_max
#         self.x_max = config.dimensions.x_max
#         self.y_max = config.dimensions.y_max
#         self.z_max = config.dimensions.z_max
        
#         # Processing conditions in scaled version
#         self.deposition_temperature_scaled = config.process_conditions.deposition_temperature / self.u_max
#         self.bed_temperature_scaled = config.process_conditions.bed_temperature / self.u_max
        
#         # Define material properties
#         self.rho = config.material_properties.density
#         self.C = config.material_properties.specific_heat

#         # Thermal conductivity coefficients (temperature-dependent)
#         self.k_0_xx = config.material_properties.k_0_xx
#         self.k_1_xx = config.material_properties.k_1_xx
#         self.k_0_yy = config.material_properties.k_0_yy
#         self.k_1_yy = config.material_properties.k_1_yy
#         self.k_0_zz = config.material_properties.k_0_zz
#         self.k_1_zz = config.material_properties.k_1_zz

#         self.h = config.material_properties.heat_transfer_coefficient
#         self.emissivity = config.material_properties.emissivity
#         self.sigma = 5.670374419e-8  # Stefan-Boltzmann constant in W/m^2·K^4
        
#         self.alpha = 1.0 / (self.rho * self.C)
        
#         # Scaled alpha
#         self.alpha_xx_scaled = self.alpha * self.t_max / (self.x_max ** 2)
#         self.alpha_yy_scaled = self.alpha * self.t_max / (self.y_max ** 2)
#         self.alpha_zz_scaled = self.alpha * self.t_max / (self.z_max ** 2)
        
#         # Scaling factors for h
#         # self.h_scaled = self.h
#         # self.sigma_scaled = self.emissivity * self.sigma * self.u_max ** 4


#         self.h_scaled = (self.h * self.x_max) / self.k_0_xx
#         self.sigma_scaled = (self.emissivity * self.sigma
#                              * (self.u_max**3) * self.x_max / self.k_0_xx)

#         # Define scaled versions of processing conditions
#         self.print_speed_scaled = config.process_conditions.print_speed * self.t_max / self.x_max
#         self.init_length_scaled = config.process_conditions.init_length / self.x_max
#         self.bead_width_scaled = config.process_conditions.bead_width / self.y_max
#         self.bead_height_scaled = config.process_conditions.bead_height / self.z_max
#         self.velocity_vector = config.process_conditions.velocity_vector
        
#         self.ambient_convection_temp = config.process_conditions.ambient_convection_temp / self.u_max
#         self.ambient_radiation_temp = config.process_conditions.ambient_radiation_temp / self.u_max
        
#         # Samplers - sequential collocation samplers in 3D
#         self.seqSampler = SeqCollocationSampler3D(
#             config.training.batch_size_per_device,
#             self.init_length_scaled,
#             self.print_speed_scaled * self.velocity_vector,
#             self.bead_width_scaled,
#             self.bead_height_scaled
#         )
        
#         self.deposition_front_sampler = DepositionFrontSampler3D(
#             config.training.batch_size_per_device,
#             self.init_length_scaled,
#             self.print_speed_scaled * self.velocity_vector,
#             self.bead_width_scaled,
#             self.bead_height_scaled
#         )

#         self.bed_temperature_sampler = BedTemperatureSampler3D(
#             config.training.batch_size_per_device,
#             self.init_length_scaled,
#             self.print_speed_scaled * self.velocity_vector,
#             self.bead_width_scaled
#         )

#         self.neumann_boundary_sampler_x0 = NeumannBoundarySamplerX0(
#             config.training.batch_size_per_device,
#             self.bead_width_scaled,
#             self.bead_height_scaled
#         )

#         self.boundary_sampler_x0 = BoundarySamplerX0(
#             config.training.batch_size_per_device,
#             self.bead_width_scaled,
#             self.bead_height_scaled
#         )

#         self.seqInitialBoundarySampler = SeqInitialBoundarySampler3D(
#             config.training.batch_size_per_device,
#             self.init_length_scaled,
#             self.print_speed_scaled * self.velocity_vector,
#             self.bead_width_scaled,
#             self.bead_height_scaled
#         )

#         # Instantiate the convection boundary samplers
#         self.convection_boundary_sampler_ymax = ConvectionBoundarySamplerYMax(
#             config.training.batch_size_per_device,
#             self.init_length_scaled,
#             self.print_speed_scaled * self.velocity_vector,
#             self.bead_width_scaled,
#             self.bead_height_scaled
#         )

#         self.convection_boundary_sampler_ymin = ConvectionBoundarySamplerYMin(
#             config.training.batch_size_per_device,
#             self.init_length_scaled,
#             self.print_speed_scaled * self.velocity_vector,
#             self.bead_width_scaled,
#             self.bead_height_scaled
#         )

#         self.convection_boundary_sampler_zmax = ConvectionBoundarySamplerZMax(
#             config.training.batch_size_per_device,
#             self.init_length_scaled,
#             self.print_speed_scaled * self.velocity_vector,
#             self.bead_width_scaled,
#             self.bead_height_scaled
#         )

#         self.neumann_boundary_sampler_zmax = NeumannBoundarySamplerZMax(
#             config.training.batch_size_per_device,
#             self.init_length_scaled,
#             self.print_speed_scaled * self.velocity_vector,
#             self.bead_width_scaled,
#             self.bead_height_scaled
#         )

#         # Predictions over a grid
#         self.u_pred_fn = vmap(self.u_net, (None, None, 0, 0, 0))
#         self.delta_time_lossMap = vmap(self.strong_res_net, (None, 0, 0, 0, 0))
#         self.delta_time_deposition_front_loss_map = vmap(self.loss_deposition_front, (None, 0, 0, 0, 0))
#         self.delta_time_bed_temperature_loss_map = vmap(self.loss_bed_temperature, (None, 0, 0, 0, 0))
#         self.delta_time_neumann_loss_map_x0 = vmap(self.loss_neumann_x0, (None, 0, 0, 0, 0))
#         self.delta_time_boundary_loss_map_x0 = vmap(self.loss_convection_x0, (None, 0, 0, 0, 0))
#         self.delta_time_neumann_loss_map_zmax = vmap(self.loss_neumann_zmax, (None, 0, 0, 0, 0))
#         self.delta_time_neumann_loss_map_ymin = vmap(self.loss_neumann_ymin, (None, 0, 0, 0, 0))
#         self.delta_time_neumann_loss_map_ymax = vmap(self.loss_neumann_ymax, (None, 0, 0, 0, 0))

#         # Loss maps for convection boundaries
#         self.delta_time_convection_loss_map_ymax = vmap(self.loss_convection_ymax, (None, 0, 0, 0, 0))
#         self.delta_time_convection_loss_map_ymin = vmap(self.loss_convection_ymin, (None, 0, 0, 0, 0))
#         self.delta_time_convection_loss_map_zmax = vmap(self.loss_convection_zmax, (None, 0, 0, 0, 0))
        
#         # For evaluation
#         self.evalfn_ = vmap(self.u_net, (None, 0, None, None, None))
        
#         self.bs = config.training.batch_size_per_device


#     def u_net(self, params, t, x, y, z):
#         inputs = jnp.stack([t, x, y, z])  # Now includes z
#         u = self.state.apply_fn(params, inputs)
#         return u[0]

#     def k_xx(self, u):
#         return self.k_0_xx + self.k_1_xx * u * self.u_max
#         # return self.k_0_xx if self.k_1_xx == 0 else self.k_0_xx + self.k_1_xx * u * self.u_max

#     def k_yy(self, u):
#         return self.k_0_yy + self.k_1_yy * u * self.u_max
#         # return self.k_0_yy if self.k_1_yy == 0 else self.k_0_yy + self.k_1_yy * u * self.u_max

#     def k_zz(self, u):
#         return self.k_0_zz + self.k_1_zz * u * self.u_max
#         # return self.k_0_zz if self.k_1_zz == 0 else self.k_0_zz + self.k_1_zz * u * self.u_max

#     def strong_res_net(self, params, t, x, y, z):
#         u_pred = self.u_net(params, t, x, y, z)
#         u_t = grad(self.u_net, argnums=1)(params, t, x, y, z)
#         u_xx = grad(grad(self.u_net, argnums=2), argnums=2)(params, t, x, y, z)
#         u_yy = grad(grad(self.u_net, argnums=3), argnums=3)(params, t, x, y, z)
#         u_zz = grad(grad(self.u_net, argnums=4), argnums=4)(params, t, x, y, z)
        
#         k_xx = self.k_xx(u_pred)
#         k_yy = self.k_yy(u_pred)
#         k_zz = self.k_zz(u_pred)

#         term_x = self.alpha_xx_scaled * k_xx * u_xx
#         term_y = self.alpha_yy_scaled * k_yy * u_yy
#         term_z = self.alpha_zz_scaled * k_zz * u_zz

#         res = term_x + term_y + term_z - u_t
        
#         return res
    
#     def loss_deposition_front(self, params, t, x, y, z):
#         u_pred = self.u_net(params, t, x, y, z)
#         u_sol = self.deposition_temperature_scaled
#         return u_pred - u_sol  # Enforce deposition temperature at the moving front

#     def loss_bed_temperature(self, params, t, x, y, z):
#         u_pred = self.u_net(params, t, x, y, z)
#         u_sol = self.bed_temperature_scaled
#         return u_pred - u_sol  # Enforce bed temperature at z = 0
    
#     def loss_neumann_x0(self, params, t, x, y, z):
#         u_x = grad(self.u_net, argnums=2)(params, t, x, y, z)
#         return u_x  # Enforce du/dx = 0 at x = 0

#     # # Without radiation
#     # def loss_convection_x0(self, params, t, x, y, z):
#     #     # Compute the derivative du/dx at x = 0
#     #     u_x = grad(self.u_net, argnums=2)(params, t, x, y, z)
#     #     u_pred = self.u_net(params, t, x, y, z)
#     #     u_inf = self.ambient_convection_temp
    
#     #     # Compute the convection term
#     #     convection_term = self.h_scaled * (u_pred - u_inf)
    
#     #     # Enforce the convection boundary condition
#     #     k_xx = self.k_xx(u_pred)
#     #     k_xx_scaled = k_xx / self.x_max  # Correct scaling
#     #     return k_xx_scaled * u_x - convection_term

#     # With radiation
#     def loss_convection_x0(self, params, t, x, y, z):
#         u_x = grad(self.u_net, argnums=2)(params, t, x, y, z)
#         u_pred = self.u_net(params, t, x, y, z)
#         u_inf_conv = self.ambient_convection_temp
#         u_inf_rad = self.ambient_radiation_temp
    
#         convection_term = self.h_scaled * (u_pred - u_inf_conv)
#         radiation_term = self.sigma_scaled * (u_pred**4 - u_inf_rad**4)
    
#         k_xx = self.k_xx(u_pred)
#         k_xx_scaled = k_xx / self.x_max
#         return k_xx_scaled * u_x - convection_term - radiation_term  # k*dT/dx = h(T-T_inf) + radiation

#     # # Without Radiation
#     # def loss_convection_ymin(self, params, t, x, y, z):
#     #     # Compute the derivative du/dy at y = 0
#     #     u_y = grad(self.u_net, argnums=3)(params, t, x, y, z)
#     #     u_pred = self.u_net(params, t, x, y, z)
#     #     u_inf = self.ambient_convection_temp
    
#     #     # Compute the convection term
#     #     convection_term = self.h_scaled * (u_pred - u_inf)
    
#     #     # Enforce the convection
#     #     k_yy = self.k_yy(u_pred)
#     #     k_yy_scaled = k_yy / self.y_max
#     #     return k_yy_scaled * u_y - convection_term

#     # With Radiation
#     def loss_convection_ymin(self, params, t, x, y, z):
#         u_y = grad(self.u_net, argnums=3)(params, t, x, y, z)
#         u_pred = self.u_net(params, t, x, y, z)
#         u_inf_conv = self.ambient_convection_temp
#         u_inf_rad = self.ambient_radiation_temp
    
#         convection_term = self.h_scaled * (u_pred - u_inf_conv)
#         radiation_term = self.sigma_scaled * (u_pred**4 - u_inf_rad**4)
    
#         k_yy = self.k_yy(u_pred)
#         k_yy_scaled = k_yy / self.y_max
#         return k_yy_scaled * u_y - convection_term - radiation_term  # k*dT/dy = h(T-T_inf) + radiation

#     # # Without Radiation
#     # def loss_convection_ymax(self, params, t, x, y, z):
#     #     # Compute the derivative du/dy at y = y_max
#     #     u_y = grad(self.u_net, argnums=3)(params, t, x, y, z)
#     #     u_pred = self.u_net(params, t, x, y, z)
#     #     u_inf = self.ambient_convection_temp
    
#     #     # Compute the convection term
#     #     convection_term = self.h_scaled * (u_pred - u_inf)
    
#     #     # Enforce the convection
#     #     k_yy = self.k_yy(u_pred)
#     #     k_yy_scaled = k_yy / self.y_max
#     #     return k_yy_scaled * u_y + convection_term

#     # With Radiation
#     def loss_convection_ymax(self, params, t, x, y, z):
#         u_y = grad(self.u_net, argnums=3)(params, t, x, y, z)
#         u_pred = self.u_net(params, t, x, y, z)
#         u_inf_conv = self.ambient_convection_temp
#         u_inf_rad = self.ambient_radiation_temp
    
#         convection_term = self.h_scaled * (u_pred - u_inf_conv)
#         radiation_term = self.sigma_scaled * (u_pred**4 - u_inf_rad**4)
    
#         k_yy = self.k_yy(u_pred)
#         k_yy_scaled = k_yy / self.y_max
#         return k_yy_scaled * u_y + convection_term + radiation_term  # k*dT/dy + h(T-T_inf) + radiation = 0

#     # # Without Radiation
#     # def loss_convection_zmax(self, params, t, x, y, z):
#     #     # Compute the derivative du/dz at z = z_max
#     #     u_z = grad(self.u_net, argnums=4)(params, t, x, y, z)
#     #     u_pred = self.u_net(params, t, x, y, z)
#     #     u_inf = self.ambient_convection_temp  # Scaled ambient temperature
    
#     #     # Compute the convection term
#     #     convection_term = self.h_scaled * (u_pred - u_inf)
    
#     #     # Enforce the convection
#     #     k_zz = self.k_zz(u_pred)
#     #     k_zz_scaled = k_zz / self.z_max
#     #     return k_zz_scaled * u_z + convection_term

#     # With Radiation
#     def loss_convection_zmax(self, params, t, x, y, z):
#         u_z = grad(self.u_net, argnums=4)(params, t, x, y, z)
#         u_pred = self.u_net(params, t, x, y, z)
#         u_inf_conv = self.ambient_convection_temp
#         u_inf_rad = self.ambient_radiation_temp
    
#         convection_term = self.h_scaled * (u_pred - u_inf_conv)
#         radiation_term = self.sigma_scaled * (u_pred**4 - u_inf_rad**4)
    
#         k_zz = self.k_zz(u_pred)
#         k_zz_scaled = k_zz / self.z_max
#         return k_zz_scaled * u_z + convection_term + radiation_term  # k*dT/dz + h(T-T_inf) + radiation = 0
    
#     def loss_neumann_zmax(self, params, t, x, y, z):
#         u_z = grad(self.u_net, argnums=4)(params, t, x, y, z)
#         return u_z  # Enforce du/dz = 0 at z = z_max

#     def loss_neumann_ymin(self, params, t, x, y, z):
#         # Compute the derivative du/dy at y = 0
#         u_y = grad(self.u_net, argnums=3)(params, t, x, y, z)
#         return u_y  # Enforce du/dy = 0 at y = 0
    
#     def loss_neumann_ymax(self, params, t, x, y, z):
#         # Compute the derivative du/dy at y = y_max
#         u_y = grad(self.u_net, argnums=3)(params, t, x, y, z)
#         return u_y  # Enforce du/dy = 0 at y = y_max

#     @partial(jit, static_argnums=(0,))
#     def delta_time_loss(self, params, step, time):
#         batch = self.seqSampler(step[0], time)
#         res = self.delta_time_lossMap(params, batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3])
#         return jnp.mean(res ** 2)

#     @partial(jit, static_argnums=(0,))
#     def delta_time_neumann_loss_b1(self, params, step, time):
#         batch = self.seqSamplerNeumann_B1(step[0], time)
#         res_neumann = self.delta_time_NeumannLossMap_b1(params, batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3])
#         return jnp.mean(res_neumann ** 2)

#     @partial(jit, static_argnums=(0,))
#     def delta_time_neumann_loss_b2(self, params, step, time):
#         batch = self.seqSamplerNeumann_B2(step[0], time)
#         res_neumann = self.delta_time_NeumannLossMap_b2(params, batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3])
#         return jnp.mean(res_neumann ** 2)

#     @partial(jit, static_argnums=(0,))
#     def delta_time_deposition_front_loss(self, params, step, time):
#         batch = self.deposition_front_sampler(step[0], time)
#         res = self.delta_time_deposition_front_loss_map(params, batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3])
#         return jnp.mean(res ** 2)

#     @partial(jit, static_argnums=(0,))
#     def delta_time_bed_temperature_loss(self, params, step, time):
#         batch = self.bed_temperature_sampler(step[0], time)
#         res = self.delta_time_bed_temperature_loss_map(params, batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3])
#         return jnp.mean(res ** 2)

#     @partial(jit, static_argnums=(0,))
#     def delta_time_neumann_loss_x0(self, params, step, time):
#         batch = self.neumann_boundary_sampler_x0(step[0], time)
#         res_neumann_x0 = self.delta_time_neumann_loss_map_x0(params, batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3])
#         return jnp.mean(res_neumann_x0 ** 2)

#     @partial(jit, static_argnums=(0,))
#     def delta_time_boundary_loss_x0(self, params, step, time):
#         batch = self.boundary_sampler_x0(step[0], time)
#         res_boundary_x0 = self.delta_time_boundary_loss_map_x0(params, batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3])
#         return jnp.mean(res_boundary_x0 ** 2)

#     @partial(jit, static_argnums=(0,))
#     def delta_time_convection_loss_ymin(self, params, step, time):
#         batch = self.convection_boundary_sampler_ymin(step[0], time)
#         res_convection_ymin = self.delta_time_convection_loss_map_ymin(params, batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3])
#         return jnp.mean(res_convection_ymin ** 2)

#     @partial(jit, static_argnums=(0,))
#     def delta_time_convection_loss_ymax(self, params, step, time):
#         batch = self.convection_boundary_sampler_ymax(step[0], time)
#         res_convection_ymax = self.delta_time_convection_loss_map_ymax(params, batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3])
#         return jnp.mean(res_convection_ymax ** 2)

#     @partial(jit, static_argnums=(0,))
#     def delta_time_convection_loss_zmax(self, params, step, time):
#         batch = self.convection_boundary_sampler_zmax(step[0], time)
#         res_convection_zmax = self.delta_time_convection_loss_map_zmax(params, batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3])
#         return jnp.mean(res_convection_zmax ** 2)

#     @partial(jit, static_argnums=(0,))
#     def delta_time_neumann_loss_zmax(self, params, step, time):
#         batch = self.neumann_boundary_sampler_zmax(step[0], time)
#         res_neumann_zmax = self.delta_time_neumann_loss_map_zmax(params, batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3])
#         return jnp.mean(res_neumann_zmax ** 2)

#     @partial(jit, static_argnums=(0,))
#     def delta_time_neumann_loss_ymin(self, params, step, time):
#         batch = self.convection_boundary_sampler_ymin(step[0], time)
#         res_neumann_ymin = self.delta_time_neumann_loss_map_ymin(params, batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3])
#         return jnp.mean(res_neumann_ymin ** 2)
    
#     @partial(jit, static_argnums=(0,))
#     def delta_time_neumann_loss_ymax(self, params, step, time):
#         batch = self.convection_boundary_sampler_ymax(step[0], time)
#         res_neumann_ymax = self.delta_time_neumann_loss_map_ymax(params, batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3])
#         return jnp.mean(res_neumann_ymax ** 2)

#     def losses(self, params, time_batch, batch_initial, *args):
#         if self.config.weighting.use_causal:
#             raise NotImplementedError('Not implemented for Additive')

#         if self.config.training.loss_type == "strong":
#             res_loss = jnp.mean(vmap(self.delta_time_loss, (None, 0, 0))(params, args[0], time_batch))
#             deposition_front_loss = jnp.mean(vmap(self.delta_time_deposition_front_loss, (None, 0, 0))(params, args[0], time_batch))
#             bed_temperature_loss = jnp.mean(vmap(self.delta_time_bed_temperature_loss, (None, 0, 0))(params, args[0], time_batch))
#             # neumann_loss_x0 = jnp.mean(vmap(self.delta_time_neumann_loss_x0, (None, 0, 0))(params, args[0], time_batch))
#             boundary_loss_x0 = jnp.mean(vmap(self.delta_time_boundary_loss_x0, (None, 0, 0))(params, args[0], time_batch))
#             convection_loss_ymax = jnp.mean(vmap(self.delta_time_convection_loss_ymax, (None, 0, 0))(params, args[0], time_batch))
#             convection_loss_ymin = jnp.mean(vmap(self.delta_time_convection_loss_ymin, (None, 0, 0))(params, args[0], time_batch))
#             # neumann_loss_ymin = jnp.mean(vmap(self.delta_time_neumann_loss_ymin, (None, 0, 0))(params, args[0], time_batch))
#             # neumann_loss_ymax = jnp.mean(vmap(self.delta_time_neumann_loss_ymax, (None, 0, 0))(params, args[0], time_batch))
#             convection_loss_zmax = jnp.mean(vmap(self.delta_time_convection_loss_zmax, (None, 0, 0))(params, args[0], time_batch))
#             # neumann_loss_zmax = jnp.mean(vmap(self.delta_time_neumann_loss_zmax, (None, 0, 0))(params, args[0], time_batch))
#         elif self.config.training.loss_type == "weak":
#             raise NotImplementedError('Weak form error not implemented for Additive')

#         loss_dict = {
#             "deposition_front": deposition_front_loss,
#             "bed_temperature": bed_temperature_loss,
#             "res": res_loss,
#             # "neumann_x0": neumann_loss_x0,
#             "boundary_x0": boundary_loss_x0,
#             "convection_ymax": convection_loss_ymax,
#             "convection_ymin": convection_loss_ymin,
#             # "neumann_ymin": neumann_loss_ymin,
#             # "neumann_ymax": neumann_loss_ymax,
#             "convection_zmax": convection_loss_zmax
#             # "neumann_zmax": neumann_loss_zmax
#         }
#         return loss_dict

#     def compute_l2_error(self, params, u_test):
#         return NotImplementedError('L2 error not implemented yet')

#     def evaluate_Uplot(self, params, time, num_points=200):
#         length_updated = self.velocity_vector[0] * self.print_speed_scaled * time + self.init_length_scaled[0]
#         width_updated = self.bead_width_scaled
#         height_updated = self.bead_height_scaled
        
#         # Create linspace for x and z (we will fix y)
#         x_batch = jnp.linspace(0., length_updated, num_points)
#         z_batch = jnp.linspace(0., height_updated, num_points)
#         y_fixed = width_updated / 2.0  # Fix y at the middle of the bead width
        
#         xx, zz = jnp.meshgrid(x_batch, z_batch)
#         x_volume = jnp.concatenate(
#             (xx.reshape(-1)[:, None],
#              y_fixed * jnp.ones_like(xx.reshape(-1))[:, None],
#              zz.reshape(-1)[:, None]),
#             axis=1
#         )
        
#         temp_scaled = self.u_pred_fn(params, time, x_volume[:, 0], x_volume[:, 1], x_volume[:, 2])
        
#         return temp_scaled * self.u_max, x_volume[:, 0] * self.x_max, x_volume[:, 2] * self.z_max

#     def evaluate_init_plot(self, params, num_points=200):
#         """Evaluate the temperature in the entire bead at time = 0"""
#         x_batch = jnp.linspace(0., 1., num_points)
#         z_batch = jnp.linspace(0., 1., num_points)
#         y_fixed = 0.5  # Fix y at the middle of the bead width
        
#         xx, zz = jnp.meshgrid(x_batch, z_batch)
#         x_volume = jnp.concatenate(
#             (xx.reshape(-1)[:, None],
#              y_fixed * jnp.ones_like(xx.reshape(-1))[:, None],
#              zz.reshape(-1)[:, None]),
#             axis=1
#         )
        
#         temp_scaled = self.u_pred_fn(params, 0., x_volume[:, 0], x_volume[:, 1], x_volume[:, 2])
#         return temp_scaled * self.u_max, x_volume[:, 0] * self.x_max, x_volume[:, 2] * self.z_max

# class A3DHeatTransferEvaluator(BaseEvaluator):
#     def __init__(self, config, model):
#         super().__init__(config, model)

#     def log_errors(self, params, u_ref):
#         l2_error = self.model.compute_l2_error(params, u_ref)
#         self.log_dict["l2_error"] = l2_error

#     def log_preds(self, params):
#         u_pred = self.model.u_pred_fn(params, self.model.t_star, self.model.x_star)
#         fig = plt.figure(figsize=(6, 5))
#         plt.imshow(u_pred.T, cmap="jet")
#         self.log_dict["u_pred"] = fig
#         plt.close()

#     def __call__(self, state, time_batch, batch_initial, *args, u_ref=None):
#         self.log_dict = super().__call__(state, time_batch, batch_initial, *args)

#         if self.config.weighting.use_causal:
#             raise NotImplementedError('Causal weighting not implemented for A3D')

#         if self.config.logging.log_errors:
#             self.log_errors(state.params, u_ref)

#         if self.config.logging.log_preds:
#             self.log_preds(state.params)

#         return self.log_dict










### Multilayer ###

from functools import partial
import jax.numpy as jnp
from jax import jit, grad, vmap
from A3DPINN.models import ForwardIVP
from A3DPINN.evaluator import BaseEvaluator
from A3DPINN.samplers import (
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
from matplotlib import pyplot as plt

class A3DHeatTransfer(ForwardIVP):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.u_max = config.process_conditions.deposition_temperature
        self.t_max = config.dimensions.t_max
        self.x_max = config.dimensions.x_max
        self.y_max = config.dimensions.y_max
        self.z_max = config.dimensions.z_max
        
        # Processing conditions in scaled version
        self.deposition_temperature_scaled = config.process_conditions.deposition_temperature / self.u_max
        self.bed_temperature_scaled = config.process_conditions.bed_temperature / self.u_max
        
        # Define material properties
        self.rho = config.material_properties.density
        self.C = config.material_properties.specific_heat

        # Thermal conductivity coefficients (temperature-dependent)
        self.k_0_xx = config.material_properties.k_0_xx
        self.k_1_xx = config.material_properties.k_1_xx
        self.k_0_yy = config.material_properties.k_0_yy
        self.k_1_yy = config.material_properties.k_1_yy
        self.k_0_zz = config.material_properties.k_0_zz
        self.k_1_zz = config.material_properties.k_1_zz

        self.h = config.material_properties.heat_transfer_coefficient
        self.emissivity = config.material_properties.emissivity
        self.sigma = 5.670374419e-8  # Stefan-Boltzmann constant in W/m^2·K^4
        
        self.alpha = 1.0 / (self.rho * self.C)
        
        # Scaled alpha
        self.alpha_xx_scaled = self.alpha * self.t_max / (self.x_max ** 2)
        self.alpha_yy_scaled = self.alpha * self.t_max / (self.y_max ** 2)
        self.alpha_zz_scaled = self.alpha * self.t_max / (self.z_max ** 2)
        
        # Scaling factors for h and sigma
        self.h_scaled = (self.h * self.x_max) / self.k_0_xx
        self.sigma_scaled = (self.emissivity * self.sigma * (self.u_max**3) * self.x_max / self.k_0_xx)

        # Define scaled versions of processing conditions
        self.print_speed_scaled = config.process_conditions.print_speed * self.t_max / self.x_max
        self.init_length_scaled = config.process_conditions.init_length / self.x_max
        self.bead_width_scaled = config.process_conditions.bead_width / self.y_max
        self.bead_height_scaled = config.process_conditions.bead_height / self.z_max
        self.velocity_vector = config.process_conditions.velocity_vector
        
        self.ambient_convection_temp = config.process_conditions.ambient_convection_temp / self.u_max
        self.ambient_radiation_temp = config.process_conditions.ambient_radiation_temp / self.u_max
        
        # Multi-layer scaled parameters
        self.layer_height_scaled = config.multi_layer.layer_height / self.z_max
        self.t_L_scaled = config.multi_layer.t_L / self.t_max
        self.num_layers = config.multi_layer.num_layers

        # Samplers - multi-layer collocation samplers in 3D
        self.seqSampler = MultiLayerCollocationSampler3D(
            config.training.batch_size_per_device,
            self.t_L_scaled,
            self.layer_height_scaled,
            x_max=1.0,  # Scaled x domain
            y_max=1.0   # Scaled y domain
        )
        
        self.bed_temperature_sampler = MultiLayerBedTemperatureSampler3D(
            config.training.batch_size_per_device,
            self.t_L_scaled,
            self.layer_height_scaled,
            x_max=1.0,
            y_max=1.0
        )

        self.convection_boundary_sampler_zmax = MultiLayerTopBoundarySampler3D(
            config.training.batch_size_per_device,
            self.t_L_scaled,
            self.layer_height_scaled,
            x_max=1.0,
            y_max=1.0
        )

        self.boundary_sampler_x0 = MultiLayerX0Sampler3D(
            config.training.batch_size_per_device,
            self.t_L_scaled,
            self.layer_height_scaled,
            x_max=1.0,
            y_max=1.0
        )

        self.convection_boundary_sampler_ymin = MultiLayerY0Sampler3D(
            config.training.batch_size_per_device,
            self.t_L_scaled,
            self.layer_height_scaled,
            x_max=1.0,
            y_max=1.0
        )

        self.convection_boundary_sampler_ymax = MultiLayerYMaxSampler3D(
            config.training.batch_size_per_device,
            self.t_L_scaled,
            self.layer_height_scaled,
            x_max=1.0,
            y_max=1.0
        )

        self.convection_boundary_sampler_xmax = MultiLayerXMaxSampler3D(
            config.training.batch_size_per_device,
            self.t_L_scaled,
            self.layer_height_scaled,
            x_max=1.0,
            y_max=1.0
        )

        self.ic_new_layer_sampler = MultiLayerInitialConditionSampler3D(
            config.training.batch_size_per_device,
            self.t_L_scaled,
            self.layer_height_scaled,
            x_max=1.0,
            y_max=1.0
        )
        
        self.ic_prev_layers_sampler = MultiLayerPrevInitialConditionSampler3D(
            config.training.batch_size_per_device,
            self.t_L_scaled,
            self.layer_height_scaled,
            x_max=1.0,
            y_max=1.0
        )

        # Predictions over a grid
        self.u_pred_fn = vmap(self.u_net, (None, None, 0, 0, 0))
        self.delta_time_lossMap = vmap(self.strong_res_net, (None, 0, 0, 0, 0))
        self.delta_time_bed_temperature_loss_map = vmap(self.loss_bed_temperature, (None, 0, 0, 0, 0))
        self.delta_time_boundary_loss_map_x0 = vmap(self.loss_convection_x0, (None, 0, 0, 0, 0))
        self.delta_time_convection_loss_map_ymax = vmap(self.loss_convection_ymax, (None, 0, 0, 0, 0))
        self.delta_time_convection_loss_map_ymin = vmap(self.loss_convection_ymin, (None, 0, 0, 0, 0))
        self.delta_time_convection_loss_map_zmax = vmap(self.loss_convection_zmax, (None, 0, 0, 0, 0))
        self.delta_time_convection_loss_map_xmax = vmap(self.loss_convection_xmax, (None, 0, 0, 0, 0))
        self.delta_time_ic_new_layer_loss_map = vmap(self.loss_ic_new_layer, (None, 0, 0, 0, 0))
        self.delta_time_ic_prev_layers_loss_map = vmap(self.loss_ic_prev_layers, (None, 0, 0, 0, 0))
        
        # For evaluation
        self.evalfn_ = vmap(self.u_net, (None, 0, None, None, None))
        
        self.bs = config.training.batch_size_per_device

    def u_net(self, params, t, x, y, z):
        inputs = jnp.stack([t, x, y, z])  # Now includes z
        u = self.state.apply_fn(params, inputs)
        return u[0]

    def k_xx(self, u):
        return self.k_0_xx + self.k_1_xx * u * self.u_max

    def k_yy(self, u):
        return self.k_0_yy + self.k_1_yy * u * self.u_max

    def k_zz(self, u):
        return self.k_0_zz + self.k_1_zz * u * self.u_max

    def strong_res_net(self, params, t, x, y, z):
        u_pred = self.u_net(params, t, x, y, z)
        u_t = grad(self.u_net, argnums=1)(params, t, x, y, z)
        u_xx = grad(grad(self.u_net, argnums=2), argnums=2)(params, t, x, y, z)
        u_yy = grad(grad(self.u_net, argnums=3), argnums=3)(params, t, x, y, z)
        u_zz = grad(grad(self.u_net, argnums=4), argnums=4)(params, t, x, y, z)
        
        k_xx = self.k_xx(u_pred)
        k_yy = self.k_yy(u_pred)
        k_zz = self.k_zz(u_pred)

        term_x = self.alpha_xx_scaled * k_xx * u_xx
        term_y = self.alpha_yy_scaled * k_yy * u_yy
        term_z = self.alpha_zz_scaled * k_zz * u_zz

        res = term_x + term_y + term_z - u_t
        
        return res

    def loss_bed_temperature(self, params, t, x, y, z):
        u_pred = self.u_net(params, t, x, y, z)
        u_sol = self.bed_temperature_scaled
        return u_pred - u_sol  # Enforce bed temperature at z = 0

    def loss_convection_x0(self, params, t, x, y, z):
        u_x = grad(self.u_net, argnums=2)(params, t, x, y, z)
        u_pred = self.u_net(params, t, x, y, z)
        u_inf_conv = self.ambient_convection_temp
        u_inf_rad = self.ambient_radiation_temp
    
        convection_term = self.h_scaled * (u_pred - u_inf_conv)
        radiation_term = self.sigma_scaled * (u_pred**4 - u_inf_rad**4)
    
        k_xx = self.k_xx(u_pred)
        k_xx_scaled = k_xx / self.x_max
        return k_xx_scaled * u_x - convection_term - radiation_term  # k*dT/dx = h(T-T_inf) + radiation

    def loss_convection_xmax(self, params, t, x, y, z):
        u_x = grad(self.u_net, argnums=2)(params, t, x, y, z)
        u_pred = self.u_net(params, t, x, y, z)
        u_inf_conv = self.ambient_convection_temp
        u_inf_rad = self.ambient_radiation_temp
    
        convection_term = self.h_scaled * (u_pred - u_inf_conv)
        radiation_term = self.sigma_scaled * (u_pred**4 - u_inf_rad**4)
    
        k_xx = self.k_xx(u_pred)
        k_xx_scaled = k_xx / self.x_max
        return k_xx_scaled * u_x + convection_term + radiation_term  # k*dT/dx + h(T-T_inf) + radiation = 0

    def loss_convection_ymin(self, params, t, x, y, z):
        u_y = grad(self.u_net, argnums=3)(params, t, x, y, z)
        u_pred = self.u_net(params, t, x, y, z)
        u_inf_conv = self.ambient_convection_temp
        u_inf_rad = self.ambient_radiation_temp
    
        convection_term = self.h_scaled * (u_pred - u_inf_conv)
        radiation_term = self.sigma_scaled * (u_pred**4 - u_inf_rad**4)
    
        k_yy = self.k_yy(u_pred)
        k_yy_scaled = k_yy / self.y_max
        return k_yy_scaled * u_y - convection_term - radiation_term  # k*dT/dy = h(T-T_inf) + radiation

    def loss_convection_ymax(self, params, t, x, y, z):
        u_y = grad(self.u_net, argnums=3)(params, t, x, y, z)
        u_pred = self.u_net(params, t, x, y, z)
        u_inf_conv = self.ambient_convection_temp
        u_inf_rad = self.ambient_radiation_temp
    
        convection_term = self.h_scaled * (u_pred - u_inf_conv)
        radiation_term = self.sigma_scaled * (u_pred**4 - u_inf_rad**4)
    
        k_yy = self.k_yy(u_pred)
        k_yy_scaled = k_yy / self.y_max
        return k_yy_scaled * u_y + convection_term + radiation_term  # k*dT/dy + h(T-T_inf) + radiation = 0

    def loss_convection_zmax(self, params, t, x, y, z):
        u_z = grad(self.u_net, argnums=4)(params, t, x, y, z)
        u_pred = self.u_net(params, t, x, y, z)
        u_inf_conv = self.ambient_convection_temp
        u_inf_rad = self.ambient_radiation_temp
    
        convection_term = self.h_scaled * (u_pred - u_inf_conv)
        radiation_term = self.sigma_scaled * (u_pred**4 - u_inf_rad**4)
    
        k_zz = self.k_zz(u_pred)
        k_zz_scaled = k_zz / self.z_max
        return k_zz_scaled * u_z + convection_term + radiation_term  # k*dT/dz + h(T-T_inf) + radiation = 0

    def loss_ic_new_layer(self, params, t, x, y, z):
        u_pred = self.u_net(params, t, x, y, z)
        return u_pred - self.deposition_temperature_scaled  # Enforce u = deposition_temperature
    
    def loss_ic_prev_layers(self, params, t, x, y, z):
        u_pred = self.u_net(params, t, x, y, z)
        return u_pred # Not implemented yet

    @partial(jit, static_argnums=(0,))
    def delta_time_loss(self, params, step, time):
        batch = self.seqSampler(step[0], time)
        res = self.delta_time_lossMap(params, batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3])
        return jnp.mean(res ** 2)

    @partial(jit, static_argnums=(0,))
    def delta_time_bed_temperature_loss(self, params, step, time):
        batch = self.bed_temperature_sampler(step[0], time)
        res = self.delta_time_bed_temperature_loss_map(params, batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3])
        return jnp.mean(res ** 2)

    @partial(jit, static_argnums=(0,))
    def delta_time_boundary_loss_x0(self, params, step, time):
        batch = self.boundary_sampler_x0(step[0], time)
        res_boundary_x0 = self.delta_time_boundary_loss_map_x0(params, batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3])
        return jnp.mean(res_boundary_x0 ** 2)

    @partial(jit, static_argnums=(0,))
    def delta_time_convection_loss_ymin(self, params, step, time):
        batch = self.convection_boundary_sampler_ymin(step[0], time)
        res_convection_ymin = self.delta_time_convection_loss_map_ymin(params, batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3])
        return jnp.mean(res_convection_ymin ** 2)

    @partial(jit, static_argnums=(0,))
    def delta_time_convection_loss_ymax(self, params, step, time):
        batch = self.convection_boundary_sampler_ymax(step[0], time)
        res_convection_ymax = self.delta_time_convection_loss_map_ymax(params, batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3])
        return jnp.mean(res_convection_ymax ** 2)

    @partial(jit, static_argnums=(0,))
    def delta_time_convection_loss_zmax(self, params, step, time):
        batch = self.convection_boundary_sampler_zmax(step[0], time)
        res_convection_zmax = self.delta_time_convection_loss_map_zmax(params, batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3])
        return jnp.mean(res_convection_zmax ** 2)

    @partial(jit, static_argnums=(0,))
    def delta_time_convection_loss_xmax(self, params, step, time):
        batch = self.convection_boundary_sampler_xmax(step[0], time)
        res_convection_xmax = self.delta_time_convection_loss_map_xmax(params, batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3])
        return jnp.mean(res_convection_xmax ** 2)

    @partial(jit, static_argnums=(0,))
    def delta_time_ic_new_layer_loss(self, params, step, time):
        batch = self.ic_new_layer_sampler(step[0], time)
        res_ic_new = self.delta_time_ic_new_layer_loss_map(params, batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3])
        return jnp.mean(res_ic_new ** 2)
    
    @partial(jit, static_argnums=(0,))
    def delta_time_ic_prev_layers_loss(self, params, step, time):
        batch = self.ic_prev_layers_sampler(step[0], time)
        res_ic_prev = self.delta_time_ic_prev_layers_loss_map(params, batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3])
        return jnp.mean(res_ic_prev ** 2)

    def losses(self, params, time_batch, batch_initial, *args):
        if self.config.weighting.use_causal:
            raise NotImplementedError('Not implemented for Additive')
    
        if self.config.training.loss_type == "strong":
            res_loss = jnp.mean(vmap(self.delta_time_loss, (None, 0, 0))(params, args[0], time_batch))
            bed_temperature_loss = jnp.mean(vmap(self.delta_time_bed_temperature_loss, (None, 0, 0))(params, args[0], time_batch))
            # boundary_loss_x0 = jnp.mean(vmap(self.delta_time_boundary_loss_x0, (None, 0, 0))(params, args[0], time_batch))
            # convection_loss_ymax = jnp.mean(vmap(self.delta_time_convection_loss_ymax, (None, 0, 0))(params, args[0], time_batch))
            # convection_loss_ymin = jnp.mean(vmap(self.delta_time_convection_loss_ymin, (None, 0, 0))(params, args[0], time_batch))
            convection_loss_zmax = jnp.mean(vmap(self.delta_time_convection_loss_zmax, (None, 0, 0))(params, args[0], time_batch))
            # convection_loss_xmax = jnp.mean(vmap(self.delta_time_convection_loss_xmax, (None, 0, 0))(params, args[0], time_batch))
            ic_new_layer_loss = jnp.mean(vmap(self.delta_time_ic_new_layer_loss, (None, 0, 0))(params, args[0], time_batch))
            # ic_prev_layers_loss = jnp.mean(vmap(self.delta_time_ic_prev_layers_loss, (None, 0, 0))(params, args[0], time_batch))
        elif self.config.training.loss_type == "weak":
            raise NotImplementedError('Weak form error not implemented for Additive')
    
        loss_dict = {
            "bed_temperature": bed_temperature_loss,
            "res": res_loss,
            # "boundary_x0": boundary_loss_x0,
            # "convection_ymax": convection_loss_ymax,
            # "convection_ymin": convection_loss_ymin,
            "convection_zmax": convection_loss_zmax,
            # "convection_xmax": convection_loss_xmax,
            "ic_new_layer": ic_new_layer_loss
            # "ic_prev_layers": ic_prev_layers_loss
        }
        return loss_dict

    def compute_l2_error(self, params, u_test):
        return NotImplementedError('L2 error not implemented yet')

    def evaluate_Uplot(self, params, time, num_points=200):
        x_batch = jnp.linspace(0., 1., num_points)
        z_batch = jnp.linspace(0., 1., num_points)
        y_fixed = 0.5  # Midpoint in y-direction
    
        xx, zz = jnp.meshgrid(x_batch, z_batch)
        x_volume = jnp.concatenate(
            (xx.reshape(-1)[:, None],
             y_fixed * jnp.ones_like(xx.reshape(-1))[:, None],
             zz.reshape(-1)[:, None]),
            axis=1
        )
    
        # Compute current_z_top based on time
        i = jnp.floor(time / self.t_L_scaled).astype(int)
        current_z_top = min((i + 1) * self.layer_height_scaled, 1.0)
    
        # Mask points where z > current_z_top
        mask = x_volume[:, 2] <= current_z_top
        x_volume_active = x_volume[mask]
    
        if x_volume_active.size == 0:
            return jnp.array([]), jnp.array([]), jnp.array([])
    
        temp_scaled = self.u_pred_fn(params, time, x_volume_active[:, 0], x_volume_active[:, 1], x_volume_active[:, 2])
        temp = temp_scaled * self.u_max
        x_active = x_volume_active[:, 0] * self.x_max
        z_active = x_volume_active[:, 2] * self.z_max
    
        return temp, x_active, z_active



    def evaluate_init_plot(self, params, num_points=200):
        """Evaluate the temperature in the domain at time = 0"""
        x_batch = jnp.linspace(0., 1., num_points)
        z_batch = jnp.linspace(0., self.layer_height_scaled, num_points)  # Only first layer at t=0
        y_fixed = 0.5  # Fix y at the middle of the domain
        
        xx, zz = jnp.meshgrid(x_batch, z_batch)
        x_volume = jnp.concatenate(
            (xx.reshape(-1)[:, None],
             y_fixed * jnp.ones_like(xx.reshape(-1))[:, None],
             zz.reshape(-1)[:, None]),
            axis=1
        )
        
        temp_scaled = self.u_pred_fn(params, 0., x_volume[:, 0], x_volume[:, 1], x_volume[:, 2])
        return temp_scaled * self.u_max, x_volume[:, 0] * self.x_max, x_volume[:, 2] * self.z_max

class A3DHeatTransferEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, u_ref):
        l2_error = self.model.compute_l2_error(params, u_ref)
        self.log_dict["l2_error"] = l2_error

    def log_preds(self, params):
        u_pred = self.model.u_pred_fn(params, self.model.t_star, self.model.x_star)
        fig = plt.figure(figsize=(6, 5))
        plt.imshow(u_pred.T, cmap="jet")
        self.log_dict["u_pred"] = fig
        plt.close()

    def __call__(self, state, time_batch, batch_initial, *args, u_ref=None):
        self.log_dict = super().__call__(state, time_batch, batch_initial, *args)

        if self.config.weighting.use_causal:
            raise NotImplementedError('Causal weighting not implemented for A3D')

        if self.config.logging.log_errors:
            self.log_errors(state.params, u_ref)

        if self.config.logging.log_preds:
            self.log_preds(state.params)

        return self.log_dict