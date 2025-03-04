import ml_collections
import jax.numpy as jnp

def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "train"

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "MultiLayer_Test"
    wandb.name = "MultiLayer_Test"
    wandb.tag = None

    # Material properties
    config.material_properties = material_properties = ml_collections.ConfigDict()
    material_properties.density = 1300  # kg/m^3 
    material_properties.specific_heat = 1250  # J/kg·K 
    material_properties.k_0_xx = 1.3  # W/m·K
    material_properties.k_1_xx = 0.0
    material_properties.k_0_yy = 1.3 / 2  # W/m·K
    material_properties.k_1_yy = 0.0 
    material_properties.k_0_zz = 1.3 / 3  # W/m·K
    material_properties.k_1_zz = 0.0 
    material_properties.heat_transfer_coefficient = 100  # W/m²·K
    material_properties.emissivity = 0.8  #

    # Processing conditions
    config.process_conditions = process_conditions = ml_collections.ConfigDict()
    process_conditions.deposition_temperature = 600  # K
    process_conditions.bed_temperature = 373  # K
    process_conditions.print_speed = 5000 / 1000 / 60  # m/s 
    process_conditions.velocity_vector = jnp.array([1.0, 0.0, 0.0])  
    process_conditions.init_length = jnp.array([0.00])  
    process_conditions.bead_width = 6 / 1000  # m
    process_conditions.bead_height = 1.5 / 1000  # m

    # Multi-layer parameters
    config.multi_layer = multi_layer = ml_collections.ConfigDict()
    multi_layer.num_layers = 3  # number of layers
    multi_layer.layer_height = 1.5 / 1000  # Height of each layer
    multi_layer.t_L = 2.0  # Time interval per layer (in seconds)

    # Ambient conditions
    process_conditions.ambient_convection_temp = 308
    process_conditions.ambient_radiation_temp = 308

    # Dimensions
    config.dimensions = dimensions = ml_collections.ConfigDict()
    dimensions.t_min = 0.0
    dimensions.x_min = 0.0 
    dimensions.y_min = 0.0 
    dimensions.z_min = 0.0 
    dimensions.t_max = 8
    dimensions.x_max = 50 / 1000
    dimensions.y_max = 6 / 1000
    dimensions.z_max = multi_layer.num_layers * multi_layer.layer_height  # multilayer

    # Neural Network Architecture
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "ModifiedMlp"
    arch.num_layers = 4
    arch.hidden_dim = 256
    arch.out_dim = 1
    arch.activation = "tanh"
    arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 1, "embed_dim": 128})
    arch.reparam = ml_collections.ConfigDict(
        {"type": "weight_fact", "mean": 0.5, "stddev": 0.1}
    )

    # Training Parameters
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 50000
    training.batch_size_per_device = 256
    training.time_batch_size_per_device = 8
    training.test_batch_size_per_device = 16
    training.loss_type = "strong"

    # Test Functions - if using weak type loss
    config.test_functions = test_functions = ml_collections.ConfigDict()
    test_functions.n_test = 200
    test_functions.lengthscale = 1e-4
    test_functions.centers_type = 'equidistant'

    # Optimizer Settings
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "Adam"
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 5e-4
    optim.decay_rate = 0.9
    optim.decay_steps = 5000
    optim.grad_accum_steps = 0

    # Loss Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"
    weighting.init_weights = ml_collections.ConfigDict({
        # "convection_xmax": 1.0,
        "bed_temperature": 1.0,
        "res": 1.0,
        # "boundary_x0": 1.0,
        # "convection_ymax": 1.0,
        # "convection_ymin": 1.0,
        "convection_zmax": 50.0,
        "ic_new_layer": 100.0
        # "ic_prev_layers": 10.0
    })
    weighting.momentum = 0.9
    weighting.update_every_steps = 25000
    weighting.use_causal = False
    weighting.causal_tol = 1.0
    weighting.num_chunks = 32

    # Logging Settings
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 100
    logging.log_errors = False
    logging.log_losses = True
    logging.log_weights = True
    logging.log_preds = False
    logging.log_grads = True
    logging.log_ntk = False

    # Checkpoint Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 10000
    saving.num_keep_ckpts = 10

    # Input shape
    config.input_dim = 4

    # Random Seed
    config.seed = 101

    return config






