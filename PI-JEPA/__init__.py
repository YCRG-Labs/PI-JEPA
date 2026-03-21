# PI-JEPA: Physics-Informed Joint Embedding Predictive Architecture
# A physics-informed joint embedding predictive architecture for multi-step
# coupled PDE surrogate modeling.

__version__ = "0.1.0"

# Models
from .models import (
    PIJEPA,
    ViTEncoder,
    TargetEncoder,
    Decoder,
    Predictor,
    MultiStepPredictor,
    MultiSpeciesPredictor,
    ChannelMixingAttention,
    update_ema,
)

# Physics modules
from .physics import (
    physics_loss_pressure,
    physics_loss_saturation,
    grad_x,
    grad_y,
    divergence,
    mobility,
    ReactiveTransportPhysics,
)
from .physics.darcy import (
    BrooksCoreyModel,
    TwoPhaseDarcyPhysics,
)

# Data loaders
from .data import (
    DarcyDataset,
    SequenceDarcyDataset,
    build_dataloader,
    Preprocessor,
    HDF5DatasetMixin,
    DatasetFactory,
    BenchmarkDataset,
    UFNODataset,
    PDEBenchADRDataset,
    SPE10Dataset,
    NavierStokesDataset,
)

# Training components
from .training import (
    LossBuilder,
    JEPAAlignmentLoss,
    PhysicsLoss,
    EMATeacher,
    Engine,
    LinearProbe,
    EMAMomentumSchedule,
    PhysicsWeightSchedule,
    K3PhysicsWeightManager,
    build_ema_schedule,
    build_physics_weight_schedule,
    build_k3_physics_weights,
)
from .training.finetune import FineTuningPipeline

# Evaluation components
from .eval import (
    rollout,
    rollout_with_metrics,
    single_step,
    NoiseSchedule,
    mse,
    rmse,
    mae,
    relative_l2,
    relative_l1,
    max_error,
    rollout_mse,
    rollout_rmse,
    rollout_mae,
    rollout_relative_l2,
    compute_l2,
)
from .eval.rollout import RolloutEvaluator
from .eval.metrics import (
    relative_l2_per_field,
    per_channel_mse,
    per_channel_mse_named,
    rollout_cumulative_error,
    data_efficiency_curve,
    ood_relative_l2,
    pde_residual_mse,
)
from .eval.visualization import VisualizationModule, AblationModule

# Benchmarks
from .benchmarks import (
    FNOWrapper,
    GeoFNOWrapper,
    PINOWrapper,
    DeepONetWrapper,
    UDeepONetWrapper,
    PINNWrapper,
    PILatentNOWrapper,
    UNetWrapper,
    UFNOWrapper,
    UFNO,
    BenchmarkConfig,
    BenchmarkTrainer,
    set_seed,
    create_data_splits,
    create_data_loaders,
    run_benchmark_comparison,
    create_benchmark_suite,
    save_benchmark_results,
    load_benchmark_results,
)

# Utilities
from .utils import (
    Config,
    load_config,
    Logger,
    save_checkpoint,
    load_checkpoint,
)

__all__ = [
    # Version
    "__version__",
    
    # Models
    "PIJEPA",
    "ViTEncoder",
    "TargetEncoder",
    "Decoder",
    "Predictor",
    "MultiStepPredictor",
    "MultiSpeciesPredictor",
    "ChannelMixingAttention",
    "update_ema",
    
    # Physics
    "physics_loss_pressure",
    "physics_loss_saturation",
    "grad_x",
    "grad_y",
    "divergence",
    "mobility",
    "ReactiveTransportPhysics",
    "BrooksCoreyModel",
    "TwoPhaseDarcyPhysics",
    
    # Data
    "DarcyDataset",
    "SequenceDarcyDataset",
    "build_dataloader",
    "Preprocessor",
    "HDF5DatasetMixin",
    "DatasetFactory",
    "BenchmarkDataset",
    "UFNODataset",
    "PDEBenchADRDataset",
    "SPE10Dataset",
    "NavierStokesDataset",
    
    # Training
    "LossBuilder",
    "JEPAAlignmentLoss",
    "PhysicsLoss",
    "EMATeacher",
    "Engine",
    "LinearProbe",
    "FineTuningPipeline",
    "EMAMomentumSchedule",
    "PhysicsWeightSchedule",
    "K3PhysicsWeightManager",
    "build_ema_schedule",
    "build_physics_weight_schedule",
    "build_k3_physics_weights",
    
    # Evaluation
    "rollout",
    "rollout_with_metrics",
    "single_step",
    "NoiseSchedule",
    "RolloutEvaluator",
    "mse",
    "rmse",
    "mae",
    "relative_l2",
    "relative_l1",
    "max_error",
    "rollout_mse",
    "rollout_rmse",
    "rollout_mae",
    "rollout_relative_l2",
    "compute_l2",
    "relative_l2_per_field",
    "per_channel_mse",
    "per_channel_mse_named",
    "rollout_cumulative_error",
    "data_efficiency_curve",
    "ood_relative_l2",
    "pde_residual_mse",
    "VisualizationModule",
    "AblationModule",
    
    # Benchmarks
    "FNOWrapper",
    "GeoFNOWrapper",
    "PINOWrapper",
    "DeepONetWrapper",
    "UDeepONetWrapper",
    "PINNWrapper",
    "PILatentNOWrapper",
    "UNetWrapper",
    "UFNOWrapper",
    "UFNO",
    "BenchmarkConfig",
    "BenchmarkTrainer",
    "set_seed",
    "create_data_splits",
    "create_data_loaders",
    "run_benchmark_comparison",
    "create_benchmark_suite",
    "save_benchmark_results",
    "load_benchmark_results",
    
    # Utilities
    "Config",
    "load_config",
    "Logger",
    "save_checkpoint",
    "load_checkpoint",
]
