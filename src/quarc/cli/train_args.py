# Adapted from chemprop v1's legacy TrainArgs class

from typing_extensions import Literal
from tap import Tap


class TrainArgs(Tap):
    stage: int = 1
    """Stage of the training."""
    no_cuda: bool = False
    """Turn off cuda (i.e., use CPU instead of GPU)."""
    gpu: int = None
    """Which GPU to use."""
    num_workers: int = 8
    """Number of workers for the parallel data loading (0 means sequential)."""
    batch_size: int = 256
    """Batch size."""
    seed: int = 42
    """Random seed."""
    max_epochs: int = 30
    """Number of epochs to run."""
    save_dir: str = "./train_log"
    """Directory where model checkpoints will be saved."""
    logger_name: str = "stage1"
    """Name of the logger."""
    checkpoint_dir: str = None
    """Directory from which to load model checkpoints (walks directory and ensembles all models that are found)."""
    checkpoint_path: str = None
    """Path to model checkpoint (:code:`.pt` file)."""
    checkpoint_paths: list[str] = None
    """List of paths to model checkpoints (:code:`.pt` files)."""

    # Model Args
    # Graph
    graph_hidden_size: int = 300
    """The size of the hidden layer in the graph conv. Same as message_hidden_dim"""
    depth: int = 3
    """Number of message passing layers"""

    FP_radius: int = 3
    """Radius of the Morgan fingerprint."""
    FP_length: int = 2048
    """Length of the Morgan fingerprint."""

    # Predictor Ag
    hidden_size: int = 1024  # refers to the ffn hidden dim
    """The size of the hidden layer in the MLP"""
    agent_hidden_size: int = 512
    """The size of the hidden layer in the agent MLP"""
    reactant_hidden_size: int = 512
    """The size of the hidden layer in the reactant MLP"""
    n_blocks: int = 2
    """Number of blocks"""
    dropout: float = 0.0
    """Dropout probability."""
    activation: Literal["ReLU", "LeakyReLU", "PReLU", "tanh", "SELU", "ELU"] = "ReLU"
    """Activation function."""
    loss: Literal["mse", "bounded_mse", "binary_cross_entropy", "cross_entropy", "focal"] = None
    """Choice of loss function."""
    num_classes: int = None
    """Number of agent classes, len(AgentEncoder)"""
    output_size: int = None
    """Number of output classes, e.g. number of temperature bins"""

    # Train Args
    """Number of epochs to run."""
    warmup_epochs: float = 2.0
    """
    Number of epochs during which learning rate increases linearly from :code:`init_lr` to :code:`max_lr`.
    Afterwards, learning rate decreases exponentially from :code:`max_lr` to :code:`final_lr`.
    """
    init_lr: float = 1e-4
    """Initial learning rate."""
    max_lr: float = 1e-3
    """Maximum learning rate."""
    final_lr: float = 1e-4
    """Final learning rate."""

    def __init__(self, *args, **kwargs) -> None:
        super(TrainArgs, self).__init__(*args, **kwargs)
        self._task_names = None
        self._crossval_index_sets = None
        self._num_tasks = None
        self._features_size = None
        self._train_data_size = None
