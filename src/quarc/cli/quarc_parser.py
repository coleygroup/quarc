def add_preprocess_opts(parser):
    group = parser.add_argument_group("quarc_preprocess")

    group.add_argument(
        "--config",
        type=str,
        default="configs/preprocess_config.yaml",
        help="Path to preprocessing configuration YAML file",
    )

    # Pipeline steps
    group.add_argument("--chunk-json", action="store_true", help="Run data organization step")
    group.add_argument(
        "--collect-dedup", action="store_true", help="Run data collection and deduplication"
    )
    group.add_argument("--generate-vocab", action="store_true", help="Run agent class generation")
    group.add_argument("--init-filter", action="store_true", help="Run initial filtering")
    group.add_argument("--split", action="store_true", help="Run train/val/test split")
    group.add_argument(
        "--stage1-filter", action="store_true", help="Run stage 1 (agent) filtering"
    )
    group.add_argument(
        "--stage2-filter", action="store_true", help="Run stage 2 (temperature) filtering"
    )
    group.add_argument(
        "--stage3-filter", action="store_true", help="Run stage 3 (reactant amount) filtering"
    )
    group.add_argument(
        "--stage4-filter", action="store_true", help="Run stage 4 (agent amount) filtering"
    )
    group.add_argument(
        "--all-filters", action="store_true", help="Run all filters to get overlap data"
    )
    group.add_argument("--all", action="store_true", help="Run complete preprocessing pipeline")


def add_model_opts(parser):
    group = parser.add_argument_group("quarc_model")

    group.add_argument(
        "--stage",
        required=True,
        type=int,
        choices=[1, 2, 3, 4],
        help="training stage: 1=agents, 2=temperature, 3=reactant_amounts, 4=agent_amounts",
    )
    group.add_argument(
        "--model-type", required=True, choices=["ffn", "gnn"], help="model architecture"
    )
    group.add_argument("--seed", type=int, default=42, help="random seed")

    # GNN
    group.add_argument("--graph-hidden-size", type=int, default=300, help="graph embedding size")
    group.add_argument("--depth", type=int, default=3, help="number of message passing layers")

    # FFN
    group.add_argument("--fp-radius", type=int, default=3, help="Morgan fingerprint radius")
    group.add_argument("--fp-length", type=int, default=2048, help="Morgan fingerprint length")

    # MLP prediction head
    group.add_argument("--hidden-size", type=int, default=1024, help="hidden layer size")
    group.add_argument("--agent-hidden-size", type=int, default=512, help="agent embedding size")
    group.add_argument("--n-blocks", type=int, default=2, help="number of FFN blocks")
    group.add_argument(
        "--activation",
        choices=["ReLU", "LeakyReLU", "PReLU", "tanh", "SELU", "ELU"],
        default="ReLU",
        help="activation function",
    )
    group.add_argument("--output-size", type=int, required=True, help="number of output classes")
    group.add_argument(
        "--num-classes",
        type=int,
        required=True,
        help="number of agent classes (len(agent_encoder))",
    )


def add_train_opts(parser):
    group = parser.add_argument_group("quarc_train")

    group.add_argument("--save-dir", type=str, default="./checkpoints", help="checkpoint folder")
    group.add_argument(
        "--logger-name", type=str, required=True, help="experiment name for logging"
    )
    group.add_argument("--no-cuda", action="store_true", help="use CPU instead of GPU")
    group.add_argument("--gpu", type=int, default=0, help="specific GPU device to use")
    group.add_argument("--num-workers", type=int, default=8, help="number of data loading workers")
    group.add_argument("--checkpoint-path", type=str, default="", help="resume from checkpoint")

    # hyperparameters
    group.add_argument("--max-epochs", type=int, default=30, help="maximum number of epochs")
    group.add_argument("--batch-size", type=int, default=256, help="batch size per GPU")
    group.add_argument("--max-lr", type=float, default=1e-3, help="peak learning rate")
    group.add_argument("--init-lr", type=float, default=1e-4, help="initial learning rate")
    group.add_argument("--final-lr", type=float, default=1e-4, help="final learning rate")
    group.add_argument(
        "--warmup-epochs",
        type=float,
        default=2.0,
        help="number of warmup epochs",
    )

    # Early stopping
    group.add_argument("--early-stop", action="store_true", help="enable early stopping")
    group.add_argument(
        "--early-stop-patience",
        type=int,
        default=5,
        help="early stopping patience",
    )

    # paths
    group.add_argument("--train-data-path", type=str, default=None, help="train data path")
    group.add_argument("--val-data-path", type=str, default=None, help="validation data path")


def add_opt_opts(parser):
    group = parser.add_argument_group("optimize_weights")

    group.add_argument(
        "--config-path",
        "-c",
        type=str,
        required=True,
        help="pipeline config (ffn_pipeline.yaml or gnn_pipeline.yaml)",
    )
    group.add_argument("--val-data", type=str, default=None, help="val data for optimization")
    group.add_argument("--n-trials", type=int, default=30, help="num of optimization trials ")
    group.add_argument("--sample-size", type=int, default=1000, help="num of validation samples")
    group.add_argument(
        "--use-top-k", type=int, default=10, help="selected top-k acc as the objective"
    )


def add_predict_opts(parser):
    group = parser.add_argument_group("quarc_predict")

    group.add_argument(
        "--config-path",
        "-c",
        type=str,
        default="ffn_pipeline.yaml",
        help="pipeline config (ffn_pipeline.yaml or gnn_pipeline.yaml)",
    )
    group.add_argument(
        "--input", "-i", type=str, required=True, help="input JSON file with reactions"
    )
    group.add_argument(
        "--output", "-o", type=str, required=True, help="output JSON file for predictions"
    )
    group.add_argument("--top-k", type=int, default=5, help="number of top predictions to return")


def add_evaluation_opts(parser):
    group = parser.add_argument_group("quarc_evaluation")

    group.add_argument(
        "--checkpoint-dir", type=str, help="directory with model checkpoints to evaluate"
    )
    group.add_argument("--test-data-path", type=str, help="test data file for evaluation")
    group.add_argument(
        "--beam-size", type=int, default=10, help="beam size for stage 1 evaluation"
    )
    group.add_argument(
        "--chunk-size", type=int, default=500, help="chunk size for parallel evaluation"
    )
    group.add_argument(
        "--n-processes", type=int, default=16, help="number of processes for evaluation"
    )
