import torch
import yaml
from pathlib import Path
from chemprop.nn import BondMessagePassing, MeanAggregation
from chemprop.featurizers import CondensedGraphOfReactionFeaturizer


def load_checkpoint_smart(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    for key in ["state_dict", "model_state_dict", "model"]:
        if key in checkpoint:
            return checkpoint[key]
    return checkpoint


def create_ffn_model(model_class_name, params):
    from quarc.models.ffn_models import (
        AgentFFNWithRxnClass,
        TemperatureFFN,
        ReactantAmountFFN,
        AgentAmountFFN,
    )
    from quarc.models.modules.ffn_heads import (
        FFNAgentHeadWithRxnClass,
        FFNTemperatureHead,
        FFNReactantAmountHead,
        FFNAgentAmountHead,
    )

    model_classes = {
        "AgentFFNWithRxnClass": (AgentFFNWithRxnClass, FFNAgentHeadWithRxnClass),
        "TemperatureFFN": (TemperatureFFN, FFNTemperatureHead),
        "ReactantAmountFFN": (ReactantAmountFFN, FFNReactantAmountHead),
        "AgentAmountFFN": (AgentAmountFFN, FFNAgentAmountHead),
    }

    model_cls, predictor_cls = model_classes[model_class_name]
    predictor_params = {
        "fp_dim": params["fp_dim"],
        "agent_input_dim": params["agent_input_dim"],
        "output_dim": params["output_dim"],
        "hidden_dim": params["hidden_dim"],
        "n_blocks": params["n_blocks"],
    }

    if "activation" in params:
        predictor_params["activation"] = params["activation"]

    predictor = predictor_cls(**predictor_params)
    return model_cls(predictor=predictor, metrics=[])


def create_gnn_model(model_class_name, params):
    from quarc.models.gnn_models import (
        AgentGNNWithRxnClass,
        TemperatureGNN,
        ReactantAmountGNN,
        AgentAmountOneshotGNN,
    )
    from quarc.models.modules.gnn_heads import (
        GNNAgentHeadWithRxnClass,
        GNNTemperatureHead,
        GNNReactantAmountHead,
        GNNAgentAmountHead,
    )

    model_classes = {
        "AgentGNNWithRxnClass": (AgentGNNWithRxnClass, GNNAgentHeadWithRxnClass),
        "TemperatureGNN": (TemperatureGNN, GNNTemperatureHead),
        "ReactantAmountGNN": (ReactantAmountGNN, GNNReactantAmountHead),
        "AgentAmountOneshotGNN": (AgentAmountOneshotGNN, GNNAgentAmountHead),
    }

    model_cls, predictor_cls = model_classes[model_class_name]
    predictor_params = {
        "graph_input_dim": params["graph_input_dim"],
        "agent_input_dim": params["agent_input_dim"],
        "output_dim": params["output_dim"],
        "hidden_dim": params["hidden_dim"],
        "n_blocks": params["n_blocks"],
    }
    predictor = predictor_cls(**predictor_params)

    fdims = CondensedGraphOfReactionFeaturizer(mode_="REAC_DIFF").shape

    return model_cls(
        message_passing=BondMessagePassing(
            *fdims,
            depth=params["gnn_depth"],
            d_h=params["graph_input_dim"],
        ),
        agg=MeanAggregation(),
        predictor=predictor,
        batch_norm=True,
        metrics=[],
    )


def load_models_from_yaml(config_path, device="cuda"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # use absolute path for running in notebooks. If running in CLI, the relative path is fine.
    base_dir = Path(config_path).parent.parent / config["base_model_dir"]

    models = {}
    model_types = {}
    for name, model_config in config["models"].items():
        if model_config["model_type"] == "ffn":
            model = create_ffn_model(
                model_config["model_class"],
                model_config["params"],
            )
            model_types[name] = "ffn"
        elif model_config["model_type"] == "gnn":
            model = create_gnn_model(
                model_config["model_class"],
                model_config["params"],
            )
            model_types[name] = "gnn"
        else:
            raise ValueError(f"Invalid model type: {model_config['model_type']}")

        checkpoint_path = base_dir / model_config["checkpoint_path"]
        state_dict = load_checkpoint_smart(checkpoint_path, device)

        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            model.load_state_dict(state_dict, strict=False)

        model.eval().to(device)
        models[name] = model

    weights = config.get("optimized_weights", {})

    return models, model_types, weights
