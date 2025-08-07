#!/bin/bash

mkdir -p checkpoints/agent
mkdir -p checkpoints/temperature
mkdir -p checkpoints/reactant_amount
mkdir -p checkpoints/agent_amount

GNN_AGENT_MODEL_URL="https://www.dropbox.com/scl/fi/5hxd7ir4wj0p5nuflxpf3/gnn_agent_model_oss.ckpt?rlkey=m9mfxssehx9xe324fhszzpy4y&st=05pescoj&dl=1"
GNN_TEMPERATURE_MODEL_URL="https://www.dropbox.com/scl/fi/8dpog5knvywpflm5p6w7s/gnn_temperature_model.ckpt?rlkey=bku14kwg9kkjmscm7qdp337f3&dl=1"
GNN_REACTANT_AMOUNT_MODEL_URL="https://www.dropbox.com/scl/fi/n62w8heoaqof1ed5h99yy/gnn_reactant_amount_model.ckpt?rlkey=wtum2xblvdi88hvz00ppppaf9&dl=1"
GNN_AGENT_AMOUNT_MODEL_URL="https://www.dropbox.com/scl/fi/6bltnef2gvm9s2m7di8zo/gnn_agent_amount_model.ckpt?rlkey=z36a2jfjzezramktv5x8gy72a&st=mnpedaq1&dl=1"

FFN_AGENT_MODEL_URL="https://www.dropbox.com/scl/fi/82l0me0n3v4peaungyi1t/ffn_agent_model.ckpt?rlkey=2dg0cxou27pidqdlluvvlt3oj&dl=1"
FFN_TEMPERATURE_MODEL_URL="https://www.dropbox.com/scl/fi/v2vswiy4l1hte38e81sij/ffn_temperature_model.ckpt?rlkey=9zajegwnlar3rfokejkizwz6j&st=wx1w4dyd&dl=1"
FFN_REACTANT_AMOUNT_MODEL_URL="https://www.dropbox.com/scl/fi/3f1lm61uurdpk30cvrbqo/ffn_reactant_amount_model.ckpt?rlkey=ix869p7i98bg1j5bp1dvt5vqe&st=w1q3rep4&dl=1"
FFN_AGENT_AMOUNT_MODEL_URL="https://www.dropbox.com/scl/fi/mt9vsiql4e4p5df7y9adh/ffn_agent_amount_model.ckpt?rlkey=232i309za50w8epgyk29kic3w&st=bib02ht9&dl=1"

# Download GNN agent model
if [ ! -f checkpoints/agent/gnn_agent_model.ckpt ]; then
    wget -q --show-progress -O checkpoints/agent/gnn_agent_model.ckpt \
      "${GNN_AGENT_MODEL_URL}"
    echo "checkpoints/agent/gnn_agent_model.ckpt Downloaded."
fi

# Download GNN temperature model
if [ ! -f checkpoints/temperature/gnn_temperature_model.ckpt ]; then
    wget -q --show-progress -O checkpoints/temperature/gnn_temperature_model.ckpt \
      "${GNN_TEMPERATURE_MODEL_URL}"
    echo "checkpoints/temperature/gnn_temperature_model.ckpt Downloaded."
fi

# Download GNN reactant amount model

if [ ! -f checkpoints/reactant_amount/gnn_reactant_amount_model.ckpt ]; then
    wget -q --show-progress -O checkpoints/reactant_amount/gnn_reactant_amount_model.ckpt \
      "${GNN_REACTANT_AMOUNT_MODEL_URL}"
    echo "checkpoints/reactant_amount/gnn_reactant_amount_model.ckpt Downloaded."
fi

# Download GNN agent amount model
if [ ! -f checkpoints/agent_amount/gnn_agent_amount_model.ckpt ]; then
    wget -q --show-progress -O checkpoints/agent_amount/gnn_agent_amount_model.ckpt \
      "${GNN_AGENT_AMOUNT_MODEL_URL}"
    echo "checkpoints/agent_amount/gnn_agent_amount_model.ckpt Downloaded."
fi

# Download FFN agent model
if [ ! -f checkpoints/agent/ffn_agent_model.ckpt ]; then
    wget -q --show-progress -O checkpoints/agent/ffn_agent_model.ckpt \
      "${FFN_AGENT_MODEL_URL}"
    echo "checkpoints/agent/ffn_agent_model.ckpt Downloaded."
fi

# Download FFN temperature model
if [ ! -f checkpoints/temperature/ffn_temperature_model.ckpt ]; then
    wget -q --show-progress -O checkpoints/temperature/ffn_temperature_model.ckpt \
      "${FFN_TEMPERATURE_MODEL_URL}"
    echo "checkpoints/temperature/ffn_temperature_model.ckpt Downloaded."
fi

# Download FFN reactant amount model
if [ ! -f checkpoints/reactant_amount/ffn_reactant_amount_model.ckpt ]; then
    wget -q --show-progress -O checkpoints/reactant_amount/ffn_reactant_amount_model.ckpt \
      "${FFN_REACTANT_AMOUNT_MODEL_URL}"
    echo "checkpoints/reactant_amount/ffn_reactant_amount_model.ckpt Downloaded."
fi

# Download FFN agent amount model
if [ ! -f checkpoints/agent_amount/ffn_agent_amount_model.ckpt ]; then
    wget -q --show-progress -O checkpoints/agent_amount/ffn_agent_amount_model.ckpt \
      "${FFN_AGENT_AMOUNT_MODEL_URL}"
    echo "checkpoints/agent_amount/ffn_agent_amount_model.ckpt Downloaded."
fi

echo "All models downloaded to checkpoints/ directory."
