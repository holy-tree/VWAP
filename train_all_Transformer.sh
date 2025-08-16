#!/bin/bash

# Set environment variables for UTF-8 encoding
export PYTHONIOENCODING=utf-8
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

echo "Starting predictor training script..."

# Define base directories
# This directory should contain files like c9999_1min_data.csv, rb9999_1min_data.csv etc.
LATENT_DATA_BASE_DIR="./data/latent_features"
# This directory should contain files like c9999_tdist_vae_model.pth, rb9999_tdist_vae_model.pth etc.
VAE_MODEL_BASE_DIR="./models"
# This directory is where the trained predictor models will be saved
PREDICTOR_OUTPUT_DIR="./predictor_models"

# Ensure output directory for predictor models exists
mkdir -p "$PREDICTOR_OUTPUT_DIR"

# List of symbols to train on - **USING THE EXACT CASING AS PROVIDED**
SYMBOLS=(
    "rb9999" "i9999" "cu9999" "ni9999" "sc9999"
    "pg9999" "y9999" "ag9999"
    "m9999" "c9999"
    "TA9999" "UR9999" "OI9999" "au9999" "IH9999" "T9999"
    "CF9999" "AP9999"
)

# Predictor Training Hyperparameters (as requested)
EPOCHS=10
BATCH_SIZE=1
LEARNING_RATE=2e-5
SEQ_LENGTH=345 # Default sequence length

# Model architecture hyperparameters (should match your desired predictor config)
VAE_LATENT_DIM=16
PREDICTOR_N_LAYER=4
PREDICTOR_N_HEAD=8
PREDICTOR_N_EMBD=256

# VAE model config (should match how your VAE models were trained)
VAE_FEATURE_DIM=5 # Adjust if your VAE was trained with 6 features
VAE_EMBED_DIM=64
VAE_DF_PARAM=5.0

# Random seed for reproducibility
SEED=42

# --- tmux session setup ---
SESSION_NAME="predictor_training_$(date +%Y%m%d%H%M)"
echo "Starting tmux session: $SESSION_NAME"

# Create a new tmux session (without attaching)
tmux new-session -d -s "$SESSION_NAME"

# Iterate through symbols and create a new window for each
FIRST_SYMBOL=true
for SYMBOL in "${SYMBOLS[@]}"; do
    echo "Launching predictor training for $SYMBOL in tmux window..."

    # Construct symbol-specific latent data file path
    LATENT_DATA_FILE="${LATENT_DATA_BASE_DIR}/${SYMBOL}_1min_data.csv"
    # Construct symbol-specific VAE model path
    VAE_MODEL_FILE="${VAE_MODEL_BASE_DIR}/${SYMBOL}_tdist_vae_model.pth"

    # Check if latent data file exists
    if [ ! -f "$LATENT_DATA_FILE" ]; then
        echo "Error: Latent data file for $SYMBOL not found at $LATENT_DATA_FILE. Skipping this symbol."
        continue # Skip to the next symbol
    fi

    # Check if VAE model file exists
    if [ ! -f "$VAE_MODEL_FILE" ]; then
        echo "Error: VAE model for $SYMBOL not found at $VAE_MODEL_FILE. Skipping this symbol."
        continue # Skip to the next symbol
    fi

    # If it's the first symbol, rename the default window
    if [ "$FIRST_SYMBOL" = true ]; then
        tmux rename-window -t "$SESSION_NAME:0" "$SYMBOL"
        FIRST_SYMBOL=false
    else
        # For subsequent symbols, create a new window
        tmux new-window -t "$SESSION_NAME" -n "$SYMBOL"
    fi

    # Construct the command to run the Python script
    # Pass the full paths for latent_data_path and vae_model_path
    TRAIN_COMMAND="python3 dataloader_setup.py \
        --symbol \"$SYMBOL\" \
        --latent_data_dir \"$LATENT_DATA_BASE_DIR\" \
        --vae_model_dir \"$VAE_MODEL_BASE_DIR\" \
        --predictor_output_dir \"$PREDICTOR_OUTPUT_DIR\" \
        --batch_size \"$BATCH_SIZE\" \
        --seq_length \"$SEQ_LENGTH\" \
        --epochs \"$EPOCHS\" \
        --learning_rate \"$LEARNING_RATE\" \
        --vae_latent_dim \"$VAE_LATENT_DIM\" \
        --predictor_n_layer \"$PREDICTOR_N_LAYER\" \
        --predictor_n_head \"$PREDICTOR_N_HEAD\" \
        --predictor_n_embd \"$PREDICTOR_N_EMBD\" \
        --vae_feature_dim \"$VAE_FEATURE_DIM\" \
        --vae_embed_dim \"$VAE_EMBED_DIM\" \
        --vae_df_param \"$VAE_DF_PARAM\" \
        --seed \"$SEED\""

    # Send the command to the tmux window
    tmux send-keys -t "$SESSION_NAME:$SYMBOL" "$TRAIN_COMMAND" C-m

    # Optional: Add a small delay between launching windows
    sleep 1
done

echo "All predictor training tasks launched in tmux session '$SESSION_NAME'."
echo "You can attach to the session using: tmux attach-session -t $SESSION_NAME"
echo "To list sessions: tmux ls"
echo "To detach from session: Ctrl-b d"
echo "To move between windows: Ctrl-b n (next) or Ctrl-b p (previous)"
echo "To kill session: tmux kill-session -t $SESSION_NAME"