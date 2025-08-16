#!/bin/bash

# Set environment variables for UTF-8 encoding
export PYTHONIOENCODING=utf-8
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

echo "Starting master data processing script..."

# Define the base directory for raw minute data
RAW_DATA_DIR="./data/raw_data"
# Define the directory to save models and plots
MODEL_OUTPUT_DIR="./models"

# Ensure the model output directory exists
mkdir -p "$MODEL_OUTPUT_DIR"

# List of symbols to train on - **USING THE EXACT CASING AS PROVIDED**
SYMBOLS=(
    "rb9999" "i9999" "cu9999" "ni9999" "sc9999"
    "pg9999" "y9999" "ag9999"
    "m9999" "c9999"
    "TA9999" "UR9999" "OI9999" "au9999" "IH9999" "T9999"
    "CF9999" "AP9999"
)

# Training Hyperparameters (you can adjust these)
EPOCHS=50
BATCH_SIZE=345
LEARNING_RATE=2e-4
LATENT_DIM=16
EMBED_DIM=64
NHEAD=4
NUMLAYERS=3
DF_INITIAL=5.0
TEST_SIZE=0.05

# --- tmux session setup ---
SESSION_NAME="vae_training_$(date +%Y%m%d%H%M)"
echo "Starting tmux session: $SESSION_NAME"

# Create a new tmux session (without attaching)
tmux new-session -d -s "$SESSION_NAME"

# Iterate through symbols and create a new window for each
FIRST_SYMBOL=true
for SYMBOL in "${SYMBOLS[@]}"; do
    echo "Launching training for $SYMBOL in tmux window..."

    # If it's the first symbol, rename the default window
    if [ "$FIRST_SYMBOL" = true ]; then
        tmux rename-window -t "$SESSION_NAME:0" "$SYMBOL"
        FIRST_SYMBOL=false
    else
        # For subsequent symbols, create a new window
        tmux new-window -t "$SESSION_NAME" -n "$SYMBOL"
    fi

    # Construct the command to run the Python script
    # Ensure `python3` points to your Python 3 executable
    TRAIN_COMMAND="python VAE_trainer.py \
        --symbol \"$SYMBOL\" \
        --data_file_path \"${RAW_DATA_DIR}/${SYMBOL}_1min_data.csv\" \
        --batch_size \"$BATCH_SIZE\" \
        --test_size \"$TEST_SIZE\" \
        --epochs \"$EPOCHS\" \
        --learning_rate \"$LEARNING_RATE\" \
        --latent_dim \"$LATENT_DIM\" \
        --embed_dim \"$EMBED_DIM\" \
        --nhead \"$NHEAD\" \
        --num_layers \"$NUMLAYERS\" \
        --df_initial \"$DF_INITIAL\" \
        --model_output_dir \"$MODEL_OUTPUT_DIR\""

    # Send the command to the tmux window
    tmux send-keys -t "$SESSION_NAME:$SYMBOL" "$TRAIN_COMMAND" C-m

    # Optional: Add a small delay between launching windows to prevent resource spikes
    sleep 1
done

echo "All training tasks launched in tmux session '$SESSION_NAME'."
echo "You can attach to the session using: tmux attach-session -t $SESSION_NAME"
echo "To list sessions: tmux ls"
echo "To detach from session: Ctrl-b d"
echo "To move between windows: Ctrl-b n (next) or Ctrl-b p (previous)"
echo "To kill session: tmux kill-session -t $SESSION_NAME"