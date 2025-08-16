import time
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from tqdm import tqdm

# Import VAE model and preprocessing function from VAE_trainer.py
# Make sure VAE_trainer.py is accessible in the same environment or path
from VAE_trainer import TransformerVAE_TDist, preprocess_minute_data
import torch.nn.functional as F
SEQ_LENGTH=345

class FinancialDataset(Dataset):
    def __init__(self, latent_csv_path, seq_length=345, latent_dim=16, predict_raw=True):
        """
        Args:
            latent_csv_path (str): Path to latent features CSV (output from VAE processing)
            ohlcv_csv_path (str): Path to original OHLCV data CSV
            seq_length (int): Number of minutes in each sequence (window size)
            latent_dim (int): Dimension of latent features (e.g., 16)
            predict_raw (bool): Whether to predict raw OHLCV or normalized (currently predict normalized)
        """
        self.seq_length = seq_length
        self.latent_dim = latent_dim
        self.predict_raw = predict_raw  # Not used in current implementation for prediction output

        # Load both datasets
        latent_df = pd.read_csv(latent_csv_path)

        # --- Data alignment: Merge on date column ---
        # The latent_df only contains dates that passed preprocessing.
        # We need to filter ohlcv_df similarly to ensure alignment.
        # This part assumes that the latent_csv_path contains a 'date' column that aligns
        # with the 'date' column in ohlcv_csv_path after preprocessing.
        # If your latent_csv_path does not have 'date', you need to re-think how to align.

        # Ensure 'day' column is created for proper filtering if needed for original data
        # This part is simplified for direct merge, but consider if ohlcv_df needs
        # the same filtering as done in preprocess_minute_data if not already aligned.
        self.df = latent_df

        # Extract features
        self.dates = self.df['date'].values
        # Correct latent column names: 'latent_dim_1', 'latent_dim_2', etc.
        latent_cols = [f'latent_dim_{i + 1}' for i in range(latent_dim)]

        # OHLCV features now include 'amount' as per VAE's FEATURE_DIM=6 assumption
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume',]

        self.latent_features = self.df[latent_cols].values.astype(np.float32)
        self.ohlcv_features = self.df[ohlcv_cols].values.astype(np.float32)

        # Create scalers for both latent and OHLCV features
        self.latent_scaler = StandardScaler()
        self.normalized_latent = self.latent_scaler.fit_transform(self.latent_features)

        # For OHLCV, use log transform then standardization as in VAE_trainer.py
        self.ohlcv_scaler = StandardScaler()
        # Apply log transform to OHLCV features before fitting scaler
        # Add a small constant to avoid log(0) for volume/amount if they can be zero
        log_ohlcv_features = np.log1p(self.ohlcv_features)  # Using log1p which is log(1+x)
        self.ohlcv_scaler.fit(log_ohlcv_features)
        self.normalized_ohlcv = self.ohlcv_scaler.transform(log_ohlcv_features)

    def __len__(self):
        # We need seq_length for input and seq_length for target (shifted by 1)
        # So we need seq_length + 1 total points for each sample.
        return len(self.df) - self.seq_length  # Original was - self.seq_length, keeping for consistency with prev logic

    def __getitem__(self, idx):
        # Get sequences:
        # Input sequence for latent features: from idx to idx + seq_length - 1
        # Target sequence for OHLCV features: from idx + 1 to idx + seq_length
        # This means input [T:T+seq_len-1] predicts output [T+1:T+seq_len]
        latent_input_seq = self.normalized_latent[idx: idx + self.seq_length]
        ohlcv_target_seq = self.normalized_ohlcv[idx + 1: idx + self.seq_length + 1]

        # Target is the next sequence of OHLCV
        combined_target = {
            'ohlcv': ohlcv_target_seq
        }

        # Convert to tensors
        input_tensor = torch.FloatTensor(latent_input_seq)
        target_tensors = {
            k: torch.FloatTensor(v) for k, v in combined_target.items()
        }

        return input_tensor, target_tensors


def get_data_loaders_with_validation(latent_csv_path, batch_size=32, seq_length=345, latent_dim=16,
                                     validation_split=0.15):
    """
    Creates and returns training and validation DataLoaders with a chronological split.
    """
    full_dataset = FinancialDataset(latent_csv_path, seq_length, latent_dim)

    # Perform a chronological split
    total_samples = len(full_dataset)
    split_index = int(total_samples * (1 - validation_split))

    train_dataset = Subset(full_dataset, range(split_index))
    val_dataset = Subset(full_dataset, range(split_index, total_samples))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Display split information
    num_train_sequences = len(train_dataset)
    num_val_sequences = len(val_dataset)
    # A "day" is approximated by the number of sequences of length seq_length
    num_train_days = num_train_sequences / seq_length
    num_val_days = num_val_sequences / seq_length

    print("-" * 50)
    print("Dataset Split Information")
    print(f"Total sequences: {total_samples}")
    print(f"Training set:   {num_train_sequences} sequences (~{num_train_days:.1f} days)")
    print(f"Validation set: {num_val_sequences} sequences (~{num_val_days:.1f} days)")
    print("-" * 50)

    # Return loaders and the scalers from the full dataset
    return train_loader, val_loader, full_dataset.latent_scaler, full_dataset.ohlcv_scaler


class SelfAttention(nn.Module):
    """
    Custom Multi-Head Self-Attention module.
    """

    def __init__(self, n_embd, n_head, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        # Key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # Regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        # Causal mask to ensure that attention is only paid to the left
        self.register_buffer("bias", torch.tril(torch.ones(SEQ_LENGTH, SEQ_LENGTH)).view(1, 1, SEQ_LENGTH, SEQ_LENGTH))
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x, layer_past=None, attention_mask=None, is_causal=True):
        B, T, C = x.size()  # Batch, Sequence Length, Embedding Dimension

        # calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Handle past key values
        if layer_past is not None:
            past_key, past_value = layer_past
            # Concatenate along sequence dimension
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        present = (k, v)  # Cache for next step

        # Causal self-attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.size(-1)))

        if is_causal:
            # Create causal mask for current sequence length
            mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
            att = att.masked_fill(mask == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Re-assemble all head outputs

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, present


class TransformerBlock(nn.Module):
    """
    A single Transformer block (SelfAttention + MLP).
    """

    def __init__(self, n_embd, n_head, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),  # Use GELU as common in modern Transformers
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x, layer_past=None, attention_mask=None, is_causal=True):
        attn_output, present = self.attn(self.ln_1(x), layer_past=layer_past, attention_mask=attention_mask,
                                         is_causal=is_causal)
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return x, present  # Return output and updated cache for this block


class CustomTransformer(nn.Module):
    """
    A stack of Transformer Blocks. This replaces GPT2Model.
    """

    def __init__(self, n_embd, n_head, n_layer, seq_length, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        self.n_embd = n_embd
        self.seq_length = seq_length
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, attn_pdrop, resid_pdrop)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)  # Final layer norm

    def forward(self, x, past_key_values=None, use_cache=False, position_ids=None):
        # x is inputs_embeds
        # past_key_values is a list of tuples (key, value) for each layer from previous steps

        if past_key_values is None:
            past_key_values = [None] * len(self.blocks)

        presents = []  # To store new past_key_values for current step

        # Add positional embeddings to the input
        # For full sequence (initial call), position_ids specify actual positions (0 to T-1)
        # For single token (subsequent calls), position_ids specify its absolute position
        if position_ids is None:
            # Default to 0...T-1 if not provided, handles first full sequence
            position_ids = torch.arange(x.size(1), dtype=torch.long, device=x.device).unsqueeze(0)

        # Positional embeddings should be added before first layer, based on absolute positions
        # self.pos_emb should be defined in LatentOHLCVPredictor, passed here if needed,
        # or handled by adding to x before this call.
        # For simplicity, if we assume pos_emb is handled at LatentOHLCVPredictor level.

        for i, block in enumerate(self.blocks):
            # Each block takes its previous output and corresponding past_key_value
            x, present = block(x, layer_past=past_key_values[i], is_causal=True)  # Assuming causal for all blocks
            presents.append(present)

        x = self.ln_f(x)  # Final layer normalization

        outputs = {
            'last_hidden_state': x,
        }
        if use_cache:
            outputs['past_key_values'] = presents  # Store the updated caches

        return outputs


# --- Predictor Model Class ---
class LatentOHLCVPredictor(nn.Module):
    def __init__(self, latent_dim=16, ohlcv_dim=5, seq_length=345, n_layer=4, n_head=8, n_embd=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.ohlcv_dim = ohlcv_dim
        self.seq_length = seq_length
        self.n_embd = n_embd

        self.input_proj = nn.Linear(self.latent_dim, n_embd)

        # Positional embeddings for the maximum possible sequence length
        # This will be sliced based on the actual sequence length of the input.
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_length, n_embd))

        # Replace GPT2Model with CustomTransformer
        self.transformer = CustomTransformer(
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            seq_length=seq_length,
            attn_pdrop=0.1,
            resid_pdrop=0.1
        )

        self.ohlcv_head = nn.Linear(n_embd, self.ohlcv_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x, past_key_values=None, use_cache=False, position_ids=None):
        batch_size, seq_len = x.shape[0], x.shape[1]

        # Project input latent features to transformer embedding dimension
        x = self.input_proj(x)

        # Add positional embeddings
        # If past_key_values is None, it's the first full sequence input
        # Otherwise, position_ids will be for the single new token
        if position_ids is None:  # This should only happen for the very first call with full sequence
            position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0)

        # Ensure positional embeddings cover the range defined by position_ids
        # Here, x will be the `inputs_embeds` for the CustomTransformer
        # The CustomTransformer expects `x` to be the input embeddings.
        # Positional embeddings need to be added to `x` before passing to transformer blocks.
        # We need to map `position_ids` to the `self.pos_emb` tensor.
        # If position_ids is e.g., [SEQ_LENGTH], we take self.pos_emb[:, SEQ_LENGTH:SEQ_LENGTH+1]

        x = x + self.pos_emb[:, position_ids.squeeze(0), :]  # Apply positional embeddings using correct indices

        # Pass through CustomTransformer
        transformer_outputs = self.transformer(
            x,  # This is already inputs_embeds with positional info
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_ids=position_ids  # Pass position_ids for internal handling in transformer if needed
            # (though our custom one relies on position_ids mapping before passing to blocks)
        )
        hidden_states = transformer_outputs['last_hidden_state']

        # Predict OHLCV + Amount from transformer hidden states
        ohlcv_pred = self.ohlcv_head(hidden_states)

        outputs = {
            'ohlcv_pred': ohlcv_pred,
        }

        if use_cache:
            outputs['past_key_values'] = transformer_outputs['past_key_values']

        return outputs

    def generate(self, initial_latent_seq, vae_model, ohlcv_scaler, latent_scaler, steps=345, device='cpu',
                 temperature=1.0, use_cache=True):
        self.eval()
        vae_model.eval()

        with torch.no_grad():
            past_key_values = None
            generated_ohlcv_list = []

            # First pass with full sequence
            current_input = initial_latent_seq.to(device)
            outputs = self(current_input, past_key_values=past_key_values, use_cache=use_cache)
            predicted_next_ohlcv = outputs['ohlcv_pred'][:, -1:, :]
            generated_ohlcv_list.append(predicted_next_ohlcv)

            if use_cache:
                past_key_values = outputs['past_key_values']

            # Autoregressive generation loop
            for i in tqdm(range(steps - 1), desc="Generating"):
                # Process through VAE
                predicted_next_ohlcv_np = ohlcv_scaler.inverse_transform(
                    predicted_next_ohlcv.cpu().numpy().squeeze(0))
                predicted_next_ohlcv_raw = np.expm1(predicted_next_ohlcv_np)
                # print(predicted_next_ohlcv_raw)
                input_for_vae = torch.FloatTensor(predicted_next_ohlcv_np).to(device)
                mu_vae, log_scale_vae, log_df_vae = vae_model.encode(input_for_vae)
                new_latent_z, _ = vae_model.reparameterize(mu_vae, log_scale_vae, log_df_vae)

                # Normalize latent
                new_latent_z_normalized = torch.FloatTensor(
                    latent_scaler.transform(new_latent_z.cpu().numpy())
                ).to(device).unsqueeze(1)

                # Next prediction step
                outputs = self(new_latent_z_normalized,
                               past_key_values=past_key_values,
                               use_cache=use_cache)

                predicted_next_ohlcv = outputs['ohlcv_pred'][:, -1:, :]
                generated_ohlcv_list.append(predicted_next_ohlcv)

                if use_cache:
                    past_key_values = outputs['past_key_values']

            return {
                'ohlcv': torch.cat(generated_ohlcv_list, dim=1)
            }

def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # for multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

import os, argparse

def main():
    parser = argparse.ArgumentParser(
        description="Train a Transformer Predictor model for OHLCV from VAE latent features.")

    # Data arguments
    parser.add_argument('--symbol', type=str, required=True,
                        help='The target symbol for training (e.g., c9999, rb9999).')
    parser.add_argument('--latent_data_dir', type=str, default='./data/latent_features',
                        help='Directory containing latent feature CSV files (e.g., symbol_1min_data.csv).')
    parser.add_argument('--vae_model_dir', type=str, default='./models',
                        help='Directory containing trained VAE models (e.g., symbol_tdist_vae_model.pth).')
    parser.add_argument('--batch_size', type=int, default=1,  # Changed default to 1 as requested
                        help='Batch size for training DataLoader.')
    parser.add_argument('--seq_length', type=int, default=345,
                        help='Sequence length (window size) for each sample.')
    parser.add_argument('--validation_split', type=float, default=0.05,
                        help='Proportion of data to use for validation.')

    # Model arguments
    parser.add_argument('--vae_latent_dim', type=int, default=16,
                        help='Dimension of the VAE latent space.')
    parser.add_argument('--ohlcv_dim', type=int, default=5,  # Default to 6 (OHLCV + Amount)
                        help='Dimension of OHLCV features (OHLCV + Volume + Amount).')
    parser.add_argument('--predictor_n_layer', type=int, default=4,
                        help='Number of Transformer blocks in the predictor.')
    parser.add_argument('--predictor_n_head', type=int, default=8,
                        help='Number of attention heads in the predictor Transformer.')
    parser.add_argument('--predictor_n_embd', type=int, default=256,
                        help='Embedding dimension for the predictor Transformer.')

    # VAE Model config (for loading VAE model, these should match VAE training)
    parser.add_argument('--vae_feature_dim', type=int, default=7,
                        # Assuming VAE trained with 7 features (OHLCV, Vol, Amt, Pos)
                        help='Feature dimension used when training the VAE model (input to VAE).')
    parser.add_argument('--vae_embed_dim', type=int, default=64,
                        help='Embedding dimension used when training the VAE model.')
    parser.add_argument('--vae_df_param', type=float, default=5.0,
                        help='Initial degrees of freedom used when training the VAE model.')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=1,  # Changed default to 1 as requested
                        help='Number of training epochs for the predictor.')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate for the AdamW optimizer.')
    parser.add_argument('--predictor_output_dir', type=str, default='./predictor_models',
                        # New directory for predictor models
                        help='Directory to save trained predictor models and plots.')
    parser.add_argument('--seed', type=int, default=42,  # Random seed
                        help='Random seed for reproducibility.')

    args = parser.parse_args()

    # Set random seed at the very beginning
    set_seed(args.seed)

    # Construct paths
    # Latent CSV path example: ./data/latent_features/c9999_1min_data.csv
    latent_csv_path = os.path.join(args.latent_data_dir, f'{args.symbol}_1min_data.csv')
    # VAE model path example: ./models/c9999_tdist_vae_model.pth
    vae_model_path = os.path.join(args.vae_model_dir, f'{args.symbol}_tdist_vae_model.pth')
    # Predictor model save path
    predictor_model_save_path = os.path.join(args.predictor_output_dir, f'{args.symbol}_predictor_model.pth')

    # Ensure output directory for predictor models exists
    os.makedirs(args.predictor_output_dir, exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # --- Data Preparation with Validation Split ---
    # `actual_ohlcv_dim` is returned from the dataset to ensure consistency with what's loaded
    train_loader, val_loader, latent_scaler, ohlcv_scaler = get_data_loaders_with_validation(
        latent_csv_path,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        latent_dim=args.vae_latent_dim,
        validation_split=args.validation_split
    )
    # Override ohlcv_dim with the actual dimension determined by the dataset
    print(f"Detected OHLCV dimension from data: {args.ohlcv_dim}")

    # --- Initialize and Load VAE Model (for autoregressive feedback) ---
    vae_model = TransformerVAE_TDist(
        feature_dim=args.vae_feature_dim,
        latent_dim=args.vae_latent_dim,
        embed_dim=args.vae_embed_dim,
        df=args.vae_df_param
    ).to(DEVICE)

    try:
        vae_model.load_state_dict(torch.load(vae_model_path, map_location=DEVICE))
        vae_model.eval()  # Set VAE to evaluation mode
        print(f"Successfully loaded VAE model from {vae_model_path}")
    except FileNotFoundError:
        print(f"Error: VAE model not found at {vae_model_path}. Please train VAE first or provide correct path.")
        exit(1)  # Exit if VAE model is not found

    # --- Initialize Predictor Model ---
    model = LatentOHLCVPredictor(
        latent_dim=args.vae_latent_dim,
        ohlcv_dim=args.ohlcv_dim,
        seq_length=args.seq_length,
        n_layer=args.predictor_n_layer,
        n_head=args.predictor_n_head,
        n_embd=args.predictor_n_embd
    ).to(DEVICE)

    # Optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()  # Using MSELoss for normalized OHLCV + Amount

    # --- Training & Validation Loop ---
    best_val_loss = float('inf')
    print(f"\n--- Starting Predictor Training for symbol: {args.symbol} ---")
    for epoch in range(args.epochs):
        # -- Training Phase --
        model.train()
        total_train_loss = 0
        start_time = time.time()
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} Training"):
            inputs, ohlcv_target = inputs.to(DEVICE), targets['ohlcv'].to(DEVICE)  # targets is now directly ohlcv_target
            outputs = model(inputs)
            loss = criterion(outputs['ohlcv_pred'], ohlcv_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        # -- Validation Phase --
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} Validation"):
                inputs, ohlcv_target = inputs.to(DEVICE), targets['ohlcv'].to(DEVICE)  # targets is now directly ohlcv_target
                outputs = model(inputs)
                loss = criterion(outputs['ohlcv_pred'], ohlcv_target)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)

        epoch_time = time.time() - start_time
        print(
            f"Epoch {epoch + 1}/{args.epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Time: {epoch_time:.2f}s")

        # Save model if validation loss has improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), predictor_model_save_path)
            print(
                f"Validation loss improved to {best_val_loss:.6f}. Saved new best model to {predictor_model_save_path}.")

    # --- Test Generation (using the best model saved) ---
    print("\nLoading best model for test generation...")
    # Load the best performing model before generation
    if os.path.exists(predictor_model_save_path):
        model.load_state_dict(torch.load(predictor_model_save_path, map_location=DEVICE))
        model.eval()  # Set to eval mode for generation
    else:
        print(f"Warning: Best predictor model not found at {predictor_model_save_path}. Skipping generation.")
        return

    print("Testing autoregressive generation...")
    # Get an initial sequence from the validation set for a more realistic test
    # Ensure val_loader is not empty
    if len(val_loader) > 0:
        initial_latent_for_gen, _ = next(iter(val_loader))
        initial_latent_for_gen = initial_latent_for_gen[:1, :, :].to(DEVICE)  # Take first sequence from batch

        # Generation steps should ideally be seq_length to generate one full sequence
        generated_output = model.generate(initial_latent_for_gen, vae_model, ohlcv_scaler, latent_scaler,
                                          steps=args.seq_length, device=DEVICE, use_cache=True)

        generated_ohlcv_normalized_np = generated_output['ohlcv'].squeeze(0).cpu().numpy()

        # Inverse transform the normalized (log1p'd) values back to original log1p scale
        generated_ohlcv_scaled_back = ohlcv_scaler.inverse_transform(generated_ohlcv_normalized_np)

        # Finally, inverse log1p to get raw values
        generated_ohlcv_final_raw = np.expm1(generated_ohlcv_scaled_back)

        print(f"\nGenerated OHLCV+Amount sample for {args.symbol} (first 5 steps):")
        print(generated_ohlcv_final_raw[:5])
    else:
        print("Validation loader is empty. Skipping generation test.")

    print(f"Predictor training and generation test completed for {args.symbol}.")


if __name__ == "__main__":
    main()