import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from VAE_trainer import TransformerVAE_TDist
from dataloader_setup import LatentOHLCVPredictor, FinancialDataset


def plot_ohlcv_comparison(original_data, generated_data, original_date, plot_type, base_output_dir, symbol):
    """
    Helper function to plot Close and Volume comparisons side-by-side.
    original_data and generated_data are expected to be raw OHLCV.
    plot_type is now used to create a subdirectory (e.g., "Full_Day_Generated", "Single_Step_Predicted").
    """
    # Assuming OHLCV order: Open, High, Low, Close, Volume, Turnover
    close_idx = 3  # Index for Close price
    volume_idx = 4 # Index for Volume

    # Create the symbol-specific and plot_type-specific output directory
    # Sanitize plot_type for directory name
    plot_type_dirname = plot_type.replace(' ', '_').replace('-', '_')
    current_output_dir = os.path.join(base_output_dir, symbol, plot_type_dirname)
    os.makedirs(current_output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(25, 7))

    # Plot Close Price
    axes[0].plot(original_data[:, close_idx], label='Original Close Price', color='blue', alpha=0.8)
    axes[0].plot(generated_data[:, close_idx], label=f'{plot_type} Close Price', color='red', linestyle='--')
    axes[0].set_title(f'Close Price Comparison - {symbol} ({original_date.split(" ")[0]})')
    axes[0].set_xlabel('Time Step (Minute of Day)')
    axes[0].set_ylabel('Price')
    axes[0].legend()
    axes[0].grid(True)

    # Plot Volume
    axes[1].plot(original_data[:, volume_idx], label='Original Volume', color='blue', alpha=0.8)
    axes[1].plot(generated_data[:, volume_idx], label=f'{plot_type} Volume', color='green', linestyle='--')
    axes[1].set_title(f'Volume Comparison - {symbol} ({original_date.split(" ")[0]})')
    axes[1].set_xlabel('Time Step (Minute of Day)')
    axes[1].set_ylabel('Volume')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    # Sanitize original_date for filename
    filename_date = original_date.split(" ")[0]
    plt.savefig(os.path.join(current_output_dir, f"{symbol}_{filename_date}.png")) # Removed plot_type from filename as it's in directory
    plt.close()


def visualize_last_n_days(predictor_model, vae_model, dataset, latent_scaler, ohlcv_scaler, n_days=5, seq_length=345,
                          device='cpu', output_dir='output_plots', symbol='Unknown'):
    """
    Generates predictions for the last n days and plots them against original data.
    """
    print(f"\n--- Starting Full-Day Generation for Last {n_days} Days ({symbol}) ---")
    predictor_model.eval()
    vae_model.eval()

    total_minutes = len(dataset.df)

    if total_minutes < (n_days + 1) * seq_length:
        print(f"Error: Not enough data. Need at least {(n_days + 1) * seq_length} minutes, but found {total_minutes}.")
        return

    # No need to create base output_dir here, it's done in plot_ohlcv_comparison based on symbol and type
    # os.makedirs(output_dir, exist_ok=True) # This line is no longer necessary here

    for i in range(1, n_days + 1):
        # --- 1. Get Input: Latent sequence from the PREVIOUS day ---
        # The day before the target day. e.g., for the last day (i=1), use the second-to-last day as input.
        start_idx_input = total_minutes - (i + 1) * seq_length
        end_idx_input = total_minutes - i * seq_length

        print(f"\nGenerating for target day starting at minute index {end_idx_input}...")
        print(f"Using input data from minute indices {start_idx_input} to {end_idx_input - 1}")

        initial_latent_seq_np = dataset.normalized_latent[start_idx_input:end_idx_input]
        initial_latent_for_gen = torch.FloatTensor(initial_latent_seq_np).unsqueeze(0).to(device)

        # --- 2. Get Ground Truth: Raw OHLCV from the TARGET day ---
        start_idx_target = end_idx_input
        end_idx_target = start_idx_target + seq_length

        original_ohlcv_raw = dataset.ohlcv_features[start_idx_target:end_idx_target]
        original_date = dataset.dates[start_idx_target]

        # --- 3. Generate ---
        generated_output = predictor_model.generate(
            initial_latent_for_gen, vae_model, ohlcv_scaler, latent_scaler,
            steps=seq_length, device=device, use_cache=True
        )

        # --- 4. Process Generated Data (Inverse Transform) ---
        generated_ohlcv_normalized_np = generated_output['ohlcv'].squeeze(0).cpu().numpy()
        generated_ohlcv_scaled_back = ohlcv_scaler.inverse_transform(generated_ohlcv_normalized_np)
        generated_ohlcv_final_raw = np.expm1(generated_ohlcv_scaled_back)

        # --- 5. Visualize Comparison ---
        plot_ohlcv_comparison(original_ohlcv_raw, generated_ohlcv_final_raw, original_date, "Full-Day Generated", output_dir, symbol)


def visualize_last_n_days_single_step(predictor_model, dataset, ohlcv_scaler, n_days=5, seq_length=345, device='cpu',
                                      output_dir='output_plots', symbol='Unknown'):
    """
    Generates single-step predictions for the last n days and plots them.
    Each prediction is based on the true historical data preceding it (rolling forecast).
    """
    print(f"\n--- Starting Single-Step Prediction for Last {n_days} Days ({symbol}) ---")
    predictor_model.eval()

    total_minutes = len(dataset.df)

    # We need at least seq_length minutes of history before the first prediction point.
    if total_minutes < (n_days * seq_length) + seq_length:
        print(f"Error: Not enough data. Need at least {(n_days * seq_length) + seq_length} minutes, but found {total_minutes}.")
        return

    # No need to create base output_dir here, it's done in plot_ohlcv_comparison based on symbol and type
    # os.makedirs(output_dir, exist_ok=True) # This line is no longer necessary here

    # Outer loop for each of the last N days
    for i in range(1, n_days + 1):
        day_start_index = total_minutes - i * seq_length
        day_end_index = day_start_index + seq_length

        original_ohlcv_raw = dataset.ohlcv_features[day_start_index:day_end_index]
        original_date = dataset.dates[day_start_index]

        print(f"\nPredicting for target day starting at minute index {day_start_index} (Date: {original_date})")

        all_single_step_preds_normalized = []

        # Inner loop for each minute within the target day
        for j in tqdm(range(seq_length), desc=f"Predicting Day T-{(n_days - i)} for {symbol}"):
            # Define the sliding window of historical data (seq_length minutes immediately preceding the target)
            start_idx_input = day_start_index + j - seq_length
            end_idx_input = day_start_index + j

            # Ensure start_idx_input is not negative
            if start_idx_input < 0:
                print(f"Warning: Not enough history for single-step prediction at minute {j}. Skipping this day.")
                all_single_step_preds_normalized = [] # Clear any partial predictions for this day
                break

            input_latent_seq_np = dataset.normalized_latent[start_idx_input:end_idx_input]
            input_tensor = torch.FloatTensor(input_latent_seq_np).unsqueeze(0).to(device)

            # --- Predict just one step forward using the standard forward pass ---
            with torch.no_grad():
                outputs = predictor_model(input_tensor)
                # The prediction for the next step is the last element in the output sequence
                next_step_pred_normalized = outputs['ohlcv_pred'][:, -1, :]  # Shape: [1, ohlcv_dim]

            all_single_step_preds_normalized.append(next_step_pred_normalized)

        if not all_single_step_preds_normalized: # Skip if no predictions were made for this day
            continue

        # --- Process all collected predictions for the day ---
        generated_ohlcv_normalized = torch.cat(all_single_step_preds_normalized, dim=0)
        generated_ohlcv_normalized_np = generated_ohlcv_normalized.cpu().numpy()
        generated_ohlcv_scaled_back = ohlcv_scaler.inverse_transform(generated_ohlcv_normalized_np)
        generated_ohlcv_final_raw = np.expm1(generated_ohlcv_scaled_back)

        # --- Visualize Comparison ---
        plot_ohlcv_comparison(original_ohlcv_raw, generated_ohlcv_final_raw, original_date, "Single-Step Predicted", output_dir, symbol)


def main():
    parser = argparse.ArgumentParser(description='Visualize OHLCV predictions.')
    parser.add_argument('--symbol', type=str, required=True, help='The financial symbol (e.g., c9999, ag9999).')
    parser.add_argument('--vae_path', type=str, default='best_tdist_vae_model.pth',
                        help='Path to the VAE model state_dict.')
    parser.add_argument('--predictor_path', type=str, default='best_GPTpredictor_model.pth',
                        help='Path to the Predictor model state_dict.')
    parser.add_argument('--output_dir', type=str, default='output_plots',
                        help='Base directory to save the visualization plots.') # Changed help message
    parser.add_argument('--n_days', type=int, default=5,
                        help='Number of last days to visualize.')

    args = parser.parse_args()

    # --- Configuration ---
    symbol = args.symbol
    latent_csv_path = f"data/latent_features/{symbol}_1min_data.csv"
    vae_model_path = args.vae_path
    predictor_model_path = args.predictor_path
    base_output_dir = args.output_dir # Renamed for clarity
    n_days = args.n_days

    seq_length = 345
    vae_latent_dim = 16
    ohlcv_dim = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Data Preparation ---
    print(f"Loading and preparing dataset for {symbol}...")
    try:
        full_dataset = FinancialDataset(
            latent_csv_path=latent_csv_path,
            seq_length=seq_length,
            latent_dim=vae_latent_dim
        )
    except FileNotFoundError:
        print(f"Error: Data file not found at {latent_csv_path}. Please check the path.")
        return
    print("Dataset loaded successfully.")

    # --- Load VAE Model ---
    vae_model = TransformerVAE_TDist(
        feature_dim=ohlcv_dim, latent_dim=vae_latent_dim, embed_dim=64, df=5.0
    ).to(device)
    try:
        vae_model.load_state_dict(torch.load(vae_model_path, map_location=device))
        vae_model.eval()
        print(f"Successfully loaded VAE model from {vae_model_path}")
    except FileNotFoundError:
        print(f"Error: VAE model not found at {vae_model_path}. Please ensure it is trained and the path is correct.")
        return

    # --- Load Predictor Model ---
    model = LatentOHLCVPredictor(
        latent_dim=vae_latent_dim, ohlcv_dim=ohlcv_dim, seq_length=seq_length
    ).to(device)
    try:
        model.load_state_dict(torch.load(predictor_model_path, map_location=device))
        model.eval()
        print(f"Successfully loaded Predictor model from {predictor_model_path}")
    except FileNotFoundError:
        print(
            f"Error: Predictor model not found at {predictor_model_path}. Please ensure it is trained and the path is correct.")
        return

    # --- Run Visualization ---
    visualize_last_n_days(
        predictor_model=model,
        vae_model=vae_model,
        dataset=full_dataset,
        latent_scaler=full_dataset.latent_scaler,
        ohlcv_scaler=full_dataset.ohlcv_scaler,
        n_days=n_days,
        seq_length=seq_length,
        device=device,
        output_dir=base_output_dir, # Pass the base directory
        symbol=symbol
    )

    visualize_last_n_days_single_step(
        predictor_model=model,
        dataset=full_dataset,
        ohlcv_scaler=full_dataset.ohlcv_scaler,
        n_days=n_days,
        seq_length=seq_length,
        device=device,
        output_dir=base_output_dir, # Pass the base directory
        symbol=symbol
    )


if __name__ == "__main__":
    main()