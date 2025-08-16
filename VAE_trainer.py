import pandas as pd
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


def preprocess_minute_data(file_path, num_features=7):
    """
    加载并预处理分钟级数据。
    V2版: 增加了数据质量检查和最终特征分布的打印。
    """
    print("Loading data...")
    df = pd.read_csv(file_path, parse_dates=['date'])

    # Get initial unique days
    initial_days = set(df['date'].dt.date.unique())
    print(f"Initial unique days in dataset: {len(initial_days)}")

    # 创建一个 'day' 列用于分组
    df['day'] = df['date'].dt.date

    # # 1. 筛选出每日恰好有345分钟的数据
    # print("Filtering days with exactly 345 minutes...")
    # day_counts = df.groupby('day').size()
    # # Identify days that do not have 345 minutes
    # days_not_345_minutes = day_counts[day_counts != 345].index.tolist()
    # if days_not_345_minutes:
    #     print(
    #         f"⚠️ Warning: Found {len(days_not_345_minutes)} days not having exactly 345 minutes. These days will be removed.")
    #     for d in days_not_345_minutes:
    #         print(f"  - Day: {d}, Minutes found: {day_counts[d]}")
    #
    # df_filtered = df.groupby('day').filter(lambda x: len(x) == 345)
    #
    # if df_filtered.empty:
    #     raise ValueError("No days with exactly 345 minutes found. Please check your data.")

    df_filtered = df

    # 增加检查：确保每日开盘价不为0，避免除零错误
    daily_opens = df_filtered.groupby('day')['open'].transform('first')
    if (daily_opens == 0).any():
        print("⚠️ Warning: Found days with an opening price of 0. These days will be removed.")
        bad_days = df_filtered[daily_opens == 0]['day'].unique()
        print("Found bad days: ", bad_days)
        df_filtered = df_filtered[~df_filtered['day'].isin(bad_days)]
        # 重新计算daily_opens
        daily_opens = df_filtered.groupby('day')['open'].transform('first')

    print(f"Found {len(df_filtered['day'].unique())} valid trading days after all checks.")

    # 2. 特征工程与归一化
    print("Performing feature engineering and scaling...")

    # 价格归一化
    # df_filtered['open'] = (df_filtered['open'] / daily_opens) - 1
    # df_filtered['high'] = (df_filtered['high'] / daily_opens) - 1
    # df_filtered['low'] = (df_filtered['low'] / daily_opens) - 1
    # df_filtered['close'] = (df_filtered['close'] / daily_opens) - 1

    # 对数化 + 标准化
    feature_columns = ['open', 'high', 'low', 'close', 'volume']
    # for col in ['volume', 'amount', 'position']:
    for col in feature_columns:
        df_filtered[col] = np.log1p(df_filtered[col])
        scaler = StandardScaler()
        df_filtered[col] = scaler.fit_transform(df_filtered[[col]])

    processed_data = df_filtered[feature_columns].values.astype(np.float32)

    # --- 3. 数据诊断 ---
    print("\n--- Data Diagnostics ---")

    # 检查是否存在 NaN 或 Inf
    has_nan = np.isnan(processed_data).any()
    has_inf = np.isinf(processed_data).any()

    if has_nan or has_inf:
        print(f"❌ Error: Processed data contains invalid values!")
        print(f"  - Contains NaN: {has_nan}")
        print(f"  - Contains Inf: {has_inf}")
        # 如果有问题，可以进一步定位具体是哪一列
        if has_nan:
            print("Columns with NaN:",
                  [feature_columns[i] for i, col_has_nan in enumerate(np.isnan(processed_data).any(axis=0)) if
                   col_has_nan])
        if has_inf:
            print("Columns with Inf:",
                  [feature_columns[i] for i, col_has_inf in enumerate(np.isinf(processed_data).any(axis=0)) if
                   col_has_inf])

    else:
        print("✅ Success: Processed data is clean (No NaN or Inf values).")

    # 计算并显示每个特征的均值和方差
    means = np.mean(processed_data, axis=0)
    variances = np.var(processed_data, axis=0)

    print("\nStatistics of Processed Features:")
    print("-" * 40)
    print(f"{'Feature':<12} | {'Mean':>10} | {'Variance':>12}")
    print("-" * 40)
    for i, name in enumerate(feature_columns):
        print(f"{name:<12} | {means[i]:>10.4f} | {variances[i]:>12.4f}")
    print("-" * 40)

    print("\nData processing complete.")
    return processed_data


class TransformerVAE_TDist(nn.Module):
    def __init__(self, feature_dim, latent_dim, embed_dim=64, nhead=4, num_layers=3, df=5.0):
        super(TransformerVAE_TDist, self).__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.df = df  # T分布的自由度参数

        # 编码器部分保持不变
        self.encoder_input_fc = nn.Linear(feature_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出T分布参数: mu, log_scale, log_df
        self.fc_mu = nn.Linear(embed_dim, latent_dim)
        self.fc_log_scale = nn.Linear(embed_dim, latent_dim)
        self.fc_log_df = nn.Linear(embed_dim, latent_dim)

        # 解码器部分保持不变
        self.decoder_input_fc = nn.Linear(latent_dim, embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.decoder_output_fc = nn.Linear(embed_dim, feature_dim)

        # 初始化为接近目标df的值 (df=5 -> log_df≈1.0986)
        nn.init.constant_(self.fc_log_df.bias, np.log(df - 2))
        # 限制权重范围
        self.fc_log_df.weight.data.uniform_(-0.001, 0.001)

    def encode(self, x):
        x = x.unsqueeze(1)
        x = F.silu(self.encoder_input_fc(x))
        x = self.transformer_encoder(x)
        x = x.squeeze(1)

        mu = self.fc_mu(x)
        log_scale = self.fc_log_scale(x)
        log_df = self.fc_log_df(x)
        return mu, log_scale, log_df

    def reparameterize(self, mu, log_scale, log_df):
        """T分布的重参数化技巧"""
        scale = torch.exp(log_scale)
        df = 3 + 27 * torch.sigmoid(log_df)  # 3 + 27*(0~1)

        # 使用StudentT分布
        t_dist = torch.distributions.StudentT(
            df=df,
            loc=mu,
            scale=scale
        )

        # 重参数化采样
        z = t_dist.rsample()
        return z, (mu, log_scale, log_df)

    def decode(self, z):
        z = F.silu(self.decoder_input_fc(z))
        z = z.unsqueeze(1)
        output = self.transformer_decoder(tgt=z, memory=z)
        output = output.squeeze(1)
        recon_x = self.decoder_output_fc(output)
        return recon_x

    def forward(self, x):
        mu, log_scale, log_df = self.encode(x)
        z, params = self.reparameterize(mu, log_scale, log_df)
        recon_x = self.decode(z)
        return recon_x, params, z


import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt


# 定义细粒度损失计算函数
def detailed_loss_function(recon_x, x):
    """
    计算各特征的独立重建损失
    返回: 字典包含各项指标
    """

    # 价格指标 (OHLC)
    price_cols = slice(0, 4)  # 前4列是OHLC
    price_mae = F.l1_loss(recon_x[:, price_cols], x[:, price_cols]).item()
    price_mse = F.mse_loss(recon_x[:, price_cols], x[:, price_cols]).item()

    # 成交量指标 (volume, amount, position)
    volume_cols = slice(4, 7)
    volume_mae = F.l1_loss(recon_x[:, volume_cols], x[:, volume_cols]).item()
    volume_mse = F.mse_loss(recon_x[:, volume_cols], x[:, volume_cols]).item()

    # 整体指标
    total_mae = F.l1_loss(recon_x, x).item()
    total_mse = F.mse_loss(recon_x, x).item()

    results = {
        'price_mae': price_mae,
        'price_mse': price_mse,
        'volume_mae': volume_mae,
        'volume_mse': volume_mse,
        'total_mae': total_mae,
        'total_mse': total_mse
    }

    return results



def t_dist_kl_divergence(mu_q, log_scale_q, log_df_q, mu_p=0, scale_p=1, df_p=5.0, df_penalty_weight=0.1):
    """
    改进的KL散度计算，添加对自由度的显式约束
    df_penalty_weight: 控制对自由度的惩罚强度
    """
    scale_q = torch.exp(log_scale_q)
    df_q = torch.exp(log_df_q) + 2.0

    df_p = torch.tensor(df_p, dtype=torch.float)
    scale_p = torch.tensor(scale_p, dtype=torch.float)

    # 原始KL计算
    term1 = torch.lgamma((df_q + 1) / 2) - torch.lgamma(df_q / 2)
    term2 = torch.lgamma((df_p + 1) / 2) - torch.lgamma(df_p / 2)
    term3 = 0.5 * (torch.log(df_q) + torch.log(scale_q ** 2) - torch.log(df_p) - torch.log(scale_p ** 2))
    term4 = ((df_q + 1) / 2) * (
        torch.log1p(
            (mu_q ** 2 + scale_q ** 2 * (df_q / (df_q - 2)) - 2 * mu_q * mu_p + mu_p ** 2) / (df_q * scale_p ** 2))
    )
    term5 = ((df_p + 1) / 2) * (
        torch.log1p(
            (mu_q ** 2 + scale_p ** 2 * (df_p / (df_p - 2)) - 2 * mu_q * mu_p + mu_p ** 2) / (df_p * scale_p ** 2))
    )

    kl = term1 - term2 - term3 + term4 - term5

    # 添加自由度惩罚项 (鼓励df接近df_p)
    df_penalty = df_penalty_weight * F.mse_loss(log_df_q, torch.log(torch.tensor(df_p - 2.0).to(log_df_q.device)))

    return torch.sum(kl) + df_penalty


def kl_divergence_monte_carlo(q_dist, p_dist, z):
    """
    使用蒙特卡洛采样来近似计算KL散度: E_q[log q(z) - log p(z)]

    参数:
    q_dist (torch.distributions.Distribution): 后验分布 q(z|x)
    p_dist (torch.distributions.Distribution): 先验分布 p(z)
    z (torch.Tensor): 从 q_dist 中采样的样本

    返回:
    torch.Tensor: KL散度的一个批次的估计值
    """
    log_q_z = q_dist.log_prob(z)
    log_p_z = p_dist.log_prob(z)

    # KL divergence is log_q - log_p. Sum over the latent dimensions.
    # The result is a tensor of shape [batch_size].
    kl_per_sample = torch.sum(log_q_z - log_p_z, dim=1)

    return kl_per_sample

def vae_loss_function_tdist_rewritten(recon_x, x, params, z, df_prior=5.0, kl_beta=1.0, df_penalty_weight=0.1):


    """
    重写后的T分布VAE损失函数，使用蒙特卡洛KL散度和独立的df惩罚。

    参数:
    recon_x (torch.Tensor): 重建的输出
    x (torch.Tensor): 原始输入
    params (tuple): 从编码器输出的后验分布参数 (mu, log_scale, log_df)
    z (torch.Tensor): 从后验分布采样的潜在变量
    df_prior (float): 先验分布p(z)的自由度
    kl_beta (float): KL散度项的权重 (Beta-VAE中的beta)
    df_penalty_weight (float): 对df偏离先验的惩罚权重

    返回:
    torch.Tensor: 该批次的总损失
    """
    # --- 1. 重建损失 (与之前相同) ---
    # 价格部分使用Huber损失
    price_loss = F.huber_loss(recon_x[:, :4], x[:, :4], reduction='sum')
    # 成交量部分使用MSE
    volume_loss = F.mse_loss(recon_x[:, 4:], x[:, 4:], reduction='sum')
    recon_loss = price_loss + volume_loss

    # --- 2. KL散度 (新方法) ---
    mu_q, log_scale_q, log_df_q = params

    # a. 创建后验分布 q(z|x)
    scale_q = torch.exp(log_scale_q)
    df_q = F.softplus(log_df_q) + 2.0  # 使用softplus确保 df > 2, 且无上限

    q_dist = torch.distributions.StudentT(
        df=df_q,
        loc=mu_q,
        scale=scale_q
    )

    # b. 创建先验分布 p(z)
    # 先验的均值为0，尺度为1，自由度为df_prior
    p_dist = torch.distributions.StudentT(
        df=torch.full_like(df_q, df_prior),
        loc=torch.zeros_like(mu_q),
        scale=torch.ones_like(scale_q)
    )

    # c. 使用蒙特卡洛方法计算KL散度
    kl_div_per_sample = kl_divergence_monte_carlo(q_dist, p_dist, z)
    kl_div = torch.sum(kl_div_per_sample)  # 在批次维度上求和

    # --- 3. 自由度(df)惩罚项 (可选但推荐) ---
    # 鼓励模型学习到的df_q接近我们设定的先验df_prior
    # 我们直接对df进行惩罚，而不是log_df
    df_penalty = F.mse_loss(df_q, torch.full_like(df_q, df_prior))

    # --- 4. 计算总损失 ---
    # 加权组合三个部分
    total_loss = recon_loss + (kl_beta * kl_div) + (df_penalty_weight * df_penalty)

    return total_loss

def vae_loss_function_tdist(recon_x, x, params):
    """T分布VAE的损失函数"""
    mu_q, log_scale_q, log_df_q = params

    # 重建损失 (区分价格和成交量)
    price_loss = F.huber_loss(recon_x[:, :4], x[:, :4], reduction='sum')
    volume_loss = F.mse_loss(recon_x[:, 4:], x[:, 4:], reduction='sum')
    recon_loss = price_loss + volume_loss

    # KL散度 (T分布)
    kl_div = t_dist_kl_divergence(mu_q, log_scale_q, log_df_q)

    return recon_loss + kl_div


def prepare_dataloaders(processed_data, batch_size=345, test_size=0.05):
    """准备训练和验证集的DataLoader"""
    # 按天数分割确保时间连续性
    n_days = len(processed_data) // 345
    train_size = int((1 - test_size) * n_days)

    # 按天分割
    train_data = processed_data[:train_size * 345]
    val_data = processed_data[train_size * 345:]

    train_dataset = TensorDataset(torch.from_numpy(train_data).float())
    val_dataset = TensorDataset(torch.from_numpy(val_data).float())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_model_tdist(model, train_loader, val_loader, optimizer, epochs, device, model_save_path): # Added model_save_path
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_metrics': [],
        'val_metrics': []
    }

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_metrics = {
            'price_mae': 0, 'price_mse': 0,
            'volume_mae': 0, 'volume_mse': 0
        }

        for data, in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            recon_batch, params,z = model(data)
            loss = vae_loss_function_tdist_rewritten(
                recon_batch,
                data,
                params,
                z,
                kl_beta=0.1,  # 建议从一个较小的值开始
                df_penalty_weight=5.0  # 可以适当增大惩罚
            )
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            batch_metrics = detailed_loss_function(recon_batch.detach(), data.detach())
            for k in train_metrics:
                train_metrics[k] += batch_metrics[k]

        # 验证阶段
        model.eval()
        val_loss = 0
        val_metrics = {
            'price_mae': 0, 'price_mse': 0,
            'volume_mae': 0, 'volume_mse': 0
        }
        avg_df = 0 # Initialize avg_df for calculation

        with torch.no_grad():
            # Sample to get average df (from first batch of validation data)
            if len(val_loader) > 0:
                sample_data = next(iter(val_loader))[0].to(device)
                _, _, log_df_raw = model.encode(sample_data)
                # Apply the sigmoid and scaling used in reparameterize to get actual df value
                avg_df = (3 + F.softplus(log_df_raw)).mean().item()
                print(f"Average learned degrees of freedom (df) at Epoch {epoch + 1}: {avg_df:.2f}")

            for data, in val_loader:
                data = data.to(device)
                recon_batch, params, z = model(data)
                val_loss += vae_loss_function_tdist_rewritten(
                    recon_batch,
                    data,
                    params,
                    z,
                    kl_beta=0.1,  # 建议从一个较小的值开始
                    df_penalty_weight=5.0  # 可以适当增大惩罚
                ).item()

                batch_metrics = detailed_loss_function(recon_batch, data)
                for k in val_metrics:
                    val_metrics[k] += batch_metrics[k]

        # 记录和打印
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        for k in train_metrics:
            train_metrics[k] /= len(train_loader)
            val_metrics[k] /= len(val_loader)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_metrics'].append(train_metrics)
        history['val_metrics'].append(val_metrics)

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print("Train Metrics:")
        print(f"  Price - MAE: {train_metrics['price_mae']:.4f}, MSE: {train_metrics['price_mse']:.4f}")
        print(f"  Volume - MAE: {train_metrics['volume_mae']:.4f}, MSE: {train_metrics['volume_mse']:.4f}")
        print("Val Metrics:") # Added val metrics printing
        print(f"  Price - MAE: {val_metrics['price_mae']:.4f}, MSE: {val_metrics['price_mse']:.4f}")
        print(f"  Volume - MAE: {val_metrics['volume_mae']:.4f}, MSE: {val_metrics['volume_mse']:.4f}")


        # 早停和模型保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save model with symbol in its name
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved new best model to {model_save_path}")

    return history

def plot_training_history(history, symbol, output_dir): # Added symbol and output_dir
    """绘制训练过程图表"""
    plt.figure(figsize=(15, 10))

    # 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'{symbol} - Training and Validation Loss') # Added symbol
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 价格MAE
    plt.subplot(2, 2, 2)
    train_price_mae = [m['price_mae'] for m in history['train_metrics']]
    val_price_mae = [m['price_mae'] for m in history['val_metrics']]
    plt.plot(train_price_mae, label='Train Price MAE')
    plt.plot(val_price_mae, label='Val Price MAE')
    plt.title(f'{symbol} - Price MAE') # Added symbol
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    # 成交量MAE
    plt.subplot(2, 2, 3)
    train_volume_mae = [m['volume_mae'] for m in history['train_metrics']]
    val_volume_mae = [m['volume_mae'] for m in history['val_metrics']]
    plt.plot(train_volume_mae, label='Train Volume MAE')
    plt.plot(val_volume_mae, label='Val Volume MAE')
    plt.title(f'{symbol} - Volume MAE') # Added symbol
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    # 价格MSE
    plt.subplot(2, 2, 4)
    train_price_mse = [m['price_mse'] for m in history['train_metrics']]
    val_price_mse = [m['price_mse'] for m in history['val_metrics']]
    plt.plot(train_price_mse, label='Train Price MSE')
    plt.plot(val_price_mse, label='Val Price MSE')
    plt.title(f'{symbol} - Price MSE') # Added symbol
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()

    plt.tight_layout()
    # Save plot with symbol in filename
    plot_filename = os.path.join(output_dir, f'{symbol}_training_metrics.png')
    plt.savefig(plot_filename)
    plt.close()
    print(f"Training plot saved to: {plot_filename}")


import random
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

# Call this function at the beginning of your main execution block
# For example, in VAE_trainer.py, within if __name__ == '__main__':
# set_seed(args.seed)  # Assuming 'seed' is an argument from argparse

if __name__ == '__main__':
    import os, argparse

    parser = argparse.ArgumentParser(description="Train a Transformer VAE T-Distribution model on minute data.")

    # Data arguments
    parser.add_argument('--data_file_path', type=str,
                        default='data/raw_data/{symbol}_1min_data.csv',
                        help='Path to the input minute data CSV file. Use {symbol} as a placeholder.')
    parser.add_argument('--symbol', type=str, required=True,
                        help='The target symbol for training (e.g., c9999, rb9999).')
    parser.add_argument('--batch_size', type=int, default=345,
                        help='Batch size for training DataLoader (e.g., 345 minutes per day).')
    parser.add_argument('--test_size', type=float, default=0.05,
                        help='Proportion of data to use for validation (e.g., 0.05 for 5%%).')

    # Model arguments
    parser.add_argument('--feature_dim', type=int, default=5, # Changed to 7 assuming position is often included
                        help='Dimension of input features (OHLC, Volume, Amount, Position).')
    parser.add_argument('--latent_dim', type=int, default=16,
                        help='Dimension of the latent space.')
    parser.add_argument('--embed_dim', type=int, default=64,
                        help='Embedding dimension for Transformer layers.')
    parser.add_argument('--nhead', type=int, default=4,
                        help='Number of attention heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of Transformer encoder/decoder layers.')
    parser.add_argument('--df_initial', type=float, default=5.0,
                        help='Initial degrees of freedom for the T-distribution prior.')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate for the Adam optimizer.')
    parser.add_argument('--model_output_dir', type=str, default='./models',
                        help='Directory to save trained models and plots.')

    parser.add_argument('--seed', type=int, default=42, # NEW: Add seed argument
                        help='Random seed for reproducibility.')

    args = parser.parse_args()

    # Set random seed at the very beginning of execution
    set_seed(args.seed) # NEW: Call set_seed

    # Construct the actual file path using the provided symbol
    actual_file_path = args.data_file_path.format(symbol=args.symbol)
    model_save_file = os.path.join(args.model_output_dir, f'{args.symbol}_tdist_vae_model.pth')
    plot_save_dir = args.model_output_dir # Plots will be saved in the same model output directory

    # Ensure model output directory exists
    os.makedirs(args.model_output_dir, exist_ok=True)


    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Data preparation
    try:
        processed_data = preprocess_minute_data(actual_file_path, num_features=args.feature_dim)
        train_loader, val_loader = prepare_dataloaders(processed_data, args.batch_size, args.test_size)
    except Exception as e:
        print(f"Error during data loading or preprocessing for {args.symbol}: {e}")
        exit(1) # Exit with an error code

    # Initialize T-distribution VAE model
    model = TransformerVAE_TDist(
        feature_dim=args.feature_dim, # Use actual_feature_dim from preprocessing
        latent_dim=args.latent_dim,
        embed_dim=args.embed_dim,
        nhead=args.nhead,
        num_layers=args.num_layers,
        df=args.df_initial
    ).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    print(f"\n--- Starting training for symbol: {args.symbol} ---")
    # Train model
    history = train_model_tdist(
        model, train_loader, val_loader,
        optimizer, args.epochs, DEVICE,
        model_save_path=model_save_file # Pass model save path
    )

    # Visualize results
    plot_training_history(history, args.symbol, plot_save_dir) # Pass symbol and plot save dir

    print(f"Training completed for {args.symbol}. Best T-distribution VAE saved to '{model_save_file}'")


