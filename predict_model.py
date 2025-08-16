import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Predictor Model Class ---
class LatentOHLCVRNN(nn.Module):
    def __init__(self, latent_dim=16, ohlcv_dim=5, hidden_dim=128, num_layers=2, seq_length=345, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.ohlcv_dim = ohlcv_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_length = seq_length

        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.rnn = nn.RNN(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity='tanh',
            dropout=dropout if num_layers > 1 else 0
        )
        self.output_proj = nn.Linear(hidden_dim, ohlcv_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.RNN):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(self, x):
        # x: [batch, seq_length, latent_dim]
        x = self.input_proj(x)  # [batch, seq_length, hidden_dim]
        rnn_out, _ = self.rnn(x)  # [batch, seq_length, hidden_dim]
        ohlcv_pred = self.output_proj(rnn_out)  # [batch, seq_length, ohlcv_dim]
        return {'ohlcv_pred': ohlcv_pred}

    def generate(self, initial_latent_seq, vae_model, ohlcv_scaler, latent_scaler, steps=345, device='cpu'):
        self.eval()
        vae_model.eval()
        with torch.no_grad():
            generated_ohlcv_list = []
            current_input = initial_latent_seq.to(device)
            rnn_state = None

            # First pass: full sequence
            x = self.input_proj(current_input)
            rnn_out, rnn_state = self.rnn(x)
            predicted_next_ohlcv = self.output_proj(rnn_out)[:, -1:, :]
            generated_ohlcv_list.append(predicted_next_ohlcv)

            for _ in range(steps - 1):
                # 逆标准化和逆log变换
                predicted_next_ohlcv_np = ohlcv_scaler.inverse_transform(
                    predicted_next_ohlcv.cpu().numpy().squeeze(0))
                predicted_next_ohlcv_raw = np.expm1(predicted_next_ohlcv_np)
                input_for_vae = torch.FloatTensor(predicted_next_ohlcv_np).to(device)
                mu_vae, log_scale_vae, log_df_vae = vae_model.encode(input_for_vae)
                new_latent_z, _ = vae_model.reparameterize(mu_vae, log_scale_vae, log_df_vae)
                new_latent_z_normalized = torch.FloatTensor(
                    latent_scaler.transform(new_latent_z.cpu().numpy())
                ).to(device).unsqueeze(1)

                # RNN单步递推
                x = self.input_proj(new_latent_z_normalized)
                rnn_out, rnn_state = self.rnn(x, rnn_state)
                predicted_next_ohlcv = self.output_proj(rnn_out)
                generated_ohlcv_list.append(predicted_next_ohlcv)

            return {
                'ohlcv': torch.cat(generated_ohlcv_list, dim=1)
            }




# --- Predictor Model Class ---
class LatentOHLCVLSTM(nn.Module):
    def __init__(self, latent_dim=16, ohlcv_dim=5, hidden_dim=128, num_layers=2, seq_length=345, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.ohlcv_dim = ohlcv_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_length = seq_length

        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.output_proj = nn.Linear(hidden_dim, ohlcv_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(self, x):
        # x: [batch, seq_length, latent_dim]
        x = self.input_proj(x)  # [batch, seq_length, hidden_dim]
        lstm_out, _ = self.lstm(x)  # [batch, seq_length, hidden_dim]
        ohlcv_pred = self.output_proj(lstm_out)  # [batch, seq_length, ohlcv_dim]
        return {'ohlcv_pred': ohlcv_pred}

    def generate(self, initial_latent_seq, vae_model, ohlcv_scaler, latent_scaler, steps=345, device='cpu'):
        self.eval()
        vae_model.eval()
        with torch.no_grad():
            generated_ohlcv_list = []
            current_input = initial_latent_seq.to(device)
            lstm_state = None

            # First pass: full sequence
            x = self.input_proj(current_input)
            lstm_out, lstm_state = self.lstm(x)
            predicted_next_ohlcv = self.output_proj(lstm_out)[:, -1:, :]
            generated_ohlcv_list.append(predicted_next_ohlcv)

            for _ in range(steps - 1):
                # 逆标准化和逆log变换
                predicted_next_ohlcv_np = ohlcv_scaler.inverse_transform(
                    predicted_next_ohlcv.cpu().numpy().squeeze(0))
                predicted_next_ohlcv_raw = np.expm1(predicted_next_ohlcv_np)
                input_for_vae = torch.FloatTensor(predicted_next_ohlcv_np).to(device)
                mu_vae, log_scale_vae, log_df_vae = vae_model.encode(input_for_vae)
                new_latent_z, _ = vae_model.reparameterize(mu_vae, log_scale_vae, log_df_vae)
                new_latent_z_normalized = torch.FloatTensor(
                    latent_scaler.transform(new_latent_z.cpu().numpy())
                ).to(device).unsqueeze(1)

                # LSTM单步递推
                x = self.input_proj(new_latent_z_normalized)
                lstm_out, lstm_state = self.lstm(x, lstm_state)
                predicted_next_ohlcv = self.output_proj(lstm_out)
                generated_ohlcv_list.append(predicted_next_ohlcv)

            return {
                'ohlcv': torch.cat(generated_ohlcv_list, dim=1)
            }


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
