import streamlit as st
import pandas as pd
import numpy as np
import torch
import os  # 引入os模块
from new_utils import generate_bollinger_signals, generate_bollinger_signals_with_strength
import argparse

from dataloader_setup import FinancialDataset
from VAE_trainer import TransformerVAE_TDist
from predict_model import LatentOHLCVRNN, LatentOHLCVLSTM
from vis import LatentOHLCVPredictor

SEQ_LENGTH = 345

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 模型与数据加载函数 (无变化) ---
@st.cache_resource
def load_vae_model(vae_path, feature_dim, latent_dim, embed_dim, df, device):
    model = TransformerVAE_TDist(feature_dim=feature_dim, latent_dim=latent_dim, embed_dim=embed_dim, df=df).to(device)
    model.load_state_dict(torch.load(vae_path, map_location=device))
    model.eval()
    return model


# --- 【关键修改 A】: load_predictor_model 现在不再依赖于动态的 seq_length ---
# 我们把它从@st.cache_resource中移除，因为它现在依赖于固定的超参数，或者确保它的参数在一次运行中是恒定的
# 为了避免缓存冲突，最简单的方法是确保传入的seq_length是恒定的
@st.cache_resource
def load_predictor_model(model_name, predictor_path, device):
    # 使用固定的seq_length来初始化模型
    new_model = None
    if model_name == "rnn":
        new_model = LatentOHLCVRNN().to(device)
    elif model_name == "lstm":
        new_model = LatentOHLCVLSTM().to(device)
    else:
        new_model = LatentOHLCVPredictor().to(device)

    model = new_model
    model.load_state_dict(torch.load(predictor_path, map_location=device))
    model.eval()
    return model


@st.cache_data
def load_financial_data(csv_path, seq_length, latent_dim):
    # 这个函数现在主要用于1分钟原始数据的Dataset构建
    dataset = FinancialDataset(latent_csv_path=csv_path, seq_length=seq_length, latent_dim=latent_dim)
    return dataset


@st.cache_data
def resample_data(df, interval):
    if interval == '1min':
        return df.copy()

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    latent_cols = [col for col in df.columns if 'latent_dim' in col]

    aggregation_rules = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'amount': 'sum',
        'position': 'last'
    }

    for col in latent_cols:
        aggregation_rules[col] = 'mean'

    resampled_df = df.resample(interval).agg(aggregation_rules)
    resampled_df = resampled_df.dropna(subset=['open'])
    return resampled_df.reset_index()


# --- 【关键修改 B】: 回测逻辑现在需要同时访问重采样数据和原始1分钟数据 ---
def run_adaptive_backtest(
        predictor_model,
        dataset_1min,  # 1分钟数据的dataset（标准化、特征齐全）
        raw_df_1min,  # 原始1分钟数据
        signal_idx_1min,  # 信号在1min数据中的起始索引
        model_input_seq_length,
        execution_window,  # VWAP实际执行窗口（分钟数）
        total_quantity,
        trade_direction,
        failsafe_ratio,
        device,
        vae_model
):
    remaining_quantity = float(total_quantity)
    results_log = []

    for t in range(execution_window):
        order_quantity_for_this_minute = 0.0
        logic_used = ""
        predicted_close_for_this_minute = np.nan

        # --- 边界条件1: 提前完成 ---
        if remaining_quantity <= 1e-6:
            logic_used = "已完成"
            order_quantity_for_this_minute = 0.0

        # --- 边界条件2: 收盘冲刺 ---
        elif t >= execution_window - 5:
            logic_used = f"收盘冲刺 (r={failsafe_ratio})"
            minutes_left = execution_window - t
            if minutes_left == 1:
                order_quantity_for_this_minute = remaining_quantity
            else:
                denominator = 1 - (failsafe_ratio ** minutes_left)
                if abs(denominator) < 1e-9:
                    order_quantity_for_this_minute = remaining_quantity / minutes_left
                else:
                    first_term = remaining_quantity * (1 - failsafe_ratio) / denominator
                    order_quantity_for_this_minute = first_term

        # --- 主要逻辑: 模型预测 ---
        else:
            logic_used = "模型预测"
            input_start_idx = signal_idx_1min + t - model_input_seq_length
            input_end_idx = signal_idx_1min + t

            if input_start_idx >= 0:
                input_latent_seq_np = dataset_1min.normalized_latent[input_start_idx:input_end_idx]
                input_tensor = torch.FloatTensor(input_latent_seq_np).unsqueeze(0).to(device)

                latent_scaler = None
                ohlcv_scaler = None
                if hasattr(dataset_1min, "latent_scaler"):
                    latent_scaler = dataset_1min.latent_scaler
                if hasattr(dataset_1min, "ohlcv_scaler"):
                    ohlcv_scaler = dataset_1min.ohlcv_scaler

                if vae_model is None or latent_scaler is None or ohlcv_scaler is None:
                    print(f"{vae_model}, {latent_scaler}, {ohlcv_scaler} must be available for .generate()")
                    exit(1)

                with torch.no_grad():
                    gen_result = predictor_model.generate(
                        input_tensor,
                        vae_model=vae_model,
                        ohlcv_scaler=ohlcv_scaler,
                        latent_scaler=latent_scaler,
                        steps=execution_window,
                        device=device
                    )
                    outputs = gen_result['ohlcv'].squeeze(0).cpu().numpy()
                    outputs = ohlcv_scaler.inverse_transform(outputs)
                    preds_raw = np.expm1(outputs)

                    minutes_left_in_window = execution_window - t
                    relevant_future_preds = preds_raw[t:]
                    predicted_volumes_in_window = relevant_future_preds[:, 4]  # volume index
                    predicted_volume_for_now = predicted_volumes_in_window[0]
                    sum_of_future_predicted_volumes = np.sum(predicted_volumes_in_window)
                    predicted_close_for_this_minute = relevant_future_preds[0, 3]  # close index

                    if sum_of_future_predicted_volumes > 1e-6:
                        weight_for_this_minute = predicted_volume_for_now / sum_of_future_predicted_volumes
                        order_quantity_for_this_minute = remaining_quantity * weight_for_this_minute
                    else:
                        order_quantity_for_this_minute = remaining_quantity / minutes_left_in_window
            else:
                order_quantity_for_this_minute = remaining_quantity / (execution_window - t)

        # --- 执行与记录 ---
        final_order_quantity = max(0.0, min(remaining_quantity, order_quantity_for_this_minute))
        current_minute_raw_data = raw_df_1min.iloc[signal_idx_1min + t]
        execution_price = (current_minute_raw_data['high'] + current_minute_raw_data['low'] + current_minute_raw_data[
            'close']) / 3.0

        # ；添加执行窗口开始和结束标志
        time_mark = ""
        if t == 0:  # t为0，代表循环开始，即执行窗口的开始
            time_mark = "start"
        elif t == execution_window - 1:  # t为execution_window-1，代表循环结束，即执行窗口的结束
            time_mark = "end"

        remaining_quantity -= final_order_quantity

        results_log.append({
            'timestamp': current_minute_raw_data['date'],
            'execution': time_mark,
            'logic_used': logic_used,
            'predicted_price': predicted_close_for_this_minute,
            'execution_price': execution_price,
            'actual_volume': current_minute_raw_data['volume'],
            'order_quantity': final_order_quantity,
            'trade_value': final_order_quantity * execution_price,
            'remaining_quantity': remaining_quantity
        })

    # --- 后处理与指标计算 ---
    results_df = pd.DataFrame(results_log)
    if results_df.empty or results_df['order_quantity'].sum() < 1e-6:
        return pd.DataFrame(), {}

    total_actual_value_for_benchmark = (results_df['execution_price'] * results_df['actual_volume']).sum()
    total_actual_volume_for_benchmark = results_df['actual_volume'].sum()
    benchmark_vwap = total_actual_value_for_benchmark / total_actual_volume_for_benchmark if total_actual_volume_for_benchmark > 0 else 0
    model_total_trade_value = results_df['trade_value'].sum()
    model_total_quantity_traded = results_df['order_quantity'].sum()
    model_achieved_price = model_total_trade_value / model_total_quantity_traded if model_total_quantity_traded > 0 else 0
    if trade_direction.lower() == 'buy':
        slippage_per_share = benchmark_vwap - model_achieved_price
    else:
        slippage_per_share = model_achieved_price - benchmark_vwap
    total_cost_savings = slippage_per_share * total_quantity
    slippage_bps = (slippage_per_share / benchmark_vwap) * 10000 if benchmark_vwap > 0 else 0
    results_df['cumulative_benchmark_value'] = (results_df['execution_price'] * results_df['actual_volume']).cumsum()
    results_df['cumulative_actual_volume'] = results_df['actual_volume'].cumsum()
    results_df['traditional_vwap_line'] = results_df['cumulative_benchmark_value'] / results_df[
        'cumulative_actual_volume']
    results_df['cumulative_model_value'] = results_df['trade_value'].cumsum()
    results_df['cumulative_model_volume'] = results_df['order_quantity'].cumsum()
    results_df['model_vwap_line'] = (
            results_df['cumulative_model_value'] / results_df['cumulative_model_volume']).replace([np.inf, -np.inf],
                                                                                                  np.nan).ffill()
    metrics = {"Benchmark VWAP": benchmark_vwap, "Model Achieved Price": model_achieved_price,
               "Slippage Reduction (BPS)": slippage_bps, "Total Cost Savings": total_cost_savings}
    return results_df.set_index('timestamp'), metrics


import random


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


if __name__ == "__main__":
    # 定义周期与可选execution_window的映射
    interval_to_windows = {
        '1min': [1],
        '5min': [1, 5],
        '15min': [1, 5, 15],
        '30min': [1, 5, 15, 30],
        '60min': [1, 5, 15, 30, 60],
        'D': [1, 5, 15, 30, 60, SEQ_LENGTH]  # 假设日线SEQ_LENGTH分钟
    }
    for finaldirection in ["buy", "sell"]:
        for model_name in ["our-transformer", "rnn", "lstm"]:
            for symbol in ["rb9999", "i9999", "cu9999", "ni9999", "sc9999", "pg9999", "y9999", "ag9999", "m9999",
                           "c9999", "TA9999", "UR9999", "OI9999", "au9999", "IH9999", "T9999", "CF9999", "AP9999"]:
                set_seed(42)
                parser = argparse.ArgumentParser(description="VWAP批量回测参数")
                parser.add_argument('--data_file', type=str, default=f'data/latent_features/{symbol}_1min_data.csv')
                parser.add_argument('--vae_model_path', type=str, default=f'./models/{symbol}_tdist_vae_model.pth')
                parser.add_argument('--predictor_model_path', type=str,
                                    default=f'./predictor_models/{symbol}_{model_name}_predictor_model.pth')
                parser.add_argument('--total_quantity', type=float, default=10000)
                parser.add_argument('--failsafe_ratio', type=float, default=0.75)
                parser.add_argument('--execution_window', type=int, default=30)
                parser.add_argument('--length_param', type=int, default=20)
                parser.add_argument('--std_param', type=float, default=2.0)
                parser.add_argument('--stl_param', type=float, default=5.0)
                parser.add_argument('--n_param', type=float, default=6.0)
                parser.add_argument('--model_input_seq_length', type=int, default=345)
                parser.add_argument('--latent_dim', type=int, default=16)
                parser.add_argument('--ohlcv_dim', type=int, default=5)
                parser.add_argument('--intervals', type=str, default='1min,5min,15min,30min,60min,D',
                                    help='逗号分隔的周期列表，如1min,5min,15min,30min,60min,D')

                args = parser.parse_args()

                # 解析周期映射
                time_interval_map = {
                    '1分钟': '1min', '5分钟': '5min', '15分钟': '15min',
                    '30分钟': '30min', '1小时': '60min', '日线': 'D'
                }
                # 如果只想用部分周期，可以用args.intervals
                intervals = args.intervals.split(',')
                time_interval_map = {k: v for k, v in time_interval_map.items() if v in intervals}

                data_file = args.data_file
                vae_model_path = args.vae_model_path
                predictor_model_path = None
                if model_name != "our-transformer":
                    predictor_model_path = args.predictor_model_path
                else:
                    predictor_model_path = f"./predictor_models/{symbol}_predictor_model.pth"
                total_quantity = args.total_quantity
                failsafe_ratio = args.failsafe_ratio
                execution_window = args.execution_window
                length_param = args.length_param
                std_param = args.std_param
                stl_param = args.stl_param
                n_param = args.n_param
                model_input_seq_length = args.model_input_seq_length
                latent_dim = args.latent_dim
                ohlcv_dim = args.ohlcv_dim

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                print(f"---\n**分析开始** | 设备: `{device}`")

                for selected_interval_label, time_interval in time_interval_map.items():
                    try:
                        print(f"[{selected_interval_label}] 1. 加载并重采样数据...")
                        raw_df_1min = pd.read_csv(data_file, parse_dates=['date'])
                        resampled_df = resample_data(raw_df_1min.copy(), time_interval)
# --------------------------------------------------------------------------------------------------
                        # --- 【新功能】: 基于阈值筛选信号并保存到 CSV ---
                        # 定义信号保存的文件夹
                        signals_dir = "bollinger_signals"
                        os.makedirs(signals_dir, exist_ok=True)

                        # 使用新的函数生成信号和信号强度
                        # 注意：这里需要确保 generate_bollinger_signals_with_strength 函数是可用的
                        bollinger_params = {"length": length_param, "std": std_param, "stl_param": stl_param,
                                            "n_param": n_param}
                        # 使用 resampled_df 作为输入，并确保它有时间索引
                        signals_df = resampled_df.copy()
                        bollinger_signal_df = generate_bollinger_signals_with_strength(signals_df, **bollinger_params)

                        # 遍历所有允许的执行窗口，并将它们包含在文件名和信号中
                        # 注意：这段代码应该在主 for 循环内部，或者将其重构到主循环外部
                        # 如下所示，它应该嵌套在 symbol 和 time_interval 的循环中
                        # 并且要使用 resampled_df，而不是每次都重新加载
                        allowed_windows = interval_to_windows.get(time_interval, [1])
                        for execution_window in allowed_windows:
                            # 基于阈值和信号类型筛选数据
                            # 这一步是核心，只保留 'Buy' 或 'Sell' 且强度超过阈值的信号
                            signal_threshold = 0.6
                            filtered_signals_df = bollinger_signal_df[
                                (bollinger_signal_df['signal'].isin(['Buy', 'Sell'])) &
                                (bollinger_signal_df['signal_strength'] > signal_threshold)
                                ].copy()  # 使用 .copy() 避免 SettingWithCopyWarning

                            # 格式化输出 DataFrame
                            if not filtered_signals_df.empty:
                                # 'date' 列已经存在于原始数据框中，直接使用它作为时间戳

                                # 构造文件名
                                signal_filename = f"bollinger_signals_{time_interval}_{symbol}_win{execution_window}.csv"
                                signal_filepath = os.path.join(signals_dir, signal_filename)

                                # 保存 CSV 文件，只包含 'date' 列（时间戳）和格式化后的信号字符串
                                filtered_signals_df[['date', 'signal']].to_csv(signal_filepath, index=False)
                                print(f"信号已保存到 {signal_filepath}")
                            else:
                                print(
                                    f"警告：在 {time_interval}, {symbol}, window={execution_window} 的回测中，没有生成满足阈值要求的信号。")
                        # --------------------------------------------------------------------------------------------------


                        unique_dates = sorted(resampled_df['date'].dt.date.unique(), reverse=True)
                        if not unique_dates:
                            print(f"[{selected_interval_label}] 无可用日期，跳过。")
                            continue
                        selected_date = unique_dates[0]
                        bars_in_day = len(resampled_df[resampled_df['date'].dt.date == selected_date])
                        print(
                            f"[{selected_interval_label}] 数据已重采样，选定日期 {selected_date} 包含 {bars_in_day} 根K线。")

                        print(f"[{selected_interval_label}] 2. 加载模型...")
                        vae_model = load_vae_model(vae_model_path, ohlcv_dim, latent_dim, 64, 5.0, device)
                        # predictor_model = load_predictor_model(predictor_model_path, latent_dim, ohlcv_dim, model_input_seq_length, device = device)
                        predictor_model = load_predictor_model(model_name, predictor_model_path, device=device)
                        print(f"[{selected_interval_label}] 3. 生成bollinger信号...")
                        bollinger_params = {"length": length_param, "std": std_param, "stl_param": stl_param,
                                            "n_param": n_param}
                        bollinger_signal_df = generate_bollinger_signals(resampled_df.copy(), **bollinger_params)

                        selected_date_df = bollinger_signal_df[
                            bollinger_signal_df['date'].dt.date == selected_date].copy()
                        entry_signals = None
                        if finaldirection.lower() == "buy":
                            entry_signals = selected_date_df[selected_date_df['signal'] == 'Buy']
                        else:
                            entry_signals = selected_date_df[selected_date_df['signal'] == 'Sell']
                        if entry_signals.empty:
                            print(f"[{selected_interval_label}] {selected_date} 无bollinger买入信号，跳过。")
                            continue

                        first_signal = entry_signals.iloc[0]
                        trade_direction = first_signal['signal']
                        signal_time = first_signal['date']

                        signal_idx = raw_df_1min.index[
                            raw_df_1min['date'] == pd.Timestamp(signal_time).as_unit('ns')].tolist()
                        if not signal_idx:
                            signal_idx = [raw_df_1min.index[raw_df_1min['date'] <= signal_time][-1]]
                        signal_idx = signal_idx[0]

                        # 1min数据的dataset，用于标准化和特征提取
                        dataset_1min = load_financial_data(data_file, model_input_seq_length, latent_dim)

                        # 获取本周期允许的execution_window步长
                        allowed_windows = interval_to_windows.get(time_interval, [1])
                        for execution_window in allowed_windows:
                            if signal_idx + execution_window > len(raw_df_1min):
                                print(f"[{selected_interval_label}] execution_window={execution_window} 数据不足，跳过。")
                                continue

                            print(
                                f"[{selected_interval_label}] 选定信号时间: {signal_time}，信号索引: {signal_idx}, 执行窗口: {execution_window}分钟")
                            print(
                                f"[{selected_interval_label}] 交易方向: {trade_direction}, 总交易量: {total_quantity}, 失败安全系数: {failsafe_ratio}")
                            print(f"[{selected_interval_label}] boll参数: {bollinger_params}")
                            print(f"[{selected_interval_label}] 4. 执行回测... execution_window={execution_window}")
                            results_df, metrics = run_adaptive_backtest(
                                predictor_model,
                                dataset_1min,
                                raw_df_1min,
                                signal_idx,
                                model_input_seq_length,
                                execution_window,
                                total_quantity,
                                trade_direction,
                                failsafe_ratio,
                                device,
                                vae_model
                            )

                            # --- 保存结果 ---
                            os.makedirs(f"{finaldirection}_results", exist_ok=True)
                            base_filename = os.path.basename(data_file).replace('.csv', '')
                            param_str = f"{length_param}_{std_param}_{stl_param}_{n_param}_win{execution_window}"
                            results_df_name = f"aanew_boll_results_df_{finaldirection}_{execution_window}MINexecution_window_{model_name}_{symbol}_{time_interval}_{base_filename}_{param_str}.csv"
                            metrics_name = f"aanew_boll_metrics_{finaldirection}_{execution_window}MINexecution_window_{model_name}_{symbol}_{time_interval}_{base_filename}_{param_str}.csv"
                            results_df.to_csv(os.path.join(f"{finaldirection}_results", results_df_name))
                            pd.DataFrame([metrics]).to_csv(os.path.join(f"{finaldirection}_results", metrics_name),
                                                           index=False)
                            print(
                                f"[{selected_interval_label}] 回测完成，结果已保存为 {results_df_name} 和 {metrics_name}")

                    except Exception as e:
                        print(f"[{selected_interval_label}] 发生错误: {e}")
                        import traceback

                        print(traceback.format_exc())





