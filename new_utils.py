# -*- coding: utf-8 -*-
import pandas_ta as ta
import pandas as pd
import numpy as np

def generate_c53_signals(df, cl_period=35, cd_period=0, stl_param=5.0, n_param=6.0, long_ma=240, short_ma=26):
    """
    根据C53策略逻辑生成交易信号。
    此函数使用pandas重现backtrader策略，以便在Streamlit应用中集成。

    Args:
        df (pd.DataFrame): 包含'high', 'low', 'close'列的原始数据。
        cl_period (int): 通道周期 (C53中的CL)。
        cd_period (int): 通道偏移 (C53中的CD)。
        stl_param (float): 百分比止损 (C53中的STL)。
        n_param (float): ATR止损倍数 (C53中的N)。
        long_ma (int): 长期均线周期。
        short_ma (int): 用于ATR的短期均线周期。

    Returns:
        pd.DataFrame: 带有 'signal' 列的DataFrame。
    """
    if df.empty:
        return df

    # 1. 使用pandas_ta计算所需指标
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=short_ma)
    df['mas'] = ta.ema(df['close'], length=long_ma)
    df['upperc'] = df['high'].rolling(window=cl_period).max()
    df['lowerc'] = df['low'].rolling(window=cl_period).min()

    # 2. 为匹配backtrader逻辑，将指标值向后移动
    df['upperc_shifted'] = df['upperc'].shift(cd_period + 1)
    df['lowerc_shifted'] = df['lowerc'].shift(cd_period + 1)
    df['mas_shifted'] = df['mas'].shift(1)
    df['high_prev'] = df['high'].shift(1)
    df['low_prev'] = df['low'].shift(1)
    df['close_prev'] = df['close'].shift(1)

    # 3. 定义入场条件
    duo_condition = (df['high'] >= df['upperc_shifted']) & \
                    (df['high_prev'] < df['upperc_shifted']) & \
                    (df['close_prev'] > df['mas_shifted'])

    kong_condition = (df['low'] <= df['lowerc_shifted']) & \
                     (df['low_prev'] > df['lowerc_shifted']) & \
                     (df['close_prev'] < df['mas_shifted'])

    # 4. 使用状态机模拟持仓和止损
    signals = ['Hold'] * len(df)
    position = 0  # -1代表空仓, 0代表无仓位, 1代表多仓
    bkhigh = 0.0    # 多头入场后的最高价
    sklow = float('inf')  # 空头入场后的最低价

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]

        # --- 止损逻辑 ---
        if position == 1:  # 持有多仓
            stop_loss_atr = bkhigh - n_param * row['atr']
            stop_loss_stl = prev_row['close'] * (1 - 0.01 * stl_param)
            if row['close'] <= stop_loss_atr or prev_row['low'] < stop_loss_stl:
                signals[i] = 'Close_Buy'
                position = 0
                continue
            bkhigh = max(bkhigh, row['high'])

        elif position == -1:  # 持有空仓
            stop_loss_atr = sklow + n_param * row['atr']
            stop_loss_stl = prev_row['close'] * (1 + 0.01 * stl_param)
            if row['close'] >= stop_loss_atr or prev_row['high'] > stop_loss_stl:
                signals[i] = 'Close_Sell'
                position = 0
                continue
            sklow = min(sklow, row['low'])

        # --- 入场逻辑 (优先处理止损) ---
        if position == 0:
            if duo_condition.iloc[i]:
                signals[i] = 'Buy'
                position = 1
                bkhigh = row['high']
            elif kong_condition.iloc[i]:
                signals[i] = 'Sell'
                position = -1
                sklow = row['low']

    df['signal'] = signals
    return df


def generate_macd_signals_dual(df, fast=12, slow=26, signal_period=9, stl_param=5.0, n_param=6.0):
    """
    基于MACD的双向策略生成交易信号：
    - 二次金叉或底背离买入
    - 顶背离卖出
    - 加入止损逻辑（ATR + 百分比）
    
    Returns:
        DataFrame 带 'signal' 列
    """
    if df.empty:
        df['signal'] = []
        return df

    # 计算指标
    macd = ta.macd(df['close'], fast=fast, slow=slow, signal=signal_period)
    df['MACD'] = macd[f"MACD_{fast}_{slow}_{signal_period}"]
    df['MACD_signal'] = macd[f"MACDs_{fast}_{slow}_{signal_period}"]
    df['MACD_hist'] = macd[f"MACDh_{fast}_{slow}_{signal_period}"]
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=fast)

    signals = ['Hold'] * len(df)
    position = 0  # 1=多仓，-1=空仓，0=空仓
    bkhigh = 0.0
    sklow = float('inf')
    last_buy_index = -5  # 控制二次金叉间隔

    for i in range(2, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        # 止损逻辑
        if position == 1:
            stop_loss_atr = bkhigh - n_param * row['atr']
            stop_loss_stl = prev_row['close'] * (1 - 0.01 * stl_param)
            if row['close'] <= stop_loss_atr or prev_row['low'] < stop_loss_stl:
                signals[i] = 'Close_Buy'
                position = 0
                continue
            bkhigh = max(bkhigh, row['high'])

        elif position == -1:
            stop_loss_atr = sklow + n_param * row['atr']
            stop_loss_stl = prev_row['close'] * (1 + 0.01 * stl_param)
            if row['close'] >= stop_loss_atr or prev_row['high'] > stop_loss_stl:
                signals[i] = 'Close_Sell'
                position = 0
                continue
            sklow = min(sklow, row['low'])

        # 入场逻辑
        if position == 0:
            # 二次金叉
            if df['MACD'].iloc[i] > df['MACD_signal'].iloc[i] and df['MACD'].iloc[i - 1] <= df['MACD_signal'].iloc[i - 1]:
                if i - last_buy_index >= 3:
                    signals[i] = 'Buy'
                    position = 1
                    bkhigh = row['high']
                    last_buy_index = i

            # 底背离买入
            elif df['MACD_hist'].iloc[i] > df['MACD_hist'].iloc[i - 1] and df['close'].iloc[i] < df['close'].iloc[i - 1]:
                signals[i] = 'Buy'
                position = 1
                bkhigh = row['high']
                last_buy_index = i

            # 顶背离卖出开空
            elif df['MACD_hist'].iloc[i] < df['MACD_hist'].iloc[i - 1] and df['close'].iloc[i] > df['close'].iloc[i - 1]:
                signals[i] = 'Sell'
                position = -1
                sklow = row['low']

    df['signal'] = signals
    return df




def generate_bollinger_signals(df, length=20, std=2.0, stl_param=5.0, n_param=6.0):
    """
    基于布林带的双向交易信号生成策略。

    Args:
        df (pd.DataFrame): 包含'close', 'high', 'low'列的数据。
        length (int): 布林带均线周期。
        std (float): 标准差倍数。
        stl_param (float): 百分比止损参数。
        n_param (float): ATR止损倍数。

    Returns:
        pd.DataFrame: 添加'signal'列后的数据。
    """
    if df.empty:
        df['signal'] = []
        return df

    # 计算布林带和ATR
    bbands = ta.bbands(df['close'], length=length, std=std)
    df['bb_upper'] = bbands[f'BBU_{length}_{std}']
    df['bb_middle'] = bbands[f'BBM_{length}_{std}']
    df['bb_lower'] = bbands[f'BBL_{length}_{std}']
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=length)

    signals = ['Hold'] * len(df)
    position = 0  # 1 多仓，-1 空仓，0 空仓
    bkhigh = 0.0
    sklow = float('inf')

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        # --- 止损或止盈逻辑 ---
        if position == 1:
            stop_loss_atr = bkhigh - n_param * row['atr']
            stop_loss_stl = prev_row['close'] * (1 - 0.01 * stl_param)
            if row['close'] <= stop_loss_atr or prev_row['low'] < stop_loss_stl:
                signals[i] = 'Close_Buy'
                position = 0
                continue
            bkhigh = max(bkhigh, row['high'])

        elif position == -1:
            stop_loss_atr = sklow + n_param * row['atr']
            stop_loss_stl = prev_row['close'] * (1 + 0.01 * stl_param)
            if row['close'] >= stop_loss_atr or prev_row['high'] > stop_loss_stl:
                signals[i] = 'Close_Sell'
                position = 0
                continue
            sklow = min(sklow, row['low'])

        # --- 入场逻辑 ---
        if position == 0:
            # 开多：由下轨下穿反转向上
            if prev_row['close'] < prev_row['bb_lower'] and row['close'] > row['bb_lower']:
                signals[i] = 'Buy'
                position = 1
                bkhigh = row['high']
            # 开空：由上轨上穿反转向下
            elif prev_row['close'] > prev_row['bb_upper'] and row['close'] < row['bb_upper']:
                signals[i] = 'Sell'
                position = -1
                sklow = row['low']

    df['signal'] = signals
    return df



def generate_dual_thrust_signals(df, k1=0.5, k2=0.5, stl_param=5.0, n_param=6.0):
    """
    基于 Dual Thrust 策略生成双向交易信号（开多、开空、平多、平空）

    Args:
        df (pd.DataFrame): 包含 'open', 'high', 'low', 'close' 列的日内数据。
        k1 (float): 向上突破倍数。
        k2 (float): 向下突破倍数。
        stl_param (float): 百分比止损比例。
        n_param (float): ATR 止损倍数。

    Returns:
        pd.DataFrame: 添加 'signal' 列。
    """
    if df.empty:
        df['signal'] = []
        return df

    # 计算前一日的波动区间
    df['high_prev'] = df['high'].shift(1)
    df['low_prev'] = df['low'].shift(1)
    df['close_prev'] = df['close'].shift(1)
    df['range'] = (df[['high_prev', 'close_prev']].max(axis=1) - df[['low_prev', 'close_prev']].min(axis=1))

    # 计算上下突破边界
    df['upper_bound'] = df['open'] + k1 * df['range']
    df['lower_bound'] = df['open'] - k2 * df['range']

    # ATR止损参考
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=20)

    signals = ['Hold'] * len(df)
    position = 0  # 0=空仓，1=多头，-1=空头
    bkhigh = 0.0
    sklow = float('inf')

    for i in range(2, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        # --- 平仓逻辑 ---
        if position == 1:
            stop_loss_atr = bkhigh - n_param * row['atr']
            stop_loss_stl = prev_row['close'] * (1 - 0.01 * stl_param)
            if row['close'] <= stop_loss_atr or row['low'] < stop_loss_stl:
                signals[i] = 'Close_Buy'
                position = 0
                continue
            bkhigh = max(bkhigh, row['high'])

        elif position == -1:
            stop_loss_atr = sklow + n_param * row['atr']
            stop_loss_stl = prev_row['close'] * (1 + 0.01 * stl_param)
            if row['close'] >= stop_loss_atr or row['high'] > stop_loss_stl:
                signals[i] = 'Close_Sell'
                position = 0
                continue
            sklow = min(sklow, row['low'])

        # --- 入场逻辑 ---
        if position == 0:
            if row['high'] > row['upper_bound']:
                signals[i] = 'Buy'
                position = 1
                bkhigh = row['high']
            elif row['low'] < row['lower_bound']:
                signals[i] = 'Sell'
                position = -1
                sklow = row['low']

    df['signal'] = signals
    return df



def generate_kdj_signals(df, length=9, signal_smooth=3, stl_param=5.0, n_param=6.0):
    """
    基于KDJ指标生成多头和空头交易信号，包含止损平仓。

    Args:
        df (pd.DataFrame): 包含'high', 'low', 'close'的DataFrame
        length (int): RSV的周期（一般9）
        signal_smooth (int): K与D的平滑周期（一般3）
        stl_param (float): 百分比止损
        n_param (float): ATR止损倍数

    Returns:
        pd.DataFrame: 添加'signal'列
    """
    if df.empty:
        df['signal'] = []
        return df

    # 计算KDJ
    kdj = ta.stoch(df['high'], df['low'], df['close'], k=length, d=signal_smooth, smooth_k=signal_smooth)
    df['K'] = kdj['STOCHk_9_3_3']
    df['D'] = kdj['STOCHd_9_3_3']
    df['J'] = 3 * df['K'] - 2 * df['D']
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=length)

    signals = ['Hold'] * len(df)
    position = 0  # 1=多头，-1=空头，0=空仓
    bkhigh = 0.0
    sklow = float('inf')

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        # --- 止损 ---
        if position == 1:
            stop_loss_atr = bkhigh - n_param * row['atr']
            stop_loss_stl = prev_row['close'] * (1 - 0.01 * stl_param)
            if row['close'] <= stop_loss_atr or row['low'] < stop_loss_stl:
                signals[i] = 'Close_Buy'
                position = 0
                continue
            bkhigh = max(bkhigh, row['high'])

        elif position == -1:
            stop_loss_atr = sklow + n_param * row['atr']
            stop_loss_stl = prev_row['close'] * (1 + 0.01 * stl_param)
            if row['close'] >= stop_loss_atr or row['high'] > stop_loss_stl:
                signals[i] = 'Close_Sell'
                position = 0
                continue
            sklow = min(sklow, row['low'])

        # --- 入场 ---
        if position == 0:
            if prev_row['K'] <= prev_row['D'] and row['K'] > row['D']:  # 金叉
                signals[i] = 'Buy'
                position = 1
                bkhigh = row['high']
            elif prev_row['K'] >= prev_row['D'] and row['K'] < row['D']:  # 死叉
                signals[i] = 'Sell'
                position = -1
                sklow = row['low']

    df['signal'] = signals
    return df


def generate_rsi_signals(df, rsi_length=14, overbought=70, oversold=30, stl_param=5.0, n_param=6.0):
    """
    基于 RSI 指标生成多空交易信号，附带止损逻辑。

    Args:
        df (pd.DataFrame): 包含 'close', 'high', 'low' 的DataFrame
        rsi_length (int): RSI周期长度
        overbought (float): 超买阈值
        oversold (float): 超卖阈值
        stl_param (float): 百分比止损（5表示5%）
        n_param (float): ATR止损倍数

    Returns:
        pd.DataFrame: 添加 'signal' 列的DataFrame
    """
    if df.empty:
        df['signal'] = []
        return df

    # 计算RSI和ATR
    df['rsi'] = ta.rsi(df['close'], length=rsi_length)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=rsi_length)

    signals = ['Hold'] * len(df)
    position = 0  # 0空仓，1多头，-1空头
    bkhigh = 0.0
    sklow = float('inf')

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        # --- 平仓止损 ---
        if position == 1:
            stop_loss_atr = bkhigh - n_param * row['atr']
            stop_loss_stl = prev_row['close'] * (1 - 0.01 * stl_param)
            if row['close'] <= stop_loss_atr or row['low'] < stop_loss_stl:
                signals[i] = 'Close_Buy'
                position = 0
                continue
            bkhigh = max(bkhigh, row['high'])

        elif position == -1:
            stop_loss_atr = sklow + n_param * row['atr']
            stop_loss_stl = prev_row['close'] * (1 + 0.01 * stl_param)
            if row['close'] >= stop_loss_atr or row['high'] > stop_loss_stl:
                signals[i] = 'Close_Sell'
                position = 0
                continue
            sklow = min(sklow, row['low'])

        # --- 开仓信号 ---
        if position == 0:
            # RSI金叉超卖线：开多
            if prev_row['rsi'] < oversold and row['rsi'] >= oversold:
                signals[i] = 'Buy'
                position = 1
                bkhigh = row['high']
            # RSI死叉超买线：开空
            elif prev_row['rsi'] > overbought and row['rsi'] <= overbought:
                signals[i] = 'Sell'
                position = -1
                sklow = row['low']

    df['signal'] = signals
    return df
