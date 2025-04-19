import math
import logging
import pandas as pd
import numpy as np
import talib

logging.basicConfig(level=logging.DEBUG)  # Set the logging level to DEBUG
format_pos = lambda price: ('-$' if price < 0 else '+$') + '{0:.2f}'.format(abs(price))
format_curr = lambda price: '${0:.2f}'.format(abs(price))

def display_train_result(result, val_pos, initial_offset):
    if val_pos == initial_offset or val_pos == 0.0:
        print('Episode {}/{} - Train Pos: USELESS  Val Pos: USELESS  Train Loss: {:.4f}'.format(result[0], result[1], result[3]))
        # logging.info('Episode {}/{} - Train Pos: {}  Val Pos: USELESS  Train Loss: {:.4f}'
        #              .format(result[0], result[1], format_pos(result[2]), result[3]))
    else:
        print('Episode {}/{} - Train Pos: {}  Val Pos: {}  Train Loss: {:.4f}'.format(result[0], result[1], format_pos(result[2]), format_pos(val_pos), result[3]))
        # logging.info('Episode {}/{} - Train Pos: {}  Val Pos: {}  Train Loss: {:.4f})'
        #              .format(result[0], result[1], format_pos(result[2]), format_pos(val_pos), result[3],))

def display_eval_result(model_name, profit, initial_offset):
    if profit == initial_offset or profit == 0.0:
        print('{}: USELESS\n'.format(model_name))
        # logging.info('{}: USELESS\n'.format(model_name))
    else:
        print('{}: {}\n'.format(model_name, format_pos(profit)))
        # logging.info('{}: {}\n'.format(model_name, format_pos(profit)))

def data_sample(df, sample_interval=4):
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df_sampled = df.resample(str(sample_interval)+'D').last()
    df_sampled = df_sampled.reset_index()
    return df_sampled

def read_stock_data(stock_file):
    df = pd.read_csv(stock_file)
    print("Loaded Data Size - {}".format(df.shape))
    df = data_sample(df)
    print("Sampled Data Size - {}".format(df.shape))
    return df

def sigmoid_activation(x):
    try:
        if x < 0:
            return 1 - 1 / (1 + math.exp(x))
        return 1 / (1 + math.exp(-x))
    except Exception as e:
        print("Error in sigmoid: " + str(e))
    
def indicator(df):
    df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=12)
    df['RSI'] = talib.RSI(df['Close'], timeperiod=12)
    df['SMA'] = talib.SMA(df['Close'], timeperiod=14)
    df['EMA'] = talib.EMA(df['Close'], timeperiod=14)
    df['WILLR'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=12)
    #bollinger band
    df['BBANDS_upper'], df['BBANDS_middle'], df['BBANDS_lower'] = talib.BBANDS(df['Close'], timeperiod=12, nbdevup=2, nbdevdn=2, matype=0)
    #volume indicator
    df['VMA'] = talib.MOM(df['Volume'], timeperiod=12)
    # stochastic oscillator
    df['STOCH_slowk'], df['STOCH_slowd'] = talib.STOCH(df['High'], df['Low'], df['Close'], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df.loc[:11, 'RSI'] = df['RSI'][12]
    df.loc[:11, 'ADX'] = df['ADX'][12]
    df.loc[:11, 'SMA'] = df['SMA'][12]
    df.loc[:11, 'EMA'] = df['EMA'][12]
    df.loc[:11, 'WILLR'] = df['WILLR'][12]
    df.loc[:11, 'BBANDS_upper'] = df['BBANDS_upper'][12]
    df.loc[:11, 'BBANDS_middle'] = df['BBANDS_middle'][12]
    df.loc[:11, 'BBANDS_lower'] = df['BBANDS_lower'][12]
    df.loc[:11, 'VMA'] = df['VMA'][12]
    df.loc[:11, 'STOCH_slowk'] = df['STOCH_slowk'][12]
    df.loc[:11, 'STOCH_slowd'] = df['STOCH_slowd'][12]

    return df        

def get_state_representation(df, t, n_days, indicator):

    data = list(df['Adj Close'])
    d = t - n_days + 1 
    block = data[d: t + 1] if d >= 0 else -d * [data[0]] + data[0: t + 1]
    res = []
    for i in range(n_days - 1):
        res.append(sigmoid_activation(block[i + 1] - block[i]))
    rsi = indicator['RSI'][d + n_days - 1]
    adx = indicator['ADX'][d + n_days - 1]
    sma = indicator['SMA'][d + n_days - 1]
    ema = indicator['EMA'][d + n_days - 1]
    willr = indicator['WILLR'][d + n_days - 1]
    BBANDS_upper = indicator['BBANDS_upper'][d + n_days - 1]
    BBANDS_middle = indicator['BBANDS_middle'][d + n_days - 1]
    BBANDS_lower = indicator['BBANDS_lower'][d + n_days - 1]
    VMA = indicator['VMA'][d + n_days - 1]
    STOCH_slowk = indicator['STOCH_slowk'][d + n_days - 1]
    STOCH_slowd = indicator['STOCH_slowd'][d + n_days - 1]

    res.append(rsi)
    res.append(adx)
    res.append(sma)
    res.append(ema)
    res.append(willr)
    res.append(BBANDS_upper)
    res.append(BBANDS_middle)
    res.append(BBANDS_lower)
    res.append(VMA)
    res.append(STOCH_slowk)
    res.append(STOCH_slowd)
    return np.array([res])
