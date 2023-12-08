import pandas as pd
from technical_analysis.indicators import trend_down, adx, tr, sma, ema
import matplotlib.pyplot as plt

df = pd.read_csv("ale.csv")
#dfx=pd.read_csv('adx.csv', skiprows=1)

df['adx']=adx(df, output=['adx'])

# Assuming your dfFrame is named 'df'
# Calculate the difference between the highest and lowest prices for each day
df['HighLowDiff'] = df['high'] - df['low']

# Calculate the difference between the open and close prices for each day
df['OpenCloseDiff'] = df['close'] - df['open']

# Calculate the absolute value of the above differences
df['HighLowAbs'] = df['HighLowDiff'].abs()
df['OpenCloseAbs'] = df['OpenCloseDiff'].abs()

# Calculate the moving average of the absolute value of the differences
df['EMA_HighLow'] = df['HighLowAbs'].ewm(span=14).mean()
df['EMA_OpenClose'] = df['OpenCloseAbs'].ewm(span=14).mean()

# Calculate the ratio of the moving averages
df['ADX_Ratio'] = df['EMA_HighLow'] / df['EMA_OpenClose']

# Calculate the ADX index
df['aiADX'] = 14 * (df['ADX_Ratio'].rolling(window=14).mean())



high = df['high']
low = df['low']
price = df['close']

plusDM = pd.Series(0.0, index=high.index)
minusDM = pd.Series(0.0, index=high.index)

up_move = high - high.shift(1)
down_move = low.shift(1) - low
trn = tr(high, low, price)
plusDM[(up_move > down_move)] = pd.DataFrame({'up':up_move,'zer':0}).max(axis=1)
minusDM[(down_move > up_move)] = pd.DataFrame({'down':down_move,'zer':0}).max(axis=1)

plusDI = sma(plusDM/trn,14) 
minusDI = sma(minusDM/trn,14) 

dx= 100 * (abs(plusDI - minusDI)/(plusDI + minusDI))

adx= ema(dx,14)
adxr = (adx - adx.shift(14)) / 2
# plusDI = 100 * (sma(plusDM, 14) / sma(tr(high, low, price),14))
# minusDI = 100 * (sma(minusDM, 14) / sma(tr(high, low, price),14))
# adx = ema((abs(plusDI - minusDI) / (plusDI + minusDI))* 100, 14)


df['exADX'] = adx


fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

#axs[1].plot(df["ta"])
# axs[2].plot(df["obv"], label="obv")
# axs[2].plot(df["ad"], label="ad")
axs[1].plot(df["adx"], label="adx")
axs[1].plot(df["aiADX"], label="ai_adx")
axs[1].plot(df["exADX"], label="my_ex_adx")
#axs[1].plot(df["ADX"], label="ex_adx")
# axs[2].plot(df["+DI"], label="+DI")
# axs[2].plot(df["-DI"], label="-DI")
# axs[2].plot(df["ema20"])
# axs[2].plot(df["ema50"])
# axs[2].plot(df["ema200"])
axs[0].plot(df["close"])
axs[1].legend()
plt.show(block=True)
