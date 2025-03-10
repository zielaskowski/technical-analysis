import os
import pandas as pd
from technical_analysis.indicators import ema
from technical_analysis.indicators import trend_down, atr, sma, adx, tr
from technical_analysis.candles import *
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


# df = pd.read_csv("./src/pkn.csv")
# df = df[["date", "open", "high", "low", "val", "vol"]]
# df = df.rename(columns={"val": "close"})
# df["date"] = pd.to_datetime(df["date"])
# resample_agg = {
#     "open": "first",
#     "high": "max",
#     "low": "min",
#     "close": "last",
#     "vol": "sum",
# }
# df = df.resample(rule="D", on="date").agg(resample_agg)
# df.reset_index(inplace=True)


# https://github.com/twopirllc/pandas-ta
sample_data = pd.read_csv(
    f"./src/SPY_D.csv",
    index_col=0,
    parse_dates=True,
    infer_datetime_format=True
)
sample_data['vol'] = sample_data['volume']
sample_data["date"] = pd.to_datetime(sample_data["date"])
# sample_data = sample_data.loc[sample_data['date']<pd.to_datetime('2004-04-20')]
# sample_data = sample_data.loc[sample_data['date']<pd.to_datetime('2003-08-05')]
# sample_data = sample_data.loc[(sample_data['date']<pd.to_datetime('2004-01-28')) & (sample_data['date']>pd.to_datetime('2003-04-20'))]
# sample_data = sample_data.loc[(sample_data['date']<pd.to_datetime('2007-05-23')) & (sample_data['date']>pd.to_datetime('2007-04-01'))]
# sample_data = sample_data.loc[(sample_data['date']<pd.to_datetime('2002-10-12')) & (sample_data['date']>pd.to_datetime('2002-08-01'))]
# sample_data = sample_data.loc[(sample_data['date']<pd.to_datetime('2001-05-30')) & (sample_data['date']>pd.to_datetime('2001-01-01'))]
df= sample_data



fig = go.Figure()
fig.add_trace(
    go.Candlestick(
        x=df["date"], 
        open=df["open"], 
        high=df["high"], 
        low=df["low"], 
        close=df["close"]
    )
)

fig.show()
# px.bar(df, x='date',y=['open', 'high', 'low', 'close'],title='candles', barmode='group').show()


#*********************
#CANDLES
#*********************
rt = rising_three(df, lookback=20)
print(df[rt]) # 2004-04-19
rt = rising_n(df, n=5, lookback=20)
print(df[rt]) # 2003-08-04

be = bearish_engulfing(df, trend_lookback=20)
print(df[be])
be = bullish_engulfing(df, trend_lookback=20)
print(df[be])

dc = dark_cloud(
    df, 
    trend_lookback=20, 
    trend_threshold=0.03, 
    min_body_size=0.7, 
    new_high_periods=20
)
print(df[dc]) # 2004-01-27

bs = bearish_star(df,
            lookback= 20,
            min_body_size= 0.7,
            relative_threshold= 0.3,
            min_gap_size= 0.001
 )
print(df[bs]) # 2007-05-22 

bs = bullish_star(df,
            lookback= 20,
            min_body_size= 0.7,
            relative_threshold= 0.3,
            min_gap_size= 0.001
        )
print(df[bs]) # 2002-02-08


bi=bearish_island(df,
               min_gap_size= 0.001, 
               lookback= 30
               )
print(df[bi])

bi = bullish_island(df,
            min_gap_size= 0.001,
            lookback= 30)
print(df[bi])

# add island cluster!
# bullish reversal:
# 2002-10-11
open = df['open']
high = df['high']
low = df['low']
close = df['close']
min_gap_size: float = 0.001
lookback: int = 20
cluster_min = 3
cluster_max = 7

downtrend = is_bearish_trend(close, lookback)
open_red = negative_close(open, close)
down_gap = is_gap_down(high, low, min_gap_size)
up_gap = is_gap_up(high, low, min_gap_size)
close_green = positive_close(open, close)

is_formation = open != open # trick to have correct index; hate pandas!
for n in range(cluster_min, cluster_max + 1):
    cluster_no_gap = open == open
    cluster_below = open == open
    cluster_high_limit = pd.DataFrame({"low": low, "shifted": low.shift(n+1)}).min(axis=1)
    for i in range(1, n):
        # no gap in cluster
        cluster_no_gap = (cluster_no_gap | ~(down_gap.shift(i, fill_value=False) | 
                                            up_gap.shift(i, fill_value=False)))
        # cluster below opening and closing candle
        cluster_below = (cluster_below & (high.shift(i) < cluster_high_limit))
    is_formation = (is_formation | (downtrend & 
                                    open_red.shift(n+1) & 
                                    down_gap.shift(n) & 
                                    cluster_no_gap &
                                    cluster_below &
                                    up_gap & 
                                    close_green))
print(df[is_formation])
# bearish reversal
# 2001-05-29
uptrend = is_bullish_trend(close, lookback)
open_green = positive_close(open, close)
up_gap = is_gap_up(high, low, min_gap_size)
down_gap = is_gap_down(high, low, min_gap_size)
close_red = negative_close(open, close)

is_formation = open != open
for n in range(cluster_min, cluster_max+1):
    cluster_no_gap = open == open
    cluster_above = open == open
    cluster_low_limit = pd.DataFrame({"high": high, "shifted": high.shift(n+1)}).max(axis=1)
    for i in range(1,n):
        # no gap in cluster
        cluster_no_gap = (cluster_no_gap | ~(down_gap.shift(i, fill_value=False) & 
                                      up_gap.shift(i, fill_value=False)))
        cluster_above = (cluster_above & (low.shift(i) > cluster_low_limit))
    is_formation = (is_formation | (uptrend & 
                                    open_green.shift(n+1) & 
                                    up_gap.shift(n) & 
                                    cluster_no_gap &
                                    cluster_above &
                                    down_gap & 
                                    close_red))
print(df[is_formation])    

bt = bearish_tasuki_gap(df,
            trend_lookback= 30,
            trend_threshold= -0.03,
            min_body_size= 0.75,
            min_gap_size= 0.002
        )
print(df[bt])


bt = bullish_tasuki_gap(df,
            trend_lookback= 30,
            trend_threshold= -0.03,
            min_body_size= 0.75,
            min_gap_size= 0.002
        )
print(df[bt])

bc= n_black_crows(df, 
            n= 5,
            lookback= 20,
            min_body_size= 0.75,
            close_threshold= 0.002
        )
print(df[bc])

ws=n_white_soldiers(df, 
            n= 5,
            lookback= 20,
            min_body_size= 0.75,
            close_threshold= 0.002
        )
print(df[ws])



ema = ema(df, period=20)
fig.add_trace(go.Line(x=df["date"], y=ema, name="ema"))
fig.show()


df["adx"] = adx(df, output=["adx"])


high = df["high"]
low = df["low"]
price = df["close"]

plusDM = pd.Series(0.0, index=high.index, dtype=float)
minusDM = pd.Series(0.0, index=high.index, dtype=float)

up_move = high - high.shift(1)
down_move = low.shift(1) - low
trs = tr(high, low, price)
trs = ema(trs, 14)

plusDM[up_move > down_move] = pd.DataFrame({"up": up_move, "zer": 0}).max(axis=1)
minusDM[down_move > up_move] = pd.DataFrame({"down": down_move, "zer": 0}).max(axis=1)

plusDI = ema(plusDM, period=14) / trs * 100
plusDI.ffill(inplace=True)
minusDI = ema(minusDM, period=14) / trs * 100
minusDI.ffill(inplace=True)

dx = 100 * abs((plusDI - minusDI) / (plusDI + minusDI))

# adx= ema(dx,14)


df["exADX"] = adx
df["minusDI"] = minusDI
df["plusDI"] = plusDI
df["trs"] = trs
df["tr"] = tr(high, low, price)


# px.line(x=df.loc[~df['plusDM'].isna(),'date'],y=df.loc[~df['plusDM'].isna(),'plusDM'],title='plusDM').show()
px.line(x=df["date"], y=df["adx"], title="adx").show()

# fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# axs[1].plot(df["ta"])
# axs[2].plot(df["obv"], label="obv")
# axs[2].plot(df["ad"], label="ad")
# axs[1].plot(df["adx"], label="adx")
# axs[1].plot(df["exADX"], label="my_ex_adx")
# axs[1].plot(df["minusDI"], label="minusDI")
# axs[1].plot(df["plusDI"], label="plusDI")
# axs[1].plot(df["trs"], label="trs")
# axs[1].plot(df["tr"], label="tr")
# #axs[1].plot(df["ADX"], label="ex_adx")
# # axs[2].plot(df["+DI"], label="+DI")
# # axs[2].plot(df["-DI"], label="-DI")
# # axs[2].plot(df["ema20"])
# # axs[2].plot(df["ema50"])
# # axs[2].plot(df["ema200"])
# axs[0].plot(df["close"])
# axs[1].legend()
# plt.show(block=True)
