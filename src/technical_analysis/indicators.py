from typing import Tuple, Callable, Union, List

import numpy as np
import pandas as pd

from technical_analysis._common import _atr, _bbands, _dbands, _true_range
from technical_analysis.moving_average import ema, sma, wilder_ma
from technical_analysis.utils import log_returns


def volatility(price: pd.Series, period: int, use_log: bool = True) -> pd.Series:
    if use_log:
        price = log_returns(price)
    else:
        price = price.pct_change()
    return price.rolling(period).std()


true_range = _true_range
atr = _atr


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    use_wilder_ma: bool = True,
) -> pd.Series:
    """
    Average True Range (measures volatility)
    and is the 14 day moving average of the following:
    ```
        max(
            high - low
            abs(high - prev_close)
            abs(low - prev_close)
        )
    ```
    """
    high_low = high - low
    high_cp = np.abs(high - close.shift(1))
    low_cp = np.abs(low - close.shift())
    df = pd.concat([high_low, high_cp, low_cp], axis=1)
    true_range = np.max(df, axis=1)
    if use_wilder_ma:
        average_true_range = wilder_ma(true_range, period)
    else:
        average_true_range = sma(true_range, period)
    return average_true_range


def rsi(
    price: pd.Series, period: int, ma_fn: Callable = sma, use_wilder_ma: bool = True
) -> pd.Series:
    """
    Relative Strength Index

    Calculation:
    -----------
        Average Gain = sum(gains over period) / period
        Average Loss = sum(losses over period) / period
        RS = Average Gain / Average Loss
        RSI = 100 - (100/(1+RS))
    """
    if use_wilder_ma:
        ma_fn = wilder_ma

    delta = price.diff()[1:]  # first row is nan
    gains, losses = delta.clip(lower=0), delta.clip(upper=0).abs()
    gains = ma_fn(gains, period)
    losses = ma_fn(losses, period)
    rs = gains / losses
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:] = np.select([losses == 0, gains == 0, True], [100, 0, rsi])
    return rsi


def perc_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """
    -------
    Williams %R
    Reflects the level of the close relative to the high-low range over a given period of time
    Oscillator between 0 and 1
    -------
    Calculation:
        %R = (Highest High - Close)/(Highest High - Lowest Low)

    Params:
        1. 'period' -> lookback period for highest high and lowest low
    -------
    Reference: https://school.stockcharts.com/doku.php?id=technical_indicators:williams_r
    """
    lowest_low = low.rolling(period).min()
    highest_high = high.rolling(period).max()
    return (highest_high - close) / (highest_high - lowest_low)


def perc_b(price: pd.Series, period: int = 20, num_std: int = 2) -> pd.Series:
    """
    %B measures a security's price in relation to the Bollinger Bands
    https://school.stockcharts.com/doku.php?id=technical_indicators:bollinger_band_perce
    %B = (Price - Lower Band) / (Upper Band - Lower Band)
    """
    lower_band, upper_band = _bbands(price, period=period, num_std=num_std)
    return (price - lower_band) / (upper_band - lower_band)


def perc_d(price: pd.Series, period: int = 20) -> pd.Series:
    """
    %D measures a security's price in relation to the Donchian Bands
    """
    lower_band, upper_band = _dbands(price, period=period)
    return (price - lower_band) / (upper_band - lower_band)


def tsi(price: pd.Series, period1: int = 25, period2: int = 13) -> pd.Series:
    """
    True Strength Index
    ------------

    Calculation:
    ------------
        Double Smoothed PC
        ------------------
        PC = Current Price minus Prior Price
        First Smoothing = 25-period EMA of PC
        Second Smoothing = 13-period EMA of 25-period EMA of PC

        Double Smoothed Absolute PC
        ---------------------------
        Absolute Price Change |PC| = Absolute Value of Current Price minus Prior Price
        First Smoothing = 25-period EMA of |PC|
        Second Smoothing = 13-period EMA of 25-period EMA of |PC|

        TSI = 100 x (Double Smoothed PC / Double Smoothed Absolute PC)

    Reference:
    ------------
        https://school.stockcharts.com/doku.php?id=technical_indicators:true_strength_index
    """
    shifted_price = price.shift(1)
    pc = price - shifted_price
    double_smoothed_pc = ema(ema(pc, period1), period2)

    pc = np.abs(pc)
    double_smoothed_abs_pc = ema(ema(pc, period1), period2)
    return 100 * (double_smoothed_pc / double_smoothed_abs_pc)


def trix(price: pd.Series, period: int = 15) -> pd.Series:
    """
    Displays percent rate of change as a triply smoothed moving average
        > similar to MACD, but smoother

    Calculation:
    ------------
        1. Single-Smoothed EMA = 15-period EMA of the closing price
        2. Double-Smoothed EMA = 15-period EMA of Single-Smoothed EMA
        3. Triple-Smoothed EMA = 15-period EMA of Double-Smoothed EMA
        4. TRIX = 1-period percent change in Triple-Smoothed EMA

    Reference:
    -----------
        https://school.stockcharts.com/doku.php?id=technical_indicators:trix
    """
    trix = ema(ema(ema(price, period), period), period)
    return trix.pct_change(1)


def stochastic(
    high: Union[pd.Series,pd.DataFrame],
    period: int,
    low: Union[pd.Series, None]=None,
    close: Union[pd.Series, None]=None,
    output: List[str] = ['perc_k', 'perc_d'],
    perc_k_smoothing: int = 0,
    perc_d_smoothing: int = 3,
) -> Union[pd.Series, Tuple[pd.Series]]:
    """
    Stochastic Oscillator
    ----------

    Calculation:
    ----------
        %K = (Current Close - Lowest Low)/(Highest High - Lowest Low) * 100
        %D = 3-day SMA of %K

    Three modes:
    ----------
        1. Fast
            perc_k_smoothing = 0
            perc_d_smoothing = 3

        2. Slow
            perc_k_smoothing = 3
            perc_d_smoothing = 3

        3. Full
            perc_k_smoothing > 3
            perc_d_smoothing > 3

    Reference:
    -----------
        https://school.stockcharts.com/doku.php?id=technical_indicators:stochastic_oscillator_fast_slow_and_full

    """
    if isinstance(high, pd.DataFrame):
        low = high['low']
        close = high['close']
        high = high['high']

    lowest_low = low.rolling(period).min()
    highest_high = high.rolling(period).max()

    perc_k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    if perc_k_smoothing:
        perc_k = sma(perc_k, perc_k_smoothing)
    perc_d = sma(perc_k, perc_d_smoothing)  # the trigger line
    output_data = {"perc_k": perc_k, "perc_d": perc_d}
    if len(output) == 1:
        return output_data[output[0]]
    return (output_data[o] for o in output)


def macd(
    price: Union[pd.Series,pd.DataFrame],
    output: List[str] = ["macd"],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Union[pd.Series, Tuple[pd.Series]]:
    """
    Moving Average Convergence/Divergence (MACD)

    Calculation:
    -----------
        MACD Line: (12-day EMA - 26-day EMA)
        Signal Line: 9-day EMA of MACD Line
        MACD Histogram: MACD Line - Signal Line

    Returns:
    -----------
    defined by 'output' argument [macd,signal,hist]
    - tuple(pd.Series) if more then one selected, otherway pd.Series

    Reference:
    -----------
        https://school.stockcharts.com/doku.php?id=technical_indicators:moving_average_convergence_divergence_macd

    """
    macd_line = ema(price, period=fast_period) - ema(price, period=slow_period)
    signal_line = ema(macd_line, period=signal_period)
    histogram = macd_line - signal_line
    output_data = {"macd": macd_line, "signal": signal_line, "hist": histogram}
    if len(output) == 1:
        return output_data[output[0]]
    return (output_data[o] for o in output)


def trend_up(price: pd.Series, period: int = 5) -> pd.Series:
    """
    Return True when trend is up, false otherway
    for down trend on stock, makes more sense to provide high price
    """
    return trend_down(price *-1, period)


def trend_down(price: pd.Series, period: int = 5) -> pd.Series:
    """
    Return True when trend is down, false otherway
    for down trend on stock, makes more sense to provide low price
    check price against average, simply but usefull in most cases
    """
    df = pd.DataFrame(price)
    df.reset_index(drop=True, inplace=True)
    col = df.columns[0]
    trend = []
    i_start = 0

    while True:
        dat = df.loc[i_start:].copy()
        dat.reset_index(inplace=True)  # will move 'global' index to column 'index'
        dat["cummean"] = dat[col].cumsum() / (dat.index + 1)
        dat["trend"] = dat[col] < dat["cummean"]

        ##########
        # OPENING TREND
        # checks next 'window' points are below mean
        from_trend = (
            dat["trend"]
            .sort_index(ascending=False)
            .rolling(period)
            .apply(lambda x: min(x.index) if all(x) else np.nan)
        ).min()
        # drop before so cumulative mean is calculated during trend only
        if from_trend > 1:
            # max value may be in droped rows
            i_start += dat.loc[1:from_trend, col].idxmax()
            continue

        ############
        # CLOSING TREND
        # if last 'window' points were above mean
        # we are in trend
        to_trend_s = (
            dat["trend"]
            .rolling(period)
            .apply(lambda x: max(x.index) if all(x) else np.nan)
        )
        # closing trend when last digit (not nan()) followed by nan()
        to_trend = (
            (to_trend_s.isna().shift(-1) & ~to_trend_s.isna())
            .map({False: np.nan, True: 0})
            .first_valid_index()
        )

        if not to_trend or to_trend > len(df) - period:
            break
        # close the trend at minimum
        to_trend = dat.loc[from_trend:to_trend, col].idxmin()
        trend.append([dat.loc[from_trend, "index"], dat.loc[to_trend, "index"]])
        i_start = dat.loc[to_trend, "index"] + 1
    
    price=price.astype(str).replace(regex=r'.+',value=False)
    for t in trend:
        price.iloc[t[0]:t[1]] = True
    return price


def obv(price:Union[pd.Series, pd.DataFrame],volume: Union[pd.Series, None]=None) -> pd.Series:
    """
    On Balance Volume
    """
    if isinstance(price, pd.DataFrame):
        volume = price['volume']
        price = price["close"]
    obv = pd.Series(0.0, index=price.index)
    obv[price > price.shift(1)] = volume[price > price.shift(1)]
    obv[price < price.shift(1)] = -volume[price < price.shift(1)]
    return obv.cumsum()