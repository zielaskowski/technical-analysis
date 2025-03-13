import numpy as np
import pandas as pd

from technical_analysis.decorators import df_ohlc_to_series
from technical_analysis.utils import is_bullish_trend, is_bearish_trend, get_body
from technical_analysis.candles.single import (
    body_outside_body,
    is_doji,
    is_gap_down,
    is_gap_up,
    is_long_body,
    negative_close,
    positive_close,
)
from technical_analysis.utils import get_body, is_bearish_trend, is_bullish_trend


@df_ohlc_to_series
def dark_cloud(
    open: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    trend_lookback: int = 30,
    trend_threshold: float = 0.03,
    min_body_size: float = 0.7,
    new_high_periods: int = 30,
) -> pd.Series:
    """
    Bearish Reversal Pattern

    Candles:
    ---------
        1. a long bullish body
            - We define this as 75% of price action taking place between open and close
        2. next opens at a new high, then closes below midpoint of the body of (1)
            - We define new high as within the previous 100 periods
            - The midpoint is the timesteps (high(t-1) - low(t-1))
    """
    uptrend = is_bullish_trend(close, lookback=trend_lookback, threshold=trend_threshold)
    bullish_long_body = is_long_body(open, high, low, close, min_body_size=min_body_size) & positive_close(open, close)
    new_high_comparator = high == high.rolling(new_high_periods).max()
    close_below_midpoint = close < (high.shift(1) + low.shift(1)) / 2
    return uptrend & bullish_long_body.shift(1) & new_high_comparator & close_below_midpoint


@df_ohlc_to_series
def bullish_engulfing(
    open: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    trend_lookback: int = 30,
    trend_threshold: float = 0.03,
) -> pd.Series:
    """
    Bullish Englufing Pattern (Reversal)
    ---------

    Candles:
    ---------
    In a bearish trend:
        1. a small body
        2. a body that completely englufs the body of (1)
    """
    downtrend = is_bearish_trend(close, lookback=trend_lookback, threshold=trend_threshold)
    outisde_body = body_outside_body(open, close)
    return downtrend & outisde_body & positive_close(open, close)


@df_ohlc_to_series
def bearish_engulfing(
    open: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    trend_lookback: int = 30,
    trend_threshold: float = 0.03,
) -> pd.Series:
    """
    Bullish Englufing Pattern (Reversal)
    ---------

    Candles:
    ---------
    In a bearish trend:
        1. a small body
        2. a body that completely englufs the body of (1)
    """
    uptrend = is_bullish_trend(close, lookback=trend_lookback, threshold=trend_threshold)
    outisde_body = body_outside_body(open, close)
    return uptrend & outisde_body & negative_close(open, close)


@df_ohlc_to_series
def n_black_crows(
    open: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n: int,
    lookback: int = 30,
    min_body_size: float = 0.75,
    close_threshold: float = 0.002,
) -> pd.Series:
    """
    'n' black crows algorithm
    ---------

    Candles:
    ---------
    In uptrend:
        1. three consecutive black (red), long bodies that:
            - close negative near the lows
            - open inside the body of the previous candle
    """
    uptrend = is_bullish_trend(close, lookback)

    long_body_exists = is_long_body(open, high, low, close, min_body_size=min_body_size, lookback=0)
    long_body_exists = long_body_exists & negative_close(open, close)

    prev_lower_body, prev_upper_body = get_body(open, close)
    prev_lower_body = prev_lower_body.shift(1)
    prev_upper_body = prev_upper_body.shift(1)

    open_in_body = (open > prev_lower_body) & (open < prev_upper_body)
    close_near_lows = (np.abs(close - low) / low) < close_threshold
    are_crows = long_body_exists & open_in_body & close_near_lows
    for shift in list(range(n - 1, 0, -1)):
        are_crows = are_crows & are_crows.shift(shift)
    return uptrend & are_crows


@df_ohlc_to_series
def n_white_soldiers(
    open: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n: int,
    lookback: int = 30,
    min_body_size: float = 0.75,
    close_threshold: float = 0.002,
) -> pd.Series:
    """
    'n' white soldiers algorithm
    ---------

    Candles:
    ---------
    In downtrend:
        1. three consecutive white (green), long bodies that:
            - close positive near the highs
            - open inside the body of the previous candle
    """
    downtrend = is_bearish_trend(close, lookback)

    long_body_exists = is_long_body(open, high, low, close, min_body_size=min_body_size, lookback=0)
    long_body_exists = long_body_exists & positive_close(open, close)

    prev_lower_body, prev_upper_body = get_body(open, close)
    prev_lower_body = prev_lower_body.shift(1)
    prev_upper_body = prev_upper_body.shift(1)

    open_in_body = (open > prev_lower_body) & (open < prev_upper_body)
    close_near_highs = (np.abs(high - close) / close) < close_threshold
    are_crows = long_body_exists & open_in_body & close_near_highs
    for shift in list(range(n - 1, 0, -1)):
        are_crows = are_crows & are_crows.shift(shift)
    return downtrend & are_crows


@df_ohlc_to_series
def bullish_island(
    open: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    min_gap_size: float = 0.001,
    lookback: int = 30,
    cluster_min = 1,
    cluster_max = 1
) -> pd.Series:
    """
    Bullish island reversal
    
    Candles:
    ----------
        1. start with red candle
        2. gap below(1)
        3. cluster of candles, number of candles between cluster_min and cluster_max, without gaps
        4. gap above 
        5. white candle
    """
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
    return is_formation


@df_ohlc_to_series
def bearish_island(
    open: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    min_gap_size: float = 0.001,
    lookback: int = 30,
    cluster_min = 1,
    cluster_max = 1
) -> pd.Series:
    """
    Bullish island reversal

    Candles:
    ----------
        1. start with white candle
        2. gap above(1)
        3. cluster of candles, number of candles between cluster_min and cluster_max, without gaps
        4. gap below
        5. red candle
    """
    uptrend = is_bullish_trend(close, lookback)
    up_gap = is_gap_up(high, low, min_gap_size)
    open_green = positive_close(open, close)
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
            # cluster above opening and closing candle
            cluster_above = (cluster_above & (low.shift(i) > cluster_low_limit))
        is_formation = (is_formation | (uptrend & 
                                        open_green.shift(n+1) & 
                                        up_gap.shift(n) & 
                                        cluster_no_gap &
                                        cluster_above &
                                        down_gap & 
                                        close_red))
    return is_formation


@df_ohlc_to_series
def bullish_star(
    open: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    lookback: int = 30,
    min_body_size: float = 0.7,
    relative_threshold: float = 0.3,
    min_gap_size: float = 0.001,
) -> pd.Series:
    """
    Morning Doji Start Reversal
    https://www.candlescanner.com/candlestick-patterns/morning-doji-star/

    Candles:
    ----------
        1. long black body
        2. short doji-like candle with a gap-down
        3. long-ish white candle that , start above doji and ends above half of (1)
    """
    downtrend = is_bearish_trend(close, lookback)
    long_body_exists = is_long_body(open, high, low, close, min_body_size=min_body_size, lookback=0)
    long_red = long_body_exists & negative_close(open, close)
    valid_star = is_doji(open, high, low, close, relative_threshold=relative_threshold)
    valid_star = valid_star & is_gap_down(high, low, min_gap_size=min_gap_size)
    reverse_candle = long_body_exists & positive_close(open, close)
    above_midpoint = close > (close.shift(2) + (open.shift(2) - close.shift(2)) / 2)
    return (downtrend & 
            long_red.shift(2) & 
            valid_star.shift(1) & 
            reverse_candle & 
            is_gap_up(high, low, min_gap_size) &
            above_midpoint)


@df_ohlc_to_series
def bearish_star(
    open: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    lookback: int = 30,
    min_body_size: float = 0.7,
    relative_threshold: float = 0.3,
    min_gap_size: float = 0.001,
) -> pd.Series:
    """
    Evening Doji Start Reversal
    https://www.candlescanner.com/candlestick-patterns/evening-doji-star/

    Candles:
    ----------
        1. long white body
        2. short doji-like candle with a gap-up
        3. long-ish black candle that reverses direction, start below doji and ends below half of (1)
    """
    uptrend = is_bullish_trend(close, lookback)
    long_body_exists = is_long_body(open, high, low, close, min_body_size=min_body_size, lookback=0)
    long_green = long_body_exists & positive_close(open, close)
    valid_star = is_doji(open, high, low, close, relative_threshold=relative_threshold)
    valid_star = valid_star & is_gap_up(high, low, min_gap_size=min_gap_size)
    reverse_candle = long_body_exists & negative_close(open, close)
    below_midpoint = close < (open.shift(2) + (close.shift(2) - open.shift(2)) / 2)
    return (uptrend & 
            long_green.shift(2) & 
            valid_star.shift(1) & 
            reverse_candle & 
            is_gap_down(high, low, min_gap_size) & 
            below_midpoint)
