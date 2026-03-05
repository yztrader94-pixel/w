import asyncio
import ccxt.async_support as ccxt
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.constants import ParseMode
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta
import logging
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════
#  ORDER BLOCK DETECTION  (swing-optimised)
#  Higher swing_strength + lookback for clean structural OBs only
# ═══════════════════════════════════════════════════════════════

def detect_order_blocks(df, lookback=100, swing_strength=6):
    """
    Swing-grade Order Block detection.

    Uses a wider swing_strength window so only clean structural OBs
    are captured — removes noise from minor pullbacks.

    Bullish OB = last BEARISH candle before a strong bullish impulse
                 that broke a structural swing high.
    Bearish OB = last BULLISH candle before a strong bearish impulse
                 that broke a structural swing low.

    Returns list of dicts:
        type, top, bottom, index, mitigated,
        body_size (candle body as % of ATR — quality proxy),
        impulse_strength (how hard the breakout was)
    """
    obs = []
    highs  = df['high'].values
    lows   = df['low'].values
    opens  = df['open'].values
    closes = df['close'].values
    n      = len(df)

    if n < lookback + swing_strength * 2:
        return obs

    # Estimate ATR for quality scoring
    atr_vals = [abs(highs[i] - lows[i]) for i in range(max(0, n-20), n)]
    atr_est  = np.mean(atr_vals) if atr_vals else 1

    start = max(swing_strength, n - lookback)

    for i in range(start, n - swing_strength):
        # ── BULLISH OB ───────────────────────────────────────────
        if closes[i] < opens[i]:   # bearish candle
            impulse_high    = max(highs[i+1 : i+1+swing_strength])
            prior_swing_high = max(highs[max(0, i-swing_strength) : i])

            if impulse_high > prior_swing_high:
                body_size        = abs(opens[i] - closes[i]) / atr_est
                impulse_strength = (impulse_high - prior_swing_high) / atr_est
                obs.append({
                    'type':             'bullish',
                    'top':              max(opens[i], closes[i]),
                    'bottom':           min(opens[i], closes[i]),
                    'index':            i,
                    'mitigated':        False,
                    'body_size':        round(body_size, 2),
                    'impulse_strength': round(impulse_strength, 2),
                })

        # ── BEARISH OB ───────────────────────────────────────────
        elif closes[i] > opens[i]:   # bullish candle
            impulse_low     = min(lows[i+1 : i+1+swing_strength])
            prior_swing_low  = min(lows[max(0, i-swing_strength) : i])

            if impulse_low < prior_swing_low:
                body_size        = abs(opens[i] - closes[i]) / atr_est
                impulse_strength = (prior_swing_low - impulse_low) / atr_est
                obs.append({
                    'type':             'bearish',
                    'top':              max(opens[i], closes[i]),
                    'bottom':           min(opens[i], closes[i]),
                    'index':            i,
                    'mitigated':        False,
                    'body_size':        round(body_size, 2),
                    'impulse_strength': round(impulse_strength, 2),
                })

    # Mark mitigated — swing grade: only mark if a FULL candle CLOSED through the OB
    current_price = closes[-1]
    for ob in obs:
        if ob['type'] == 'bullish' and current_price < ob['bottom']:
            ob['mitigated'] = True
        elif ob['type'] == 'bearish' and current_price > ob['top']:
            ob['mitigated'] = True

    return obs


def price_at_order_block(current_price, obs, tolerance=0.005):
    """
    Returns the best (most recent, strongest) unmitigated OB that
    price is currently touching or sitting inside.
    tolerance = 0.5% cushion (wider than day-trade to account for daily wicks)
    """
    best      = None
    best_type = None

    for ob in obs:
        if ob['mitigated']:
            continue

        cushion = current_price * tolerance
        in_zone = (current_price >= ob['bottom'] - cushion and
                   current_price <= ob['top']    + cushion)

        if in_zone:
            # Prefer most recent; break ties by impulse strength
            if best is None:
                best      = ob
                best_type = ob['type']
            elif ob['index'] > best['index']:
                best      = ob
                best_type = ob['type']
            elif ob['index'] == best['index'] and ob['impulse_strength'] > best['impulse_strength']:
                best      = ob
                best_type = ob['type']

    return best_type, best


# ═══════════════════════════════════════════════════════════════
#  SMART MONEY CONCEPTS  (swing-optimised)
# ═══════════════════════════════════════════════════════════════

def detect_swing_points(df, swing_length=8):
    """Swing highs / lows — wider window for swing-grade pivots."""
    highs = df['high'].values
    lows  = df['low'].values
    n     = len(df)
    s     = swing_length

    swing_highs = []
    swing_lows  = []

    for i in range(s, n - s):
        if highs[i] == max(highs[i-s : i+s+1]):
            swing_highs.append((i, highs[i]))
        if lows[i] == min(lows[i-s : i+s+1]):
            swing_lows.append((i, lows[i]))

    return swing_highs, swing_lows


def detect_bos_choch(df, swing_length=8):
    """BOS / CHoCH on swing timeframe (wider pivots)."""
    closes = df['close'].values
    n      = len(df)

    swing_highs, swing_lows = detect_swing_points(df, swing_length)

    empty = {
        'last_bos_bull': None, 'last_bos_bear': None,
        'last_choch_bull': None, 'last_choch_bear': None,
        'swing_trend': 'neutral',
        'recent_bull_structure': False,
        'recent_bear_structure': False,
    }

    if not swing_highs or not swing_lows:
        return empty

    swing_trend     = 'neutral'
    last_bos_bull   = last_bos_bear   = None
    last_choch_bull = last_choch_bear = None

    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        if swing_highs[-1][1] > swing_highs[-2][1] and swing_lows[-1][1] > swing_lows[-2][1]:
            swing_trend = 'bullish'
        elif swing_highs[-1][1] < swing_highs[-2][1] and swing_lows[-1][1] < swing_lows[-2][1]:
            swing_trend = 'bearish'

    last_sh_price = swing_highs[-1][1]
    last_sl_price = swing_lows[-1][1]

    # Scan recent bars — use last 15 bars for swing (daily bars)
    lookback = min(15, n - 1)
    for i in range(n - lookback, n):
        if closes[i] > last_sh_price and (i == 0 or closes[i-1] <= last_sh_price):
            if swing_trend == 'bearish':
                last_choch_bull = i
            else:
                last_bos_bull   = i

        if closes[i] < last_sl_price and (i == 0 or closes[i-1] >= last_sl_price):
            if swing_trend == 'bullish':
                last_choch_bear = i
            else:
                last_bos_bear   = i

    recent_cutoff = n - 8   # within last 8 bars (daily = 8 days)
    return {
        'last_bos_bull':         last_bos_bull,
        'last_bos_bear':         last_bos_bear,
        'last_choch_bull':       last_choch_bull,
        'last_choch_bear':       last_choch_bear,
        'swing_trend':           swing_trend,
        'recent_bull_structure': (
            (last_bos_bull   is not None and last_bos_bull   >= recent_cutoff) or
            (last_choch_bull is not None and last_choch_bull >= recent_cutoff)),
        'recent_bear_structure': (
            (last_bos_bear   is not None and last_bos_bear   >= recent_cutoff) or
            (last_choch_bear is not None and last_choch_bear >= recent_cutoff)),
    }


def detect_fair_value_gaps(df, min_gap_pct=0.002):
    """FVGs with tighter min_gap_pct filter — only clean institutional gaps."""
    highs  = df['high'].values
    lows   = df['low'].values
    closes = df['close'].values
    n      = len(df)

    bull_fvgs = []
    bear_fvgs = []

    for i in range(2, n):
        if lows[i] > highs[i-2]:
            gap = (lows[i] - highs[i-2]) / highs[i-2]
            if gap >= min_gap_pct:
                bull_fvgs.append({'top': lows[i], 'bottom': highs[i-2],
                                   'mid': (lows[i] + highs[i-2]) / 2, 'index': i, 'gap_pct': gap})
        if highs[i] < lows[i-2]:
            gap = (lows[i-2] - highs[i]) / lows[i-2]
            if gap >= min_gap_pct:
                bear_fvgs.append({'top': lows[i-2], 'bottom': highs[i],
                                   'mid': (lows[i-2] + highs[i]) / 2, 'index': i, 'gap_pct': gap})

    current = closes[-1]
    active_bull = [f for f in bull_fvgs if current > f['bottom']]
    active_bear = [f for f in bear_fvgs if current < f['top']]

    nearest_bull = min(active_bull, key=lambda f: abs(current - f['mid']), default=None)
    nearest_bear = min(active_bear, key=lambda f: abs(current - f['mid']), default=None)

    return {
        'bull_fvgs': active_bull[-5:],
        'bear_fvgs': active_bear[-5:],
        'nearest_bull': nearest_bull,
        'nearest_bear': nearest_bear,
    }


def detect_premium_discount(df, swing_length=8):
    """
    Premium / Discount using structural swing high/low (not all-time range).
    Uses last clean swing HH and LL for a more relevant range.
    """
    swing_highs, swing_lows = detect_swing_points(df, swing_length)

    if len(swing_highs) >= 1 and len(swing_lows) >= 1:
        swing_high = max(h[1] for h in swing_highs[-3:])
        swing_low  = min(l[1] for l in swing_lows[-3:])
    else:
        swing_high = df['high'].values.max()
        swing_low  = df['low'].values.min()

    rng = swing_high - swing_low
    if rng == 0:
        return {'zone': 'neutral', 'range_pct': 50, 'swing_high': swing_high, 'swing_low': swing_low}

    current   = df['close'].iloc[-1]
    range_pct = (current - swing_low) / rng * 100

    if range_pct >= 70:
        zone = 'premium'
    elif range_pct <= 30:
        zone = 'discount'
    elif 45 <= range_pct <= 55:
        zone = 'equilibrium'
    else:
        zone = 'neutral'

    return {'zone': zone, 'range_pct': range_pct, 'swing_high': swing_high, 'swing_low': swing_low}


def detect_equal_highs_lows(df, length=8, threshold_atr_mult=0.15):
    """EQH / EQL liquidity pools — swing grade uses wider ATR threshold."""
    highs  = df['high'].values
    lows   = df['low'].values
    closes = df['close'].values
    n      = len(df)

    atr       = np.mean([abs(highs[i] - lows[i]) for i in range(max(0, n-14), n)])
    threshold = threshold_atr_mult * atr

    swing_highs, swing_lows = detect_swing_points(df, length)

    eqh = []
    for i in range(len(swing_highs) - 1):
        for j in range(i+1, len(swing_highs)):
            if abs(swing_highs[i][1] - swing_highs[j][1]) < threshold:
                eqh.append(round((swing_highs[i][1] + swing_highs[j][1]) / 2, 8))

    eql = []
    for i in range(len(swing_lows) - 1):
        for j in range(i+1, len(swing_lows)):
            if abs(swing_lows[i][1] - swing_lows[j][1]) < threshold:
                eql.append(round((swing_lows[i][1] + swing_lows[j][1]) / 2, 8))

    current     = closes[-1]
    nearest_eqh = min([h for h in eqh if h > current], default=None) if eqh else None
    nearest_eql = max([l for l in eql if l < current], default=None) if eql else None

    return {'eqh': sorted(set(eqh)), 'eql': sorted(set(eql)),
            'nearest_eqh': nearest_eqh, 'nearest_eql': nearest_eql}


# ═══════════════════════════════════════════════════════════════
#  MAIN SWING SCANNER
# ═══════════════════════════════════════════════════════════════

class SwingTradingScanner:
    def __init__(self, telegram_token, telegram_chat_id, binance_api_key=None, binance_secret=None):
        self.telegram_token = telegram_token
        self.telegram_bot   = Bot(token=telegram_token)
        self.chat_id        = telegram_chat_id
        self.exchange = ccxt.binance({
            'apiKey':          binance_api_key,
            'secret':          binance_secret,
            'enableRateLimit': True,
            'options':         {'defaultType': 'future'}
        })
        self.signal_history = deque(maxlen=200)
        self.active_trades  = {}
        self.stats = {
            'total_signals': 0, 'long_signals': 0, 'short_signals': 0,
            'premium_signals': 0, 'tp1_hits': 0, 'tp2_hits': 0, 'tp3_hits': 0,
            'ob_signals': 0, 'last_scan_time': None, 'pairs_scanned': 0,
            'daily_signals': 0, 'daily_wins': 0, 'daily_losses': 0,
            'daily_be': 0, 'report_start': datetime.now(),
        }
        self.is_scanning = False
        self.is_tracking = False

    # ─────────────────────────────────────────────────────────
    #  DATA FETCHING
    # ─────────────────────────────────────────────────────────

    async def get_all_usdt_pairs(self):
        try:
            await self.exchange.load_markets()
            tickers = await self.exchange.fetch_tickers()
            pairs = []
            for symbol in self.exchange.symbols:
                if symbol.endswith('/USDT:USDT') and 'PERP' not in symbol:
                    ticker = tickers.get(symbol)
                    # Swing: require higher minimum volume — less liquid pairs = worse OBs
                    if ticker and ticker.get('quoteVolume', 0) > 5_000_000:
                        pairs.append(symbol)
            sorted_pairs = sorted(pairs,
                                   key=lambda x: tickers.get(x, {}).get('quoteVolume', 0),
                                   reverse=True)
            logger.info(f"✅ Found {len(sorted_pairs)} swing-eligible pairs")
            return sorted_pairs
        except Exception as e:
            logger.error(f"Error fetching pairs: {e}")
            return []

    async def fetch_swing_data(self, symbol):
        """
        Swing timeframes:
          1D  — primary OB detection + structure (200 candles = ~200 days)
          4H  — secondary OB + entry zone confirmation (200 candles = ~33 days)
          1H  — entry confirmation only (100 candles)
        """
        timeframes = {'1d': 200, '4h': 200, '1h': 100}
        data = {}
        try:
            for tf, limit in timeframes.items():
                ohlcv = await self.exchange.fetch_ohlcv(symbol, tf, limit=limit)
                df    = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                data[tf] = df
                await asyncio.sleep(0.05)
            return data
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None

    # ─────────────────────────────────────────────────────────
    #  INDICATORS
    # ─────────────────────────────────────────────────────────

    def calculate_supertrend(self, df, period=14, multiplier=3):
        try:
            hl2 = (df['high'] + df['low']) / 2
            atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'],
                                                  window=period).average_true_range()
            upper = hl2 + multiplier * atr
            lower = hl2 - multiplier * atr
            st    = [0.0] * len(df)
            for i in range(1, len(df)):
                if df['close'].iloc[i] > upper.iloc[i-1]:
                    st[i] = lower.iloc[i]
                elif df['close'].iloc[i] < lower.iloc[i-1]:
                    st[i] = upper.iloc[i]
                else:
                    st[i] = st[i-1]
            return pd.Series(st, index=df.index)
        except:
            return pd.Series([0.0] * len(df), index=df.index)

    def calculate_indicators(self, df):
        try:
            if len(df) < 50:
                return df

            # Trend
            df['ema_21']  = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
            df['ema_50']  = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
            df['ema_200'] = ta.trend.EMAIndicator(df['close'], window=min(200, len(df)-1)).ema_indicator()
            df['supertrend'] = self.calculate_supertrend(df)

            # Momentum
            df['rsi']    = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            df['rsi_w']  = ta.momentum.RSIIndicator(df['close'], window=21).rsi()   # "weekly" feel

            macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
            df['macd']        = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_hist']   = macd.macd_diff()

            df['uo']        = ta.momentum.UltimateOscillator(df['high'], df['low'], df['close']).ultimate_oscillator()
            df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
            df['roc']       = ta.momentum.ROCIndicator(df['close'], window=10).roc()

            # Volatility
            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_upper']  = bb.bollinger_hband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_lower']  = bb.bollinger_lband()
            df['bb_pband']  = bb.bollinger_pband()
            df['bb_width']  = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'],
                                                        window=14).average_true_range()

            # Volume
            df['volume_sma']   = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['obv']          = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
            df['obv_ema']      = df['obv'].ewm(span=21).mean()
            df['mfi']          = ta.volume.MFIIndicator(df['high'], df['low'], df['close'],
                                                         df['volume'], window=14).money_flow_index()
            df['cmf']          = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'],
                                                                       df['volume']).chaikin_money_flow()

            # Trend strength
            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
            df['adx']      = adx.adx()
            df['di_plus']  = adx.adx_pos()
            df['di_minus'] = adx.adx_neg()

            df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()

            aroon = ta.trend.AroonIndicator(df['high'], df['low'], window=25)
            df['aroon_up']   = aroon.aroon_up()
            df['aroon_down'] = aroon.aroon_down()
            df['aroon_ind']  = df['aroon_up'] - df['aroon_down']

            # VWAP (cumulative — meaningful for swing on 4H)
            tp         = (df['high'] + df['low'] + df['close']) / 3
            df['vwap'] = (tp * df['volume']).cumsum() / df['volume'].cumsum()
            df['vwap'] = df['vwap'].fillna(df['close'])

            # Divergences
            df['bullish_divergence'] = (
                (df['low'] < df['low'].shift(2)) &
                (df['rsi'] > df['rsi'].shift(2))
            ).astype(int)
            df['bearish_divergence'] = (
                (df['high'] > df['high'].shift(2)) &
                (df['rsi'] < df['rsi'].shift(2))
            ).astype(int)

            # Swing candle patterns (larger body requirement for swing)
            df['swing_bull_engulf'] = (
                (df['close'].shift(1) < df['open'].shift(1)) &
                (df['close'] > df['open']) &
                (df['open'] < df['close'].shift(1)) &
                (df['close'] > df['open'].shift(1)) &
                (abs(df['close'] - df['open']) > abs(df['close'].shift(1) - df['open'].shift(1)) * 1.2)
            ).astype(int)
            df['swing_bear_engulf'] = (
                (df['close'].shift(1) > df['open'].shift(1)) &
                (df['close'] < df['open']) &
                (df['open'] > df['close'].shift(1)) &
                (df['close'] < df['open'].shift(1)) &
                (abs(df['close'] - df['open']) > abs(df['close'].shift(1) - df['open'].shift(1)) * 1.2)
            ).astype(int)

            return df
        except Exception as e:
            logger.error(f"Indicator error: {e}")
            return df

    # ─────────────────────────────────────────────────────────
    #  VOLUME DIRECTION  (longer lookback for swing)
    # ─────────────────────────────────────────────────────────

    def analyze_volume_direction(self, df, lookback=10):
        if len(df) < max(lookback, 20):
            return False, False, 1.0, 50.0

        closes  = df['close'].values
        opens   = df['open'].values
        volumes = df['volume'].values
        highs   = df['high'].values
        lows    = df['low'].values

        avg_vol = volumes[-20:].mean()
        if avg_vol == 0 or np.isnan(avg_vol):
            return False, False, 1.0, 50.0

        vol_ratio = volumes[-lookback:].mean() / avg_vol

        buy_vol  = 0.0
        sell_vol = 0.0
        for i in range(-lookback, 0):
            cr = highs[i] - lows[i]
            if cr == 0:
                continue
            buy_vol  += volumes[i] * (closes[i] - lows[i])  / cr
            sell_vol += volumes[i] * (highs[i]  - closes[i]) / cr

        total   = buy_vol + sell_vol
        buy_pct = (buy_vol / total * 100) if total > 0 else 50.0

        mid        = lookback // 2
        early      = volumes[-lookback:-mid].mean()
        late       = volumes[-mid:].mean()
        vol_rising = late > early * 1.1
        vol_fading = late < early * 0.85

        long_conviction  = 0
        short_conviction = 0
        for i in range(-lookback, 0):
            body     = abs(closes[i] - opens[i])
            rng      = highs[i] - lows[i] if highs[i] != lows[i] else 1
            body_pct = body / rng
            if volumes[i] > avg_vol * 1.2 and body_pct > 0.55:
                if closes[i] > opens[i]:
                    long_conviction += 1
                else:
                    short_conviction += 1

        long_ok  = buy_pct > 55 and vol_ratio > 0.8 and (vol_rising or long_conviction >= 2) and not (buy_pct < 45 and vol_fading)
        short_ok = buy_pct < 45 and vol_ratio > 0.8 and (vol_rising or short_conviction >= 2) and not (buy_pct > 55 and vol_fading)

        return long_ok, short_ok, vol_ratio, buy_pct

    # ─────────────────────────────────────────────────────────
    #  VOLUME PROFILE  (same LuxAlgo logic)
    # ─────────────────────────────────────────────────────────

    def calculate_volume_profile(self, df, n_rows=30):
        try:
            if len(df) < 20:
                return None

            highs   = df['high'].values
            lows    = df['low'].values
            closes  = df['close'].values
            opens   = df['open'].values
            volumes = df['volume'].values

            p_low  = lows.min()
            p_high = highs.max()
            if p_high <= p_low:
                return None

            step      = (p_high - p_low) / n_rows
            total_vol = np.zeros(n_rows)
            bull_vol  = np.zeros(n_rows)
            bear_vol  = np.zeros(n_rows)

            for i in range(len(df)):
                v       = volumes[i]
                is_bull = closes[i] > opens[i]
                for row in range(n_rows):
                    rl = p_low + row * step
                    rh = rl + step
                    if highs[i] < rl or lows[i] >= rh:
                        continue
                    br = highs[i] - lows[i]
                    if br == 0:
                        por = 1.0
                    elif lows[i] >= rl and highs[i] > rh:
                        por = (rh - lows[i]) / br
                    elif highs[i] <= rh and lows[i] < rl:
                        por = (highs[i] - rl) / br
                    elif lows[i] >= rl and highs[i] <= rh:
                        por = 1.0
                    else:
                        por = step / br
                    alloc = v * por
                    total_vol[row] += alloc
                    if is_bull:
                        bull_vol[row] += alloc
                    else:
                        bear_vol[row] += alloc

            max_vol = total_vol.max()
            if max_vol == 0:
                return None

            poc_row   = int(np.argmax(total_vol))
            poc_price = p_low + (poc_row + 0.5) * step

            htn = []
            ltn = []
            for row in range(n_rows):
                ratio = total_vol[row] / max_vol
                mid   = p_low + (row + 0.5) * step
                if ratio >= 0.53:
                    htn.append(mid)
                elif ratio <= 0.37:
                    ltn.append(mid)

            poc_sent = 'bullish' if bull_vol[poc_row] >= bear_vol[poc_row] else 'bearish'

            current     = closes[-1]
            cur_row     = max(0, min(int((current - p_low) / step), n_rows - 1))
            cur_ratio   = total_vol[cur_row] / max_vol
            current_node = 'high' if cur_ratio >= 0.53 else ('low' if cur_ratio <= 0.37 else 'average')
            current_sent = 'bullish' if bull_vol[cur_row] >= bear_vol[cur_row] else 'bearish'

            htn_above = [h for h in htn if h > current]
            htn_below = [h for h in htn if h < current]

            return {
                'poc_price': poc_price, 'poc_sentiment': poc_sent,
                'htn_levels': htn, 'ltn_levels': ltn,
                'current_node': current_node, 'current_sentiment': current_sent,
                'nearest_resistance': min(htn_above) if htn_above else None,
                'nearest_support':    max(htn_below) if htn_below else None,
            }
        except Exception as e:
            logger.error(f"VP error: {e}")
            return None

    # ─────────────────────────────────────────────────────────
    #  SIGNAL DETECTION
    # ─────────────────────────────────────────────────────────

    def detect_signal(self, data, symbol):
        """
        Swing signal scoring.

        Timeframe roles:
          1D → primary OB detection, BOS/CHoCH, premium/discount  (most weight)
          4H → secondary OB confirmation, entry zone refinement
          1H → final entry confirmation (trend, momentum checks)

        Max score : 65
        Signal    : long_score or short_score >= 50% of max
        Premium   : >= 75%
        High      : >= 62%
        Good      : >= 50%
        """
        try:
            if not data or '1d' not in data or '4h' not in data:
                return None

            for tf in data:
                data[tf] = self.calculate_indicators(data[tf])

            df_1d = data['1d']
            df_4h = data['4h']
            df_1h = data['1h']

            if len(df_1d) < 50 or len(df_4h) < 50:
                return None

            latest_1d  = df_1d.iloc[-1]
            prev_1d    = df_1d.iloc[-2]
            latest_4h  = df_4h.iloc[-1]
            prev_4h    = df_4h.iloc[-2]
            latest_1h  = df_1h.iloc[-1]

            required = ['ema_21', 'ema_50', 'rsi', 'macd', 'atr']
            for col in required:
                if col not in latest_1d.index or pd.isna(latest_1d[col]):
                    return None

            # ── Volume analysis (10-bar swing lookback) ──────────
            long_vol_ok, short_vol_ok, vol_ratio, buy_pct = self.analyze_volume_direction(df_4h, lookback=10)

            # ── Volume Profile on daily ───────────────────────────
            vp = self.calculate_volume_profile(df_1d, n_rows=30)

            # ── SMC on DAILY (primary) ────────────────────────────
            smc_struct_1d = detect_bos_choch(df_1d,  swing_length=8)
            smc_fvg_1d    = detect_fair_value_gaps(df_1d, min_gap_pct=0.003)
            smc_pd_1d     = detect_premium_discount(df_1d, swing_length=8)
            smc_eq_1d     = detect_equal_highs_lows(df_1d, length=8)

            # ── SMC on 4H (secondary) ─────────────────────────────
            smc_struct_4h = detect_bos_choch(df_4h, swing_length=6)
            smc_pd_4h     = detect_premium_discount(df_4h, swing_length=6)

            # ── Order Blocks ──────────────────────────────────────
            # Daily OBs — swing-grade: lookback 150, swing_strength 6
            obs_1d = detect_order_blocks(df_1d, lookback=150, swing_strength=6)
            # 4H OBs — lookback 150, swing_strength 5
            obs_4h = detect_order_blocks(df_4h, lookback=150, swing_strength=5)

            current_price = latest_4h['close']

            ob_type_1d, ob_1d = price_at_order_block(current_price, obs_1d, tolerance=0.006)
            ob_type_4h, ob_4h = price_at_order_block(current_price, obs_4h, tolerance=0.005)

            long_score    = 0
            short_score   = 0
            max_score     = 65
            long_reasons  = []
            short_reasons = []

            # ══ ORDER BLOCKS  (max 18 pts) ═══════════════════════

            ob_active = None   # used for SL placement

            # Daily OB — highest weight (institutional level)
            if ob_type_1d == 'bullish':
                # Bonus if strong impulse
                bonus = 2 if ob_1d['impulse_strength'] > 2 else 0
                long_score += 8 + bonus
                long_reasons.append(
                    f"🧱 Daily Bullish OB [{ob_1d['bottom']:.4f}–{ob_1d['top']:.4f}]"
                    + (" 💪 Strong" if bonus else ""))
                ob_active = ob_1d
            elif ob_type_1d == 'bearish':
                bonus = 2 if ob_1d['impulse_strength'] > 2 else 0
                short_score += 8 + bonus
                short_reasons.append(
                    f"🧱 Daily Bearish OB [{ob_1d['bottom']:.4f}–{ob_1d['top']:.4f}]"
                    + (" 💪 Strong" if bonus else ""))
                ob_active = ob_1d

            # 4H OB — secondary confirmation
            if ob_type_4h == 'bullish':
                long_score += 5
                long_reasons.append(f"🏗️ 4H Bullish OB [{ob_4h['bottom']:.4f}–{ob_4h['top']:.4f}]")
                if ob_active is None:
                    ob_active = ob_4h
            elif ob_type_4h == 'bearish':
                short_score += 5
                short_reasons.append(f"🏗️ 4H Bearish OB [{ob_4h['bottom']:.4f}–{ob_4h['top']:.4f}]")
                if ob_active is None:
                    ob_active = ob_4h

            # ══ DAILY STRUCTURE  (max 12 pts) ════════════════════

            if smc_struct_1d['recent_bull_structure']:
                long_score += 5
                tag = 'BOS' if smc_struct_1d['last_bos_bull'] else 'CHoCH'
                long_reasons.append(f"⚡ Daily Bullish {tag}")
            if smc_struct_1d['recent_bear_structure']:
                short_score += 5
                tag = 'BOS' if smc_struct_1d['last_bos_bear'] else 'CHoCH'
                short_reasons.append(f"⚡ Daily Bearish {tag}")

            if smc_struct_1d['swing_trend'] == 'bullish':
                long_score += 3
                long_reasons.append('📈 Daily Uptrend (HH+HL)')
            elif smc_struct_1d['swing_trend'] == 'bearish':
                short_score += 3
                short_reasons.append('📉 Daily Downtrend (LH+LL)')

            # 4H structure alignment bonus
            if smc_struct_4h['swing_trend'] == 'bullish':
                long_score += 2
                long_reasons.append('📈 4H Uptrend Aligned')
            elif smc_struct_4h['swing_trend'] == 'bearish':
                short_score += 2
                short_reasons.append('📉 4H Downtrend Aligned')

            # 4H recent structure break bonus
            if smc_struct_4h['recent_bull_structure']:
                long_score += 2
                long_reasons.append('⚡ 4H Bullish Structure')
            if smc_struct_4h['recent_bear_structure']:
                short_score += 2
                short_reasons.append('⚡ 4H Bearish Structure')

            # ══ PREMIUM / DISCOUNT  (max 5 pts, hard block) ══════

            zone_1d = smc_pd_1d['zone']

            if zone_1d == 'discount':
                long_score  += 5
                long_reasons.append(f"💚 Daily Discount Zone ({smc_pd_1d['range_pct']:.0f}%)")
            elif zone_1d == 'premium':
                short_score += 5
                short_reasons.append(f"🔴 Daily Premium Zone ({smc_pd_1d['range_pct']:.0f}%)")
            elif zone_1d == 'equilibrium':
                long_score  += 1
                short_score += 1

            # HARD BLOCK: no longs in premium, no shorts in discount
            if zone_1d == 'premium' and long_score > short_score:
                logger.info(f"⛔ {symbol} LONG blocked — daily PREMIUM zone ({smc_pd_1d['range_pct']:.0f}%)")
                return None
            if zone_1d == 'discount' and short_score > long_score:
                logger.info(f"⛔ {symbol} SHORT blocked — daily DISCOUNT zone ({smc_pd_1d['range_pct']:.0f}%)")
                return None

            # ══ FAIR VALUE GAPS  (max 4 pts) ══════════════════════

            cp = current_price
            if smc_fvg_1d['nearest_bull'] and abs(cp - smc_fvg_1d['nearest_bull']['mid']) / cp < 0.012:
                long_score += 4
                f = smc_fvg_1d['nearest_bull']
                long_reasons.append(f"🟩 Daily Bullish FVG ({f['bottom']:.4f}–{f['top']:.4f})")
            if smc_fvg_1d['nearest_bear'] and abs(cp - smc_fvg_1d['nearest_bear']['mid']) / cp < 0.012:
                short_score += 4
                f = smc_fvg_1d['nearest_bear']
                short_reasons.append(f"🟥 Daily Bearish FVG ({f['bottom']:.4f}–{f['top']:.4f})")

            # ══ LIQUIDITY  (EQH / EQL, max 2 pts) ════════════════

            if smc_eq_1d['nearest_eqh'] and cp < smc_eq_1d['nearest_eqh']:
                if (smc_eq_1d['nearest_eqh'] - cp) / cp * 100 < 3.0:
                    long_score += 2
                    long_reasons.append(f"💧 EQH Liquidity Target ({smc_eq_1d['nearest_eqh']:.4f})")
            if smc_eq_1d['nearest_eql'] and cp > smc_eq_1d['nearest_eql']:
                if (cp - smc_eq_1d['nearest_eql']) / cp * 100 < 3.0:
                    short_score += 2
                    short_reasons.append(f"💧 EQL Liquidity Target ({smc_eq_1d['nearest_eql']:.4f})")

            # ══ VOLUME PROFILE  (max 6 pts) ═══════════════════════

            if vp:
                if vp['current_node'] == 'low':
                    if vp['current_sentiment'] == 'bullish':
                        long_score += 2
                        long_reasons.append('🔵 VP Low Node (bullish) — fast move likely')
                    else:
                        short_score += 2
                        short_reasons.append('🔵 VP Low Node (bearish) — fast move likely')

                if vp['poc_sentiment'] == 'bullish':
                    long_score += 2
                    long_reasons.append(f"📍 VP POC Bullish ({vp['poc_price']:.4f})")
                else:
                    short_score += 2
                    short_reasons.append(f"📍 VP POC Bearish ({vp['poc_price']:.4f})")

                if vp['nearest_support'] and abs(cp - vp['nearest_support']) / cp < 0.02:
                    long_score += 1.5
                    long_reasons.append(f"🟨 VP Support HTN ({vp['nearest_support']:.4f})")
                if vp['nearest_resistance'] and abs(cp - vp['nearest_resistance']) / cp < 0.02:
                    short_score += 1.5
                    short_reasons.append(f"🟨 VP Resistance HTN ({vp['nearest_resistance']:.4f})")

                if cp < vp['poc_price'] * 0.993:
                    long_score += 0.5
                    long_reasons.append('🧲 VP POC Magnet Above')
                elif cp > vp['poc_price'] * 1.007:
                    short_score += 0.5
                    short_reasons.append('🧲 VP POC Magnet Below')

            # ══ TREND — EMA STACK  (max 7 pts) ════════════════════

            # Daily EMA alignment
            if latest_1d['ema_21'] > latest_1d['ema_50'] > latest_1d['ema_200']:
                long_score += 4
                long_reasons.append('🔥 Daily EMA Bullish Stack (21>50>200)')
            elif latest_1d['ema_21'] < latest_1d['ema_50'] < latest_1d['ema_200']:
                short_score += 4
                short_reasons.append('🔥 Daily EMA Bearish Stack (21<50<200)')
            elif latest_1d['ema_21'] > latest_1d['ema_50']:
                long_score += 2
                long_reasons.append('1D EMA 21 > 50 Bullish')
            elif latest_1d['ema_21'] < latest_1d['ema_50']:
                short_score += 2
                short_reasons.append('1D EMA 21 < 50 Bearish')

            # Daily supertrend
            if latest_1d['close'] > latest_1d['supertrend']:
                long_score += 2
                long_reasons.append('Daily SuperTrend Bullish')
            elif latest_1d['close'] < latest_1d['supertrend']:
                short_score += 2
                short_reasons.append('Daily SuperTrend Bearish')

            # 4H EMA alignment (entry confirmation)
            if latest_4h['ema_21'] > latest_4h['ema_50']:
                long_score += 1
            elif latest_4h['ema_21'] < latest_4h['ema_50']:
                short_score += 1

            # ══ MOMENTUM  (max 10 pts) ═════════════════════════════

            # Daily RSI
            if latest_1d['rsi'] < 35:
                long_score += 4
                long_reasons.append(f'💎 Daily RSI Oversold ({latest_1d["rsi"]:.0f})')
            elif latest_1d['rsi'] < 45:
                long_score += 2
                long_reasons.append(f'Daily RSI Low ({latest_1d["rsi"]:.0f})')
            elif 45 <= latest_1d['rsi'] <= 55:
                long_score += 1

            if latest_1d['rsi'] > 65:
                short_score += 4
                short_reasons.append(f'💎 Daily RSI Overbought ({latest_1d["rsi"]:.0f})')
            elif latest_1d['rsi'] > 55:
                short_score += 2
                short_reasons.append(f'Daily RSI High ({latest_1d["rsi"]:.0f})')
            elif 45 <= latest_1d['rsi'] <= 55:
                short_score += 1

            # Daily MACD cross
            if latest_1d['macd'] > latest_1d['macd_signal'] and prev_1d['macd'] <= prev_1d['macd_signal']:
                long_score += 3
                long_reasons.append('🎯 Daily MACD Cross Up')
            elif latest_1d['macd'] < latest_1d['macd_signal'] and prev_1d['macd'] >= prev_1d['macd_signal']:
                short_score += 3
                short_reasons.append('🎯 Daily MACD Cross Down')

            # MACD histogram trending
            hist_trend = df_1d['macd_hist'].iloc[-5:].diff().mean()
            if hist_trend > 0 and latest_1d['macd_hist'] > 0:
                long_score += 1
                long_reasons.append('MACD Hist Rising')
            elif hist_trend < 0 and latest_1d['macd_hist'] < 0:
                short_score += 1
                short_reasons.append('MACD Hist Falling')

            # 4H MACD cross
            if latest_4h['macd'] > latest_4h['macd_signal'] and prev_4h['macd'] <= prev_4h['macd_signal']:
                long_score += 2
                long_reasons.append('🎯 4H MACD Cross Up')
            elif latest_4h['macd'] < latest_4h['macd_signal'] and prev_4h['macd'] >= prev_4h['macd_signal']:
                short_score += 2
                short_reasons.append('🎯 4H MACD Cross Down')

            # Ultimate Oscillator
            if latest_1d['uo'] < 30:
                long_score += 1.5
                long_reasons.append('UO Oversold')
            elif latest_1d['uo'] > 70:
                short_score += 1.5
                short_reasons.append('UO Overbought')

            # ══ VOLUME  (max 5 pts) ════════════════════════════════

            if long_vol_ok:
                long_score += 3
                long_reasons.append(f'📈 Buy Vol Confirmed ({buy_pct:.0f}% buying, {vol_ratio:.1f}x avg)')
            if short_vol_ok:
                short_score += 3
                short_reasons.append(f'📉 Sell Vol Confirmed ({100-buy_pct:.0f}% selling, {vol_ratio:.1f}x avg)')

            if latest_1d['mfi'] < 25:
                long_score += 1.5
                long_reasons.append(f'MFI Oversold ({latest_1d["mfi"]:.0f})')
            elif latest_1d['mfi'] > 75:
                short_score += 1.5
                short_reasons.append(f'MFI Overbought ({latest_1d["mfi"]:.0f})')

            obv_trend = df_1d['obv'].iloc[-5:].diff().mean()
            if obv_trend > 0 and latest_1d['obv'] > latest_1d['obv_ema']:
                long_score += 1
                long_reasons.append('OBV Accumulation')
            elif obv_trend < 0 and latest_1d['obv'] < latest_1d['obv_ema']:
                short_score += 1
                short_reasons.append('OBV Distribution')

            if latest_1d['cmf'] > 0.15:
                long_score += 0.5
                long_reasons.append('CMF Buying')
            elif latest_1d['cmf'] < -0.15:
                short_score += 0.5
                short_reasons.append('CMF Selling')

            # ══ VOLATILITY  (max 5 pts) ════════════════════════════

            if latest_1d['bb_pband'] < 0.1:
                long_score += 2.5
                long_reasons.append('💎 Daily Lower BB Touch')
            elif latest_1d['bb_pband'] > 0.9:
                short_score += 2.5
                short_reasons.append('💎 Daily Upper BB Touch')

            if latest_1d['cci'] < -150:
                long_score += 1.5
                long_reasons.append('CCI Oversold')
            elif latest_1d['cci'] > 150:
                short_score += 1.5
                short_reasons.append('CCI Overbought')

            if latest_1d['williams_r'] < -85:
                long_score += 1
                long_reasons.append('Williams Oversold')
            elif latest_1d['williams_r'] > -15:
                short_score += 1
                short_reasons.append('Williams Overbought')

            # ══ TREND STRENGTH  (max 4 pts) ════════════════════════

            if latest_1d['adx'] > 25:
                if latest_1d['di_plus'] > latest_1d['di_minus']:
                    long_score += 2
                    long_reasons.append(f'ADX Strong Up ({latest_1d["adx"]:.0f})')
                else:
                    short_score += 2
                    short_reasons.append(f'ADX Strong Down ({latest_1d["adx"]:.0f})')

            if latest_1d['aroon_ind'] > 50:
                long_score += 1
                long_reasons.append('Aroon Up')
            elif latest_1d['aroon_ind'] < -50:
                short_score += 1
                short_reasons.append('Aroon Down')

            if latest_1d['roc'] > 5:
                long_score += 1
                long_reasons.append('Strong Positive ROC')
            elif latest_1d['roc'] < -5:
                short_score += 1
                short_reasons.append('Strong Negative ROC')

            # ══ DIVERGENCE & CANDLE PATTERNS  (max 4 pts) ═════════

            if latest_1d['bullish_divergence'] == 1:
                long_score += 2.5
                long_reasons.append('🎯 Daily Bullish Divergence')
            elif latest_1d['bearish_divergence'] == 1:
                short_score += 2.5
                short_reasons.append('🎯 Daily Bearish Divergence')

            # Swing engulfing on 4H (entry-level pattern)
            if latest_4h['swing_bull_engulf'] == 1:
                long_score += 1.5
                long_reasons.append('📊 4H Bullish Engulfing')
            elif latest_4h['swing_bear_engulf'] == 1:
                short_score += 1.5
                short_reasons.append('📊 4H Bearish Engulfing')

            # ══ HTF CONFIRMATION  (2 pts) ══════════════════════════

            if latest_1d['close'] > latest_1d['vwap']:
                long_score += 1
            else:
                short_score += 1

            if latest_4h['rsi'] < 50:
                long_score += 1
            else:
                short_score += 1

            # ══ DETERMINE SIGNAL ═══════════════════════════════════

            # Swing: need at least 50% of max score, raised bar vs day trade
            min_threshold = max_score * 0.50
            signal    = None
            quality   = None
            reasons   = []

            if long_score > short_score and long_score >= min_threshold:
                if not long_vol_ok:
                    logger.info(f"⛔ {symbol} LONG blocked — no buy volume ({buy_pct:.0f}% buy)")
                    return None
                signal  = 'LONG'
                score   = long_score
                reasons = long_reasons
                if long_score >= max_score * 0.75:
                    quality = 'PREMIUM 💎'
                elif long_score >= max_score * 0.62:
                    quality = 'HIGH 🔥'
                else:
                    quality = 'GOOD ✅'

            elif short_score > long_score and short_score >= min_threshold:
                if not short_vol_ok:
                    logger.info(f"⛔ {symbol} SHORT blocked — no sell volume ({buy_pct:.0f}% buy)")
                    return None
                signal  = 'SHORT'
                score   = short_score
                reasons = short_reasons
                if short_score >= max_score * 0.75:
                    quality = 'PREMIUM 💎'
                elif short_score >= max_score * 0.62:
                    quality = 'HIGH 🔥'
                else:
                    quality = 'GOOD ✅'

            if not signal:
                return None

            # ── Swing TP / SL  (wider ATR multiples) ───────────────

            entry = latest_4h['close']
            atr   = latest_1d['atr']    # use DAILY ATR for swing sizing

            if ob_active:
                if signal == 'LONG':
                    ob_sl = ob_active['bottom'] * 0.997
                    sl    = min(entry - atr * 2.5, ob_sl)
                else:
                    ob_sl = ob_active['top'] * 1.003
                    sl    = max(entry + atr * 2.5, ob_sl)
            else:
                if signal == 'LONG':
                    sl = entry - atr * 2.5
                else:
                    sl = entry + atr * 2.5

            if signal == 'LONG':
                tp1 = entry + atr * 1.5
                tp2 = entry + atr * 3.5
                tp3 = entry + atr * 7.0
                if vp and vp['poc_price'] > entry * 1.005:
                    dist = vp['poc_price'] - entry
                    if atr * 1.5 < dist < atr * 7:
                        tp2 = vp['poc_price']
            else:
                tp1 = entry - atr * 1.5
                tp2 = entry - atr * 3.5
                tp3 = entry - atr * 7.0
                if vp and vp['poc_price'] < entry * 0.995:
                    dist = entry - vp['poc_price']
                    if atr * 1.5 < dist < atr * 7:
                        tp2 = vp['poc_price']

            targets  = [tp1, tp2, tp3]
            rr       = [abs(tp - entry) / abs(sl - entry) for tp in targets]
            risk_pct = abs((sl - entry) / entry * 100)

            trade_id = f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

            if ob_active:
                self.stats['ob_signals'] += 1

            return {
                'trade_id':    trade_id,
                'symbol':      symbol.replace('/USDT:USDT', ''),
                'full_symbol': symbol,
                'signal':      signal,
                'quality':     quality,
                'score':       score,
                'max_score':   max_score,
                'score_percent': (score / max_score) * 100,
                'entry':       entry,
                'stop_loss':   sl,
                'targets':     targets,
                'reward_ratios': rr,
                'risk_percent':  risk_pct,
                'reasons':     reasons[:14],
                'ob_zone':     ob_active,
                'ob_type':     ob_type_1d if ob_active == ob_1d else ob_type_4h,
                'ob_tf':       '1D' if ob_active == ob_1d else '4H',
                'buy_pct':     buy_pct,
                'vol_ratio':   vol_ratio,
                'vp_poc':      vp['poc_price']   if vp else None,
                'vp_node':     vp['current_node'] if vp else None,
                'vp_support':  vp['nearest_support']    if vp else None,
                'vp_resistance': vp['nearest_resistance'] if vp else None,
                'smc_trend':   smc_struct_1d['swing_trend'],
                'smc_zone':    zone_1d,
                'smc_zone_pct': smc_pd_1d['range_pct'],
                'smc_fvg_bull': smc_fvg_1d['nearest_bull'],
                'smc_fvg_bear': smc_fvg_1d['nearest_bear'],
                'smc_eqh':     smc_eq_1d['nearest_eqh'],
                'smc_eql':     smc_eq_1d['nearest_eql'],
                'smc_bos_choch': (
                    'Bullish BOS'   if smc_struct_1d['last_bos_bull']   else
                    'Bullish CHoCH' if smc_struct_1d['last_choch_bull'] else
                    'Bearish BOS'   if smc_struct_1d['last_bos_bear']   else
                    'Bearish CHoCH' if smc_struct_1d['last_choch_bear'] else None
                ),
                'daily_atr':   atr,
                'tp_hit':      [False, False, False],
                'sl_hit':      False,
                'timestamp':   datetime.now(),
                'status':      'ACTIVE',
            }

        except Exception as e:
            logger.error(f"Signal detection error {symbol}: {e}")
            return None

    # ─────────────────────────────────────────────────────────
    #  MESSAGE FORMATTING
    # ─────────────────────────────────────────────────────────

    def format_signal(self, sig):
        is_long   = sig['signal'] == 'LONG'
        dir_emoji = "🟢" if is_long else "🔴"
        dir_label = "LONG  📈" if is_long else "SHORT 📉"

        quality_line = {
            'PREMIUM 💎': "💎 PREMIUM SWING SIGNAL",
            'HIGH 🔥':    "🔥 HIGH QUALITY SWING SIGNAL",
            'GOOD ✅':    "✅ SWING SIGNAL",
        }.get(sig['quality'], "✅ SWING SIGNAL")

        ob_tf     = sig.get('ob_tf', '?')
        ob_badge  = f"  •  🧱 {ob_tf} OB" if sig.get('ob_zone')       else ""
        vp_badge  = "  •  📊 VP"          if sig.get('vp_poc')         else ""
        smc_badge = "  •  🧠 SMC"         if sig.get('smc_bos_choch')  else ""

        def fmt(p):
            if p is None:  return "—"
            if p >= 1000:  return f"{p:.1f}"
            if p >= 100:   return f"{p:.2f}"
            if p >= 1:     return f"{p:.3f}"
            if p >= 0.01:  return f"{p:.4f}"
            return f"{p:.6f}"

        entry    = sig['entry']
        sl       = sig['stop_loss']
        tp1, tp2, tp3 = sig['targets']
        rr1, rr2, rr3 = sig['reward_ratios']
        pct = lambda p: abs((p - entry) / entry * 100)

        msg  = f"<b>{quality_line}{ob_badge}{vp_badge}{smc_badge}</b>\n"
        msg += f"━━━━━━━━━━━━━━━━━━━━━━\n"
        msg += f"  {dir_emoji} <b>#{sig['symbol']}USDT  •  {dir_label}</b>\n"
        msg += f"  🕐 <i>Swing Trade  •  Days–Weeks horizon</i>\n"
        msg += f"━━━━━━━━━━━━━━━━━━━━━━\n\n"

        # Order Block zone
        if sig.get('ob_zone'):
            ob = sig['ob_zone']
            imp = ob.get('impulse_strength', 0)
            strength_tag = " 💪 Strong" if imp > 2 else ""
            msg += f"🧱 <b>{ob_tf} OB Zone:</b>  {fmt(ob['bottom'])} – {fmt(ob['top'])}{strength_tag}\n"

        # Volume Profile
        if sig.get('vp_poc'):
            node_label = {
                'high':    '🟡 Consolidation Zone',
                'low':     '⚡ Thin Air — fast move',
                'average': '⚪ Average Activity',
            }.get(sig.get('vp_node'), '')
            msg += f"📍 <b>VP POC:</b>  {fmt(sig['vp_poc'])}  <i>({node_label})</i>\n"
            if sig.get('vp_support') and is_long:
                msg += f"🟩 <b>VP Support:</b>  {fmt(sig['vp_support'])}\n"
            if sig.get('vp_resistance') and not is_long:
                msg += f"🟥 <b>VP Resist:</b>   {fmt(sig['vp_resistance'])}\n"

        # SMC line
        smc_parts = []
        if sig.get('smc_bos_choch'):
            smc_parts.append(f"⚡ {sig['smc_bos_choch']}")
        zone    = sig.get('smc_zone', 'neutral')
        z_pct   = sig.get('smc_zone_pct', 50)
        if zone == 'discount':
            smc_parts.append(f"💚 Discount ({z_pct:.0f}%)")
        elif zone == 'premium':
            smc_parts.append(f"🔴 Premium ({z_pct:.0f}%)")
        elif zone == 'equilibrium':
            smc_parts.append(f"⚖️ Equilibrium ({z_pct:.0f}%)")
        if is_long and sig.get('smc_fvg_bull'):
            f = sig['smc_fvg_bull']
            smc_parts.append(f"🟩 FVG {fmt(f['bottom'])}–{fmt(f['top'])}")
        if not is_long and sig.get('smc_fvg_bear'):
            f = sig['smc_fvg_bear']
            smc_parts.append(f"🟥 FVG {fmt(f['bottom'])}–{fmt(f['top'])}")
        if is_long and sig.get('smc_eqh'):
            smc_parts.append(f"💧 EQH {fmt(sig['smc_eqh'])}")
        if not is_long and sig.get('smc_eql'):
            smc_parts.append(f"💧 EQL {fmt(sig['smc_eql'])}")
        if smc_parts:
            msg += f"🧠 <b>SMC:</b>  {' · '.join(smc_parts)}\n"
        msg += "\n"

        # Entry + volume bar
        msg += f"💰 <b>Entry</b>       {fmt(entry)}\n"
        buy_pct   = sig.get('buy_pct', 50)
        vol_ratio = sig.get('vol_ratio', 1.0)
        filled    = int(buy_pct / 10) if is_long else int((100 - buy_pct) / 10)
        vol_bar   = "🟦" * filled + "⬜" * (10 - filled)
        vol_lbl   = f"{buy_pct:.0f}% buy pressure" if is_long else f"{100-buy_pct:.0f}% sell pressure"
        msg += f"📊 <b>Volume</b>      {vol_bar}  <i>{vol_lbl}  ({vol_ratio:.1f}x avg)</i>\n\n"

        # Targets
        msg += f"🎯 <b>TP 1</b>  →  <code>{fmt(tp1)}</code>  <i>(+{pct(tp1):.1f}%  •  RR {rr1:.1f}x)</i>\n"
        msg += f"🎯 <b>TP 2</b>  →  <code>{fmt(tp2)}</code>  <i>(+{pct(tp2):.1f}%  •  RR {rr2:.1f}x)</i>\n"
        msg += f"🎯 <b>TP 3</b>  →  <code>{fmt(tp3)}</code>  <i>(+{pct(tp3):.1f}%  •  RR {rr3:.1f}x)</i>\n\n"

        msg += f"🛑 <b>Stop Loss</b>  <code>{fmt(sl)}</code>  <i>(-{sig['risk_percent']:.1f}%)</i>\n"
        msg += f"   <i>Daily ATR: {fmt(sig['daily_atr'])}  |  Risk beyond OB zone</i>\n\n"

        # Score bar
        msg += f"━━━━━━━━━━━━━━━━━━━━━━\n"
        msg += f"⭐ Score  {sig['score']:.0f}/{sig['max_score']}  "
        msg += f"{'▰' * int(sig['score_percent']/10)}{'▱' * (10 - int(sig['score_percent']/10))}\n"
        msg += f"🔍 <i>{' · '.join(r.lstrip('🔥💎🎯⚡🚀💥📊🧱 ') for r in sig['reasons'][:5])}</i>\n"
        msg += f"━━━━━━━━━━━━━━━━━━━━━━\n"
        msg += f"<i>⏰ {sig['timestamp'].strftime('%d %b  %H:%M')}  •  📡 Live tracking on</i>"

        return msg

    # ─────────────────────────────────────────────────────────
    #  TELEGRAM
    # ─────────────────────────────────────────────────────────

    async def send_msg(self, msg):
        try:
            await self.telegram_bot.send_message(
                chat_id=self.chat_id, text=msg, parse_mode=ParseMode.HTML)
        except Exception as e:
            logger.error(f"Send error: {e}")

    async def send_tp_alert(self, trade, tp_num, price):
        emoji = "🎉" if trade['signal'] == 'LONG' else "💰"
        tp    = trade['targets'][tp_num - 1]
        pct   = abs((tp - trade['entry']) / trade['entry'] * 100)

        msg  = f"{emoji} <b>TARGET HIT!</b> {emoji}\n\n"
        msg += f"<code>{trade['trade_id']}</code>\n"
        msg += f"<b>{trade['symbol']}</b> {trade['signal']}"
        if trade.get('ob_zone'):
            msg += f" 🧱 {trade.get('ob_tf','?')} OB Setup"
        msg += f"\n\n<b>✅ TP{tp_num} HIT!</b>\n"
        msg += f"Target: ${tp:.6f}\n"
        msg += f"Current: ${price:.6f}\n"
        msg += f"Profit: +{pct:.2f}%\n\n"

        if tp_num == 1:
            msg += "📋 Take 40% profit\nMove SL to breakeven"
        elif tp_num == 2:
            msg += "📋 Take 40% profit\nTrail remaining stop"
        else:
            msg += "📋 Close remaining position\n🎊 SWING COMPLETE!"

        await self.send_msg(msg)

        if tp_num == 1:
            self.stats['tp1_hits'] += 1
            self.stats['daily_wins'] += 1
        elif tp_num == 2:
            self.stats['tp2_hits'] += 1
        else:
            self.stats['tp3_hits'] += 1

    async def send_sl_alert(self, trade, price):
        loss = abs((price - trade['entry']) / trade['entry'] * 100)
        msg  = f"⚠️ <b>STOP LOSS HIT!</b> ⚠️\n\n"
        msg += f"<code>{trade['trade_id']}</code>\n"
        msg += f"{trade['symbol']} {trade['signal']}\n\n"
        msg += f"Entry: ${trade['entry']:.6f}\n"
        msg += f"SL: ${trade['stop_loss']:.6f}\n"
        msg += f"Current: ${price:.6f}\n"
        msg += f"Loss: -{loss:.2f}%"
        await self.send_msg(msg)
        self.stats['daily_losses'] += 1

    # ─────────────────────────────────────────────────────────
    #  TRADE TRACKING  (checks every 5 min; SL skipped if TP1 hit)
    # ─────────────────────────────────────────────────────────

    async def track_trades(self):
        self.is_tracking = True
        logger.info("📡 Swing tracking started")

        while True:
            try:
                if not self.active_trades:
                    await asyncio.sleep(300)    # 5 min idle sleep
                    continue

                to_remove = []

                for tid, trade in list(self.active_trades.items()):
                    try:
                        # Swing timeout: 30 days
                        if datetime.now() - trade['timestamp'] > timedelta(days=30):
                            msg = f"⏰ 30-DAY SWING LIMIT\n<code>{tid}</code>\n{trade['symbol']}\nClose position!"
                            await self.send_msg(msg)
                            if any(trade['tp_hit']):
                                self.stats['daily_wins'] += 1
                            else:
                                self.stats['daily_be'] += 1
                            to_remove.append(tid)
                            continue

                        ticker = await self.exchange.fetch_ticker(trade['full_symbol'])
                        price  = ticker['last']

                        if trade['signal'] == 'LONG':
                            if not trade['tp_hit'][0] and price >= trade['targets'][0]:
                                await self.send_tp_alert(trade, 1, price)
                                trade['tp_hit'][0] = True
                            if not trade['tp_hit'][1] and price >= trade['targets'][1]:
                                await self.send_tp_alert(trade, 2, price)
                                trade['tp_hit'][1] = True
                            if not trade['tp_hit'][2] and price >= trade['targets'][2]:
                                await self.send_tp_alert(trade, 3, price)
                                trade['tp_hit'][2] = True
                                to_remove.append(tid)
                            # ✅ SL only fires if TP1 was never hit
                            if not trade['sl_hit'] and not trade['tp_hit'][0] and price <= trade['stop_loss']:
                                await self.send_sl_alert(trade, price)
                                trade['sl_hit'] = True
                                to_remove.append(tid)

                        else:  # SHORT
                            if not trade['tp_hit'][0] and price <= trade['targets'][0]:
                                await self.send_tp_alert(trade, 1, price)
                                trade['tp_hit'][0] = True
                            if not trade['tp_hit'][1] and price <= trade['targets'][1]:
                                await self.send_tp_alert(trade, 2, price)
                                trade['tp_hit'][1] = True
                            if not trade['tp_hit'][2] and price <= trade['targets'][2]:
                                await self.send_tp_alert(trade, 3, price)
                                trade['tp_hit'][2] = True
                                to_remove.append(tid)
                            # ✅ SL only fires if TP1 was never hit
                            if not trade['sl_hit'] and not trade['tp_hit'][0] and price >= trade['stop_loss']:
                                await self.send_sl_alert(trade, price)
                                trade['sl_hit'] = True
                                to_remove.append(tid)

                    except Exception as e:
                        logger.error(f"Track error {tid}: {e}")
                        continue

                for tid in to_remove:
                    self.active_trades.pop(tid, None)
                    logger.info(f"✅ Swing trade closed: {tid}")

                await asyncio.sleep(300)    # check every 5 minutes (swing doesn't need 30s)

            except Exception as e:
                logger.error(f"Tracking error: {e}")
                await asyncio.sleep(300)

    # ─────────────────────────────────────────────────────────
    #  SCANNER
    # ─────────────────────────────────────────────────────────

    async def scan_all(self):
        if self.is_scanning:
            logger.info("⚠️ Already scanning...")
            return []

        self.is_scanning = True
        logger.info("🔍 Starting swing scan...")

        pairs   = await self.get_all_usdt_pairs()
        signals = []
        scanned = 0

        for pair in pairs:
            try:
                logger.info(f"📊 Scanning {pair}...")
                data = await self.fetch_swing_data(pair)

                if data:
                    sig = self.detect_signal(data, pair)
                    if sig:
                        signals.append(sig)
                        self.signal_history.append(sig)
                        self.stats['total_signals']  += 1
                        self.stats['daily_signals']  += 1
                        self.stats['long_signals']   += sig['signal'] == 'LONG'
                        self.stats['short_signals']  += sig['signal'] == 'SHORT'
                        self.stats['premium_signals'] += sig['quality'] == 'PREMIUM 💎'
                        if sig.get('ob_zone'):
                            self.stats['ob_signals'] += 1

                        self.active_trades[sig['trade_id']] = sig
                        await self.send_msg(self.format_signal(sig))
                        await asyncio.sleep(2)

                scanned += 1
                if scanned % 20 == 0:
                    logger.info(f"📈 {scanned}/{len(pairs)} scanned")

                await asyncio.sleep(0.6)

            except Exception as e:
                logger.error(f"❌ {pair}: {e}")
                continue

        self.stats['last_scan_time'] = datetime.now()
        self.stats['pairs_scanned']  = scanned

        ob_count = sum(1 for s in signals if s.get('ob_zone'))
        daily_ob = sum(1 for s in signals if s.get('ob_tf') == '1D')

        summary  = f"✅ <b>SWING SCAN COMPLETE</b>\n\n"
        summary += f"📊 Pairs scanned:  {scanned}\n"
        summary += f"🎯 Signals found:  {len(signals)}\n"

        if signals:
            longs   = sum(1 for s in signals if s['signal'] == 'LONG')
            shorts  = len(signals) - longs
            premium = sum(1 for s in signals if s['quality'] == 'PREMIUM 💎')
            summary += f"  🟢 Long:    {longs}\n"
            summary += f"  🔴 Short:   {shorts}\n"
            summary += f"  💎 Premium: {premium}\n"
            summary += f"  🧱 OB setups: {ob_count}  ({daily_ob} on daily)\n"

        summary += f"  📡 Tracking: {len(self.active_trades)}\n"
        summary += f"\n⏰ {datetime.now().strftime('%d %b  %H:%M')}"
        summary += f"\n🔄 Next scan in 4 hours"

        await self.send_msg(summary)
        logger.info(f"🎉 Swing scan done — {len(signals)} signals, {ob_count} OB setups")

        self.is_scanning = False
        return signals

    # ─────────────────────────────────────────────────────────
    #  DAILY REPORT
    # ─────────────────────────────────────────────────────────

    async def _daily_report_loop(self):
        while True:
            await asyncio.sleep(24 * 60 * 60)
            try:
                await self.send_daily_report()
            except Exception as e:
                logger.error(f"Report error: {e}")

    async def send_daily_report(self):
        s        = self.stats
        total    = s['daily_signals']
        wins     = s['daily_wins']
        losses   = s['daily_losses']
        be       = s['daily_be']
        closed   = wins + losses + be
        winrate  = (wins / closed * 100) if closed > 0 else 0

        filled = int(winrate / 10)
        bar    = "🟩" * filled + "⬜" * (10 - filled)

        if winrate >= 70:   perf = "🔥 ON FIRE"
        elif winrate >= 55: perf = "💪 SOLID"
        elif winrate >= 40: perf = "😐 AVERAGE"
        else:               perf = "⚠️ ROUGH PERIOD"

        msg  = f"━━━━━━━━━━━━━━━━━━━━━━\n"
        msg += f"📊 <b>SWING 24H REPORT</b>\n"
        msg += f"━━━━━━━━━━━━━━━━━━━━━━\n\n"
        msg += f"🗓 <i>{s['report_start'].strftime('%d %b')} → {datetime.now().strftime('%d %b  %H:%M')}</i>\n\n"
        msg += f"📡 Signals:    {total}\n"
        msg += f"📊 Win Rate:   {winrate:.1f}%\n"
        msg += f"{bar}\n\n"
        msg += f"🟢 Wins:       {wins}\n"
        msg += f"🚫 Losses:     {losses}\n"
        if be:
            msg += f"⚪ Breakeven: {be}\n"
        msg += f"\n<b>TP Breakdown:</b>\n"
        msg += f"  TP1: {s['tp1_hits']} 🎯\n"
        msg += f"  TP2: {s['tp2_hits']} 🎯\n"
        msg += f"  TP3: {s['tp3_hits']} 🎯\n\n"
        msg += f"<b>{perf}</b>\n"
        msg += f"━━━━━━━━━━━━━━━━━━━━━━"

        await self.send_msg(msg)

        self.stats.update({
            'daily_signals': 0, 'daily_wins': 0, 'daily_losses': 0,
            'daily_be': 0, 'tp1_hits': 0, 'tp2_hits': 0, 'tp3_hits': 0,
            'report_start': datetime.now(),
        })

    # ─────────────────────────────────────────────────────────
    #  MAIN LOOP  (4-hour interval for swing)
    # ─────────────────────────────────────────────────────────

    async def run(self, interval=240):   # 240 min = 4 hours
        logger.info("🚀 SWING TRADING SCANNER STARTING")

        welcome  = "📈 <b>SWING TRADING SCANNER</b> 📈\n\n"
        welcome += "✅ ALL USDT pairs (>$5M daily volume)\n"
        welcome += "✅ 🧱 Daily + 4H Order Block detection\n"
        welcome += "✅ 🧠 Daily BOS / CHoCH / FVG / Zones\n"
        welcome += "✅ 📊 Volume Profile (30-row daily)\n"
        welcome += "✅ Daily EMA 21/50/200 stack\n"
        welcome += "✅ Wide TP/SL — Daily ATR sizing\n"
        welcome += "✅ SL silenced after TP1 hit\n"
        welcome += "✅ 65-point swing scoring\n\n"
        welcome += f"⏱ Scans every {interval} min  (4H)\n"
        welcome += "📅 Trade horizon: days to weeks\n\n"
        welcome += "<b>Commands:</b>  /scan /stats /trades /help"

        await self.send_msg(welcome)

        asyncio.create_task(self.track_trades())
        asyncio.create_task(self._daily_report_loop())

        while True:
            try:
                await self.scan_all()
                logger.info(f"💤 Next swing scan in {interval} min")
                await asyncio.sleep(interval * 60)
            except Exception as e:
                logger.error(f"❌ {e}")
                await asyncio.sleep(60)

    async def close(self):
        await self.exchange.close()


# ═══════════════════════════════════════════════════════════════
#  BOT COMMANDS
# ═══════════════════════════════════════════════════════════════

class BotCommands:
    def __init__(self, scanner):
        self.scanner = scanner

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg  = "📈 <b>Swing Trading Scanner</b>\n\n"
        msg += "Daily + 4H Order Block signals.\n"
        msg += "Trade horizon: days to weeks.\n\n"
        msg += "<b>Commands:</b>\n"
        msg += "/scan  — Force scan now\n"
        msg += "/stats — Statistics\n"
        msg += "/trades — Active swings\n"
        msg += "/help  — Help\n"
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def cmd_scan(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if self.scanner.is_scanning:
            await update.message.reply_text("⚠️ Scan already running!")
            return
        await update.message.reply_text("🔍 Starting swing scan...")
        await self.scanner.scan_all()

    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        s    = self.scanner.stats
        msg  = f"📊 <b>SWING STATISTICS</b>\n\n"
        msg += f"Total:    {s['total_signals']}\n"
        msg += f"Long:     {s['long_signals']} 🟢\n"
        msg += f"Short:    {s['short_signals']} 🔴\n"
        msg += f"Premium:  {s['premium_signals']} 💎\n"
        msg += f"OB Setups: {s['ob_signals']} 🧱\n\n"
        msg += f"<b>TP Hits:</b>\n"
        msg += f"  TP1: {s['tp1_hits']} 🎯\n"
        msg += f"  TP2: {s['tp2_hits']} 🎯\n"
        msg += f"  TP3: {s['tp3_hits']} 🎯\n\n"
        if s['last_scan_time']:
            msg += f"Last scan: {s['last_scan_time'].strftime('%d %b  %H:%M')}\n"
            msg += f"Pairs:     {s['pairs_scanned']}"
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def cmd_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        trades = self.scanner.active_trades
        if not trades:
            await update.message.reply_text("📭 No active swing trades")
            return

        msg = f"📡 <b>ACTIVE SWINGS ({len(trades)})</b>\n\n"
        for tid, t in list(trades.items())[:10]:
            age = datetime.now() - t['timestamp']
            days = age.days
            hrs  = int(age.total_seconds() / 3600) % 24
            tp_status = "".join("✅" if h else "⏳" for h in t['tp_hit'])
            ob_tag = f" 🧱{t.get('ob_tf','')}" if t.get('ob_zone') else ""
            msg += f"<b>{t['symbol']}</b> {t['signal']}{ob_tag}\n"
            msg += f"  {tp_status}  |  {days}d {hrs}h old\n\n"

        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg  = "📚 <b>SWING TRADING SCANNER — HELP</b>\n\n"
        msg += "<b>Timeframes:</b>\n"
        msg += "• 1D  — Primary OBs, BOS/CHoCH, P/D zones\n"
        msg += "• 4H  — Secondary OBs, entry confirmation\n"
        msg += "• 1H  — Final indicator checks\n\n"
        msg += "<b>Order Block Logic:</b>\n"
        msg += "• Daily OB = 8–10 pts (highest weight)\n"
        msg += "• 4H OB   = 5 pts\n"
        msg += "• SL placed beyond OB boundary\n"
        msg += "• SL silenced if TP1 already hit\n\n"
        msg += "<b>TP / SL Sizing:</b>\n"
        msg += "• Uses daily ATR for sizing\n"
        msg += "• TP1 = 1.5×  |  TP2 = 3.5×  |  TP3 = 7×\n"
        msg += "• SL  = 2.5× ATR beyond OB\n\n"
        msg += "<b>Signal Quality:</b>\n"
        msg += "💎 PREMIUM  ≥ 75% score\n"
        msg += "🔥 HIGH     ≥ 62%\n"
        msg += "✅ GOOD     ≥ 50%\n\n"
        msg += "<b>Commands:</b>\n"
        msg += "/scan  /stats  /trades  /help"
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

async def main():
    TELEGRAM_TOKEN   = "7732870721:AAEHG3QJdo31S9sA8xjJzf-cXj6Tn4mo2uo"
    TELEGRAM_CHAT_ID = "7500072234"
    BINANCE_API_KEY  = None
    BINANCE_SECRET   = None

    scanner = SwingTradingScanner(
        telegram_token   = TELEGRAM_TOKEN,
        telegram_chat_id = TELEGRAM_CHAT_ID,
        binance_api_key  = BINANCE_API_KEY,
        binance_secret   = BINANCE_SECRET,
    )

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    cmd = BotCommands(scanner)

    app.add_handler(CommandHandler("start",  cmd.cmd_start))
    app.add_handler(CommandHandler("scan",   cmd.cmd_scan))
    app.add_handler(CommandHandler("stats",  cmd.cmd_stats))
    app.add_handler(CommandHandler("trades", cmd.cmd_trades))
    app.add_handler(CommandHandler("help",   cmd.cmd_help))

    await app.initialize()
    await app.start()
    logger.info("🤖 Swing bot ready!")

    try:
        await scanner.run(interval=240)    # scan every 4 hours
    except KeyboardInterrupt:
        logger.info("⚠️ Shutting down...")
    finally:
        await scanner.close()
        await app.stop()
        await app.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
