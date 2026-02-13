import pandas as pd
import numpy as np


class TradingStrategy:
    """고급 기술적 분석 매매 전략
    [기본] RSI, MACD, 볼린저밴드, 이동평균
    [고급] 거래량 급증, ADX, 공포탐욕지수, 변동성 돌파,
           멀티 타임프레임, 호가창 분석, 김치프리미엄,
           일목균형표, OBV 다이버전스, Stochastic RSI
    + 손절/익절 관리
    """

    def __init__(self, rsi_period=14, rsi_oversold=30, rsi_overbought=70,
                 bb_period=20, bb_std=2.0,
                 ma_short=5, ma_long=20,
                 macd_fast=12, macd_slow=26, macd_signal=9,
                 adx_period=14, adx_threshold=25,
                 vol_spike_multiplier=2.0,
                 stop_loss_pct=3.0, take_profit_pct=5.0,
                 volatility_k=0.5,
                 kimchi_premium_threshold=3.0,
                 orderbook_imbalance_threshold=0.6,
                 fear_greed_value=None,
                 orderbook_data=None,
                 kimchi_premium_data=None,
                 multi_tf_signals=None):
        # 기본 지표
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        # ADX
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        # 거래량
        self.vol_spike_multiplier = vol_spike_multiplier
        # 손절 / 익절
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        # 변동성 돌파
        self.volatility_k = volatility_k
        # 김치프리미엄
        self.kimchi_premium_threshold = kimchi_premium_threshold
        # 호가창
        self.orderbook_imbalance_threshold = orderbook_imbalance_threshold
        # 외부 주입 데이터
        self.fear_greed_value = fear_greed_value
        self.orderbook_data = orderbook_data
        self.kimchi_premium_data = kimchi_premium_data
        self.multi_tf_signals = multi_tf_signals  # {"4h": "BUY", "1d": "HOLD", ...}

    # ================================================================
    #  기본 지표
    # ================================================================

    def calculate_rsi(self, series, period=None):
        if period is None:
            period = self.rsi_period
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, series):
        exp_fast = series.ewm(span=self.macd_fast, adjust=False).mean()
        exp_slow = series.ewm(span=self.macd_slow, adjust=False).mean()
        macd = exp_fast - exp_slow
        signal = macd.ewm(span=self.macd_signal, adjust=False).mean()
        histogram = macd - signal
        return macd, signal, histogram

    def calculate_bollinger_bands(self, series):
        mid = series.rolling(window=self.bb_period).mean()
        std = series.rolling(window=self.bb_period).std()
        upper = mid + (std * self.bb_std)
        lower = mid - (std * self.bb_std)
        return upper, mid, lower

    def calculate_moving_averages(self, series):
        ma_short = series.rolling(window=self.ma_short).mean()
        ma_long = series.rolling(window=self.ma_long).mean()
        return ma_short, ma_long

    def calculate_adx(self, df, period=None):
        if period is None:
            period = self.adx_period
        high, low, close = df["high"], df["low"], df["close"]
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
        adx = dx.rolling(window=period).mean()
        return adx, plus_di, minus_di

    def calculate_volume_spike(self, df, window=20):
        vol_ma = df["volume"].rolling(window=window).mean()
        return df["volume"] / vol_ma

    # ================================================================
    #  고급 지표 (NEW)
    # ================================================================

    def calculate_stochastic_rsi(self, series, rsi_period=None, stoch_period=14, k_period=3, d_period=3):
        """Stochastic RSI - 일반 RSI보다 민감한 과매수/과매도 판단"""
        rsi = self.calculate_rsi(series, rsi_period)
        rsi_min = rsi.rolling(window=stoch_period).min()
        rsi_max = rsi.rolling(window=stoch_period).max()
        stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min)
        k = stoch_rsi.rolling(window=k_period).mean() * 100
        d = k.rolling(window=d_period).mean()
        return k, d

    def calculate_obv(self, df):
        """OBV (On Balance Volume) - 거래량으로 가격 방향 예측"""
        obv = pd.Series(0.0, index=df.index)
        for i in range(1, len(df)):
            if df["close"].iloc[i] > df["close"].iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] + df["volume"].iloc[i]
            elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] - df["volume"].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]
        return obv

    def calculate_ichimoku(self, df, tenkan=9, kijun=26, senkou_b=52):
        """일목균형표 (Ichimoku Cloud) - 아시아권 대표 지표"""
        high, low = df["high"], df["low"]
        # 전환선 (Tenkan-sen)
        tenkan_sen = (high.rolling(window=tenkan).max() + low.rolling(window=tenkan).min()) / 2
        # 기준선 (Kijun-sen)
        kijun_sen = (high.rolling(window=kijun).max() + low.rolling(window=kijun).min()) / 2
        # 선행스팬A (Senkou Span A) - 26일 앞으로 이동
        senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
        # 선행스팬B (Senkou Span B) - 26일 앞으로 이동
        senkou_b_val = ((high.rolling(window=senkou_b).max() + low.rolling(window=senkou_b).min()) / 2).shift(kijun)
        # 후행스팬 (Chikou Span) - 26일 뒤로 이동
        chikou = df["close"].shift(-kijun)
        return tenkan_sen, kijun_sen, senkou_a, senkou_b_val, chikou

    def calculate_volatility_breakout(self, df, k=None):
        """변동성 돌파 전략 (래리 윌리엄스)
        목표가 = 당일 시가 + 전일 변동폭 * k
        현재가 > 목표가 → 매수 시그널
        """
        if k is None:
            k = self.volatility_k
        prev_range = (df["high"].shift(1) - df["low"].shift(1))
        target_price = df["open"] + prev_range * k
        return target_price

    def calculate_vwap(self, df):
        """VWAP (Volume Weighted Average Price) - 기관 투자자 핵심 지표
        거래량이 많은 가격대에 가중치를 두어 평균가를 산출
        """
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        cumulative_tp_vol = (typical_price * df["volume"]).cumsum()
        cumulative_vol = df["volume"].cumsum()
        vwap = cumulative_tp_vol / cumulative_vol
        return vwap

    def detect_rsi_divergence(self, df, lookback=14):
        """RSI 다이버전스 감지
        상승 다이버전스: 가격 저점 하락 + RSI 저점 상승 → 매수
        하락 다이버전스: 가격 고점 상승 + RSI 고점 하락 → 매도
        """
        if len(df) < lookback * 2:
            return "none"
        close = df["close"]
        rsi = df["rsi"]

        recent_close = close.iloc[-lookback:]
        prev_close = close.iloc[-lookback * 2:-lookback]
        recent_rsi = rsi.iloc[-lookback:]
        prev_rsi = rsi.iloc[-lookback * 2:-lookback]

        if recent_rsi.isna().all() or prev_rsi.isna().all():
            return "none"

        recent_low = recent_close.min()
        prev_low = prev_close.min()
        recent_rsi_low = recent_rsi.min()
        prev_rsi_low = prev_rsi.min()

        recent_high = recent_close.max()
        prev_high = prev_close.max()
        recent_rsi_high = recent_rsi.max()
        prev_rsi_high = prev_rsi.max()

        # 상승 다이버전스: 가격은 더 낮은 저점, RSI는 더 높은 저점
        if recent_low < prev_low and recent_rsi_low > prev_rsi_low:
            return "bullish"
        # 하락 다이버전스: 가격은 더 높은 고점, RSI는 더 낮은 고점
        if recent_high > prev_high and recent_rsi_high < prev_rsi_high:
            return "bearish"
        return "none"

    # ================================================================
    #  지표 통합
    # ================================================================

    def add_indicators(self, df):
        """데이터프레임에 모든 기술적 지표 추가"""
        df = df.copy()
        close = df["close"]

        # 기본 지표
        df["rsi"] = self.calculate_rsi(close)
        df["macd"], df["macd_signal"], df["macd_hist"] = self.calculate_macd(close)
        df["bb_upper"], df["bb_mid"], df["bb_lower"] = self.calculate_bollinger_bands(close)
        df["ma_short"], df["ma_long"] = self.calculate_moving_averages(close)
        df["adx"], df["plus_di"], df["minus_di"] = self.calculate_adx(df)
        df["vol_ratio"] = self.calculate_volume_spike(df)

        # 고급 지표
        df["stoch_k"], df["stoch_d"] = self.calculate_stochastic_rsi(close)
        df["obv"] = self.calculate_obv(df)
        df["obv_ma"] = df["obv"].rolling(window=20).mean()
        df["ichimoku_tenkan"], df["ichimoku_kijun"], df["ichimoku_senkou_a"], df["ichimoku_senkou_b"], df["ichimoku_chikou"] = self.calculate_ichimoku(df)
        df["vb_target"] = self.calculate_volatility_breakout(df)
        df["vwap"] = self.calculate_vwap(df)

        return df

    # ================================================================
    #  손절 / 익절
    # ================================================================

    def check_stop_loss_take_profit(self, current_price, avg_buy_price,
                                     peak_price=None, trailing_stop_pct=2.0):
        """손절/익절/트레일링 스탑 체크

        Args:
            current_price: 현재가
            avg_buy_price: 매수평균가
            peak_price: 보유 중 최고가 (트레일링 스탑용)
            trailing_stop_pct: 최고점 대비 하락 허용 % (기본 2%)

        Returns:
            "STOP_LOSS" | "TAKE_PROFIT" | "TRAILING_STOP" | None
        """
        if avg_buy_price <= 0 or current_price <= 0:
            return None
        pnl_pct = (current_price - avg_buy_price) / avg_buy_price * 100

        # 트레일링 스탑: 수익 구간에서 고점 대비 하락 시 매도
        if peak_price and peak_price > avg_buy_price:
            peak_pnl = (peak_price - avg_buy_price) / avg_buy_price * 100
            drop_from_peak = (peak_price - current_price) / peak_price * 100
            # 수익률 2% 이상 달성 후, 고점 대비 trailing_stop_pct 이상 하락하면 매도
            if peak_pnl >= 2.0 and drop_from_peak >= trailing_stop_pct:
                return "TRAILING_STOP"

        if pnl_pct <= -self.stop_loss_pct:
            return "STOP_LOSS"
        if pnl_pct >= self.take_profit_pct:
            return "TAKE_PROFIT"
        return None

    # ================================================================
    #  종합 시그널
    # ================================================================

    def get_signal(self, df):
        """복합 지표 기반 매매 시그널 생성"""
        df = self.add_indicators(df)
        if len(df) < 2:
            return {"action": "HOLD", "score": 0, "signals": {}, "indicators": {}}

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        signals = {}
        score = 0

        # ── 1. RSI ──
        if pd.notna(latest["rsi"]):
            if latest["rsi"] < self.rsi_oversold:
                signals["RSI"] = ("매수", f"RSI {latest['rsi']:.1f} < {self.rsi_oversold}")
                score += 1
            elif latest["rsi"] > self.rsi_overbought:
                signals["RSI"] = ("매도", f"RSI {latest['rsi']:.1f} > {self.rsi_overbought}")
                score -= 1
            else:
                signals["RSI"] = ("중립", f"RSI {latest['rsi']:.1f}")

        # ── 2. MACD ──
        if pd.notna(latest["macd"]) and pd.notna(prev["macd"]):
            if prev["macd"] < prev["macd_signal"] and latest["macd"] > latest["macd_signal"]:
                signals["MACD"] = ("매수", "골든크로스")
                score += 1
            elif prev["macd"] > prev["macd_signal"] and latest["macd"] < latest["macd_signal"]:
                signals["MACD"] = ("매도", "데드크로스")
                score -= 1
            else:
                label = "상승세" if latest["macd_hist"] > 0 else "하락세"
                signals["MACD"] = ("중립", label)

        # ── 3. 볼린저밴드 ──
        if pd.notna(latest["bb_lower"]) and (latest["bb_upper"] - latest["bb_lower"]) > 0:
            if latest["close"] < latest["bb_lower"]:
                signals["볼린저밴드"] = ("매수", "하단 이탈")
                score += 1
            elif latest["close"] > latest["bb_upper"]:
                signals["볼린저밴드"] = ("매도", "상단 이탈")
                score -= 1
            else:
                pct = (latest["close"] - latest["bb_lower"]) / (latest["bb_upper"] - latest["bb_lower"]) * 100
                signals["볼린저밴드"] = ("중립", f"밴드 {pct:.0f}%")

        # ── 4. 이동평균 ──
        if pd.notna(latest["ma_short"]) and pd.notna(prev["ma_short"]):
            if prev["ma_short"] < prev["ma_long"] and latest["ma_short"] > latest["ma_long"]:
                signals["이동평균"] = ("매수", "골든크로스")
                score += 1
            elif prev["ma_short"] > prev["ma_long"] and latest["ma_short"] < latest["ma_long"]:
                signals["이동평균"] = ("매도", "데드크로스")
                score -= 1
            else:
                trend = "상승세" if latest["ma_short"] > latest["ma_long"] else "하락세"
                signals["이동평균"] = ("중립", trend)

        # ── 5. 거래량 급증 ──
        if pd.notna(latest["vol_ratio"]):
            if latest["vol_ratio"] >= self.vol_spike_multiplier:
                if latest["close"] > latest["open"]:
                    signals["거래량"] = ("매수", f"급증 x{latest['vol_ratio']:.1f} (양봉)")
                    score += 1
                else:
                    signals["거래량"] = ("매도", f"급증 x{latest['vol_ratio']:.1f} (음봉)")
                    score -= 1
            else:
                signals["거래량"] = ("중립", f"x{latest['vol_ratio']:.1f}")

        # ── 6. ADX ──
        if pd.notna(latest["adx"]):
            if latest["adx"] >= self.adx_threshold:
                if latest["plus_di"] > latest["minus_di"]:
                    signals["ADX"] = ("매수", f"강한 상승추세 ({latest['adx']:.0f})")
                    score += 1
                else:
                    signals["ADX"] = ("매도", f"강한 하락추세 ({latest['adx']:.0f})")
                    score -= 1
            else:
                signals["ADX"] = ("중립", f"추세 약함 ({latest['adx']:.0f})")

        # ── 7. Stochastic RSI (NEW) ──
        if pd.notna(latest["stoch_k"]) and pd.notna(prev["stoch_k"]):
            if latest["stoch_k"] < 20 and prev["stoch_k"] < prev["stoch_d"] and latest["stoch_k"] > latest["stoch_d"]:
                signals["StochRSI"] = ("매수", f"과매도 반등 (K:{latest['stoch_k']:.0f})")
                score += 1
            elif latest["stoch_k"] > 80 and prev["stoch_k"] > prev["stoch_d"] and latest["stoch_k"] < latest["stoch_d"]:
                signals["StochRSI"] = ("매도", f"과매수 하락 (K:{latest['stoch_k']:.0f})")
                score -= 1
            else:
                signals["StochRSI"] = ("중립", f"K:{latest['stoch_k']:.0f} D:{latest['stoch_d']:.0f}")

        # ── 8. OBV 다이버전스 (NEW) ──
        if pd.notna(latest["obv"]) and pd.notna(latest["obv_ma"]):
            price_rising = latest["close"] > df["close"].iloc[-5] if len(df) >= 5 else False
            price_falling = latest["close"] < df["close"].iloc[-5] if len(df) >= 5 else False
            obv_rising = latest["obv"] > df["obv"].iloc[-5] if len(df) >= 5 else False
            obv_falling = latest["obv"] < df["obv"].iloc[-5] if len(df) >= 5 else False

            if price_falling and obv_rising:
                signals["OBV"] = ("매수", "상승 다이버전스 (가격↓ 거래량↑)")
                score += 1
            elif price_rising and obv_falling:
                signals["OBV"] = ("매도", "하락 다이버전스 (가격↑ 거래량↓)")
                score -= 1
            else:
                obv_trend = "상승" if latest["obv"] > latest["obv_ma"] else "하락"
                signals["OBV"] = ("중립", f"OBV {obv_trend}")

        # ── 9. 일목균형표 (NEW) ──
        if pd.notna(latest.get("ichimoku_tenkan")) and pd.notna(latest.get("ichimoku_kijun")):
            above_cloud = False
            below_cloud = False
            if pd.notna(latest.get("ichimoku_senkou_a")) and pd.notna(latest.get("ichimoku_senkou_b")):
                cloud_top = max(latest["ichimoku_senkou_a"], latest["ichimoku_senkou_b"])
                cloud_bottom = min(latest["ichimoku_senkou_a"], latest["ichimoku_senkou_b"])
                above_cloud = latest["close"] > cloud_top
                below_cloud = latest["close"] < cloud_bottom

            # 전환선 > 기준선 + 구름 위 = 강한 매수
            if latest["ichimoku_tenkan"] > latest["ichimoku_kijun"] and above_cloud:
                signals["일목균형표"] = ("매수", "구름 위 + 전환선 > 기준선")
                score += 1
            elif latest["ichimoku_tenkan"] < latest["ichimoku_kijun"] and below_cloud:
                signals["일목균형표"] = ("매도", "구름 아래 + 전환선 < 기준선")
                score -= 1
            elif above_cloud:
                signals["일목균형표"] = ("중립", "구름 위 (지지)")
            elif below_cloud:
                signals["일목균형표"] = ("중립", "구름 아래 (저항)")
            else:
                signals["일목균형표"] = ("중립", "구름 내부 (관망)")

        # ── 10. 변동성 돌파 ──
        if pd.notna(latest.get("vb_target")) and pd.notna(prev.get("vb_target")):
            if latest["close"] > latest["vb_target"]:
                signals["변동성돌파"] = ("매수", f"목표가 {latest['vb_target']:,.0f} 돌파")
                score += 1
            elif prev["close"] > prev["vb_target"] and latest["close"] < latest["vb_target"]:
                signals["변동성돌파"] = ("매도", f"목표가 {latest['vb_target']:,.0f} 하향 이탈")
                score -= 1
            else:
                gap = (latest["vb_target"] - latest["close"]) / latest["close"] * 100
                signals["변동성돌파"] = ("중립", f"목표가까지 {gap:.1f}%")

        # ── 15. VWAP ──
        if pd.notna(latest.get("vwap")):
            vwap_diff_pct = (latest["close"] - latest["vwap"]) / latest["vwap"] * 100
            if latest["close"] < latest["vwap"] and vwap_diff_pct < -1.0:
                signals["VWAP"] = ("매수", f"VWAP 하회 ({vwap_diff_pct:+.1f}%)")
                score += 1
            elif latest["close"] > latest["vwap"] and vwap_diff_pct > 1.0:
                signals["VWAP"] = ("매도", f"VWAP 상회 ({vwap_diff_pct:+.1f}%)")
                score -= 1
            else:
                signals["VWAP"] = ("중립", f"VWAP 근접 ({vwap_diff_pct:+.1f}%)")

        # ── 16. RSI 다이버전스 ──
        rsi_div = self.detect_rsi_divergence(df)
        if rsi_div == "bullish":
            signals["RSI다이버전스"] = ("매수", "가격↓ RSI↑ (반등 가능)")
            score += 1
        elif rsi_div == "bearish":
            signals["RSI다이버전스"] = ("매도", "가격↑ RSI↓ (하락 가능)")
            score -= 1
        else:
            signals["RSI다이버전스"] = ("중립", "다이버전스 없음")

        # ── 11. 공포탐욕지수 ──
        if self.fear_greed_value is not None:
            fg = self.fear_greed_value
            if fg <= 25:
                signals["공포탐욕"] = ("매수", f"극도의 공포 ({fg})")
                score += 1
            elif fg >= 75:
                signals["공포탐욕"] = ("매도", f"극도의 탐욕 ({fg})")
                score -= 1
            elif fg <= 40:
                signals["공포탐욕"] = ("중립", f"공포 ({fg})")
            elif fg >= 60:
                signals["공포탐욕"] = ("중립", f"탐욕 ({fg})")
            else:
                signals["공포탐욕"] = ("중립", f"중립 ({fg})")

        # ── 12. 호가창 분석 (NEW) ──
        if self.orderbook_data:
            ob = self.orderbook_data
            buy_ratio = ob.get("buy_ratio", 0.5)
            whale_bids = ob.get("whale_bid_count", 0)
            whale_asks = ob.get("whale_ask_count", 0)

            if buy_ratio >= self.orderbook_imbalance_threshold:
                detail = f"매수세 {buy_ratio:.0%}"
                if whale_bids > 0:
                    detail += f" (매수벽 {whale_bids}건)"
                signals["호가창"] = ("매수", detail)
                score += 1
            elif (1 - buy_ratio) >= self.orderbook_imbalance_threshold:
                detail = f"매도세 {1 - buy_ratio:.0%}"
                if whale_asks > 0:
                    detail += f" (매도벽 {whale_asks}건)"
                signals["호가창"] = ("매도", detail)
                score -= 1
            else:
                signals["호가창"] = ("중립", f"매수 {buy_ratio:.0%} / 매도 {1 - buy_ratio:.0%}")

        # ── 13. 김치프리미엄 (NEW) ──
        if self.kimchi_premium_data:
            kp = self.kimchi_premium_data.get("premium_pct", 0)
            if kp >= self.kimchi_premium_threshold:
                signals["김프"] = ("매도", f"김프 {kp:+.1f}% (과열)")
                score -= 1
            elif kp <= -self.kimchi_premium_threshold:
                signals["김프"] = ("매수", f"김프 {kp:+.1f}% (저평가)")
                score += 1
            else:
                signals["김프"] = ("중립", f"김프 {kp:+.1f}%")

        # ── 14. 멀티 타임프레임 (NEW) ──
        if self.multi_tf_signals:
            buy_count = sum(1 for v in self.multi_tf_signals.values() if v == "BUY")
            sell_count = sum(1 for v in self.multi_tf_signals.values() if v == "SELL")
            tf_summary = ", ".join(f"{k}:{v}" for k, v in self.multi_tf_signals.items())

            if buy_count >= 2:
                signals["멀티TF"] = ("매수", f"다중 시간대 매수 ({tf_summary})")
                score += 1
            elif sell_count >= 2:
                signals["멀티TF"] = ("매도", f"다중 시간대 매도 ({tf_summary})")
                score -= 1
            else:
                signals["멀티TF"] = ("중립", tf_summary)

        # ── 종합 판단 ──
        adx_val = latest.get("adx", 0) if pd.notna(latest.get("adx")) else 0
        if adx_val < self.adx_threshold:
            buy_threshold = 3   # 횡보장: 시그널 3개 이상
            sell_threshold = -3
        else:
            buy_threshold = 2   # 추세장: 시그널 2개 이상
            sell_threshold = -2

        if score >= buy_threshold:
            action = "BUY"
        elif score <= sell_threshold:
            action = "SELL"
        else:
            action = "HOLD"

        return {
            "action": action,
            "score": score,
            "buy_threshold": buy_threshold,
            "sell_threshold": sell_threshold,
            "signals": signals,
            "indicators": {
                "rsi": latest.get("rsi"),
                "macd": latest.get("macd"),
                "macd_signal": latest.get("macd_signal"),
                "macd_hist": latest.get("macd_hist"),
                "bb_upper": latest.get("bb_upper"),
                "bb_mid": latest.get("bb_mid"),
                "bb_lower": latest.get("bb_lower"),
                "ma_short": latest.get("ma_short"),
                "ma_long": latest.get("ma_long"),
                "adx": latest.get("adx"),
                "plus_di": latest.get("plus_di"),
                "minus_di": latest.get("minus_di"),
                "vol_ratio": latest.get("vol_ratio"),
                "stoch_k": latest.get("stoch_k"),
                "stoch_d": latest.get("stoch_d"),
                "obv": latest.get("obv"),
                "ichimoku_tenkan": latest.get("ichimoku_tenkan"),
                "ichimoku_kijun": latest.get("ichimoku_kijun"),
                "vb_target": latest.get("vb_target"),
                "vwap": latest.get("vwap"),
                "close": latest["close"],
            },
        }