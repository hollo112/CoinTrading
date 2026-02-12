import pyupbit
import json
import os
import threading
import time
from datetime import datetime
from trading.strategy import TradingStrategy
from trading.data_collector import (
    get_fear_greed_index, get_kimchi_premium,
    get_orderbook_analysis,
)

TRADE_LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trade_history.json")


class AutoTrader:
    """업비트 자동매매 엔진"""

    def __init__(self, access_key, secret_key):
        self.upbit = pyupbit.Upbit(access_key, secret_key)
        self.is_running = False
        self.ticker = "KRW-BTC"
        self.interval = "minute60"
        self.check_interval = 60
        self.invest_ratio = 0.1
        self.max_invest_amount = 1_000_000  # 1회 최대 투자금 (원)
        self.strategy = TradingStrategy()
        self.thread = None
        self.status = "대기"
        self.last_signal = None
        self.last_check_time = None
        self._load_trade_log()

    def cleanup(self):
        """연결 해제 시 민감 정보 정리"""
        self.stop()
        self.upbit = None

    # ── 거래 기록 관리 ──

    def _load_trade_log(self):
        try:
            if os.path.exists(TRADE_LOG_FILE):
                with open(TRADE_LOG_FILE, "r", encoding="utf-8") as f:
                    self.trade_log = json.load(f)
            else:
                self.trade_log = []
        except Exception:
            self.trade_log = []

    def _save_trade_log(self):
        try:
            with open(TRADE_LOG_FILE, "w", encoding="utf-8") as f:
                json.dump(self.trade_log, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _add_log(self, trade_type, message, amount=0, price=0, signal=None):
        entry = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": trade_type,
            "ticker": self.ticker,
            "message": message,
            "amount": round(amount, 2),
            "price": round(price, 2) if price else 0,
            "signal_score": signal["score"] if signal else 0,
            "signal_details": (
                {k: v[1] for k, v in signal["signals"].items()} if signal else {}
            ),
        }
        self.trade_log.append(entry)
        self._save_trade_log()

    def get_trade_log(self):
        self._load_trade_log()
        return self.trade_log

    def clear_trade_log(self):
        self.trade_log = []
        self._save_trade_log()

    # ── 잔고 조회 ──

    def get_balances(self):
        try:
            return self.upbit.get_balances()
        except Exception:
            return []

    def get_krw_balance(self):
        try:
            return float(self.upbit.get_balance("KRW"))
        except Exception:
            return 0.0

    def get_coin_balance(self, ticker=None):
        if ticker is None:
            ticker = self.ticker
        coin = ticker.split("-")[1] if "-" in ticker else ticker
        try:
            return float(self.upbit.get_balance(coin))
        except Exception:
            return 0.0

    # ── 자동매매 제어 ──

    def start(self, ticker, interval, check_interval, invest_ratio, max_invest_amount, strategy_params):
        self.ticker = ticker
        self.interval = interval
        self.check_interval = check_interval
        self.invest_ratio = invest_ratio
        self.max_invest_amount = max_invest_amount
        self.strategy = TradingStrategy(**strategy_params)
        self.is_running = True
        self.status = "실행 중"
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.is_running = False
        self.status = "중지됨"

    def _run_loop(self):
        while self.is_running:
            try:
                self._check_and_trade()
                self.last_check_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            except Exception as e:
                self._add_log("ERROR", str(e))
            time.sleep(self.check_interval)

    def _get_avg_buy_price(self):
        """현재 코인의 매수평균가 조회"""
        try:
            balances = self.upbit.get_balances()
            coin = self.ticker.split("-")[1] if "-" in self.ticker else self.ticker
            for b in balances:
                if b["currency"] == coin:
                    return float(b.get("avg_buy_price", 0))
        except Exception:
            pass
        return 0.0

    def _get_binance_symbol(self):
        """업비트 티커 → 바이낸스 심볼 변환"""
        coin = self.ticker.split("-")[1] if "-" in self.ticker else self.ticker
        return f"{coin}USDT"

    def _collect_multi_timeframe(self):
        """멀티 타임프레임 시그널 수집 (4시간, 일봉)"""
        tf_map = {"4h": "minute240", "1d": "day"}
        results = {}
        for label, interval in tf_map.items():
            if interval == self.interval:
                continue
            try:
                tf_df = pyupbit.get_ohlcv(self.ticker, interval=interval, count=200)
                if tf_df is not None and len(tf_df) > 50:
                    temp = TradingStrategy(
                        rsi_period=self.strategy.rsi_period,
                        rsi_oversold=self.strategy.rsi_oversold,
                        rsi_overbought=self.strategy.rsi_overbought,
                    )
                    sig = temp.get_signal(tf_df)
                    results[label] = sig["action"]
            except Exception:
                pass
        return results if results else None

    def _check_and_trade(self):
        df = pyupbit.get_ohlcv(self.ticker, interval=self.interval, count=200)
        if df is None or len(df) < 50:
            return

        # 외부 데이터 수집 → 전략에 주입
        try:
            fng = get_fear_greed_index()
            if fng:
                self.strategy.fear_greed_value = int(fng[0].get("value", 50))
        except Exception:
            pass

        try:
            self.strategy.orderbook_data = get_orderbook_analysis(self.ticker)
        except Exception:
            pass

        try:
            self.strategy.kimchi_premium_data = get_kimchi_premium(
                self.ticker, self._get_binance_symbol()
            )
        except Exception:
            pass

        try:
            self.strategy.multi_tf_signals = self._collect_multi_timeframe()
        except Exception:
            pass

        # 손절/익절 우선 체크
        current_price = pyupbit.get_current_price(self.ticker)
        avg_price = self._get_avg_buy_price()
        if avg_price > 0 and current_price:
            sl_tp = self.strategy.check_stop_loss_take_profit(current_price, avg_price)
            if sl_tp == "STOP_LOSS":
                pnl = (current_price - avg_price) / avg_price * 100
                self._add_log("손절", f"손절 실행 ({pnl:+.2f}%)", 0, current_price)
                self._execute_sell({"action": "SELL", "score": 0, "signals": {"손절": ("매도", f"{pnl:+.2f}%")}})
                return
            if sl_tp == "TAKE_PROFIT":
                pnl = (current_price - avg_price) / avg_price * 100
                self._add_log("익절", f"익절 실행 ({pnl:+.2f}%)", 0, current_price)
                self._execute_sell({"action": "SELL", "score": 0, "signals": {"익절": ("매도", f"{pnl:+.2f}%")}})
                return

        # 일반 시그널 매매
        signal = self.strategy.get_signal(df)
        self.last_signal = signal

        if signal["action"] == "BUY":
            self._execute_buy(signal)
        elif signal["action"] == "SELL":
            self._execute_sell(signal)

    # ── 주문 실행 ──

    def _execute_buy(self, signal):
        krw = self.get_krw_balance()
        invest = krw * self.invest_ratio
        invest = min(invest, self.max_invest_amount)  # 1회 최대 투자금 제한
        if invest < 5000:  # 업비트 최소 주문금액
            return
        try:
            result = self.upbit.buy_market_order(self.ticker, invest)
            if result and "error" not in result:
                price = pyupbit.get_current_price(self.ticker) or 0
                self._add_log("매수", f"{self.ticker} 시장가 매수", invest, price, signal)
            else:
                msg = (
                    result.get("error", {}).get("message", "알 수 없음")
                    if result else "응답 없음"
                )
                self._add_log("매수실패", msg)
        except Exception as e:
            self._add_log("매수실패", str(e))

    def _execute_sell(self, signal):
        coin_balance = self.get_coin_balance()
        current_price = pyupbit.get_current_price(self.ticker)
        if current_price is None:
            return
        if coin_balance * current_price < 5000:
            return
        try:
            result = self.upbit.sell_market_order(self.ticker, coin_balance)
            if result and "error" not in result:
                self._add_log(
                    "매도", f"{self.ticker} 시장가 매도",
                    coin_balance * current_price, current_price, signal,
                )
            else:
                msg = (
                    result.get("error", {}).get("message", "알 수 없음")
                    if result else "응답 없음"
                )
                self._add_log("매도실패", msg)
        except Exception as e:
            self._add_log("매도실패", str(e))