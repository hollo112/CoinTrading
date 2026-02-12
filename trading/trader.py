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

    COIN_LIST = [
        "KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL", "KRW-DOGE",
        "KRW-ADA", "KRW-AVAX", "KRW-DOT", "KRW-LINK", "KRW-MATIC",
        "KRW-ATOM", "KRW-TRX", "KRW-ETC", "KRW-BCH", "KRW-NEAR",
        "KRW-APT", "KRW-ARB", "KRW-OP", "KRW-SUI", "KRW-SEI",
        "KRW-AAVE", "KRW-IMX", "KRW-STX", "KRW-SAND", "KRW-MANA",
    ]

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
        self.last_reason = ""  # 마지막 판단 사유
        self.check_count = 0   # 총 체크 횟수
        # 멀티코인 모드
        self.multi_mode = False
        self.max_coins = 5
        self.coin_list = []
        self.multi_status = {}  # {ticker: {"action": ..., "score": ..., "reason": ...}}
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

    def _add_log(self, trade_type, message, amount=0, price=0, signal=None, ticker=None):
        entry = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": trade_type,
            "ticker": ticker or self.ticker,
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
        self.multi_mode = False
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
                self.check_count += 1
                self._check_and_trade()
                self.last_check_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            except Exception as e:
                self.last_reason = f"에러: {e}"
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
            self.last_reason = "데이터 부족 (50봉 미만)"
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
            # 잔고 체크
            krw = self.get_krw_balance()
            invest = min(krw * self.invest_ratio, self.max_invest_amount)
            if invest < 5000:
                self.last_reason = (
                    f"매수 시그널 (스코어:{signal['score']:+d}) "
                    f"but 투자금 부족 ({invest:,.0f}원 < 5,000원, "
                    f"KRW잔고:{krw:,.0f}원 × 비율:{self.invest_ratio:.0%})"
                )
                return
            self.last_reason = f"매수 실행 (스코어:{signal['score']:+d})"
            self._execute_buy(signal)
        elif signal["action"] == "SELL":
            self.last_reason = f"매도 실행 (스코어:{signal['score']:+d})"
            self._execute_sell(signal)
        else:
            buy_th = signal.get("buy_threshold", 3)
            sell_th = signal.get("sell_threshold", -3)
            sig_summary = ", ".join(
                f"{k}:{v[0]}" for k, v in signal["signals"].items() if v[0] != "중립"
            )
            self.last_reason = (
                f"HOLD (스코어:{signal['score']:+d}, "
                f"매수기준:{buy_th}+/매도기준:{sell_th}-) "
                f"{'| ' + sig_summary if sig_summary else ''}"
            )

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

    # ── 멀티코인 자동매매 ──

    def start_multi(self, interval, check_interval, invest_ratio, max_invest_amount,
                    max_coins, strategy_params):
        """멀티코인 모드로 자동매매 시작"""
        self.multi_mode = True
        self.interval = interval
        self.check_interval = check_interval
        self.invest_ratio = invest_ratio
        self.max_invest_amount = max_invest_amount
        self.max_coins = max_coins
        self.strategy = TradingStrategy(**strategy_params)
        self.multi_status = {}

        # 스캔 대상 코인 리스트 (실제 거래 가능한 것만)
        try:
            available = pyupbit.get_tickers(fiat="KRW")
        except Exception:
            available = []
        self.coin_list = [t for t in self.COIN_LIST if t in available]

        self.is_running = True
        self.status = "멀티코인 실행 중"
        self.thread = threading.Thread(target=self._run_multi_loop, daemon=True)
        self.thread.start()

    def _run_multi_loop(self):
        """멀티코인 매매 루프"""
        while self.is_running:
            try:
                self.check_count += 1
                self._multi_check_and_trade()
                self.last_check_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            except Exception as e:
                self.last_reason = f"멀티 루프 에러: {e}"
                self._add_log("ERROR", f"멀티 루프: {e}")
            time.sleep(self.check_interval)

    def get_held_coins(self):
        """현재 보유 중인 KRW 마켓 코인 목록과 잔고 정보 반환 (5,000원 미만 더스트 제외)"""
        held = {}
        try:
            balances = self.upbit.get_balances()
            for b in balances:
                if b["currency"] == "KRW":
                    continue
                ticker = f"KRW-{b['currency']}"
                amount = float(b.get("balance", 0)) + float(b.get("locked", 0))
                avg_price = float(b.get("avg_buy_price", 0))
                if amount <= 0 or avg_price <= 0:
                    continue
                # 더스트 필터: 현재가 기준 평가금액 5,000원 미만이면 제외
                try:
                    cur_price = pyupbit.get_current_price(ticker)
                    if cur_price and amount * cur_price < 5000:
                        continue
                except Exception:
                    pass
                held[ticker] = {"amount": amount, "avg_price": avg_price}
        except Exception:
            pass
        return held

    def _multi_check_and_trade(self):
        """멀티코인: 보유 코인 매도 체크 → 신규 매수 스캔"""
        held_coins = self.get_held_coins()
        new_status = {}

        # ── 1단계: 보유 코인 매도 체크 ──
        for ticker, info in held_coins.items():
            try:
                current_price = pyupbit.get_current_price(ticker)
                if current_price is None:
                    continue
                avg_price = info["avg_price"]
                pnl = (current_price - avg_price) / avg_price * 100

                # 손절/익절 체크
                sl_tp = self.strategy.check_stop_loss_take_profit(current_price, avg_price)
                if sl_tp == "STOP_LOSS":
                    signal = {"action": "SELL", "score": 0,
                              "signals": {"손절": ("매도", f"{pnl:+.2f}%")}}
                    self._execute_sell_ticker(ticker, info["amount"], signal)
                    new_status[ticker] = {"action": "SELL", "score": 0,
                                          "reason": f"손절 ({pnl:+.2f}%)"}
                    self._add_log("손절", f"{ticker} 손절 ({pnl:+.2f}%)",
                                  0, current_price, ticker=ticker)
                    time.sleep(0.11)
                    continue
                if sl_tp == "TAKE_PROFIT":
                    signal = {"action": "SELL", "score": 0,
                              "signals": {"익절": ("매도", f"{pnl:+.2f}%")}}
                    self._execute_sell_ticker(ticker, info["amount"], signal)
                    new_status[ticker] = {"action": "SELL", "score": 0,
                                          "reason": f"익절 ({pnl:+.2f}%)"}
                    self._add_log("익절", f"{ticker} 익절 ({pnl:+.2f}%)",
                                  0, current_price, ticker=ticker)
                    time.sleep(0.11)
                    continue

                # 일반 시그널 체크
                df = pyupbit.get_ohlcv(ticker, interval=self.interval, count=200)
                time.sleep(0.11)
                if df is None or len(df) < 50:
                    new_status[ticker] = {"action": "HOLD", "score": 0,
                                          "reason": "데이터 부족"}
                    continue
                signal = self.strategy.get_signal(df)
                if signal["action"] == "SELL":
                    self._execute_sell_ticker(ticker, info["amount"], signal)
                    new_status[ticker] = {"action": "SELL", "score": signal["score"],
                                          "reason": f"매도 시그널 (스코어:{signal['score']:+d})"}
                else:
                    new_status[ticker] = {
                        "action": "보유중", "score": signal["score"],
                        "reason": f"보유 유지 (스코어:{signal['score']:+d}, 수익률:{pnl:+.2f}%)",
                    }
            except Exception as e:
                new_status[ticker] = {"action": "ERROR", "score": 0, "reason": str(e)}

        # ── 2단계: 신규 매수 스캔 ──
        # 매도 후 보유 코인 수 재확인
        held_coins = self.get_held_coins()
        held_count = len(held_coins)
        slots_available = self.max_coins - held_count

        if slots_available > 0:
            buy_candidates = []
            for ticker in self.coin_list:
                if ticker in held_coins:
                    continue  # 이미 보유 중
                try:
                    df = pyupbit.get_ohlcv(ticker, interval=self.interval, count=200)
                    time.sleep(0.11)
                    if df is None or len(df) < 50:
                        new_status.setdefault(ticker, {
                            "action": "SKIP", "score": 0, "reason": "데이터 부족"})
                        continue
                    signal = self.strategy.get_signal(df)
                    if signal["action"] == "BUY":
                        buy_candidates.append((ticker, signal))
                        new_status[ticker] = {
                            "action": "BUY 후보", "score": signal["score"],
                            "reason": f"매수 시그널 (스코어:{signal['score']:+d})",
                        }
                    else:
                        new_status.setdefault(ticker, {
                            "action": signal["action"], "score": signal["score"],
                            "reason": f"스코어:{signal['score']:+d}",
                        })
                except Exception as e:
                    new_status.setdefault(ticker, {
                        "action": "ERROR", "score": 0, "reason": str(e)})

            # 스코어 순 정렬 → 상위부터 매수
            buy_candidates.sort(key=lambda x: x[1]["score"], reverse=True)
            bought = 0
            for ticker, signal in buy_candidates:
                if bought >= slots_available:
                    break
                if self._execute_buy_ticker(ticker, signal):
                    new_status[ticker] = {
                        "action": "BUY", "score": signal["score"],
                        "reason": f"매수 실행 (스코어:{signal['score']:+d})",
                    }
                    bought += 1
                    time.sleep(0.11)
        else:
            self.last_reason = f"보유 코인 {held_count}개 (최대 {self.max_coins}개) — 신규 매수 대기"

        self.multi_status = new_status
        held_summary = ", ".join(
            f"{t.replace('KRW-', '')}({s.get('score', 0):+d})"
            for t, s in new_status.items()
            if s["action"] in ("보유중", "BUY")
        )
        self.last_reason = (
            f"보유 {len(held_coins)}개/{self.max_coins}개 | "
            f"스캔 {len(new_status)}개 코인 완료"
            + (f" | 보유: {held_summary}" if held_summary else "")
        )

    def _execute_buy_ticker(self, ticker, signal):
        """특정 티커 시장가 매수. 성공 시 True 반환."""
        krw = self.get_krw_balance()
        invest = min(krw * self.invest_ratio, self.max_invest_amount)
        if invest < 5000:
            return False
        try:
            result = self.upbit.buy_market_order(ticker, invest)
            if result and "error" not in result:
                price = pyupbit.get_current_price(ticker) or 0
                self._add_log("매수", f"{ticker} 시장가 매수", invest, price,
                              signal, ticker=ticker)
                return True
            else:
                msg = (result.get("error", {}).get("message", "알 수 없음")
                       if result else "응답 없음")
                self._add_log("매수실패", msg, ticker=ticker)
        except Exception as e:
            self._add_log("매수실패", str(e), ticker=ticker)
        return False

    def _execute_sell_ticker(self, ticker, amount, signal):
        """특정 티커 시장가 매도. 성공 시 True 반환."""
        current_price = pyupbit.get_current_price(ticker)
        if current_price is None:
            return False
        if amount * current_price < 5000:
            return False
        try:
            result = self.upbit.sell_market_order(ticker, amount)
            if result and "error" not in result:
                self._add_log("매도", f"{ticker} 시장가 매도",
                              amount * current_price, current_price,
                              signal, ticker=ticker)
                return True
            else:
                msg = (result.get("error", {}).get("message", "알 수 없음")
                       if result else "응답 없음")
                self._add_log("매도실패", msg, ticker=ticker)
        except Exception as e:
            self._add_log("매도실패", str(e), ticker=ticker)
        return False