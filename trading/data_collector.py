import requests
import pyupbit


def get_fear_greed_index():
    """공포 & 탐욕 지수 (alternative.me - 무료)"""
    try:
        url = "https://api.alternative.me/fng/"
        params = {"limit": 30, "format": "json"}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json().get("data", [])
    except Exception:
        return []


def get_market_overview():
    """글로벌 시장 개요 (CoinGecko - 무료)"""
    try:
        url = "https://api.coingecko.com/api/v3/global"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json().get("data", {})
        return {
            "total_market_cap_krw": data.get("total_market_cap", {}).get("krw", 0),
            "total_volume_krw": data.get("total_volume", {}).get("krw", 0),
            "btc_dominance": data.get("market_cap_percentage", {}).get("btc", 0),
            "eth_dominance": data.get("market_cap_percentage", {}).get("eth", 0),
            "active_cryptos": data.get("active_cryptocurrencies", 0),
            "market_cap_change_24h": data.get("market_cap_change_percentage_24h_usd", 0),
        }
    except Exception:
        return {}


def get_top_coins(limit=10):
    """시가총액 상위 코인 (CoinGecko - 무료)"""
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "krw",
            "order": "market_cap_desc",
            "per_page": limit,
            "page": 1,
            "sparkline": "false",
            "price_change_percentage": "1h,24h,7d",
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception:
        return []


# ── 김치프리미엄 (Binance 무료 API) ──

def get_binance_price(symbol="BTCUSDT"):
    """바이낸스 현재가 조회 (무료, 키 불필요)"""
    try:
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return float(response.json()["price"])
    except Exception:
        return None


def get_usd_krw_rate():
    """USD/KRW 환율 조회 (무료)"""
    try:
        # 업비트 USDT 가격으로 대략적 환율 추정
        usdt_price = pyupbit.get_current_price("KRW-USDT")
        if usdt_price:
            return usdt_price
    except Exception:
        pass
    try:
        # 대체: exchangerate API
        url = "https://open.er-api.com/v6/latest/USD"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json()["rates"].get("KRW", 1350)
    except Exception:
        return 1350  # 기본값


def get_kimchi_premium(upbit_ticker="KRW-BTC", binance_symbol="BTCUSDT"):
    """김치프리미엄 계산 (%)
    업비트 가격 vs 바이낸스 가격 * 환율 차이
    양수 = 국내가 비쌈 (과열), 음수 = 국내가 쌈 (저평가)
    """
    try:
        upbit_price = pyupbit.get_current_price(upbit_ticker)
        binance_usd = get_binance_price(binance_symbol)
        usd_krw = get_usd_krw_rate()

        if not all([upbit_price, binance_usd, usd_krw]):
            return None

        binance_krw = binance_usd * usd_krw
        premium = (upbit_price - binance_krw) / binance_krw * 100
        return {
            "premium_pct": round(premium, 2),
            "upbit_price": upbit_price,
            "binance_krw": round(binance_krw, 0),
            "usd_krw": usd_krw,
        }
    except Exception:
        return None


# ── 호가창 분석 (pyupbit - 무료) ──

def get_orderbook_analysis(ticker="KRW-BTC"):
    """호가창 매수/매도 세력 분석
    - buy_ratio > 0.5: 매수세 우위
    - sell_ratio > 0.5: 매도세 우위
    - whale_detected: 대량 주문 감지
    """
    try:
        orderbook = pyupbit.get_orderbook(ticker)
        if not orderbook:
            return None

        ob = orderbook[0] if isinstance(orderbook, list) else orderbook
        units = ob.get("orderbook_units", [])
        if not units:
            return None

        total_bid = sum(u["bid_size"] for u in units)  # 매수 총량
        total_ask = sum(u["ask_size"] for u in units)  # 매도 총량
        total = total_bid + total_ask

        if total == 0:
            return None

        buy_ratio = total_bid / total
        sell_ratio = total_ask / total

        # 호가별 분석 (상위 5개)
        top_bids = sorted(units, key=lambda x: x["bid_size"], reverse=True)[:5]
        top_asks = sorted(units, key=lambda x: x["ask_size"], reverse=True)[:5]

        # 대량 주문 감지 (평균의 3배 이상)
        avg_bid = total_bid / len(units) if units else 0
        avg_ask = total_ask / len(units) if units else 0
        whale_bids = [u for u in units if u["bid_size"] > avg_bid * 3]
        whale_asks = [u for u in units if u["ask_size"] > avg_ask * 3]

        # 매수/매도 압력 (상위 3호가 vs 하위 호가 비중)
        top3_bid = sum(units[i]["bid_size"] for i in range(min(3, len(units))))
        top3_ask = sum(units[i]["ask_size"] for i in range(min(3, len(units))))

        return {
            "buy_ratio": round(buy_ratio, 3),
            "sell_ratio": round(sell_ratio, 3),
            "total_bid_volume": total_bid,
            "total_ask_volume": total_ask,
            "bid_ask_ratio": round(total_bid / total_ask, 2) if total_ask > 0 else 0,
            "top3_bid_pct": round(top3_bid / total_bid * 100, 1) if total_bid > 0 else 0,
            "top3_ask_pct": round(top3_ask / total_ask * 100, 1) if total_ask > 0 else 0,
            "whale_bid_count": len(whale_bids),
            "whale_ask_count": len(whale_asks),
            "spread_pct": round((units[0]["ask_price"] - units[0]["bid_price"]) / units[0]["ask_price"] * 100, 4) if units else 0,
        }
    except Exception:
        return None