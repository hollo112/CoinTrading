import streamlit as st
import pyupbit
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import time
from datetime import datetime, timedelta

from trading.strategy import TradingStrategy
from trading.data_collector import (
    get_fear_greed_index, get_market_overview, get_top_coins,
    get_kimchi_premium, get_orderbook_analysis,
)
from trading.trader import AutoTrader

# ================================================================
#  페이지 설정
# ================================================================
st.set_page_config(page_title="코인 자동매매", page_icon="chart_with_upwards_trend", layout="wide")

st.markdown("""
<style>
    .buy-tag  {background:#ff4444;color:#fff;padding:4px 12px;border-radius:6px;font-weight:700}
    .sell-tag {background:#4444ff;color:#fff;padding:4px 12px;border-radius:6px;font-weight:700}
    .hold-tag {background:#888;color:#fff;padding:4px 12px;border-radius:6px;font-weight:700}
    .profit {color:#00c853 !important;font-weight:700}
    .loss {color:#ff1744 !important;font-weight:700}
    .neutral-val {color:#9e9e9e}
    .badge {display:inline-block;padding:3px 10px;border-radius:4px;font-weight:700;font-size:0.85em}
    .badge-buy {background:#00c853;color:#fff}
    .badge-sell {background:#ff1744;color:#fff}
    .badge-hold {background:#78909c;color:#fff}
    .badge-stoploss {background:#d50000;color:#fff}
    .badge-takeprofit {background:#00bfa5;color:#fff}
    .badge-trailing {background:#ff6d00;color:#fff}
    .badge-error {background:#ff9800;color:#fff}
    .coin-table {width:100%;border-collapse:collapse;margin:10px 0}
    .coin-table th {text-align:left;padding:8px 12px;border-bottom:2px solid rgba(255,255,255,0.2);font-size:0.9em;color:#aaa}
    .coin-table td {padding:8px 12px;border-bottom:1px solid rgba(255,255,255,0.08);font-size:0.9em}
    .budget-info {background:rgba(255,255,255,0.05);border-radius:10px;padding:15px;margin:10px 0}
</style>
""", unsafe_allow_html=True)

# ================================================================
#  세션 상태 초기화
# ================================================================
if "trader" not in st.session_state:
    st.session_state.trader = None
if "connected" not in st.session_state:
    st.session_state.connected = False
if "auto_trading" not in st.session_state:
    st.session_state.auto_trading = False
if "chart_favorites" not in st.session_state:
    st.session_state.chart_favorites = ["KRW-BTC", "KRW-ETH"]

# ================================================================
#  캐시 함수
# ================================================================
@st.cache_data(ttl=300)
def cached_get_tickers():
    try:
        tickers = pyupbit.get_tickers(fiat="KRW")
        return tickers if tickers else ["KRW-BTC", "KRW-ETH", "KRW-XRP"]
    except Exception:
        return ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL", "KRW-DOGE"]

@st.cache_data(ttl=60)
def cached_get_ohlcv(ticker, interval, count=200):
    try:
        return pyupbit.get_ohlcv(ticker, interval=interval, count=count)
    except Exception:
        return None

@st.cache_data(ttl=600)
def cached_fear_greed():
    return get_fear_greed_index()

@st.cache_data(ttl=300)
def cached_market_overview():
    return get_market_overview()

@st.cache_data(ttl=300)
def cached_top_coins():
    return get_top_coins()

@st.cache_data(ttl=30)
def cached_kimchi_premium(upbit_ticker, binance_symbol):
    return get_kimchi_premium(upbit_ticker, binance_symbol)

@st.cache_data(ttl=15)
def cached_orderbook(ticker):
    return get_orderbook_analysis(ticker)

@st.cache_data(ttl=300, show_spinner="코인 스캔 중...")
def scan_coins():
    """주요 코인 전체 스캔 → 시그널 점수 순위"""
    popular = [
        "KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL", "KRW-DOGE",
        "KRW-ADA", "KRW-AVAX", "KRW-DOT", "KRW-LINK", "KRW-MATIC",
        "KRW-ATOM", "KRW-TRX", "KRW-ETC", "KRW-BCH", "KRW-NEAR",
        "KRW-APT", "KRW-ARB", "KRW-OP", "KRW-SUI", "KRW-SEI",
        "KRW-AAVE", "KRW-IMX", "KRW-STX", "KRW-SAND", "KRW-MANA",
    ]
    try:
        available = pyupbit.get_tickers(fiat="KRW")
    except Exception:
        available = []
    tickers = [t for t in popular if t in available]

    strategy = TradingStrategy()
    results = []

    for ticker in tickers:
        try:
            df = pyupbit.get_ohlcv(ticker, interval="minute60", count=100)
            if df is None or len(df) < 50:
                continue
            signal = strategy.get_signal(df)
            price = df.iloc[-1]["close"]
            change = 0.0
            if len(df) >= 24:
                change = (df.iloc[-1]["close"] - df.iloc[-24]["close"]) / df.iloc[-24]["close"] * 100
            vol_krw = df.iloc[-1]["close"] * df.iloc[-1]["volume"]

            buy_signals = [k for k, v in signal["signals"].items() if v[0] == "매수"]
            sell_signals = [k for k, v in signal["signals"].items() if v[0] == "매도"]

            results.append({
                "코인": ticker.replace("KRW-", ""),
                "ticker": ticker,
                "현재가": price,
                "24h변동": change,
                "스코어": signal["score"],
                "판단": signal["action"],
                "매수시그널": ", ".join(buy_signals) if buy_signals else "-",
                "매도시그널": ", ".join(sell_signals) if sell_signals else "-",
                "거래대금": vol_krw,
            })
            time.sleep(0.11)  # API 속도 제한 준수
        except Exception:
            continue

    results.sort(key=lambda x: x["스코어"], reverse=True)
    return results

# ================================================================
#  헬퍼 함수
# ================================================================
def format_krw(value):
    if value >= 1_0000_0000_0000:
        return f"{value / 1_0000_0000_0000:,.1f}조"
    if value >= 1_0000_0000:
        return f"{value / 1_0000_0000:,.1f}억"
    if value >= 1_0000:
        return f"{value / 1_0000:,.0f}만"
    return f"{value:,.0f}"


def colored_pnl(value_str):
    """수익률/수익금 문자열에 초록(+)/빨강(-) 색상 적용"""
    try:
        num = float(str(value_str).replace('%', '').replace('원', '').replace(',', '').replace('+', ''))
    except (ValueError, TypeError):
        return str(value_str)
    if num > 0:
        return f'<span class="profit">{value_str}</span>'
    elif num < 0:
        return f'<span class="loss">{value_str}</span>'
    return f'<span class="neutral-val">{value_str}</span>'


def type_badge(trade_type):
    """거래 유형에 색상 배지 적용"""
    badge_map = {
        "매수": "badge-buy", "매도": "badge-sell",
        "손절": "badge-stoploss", "익절": "badge-takeprofit",
        "트레일링": "badge-trailing", "매수실패": "badge-error",
        "매도실패": "badge-error", "ERROR": "badge-error",
    }
    css = badge_map.get(trade_type, "badge-hold")
    return f'<span class="badge {css}">{trade_type}</span>'


def load_trade_log_data():
    """세션 또는 파일에서 거래 로그를 로드."""
    if st.session_state.trader:
        return st.session_state.trader.get_trade_log() or []

    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_history.json")
    if not os.path.exists(log_file):
        return []
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def calculate_trade_performance(trade_log):
    """
    거래 로그 기반 실현 손익 계산(FIFO).
    - 매수: 원금(금액)과 수량(금액/가격) 누적
    - 매도/손절/익절/트레일링: FIFO로 원가 매칭 후 실현손익 계산
    """
    empty_df = pd.DataFrame()
    result = {
        "has_realized": False,
        "total_pnl": 0.0,
        "total_cost": 0.0,
        "total_proceeds": 0.0,
        "total_return_pct": 0.0,
        "today_pnl": 0.0,
        "month_pnl": 0.0,
        "daily": empty_df,
        "monthly": empty_df,
    }
    if not trade_log:
        return result

    sell_types = {"매도", "손절", "익절", "트레일링"}
    positions = {}  # {ticker: [{"qty": float, "cost": float}, ...]}
    realized_rows = []
    eps = 1e-12

    rows = []
    for t in trade_log:
        ts = pd.to_datetime(t.get("time"), errors="coerce")
        if pd.isna(ts):
            continue
        rows.append({
            "time": ts,
            "type": t.get("type"),
            "ticker": t.get("ticker", ""),
            "amount": float(t.get("amount", 0) or 0),
            "price": float(t.get("price", 0) or 0),
        })
    if not rows:
        return result

    df_log = pd.DataFrame(rows).sort_values("time")
    for row in df_log.itertuples(index=False):
        trade_type = row.type
        amount = row.amount
        price = row.price
        ticker = row.ticker
        if amount <= 0 or price <= 0:
            continue

        qty = amount / price
        if qty <= eps:
            continue

        lots = positions.setdefault(ticker, [])
        if trade_type == "매수":
            lots.append({"qty": qty, "cost": amount})
            continue

        if trade_type not in sell_types:
            continue

        original_qty = qty
        matched_qty = 0.0
        matched_cost = 0.0

        while qty > eps and lots:
            lot = lots[0]
            lot_qty = lot["qty"]
            if lot_qty <= eps:
                lots.pop(0)
                continue

            take_qty = min(qty, lot_qty)
            unit_cost = (lot["cost"] / lot_qty) if lot_qty > eps else 0.0
            take_cost = unit_cost * take_qty

            lot["qty"] -= take_qty
            lot["cost"] -= take_cost
            qty -= take_qty
            matched_qty += take_qty
            matched_cost += take_cost

            if lot["qty"] <= eps:
                lots.pop(0)

        if matched_qty <= eps:
            continue

        matched_ratio = matched_qty / original_qty
        proceeds = amount * matched_ratio
        pnl = proceeds - matched_cost
        realized_rows.append({
            "date": row.time.strftime("%Y-%m-%d"),
            "month": row.time.strftime("%Y-%m"),
            "pnl": pnl,
            "cost": matched_cost,
            "proceeds": proceeds,
            "trades": 1,
        })

    if not realized_rows:
        return result

    df_realized = pd.DataFrame(realized_rows)
    total_pnl = float(df_realized["pnl"].sum())
    total_cost = float(df_realized["cost"].sum())
    total_proceeds = float(df_realized["proceeds"].sum())
    total_return_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0.0

    daily = (
        df_realized.groupby("date", as_index=False)
        .agg({"pnl": "sum", "cost": "sum", "proceeds": "sum", "trades": "sum"})
        .sort_values("date", ascending=False)
    )
    daily["return_pct"] = daily.apply(
        lambda r: (r["pnl"] / r["cost"] * 100) if r["cost"] > 0 else 0.0,
        axis=1,
    )

    monthly = (
        df_realized.groupby("month", as_index=False)
        .agg({"pnl": "sum", "cost": "sum", "proceeds": "sum", "trades": "sum"})
        .sort_values("month", ascending=False)
    )
    monthly["return_pct"] = monthly.apply(
        lambda r: (r["pnl"] / r["cost"] * 100) if r["cost"] > 0 else 0.0,
        axis=1,
    )

    today_key = datetime.now().strftime("%Y-%m-%d")
    month_key = datetime.now().strftime("%Y-%m")
    today_pnl = float(daily.loc[daily["date"] == today_key, "pnl"].sum())
    month_pnl = float(monthly.loc[monthly["month"] == month_key, "pnl"].sum())

    result.update({
        "has_realized": True,
        "total_pnl": total_pnl,
        "total_cost": total_cost,
        "total_proceeds": total_proceeds,
        "total_return_pct": total_return_pct,
        "today_pnl": today_pnl,
        "month_pnl": month_pnl,
        "daily": daily,
        "monthly": monthly,
    })
    return result


def create_chart(df, strategy):
    """캔들스틱 + 볼린저 + RSI + MACD + 거래량 차트"""
    df = strategy.add_indicators(df)

    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.025,
        subplot_titles=("가격 / 볼린저밴드", "RSI", "MACD", "ADX (추세 강도)", "거래량"),
        row_heights=[0.40, 0.12, 0.12, 0.12, 0.16],
    )

    # 캔들스틱 (한국식: 빨강=상승, 파랑=하락)
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="가격",
        increasing_line_color="#ff4444", decreasing_line_color="#4444ff",
        increasing_fillcolor="#ff4444", decreasing_fillcolor="#4444ff",
    ), row=1, col=1)

    # 볼린저밴드
    fig.add_trace(go.Scatter(
        x=df.index, y=df["bb_upper"], name="BB 상단",
        line=dict(color="rgba(255,165,0,0.4)", width=1),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["bb_lower"], name="BB 하단",
        line=dict(color="rgba(255,165,0,0.4)", width=1),
        fill="tonexty", fillcolor="rgba(255,165,0,0.08)",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["bb_mid"], name="BB 중간",
        line=dict(color="rgba(255,165,0,0.6)", width=1, dash="dash"),
    ), row=1, col=1)

    # 이동평균
    fig.add_trace(go.Scatter(
        x=df.index, y=df["ma_short"], name=f"MA{strategy.ma_short}",
        line=dict(color="#ff6b6b", width=1.5),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["ma_long"], name=f"MA{strategy.ma_long}",
        line=dict(color="#4ecdc4", width=1.5),
    ), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=df.index, y=df["rsi"], name="RSI",
        line=dict(color="#9b59b6", width=1.5),
    ), row=2, col=1)
    fig.add_hline(y=strategy.rsi_overbought, line_dash="dash", line_color="red",
                  annotation_text=f"과매수 ({strategy.rsi_overbought})", row=2, col=1)
    fig.add_hline(y=strategy.rsi_oversold, line_dash="dash", line_color="blue",
                  annotation_text=f"과매도 ({strategy.rsi_oversold})", row=2, col=1)

    # MACD
    hist_colors = ["#ff4444" if v >= 0 else "#4444ff" for v in df["macd_hist"].fillna(0)]
    fig.add_trace(go.Bar(
        x=df.index, y=df["macd_hist"], name="MACD Hist", marker_color=hist_colors,
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["macd"], name="MACD",
        line=dict(color="#3498db", width=1.5),
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["macd_signal"], name="Signal",
        line=dict(color="#e74c3c", width=1.5),
    ), row=3, col=1)

    # ADX
    fig.add_trace(go.Scatter(
        x=df.index, y=df["adx"], name="ADX",
        line=dict(color="#f1c40f", width=1.5),
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["plus_di"], name="+DI",
        line=dict(color="#2ecc71", width=1),
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["minus_di"], name="-DI",
        line=dict(color="#e74c3c", width=1),
    ), row=4, col=1)
    fig.add_hline(y=strategy.adx_threshold, line_dash="dash", line_color="gray",
                  annotation_text=f"추세 기준 ({strategy.adx_threshold})", row=4, col=1)

    # 거래량 (급증 구간 강조)
    vol_colors = [
        "#ff4444" if df["close"].iloc[i] >= df["open"].iloc[i] else "#4444ff"
        for i in range(len(df))
    ]
    # 거래량 급증 시 색상 강조
    if "vol_ratio" in df.columns:
        for i in range(len(df)):
            if pd.notna(df["vol_ratio"].iloc[i]) and df["vol_ratio"].iloc[i] >= strategy.vol_spike_multiplier:
                vol_colors[i] = "#ffff00"  # 노란색 = 급증
    fig.add_trace(go.Bar(
        x=df.index, y=df["volume"], name="거래량",
        marker_color=vol_colors, opacity=0.7,
    ), row=5, col=1)

    fig.update_layout(
        height=950, xaxis_rangeslider_visible=False, template="plotly_dark",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=30),
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    return fig


def create_fear_greed_gauge(value, classification):
    """공포탐욕 게이지 차트"""
    color_map = {
        "Extreme Fear": "#ff4444", "Fear": "#ff8844",
        "Neutral": "#ffdd44", "Greed": "#88dd44", "Extreme Greed": "#44dd44",
    }
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "공포 & 탐욕 지수"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color_map.get(classification, "#888")},
            "steps": [
                {"range": [0, 25], "color": "rgba(255,68,68,0.25)"},
                {"range": [25, 45], "color": "rgba(255,136,68,0.25)"},
                {"range": [45, 55], "color": "rgba(255,221,68,0.25)"},
                {"range": [55, 75], "color": "rgba(136,221,68,0.25)"},
                {"range": [75, 100], "color": "rgba(68,221,68,0.25)"},
            ],
            "threshold": {"line": {"color": "white", "width": 4}, "thickness": 0.75, "value": value},
        },
    ))
    fig.update_layout(height=300, template="plotly_dark", margin=dict(l=20, r=20, t=60, b=20))
    return fig

# ================================================================
#  기본 전략 파라미터 (연결 전에도 사용)
# ================================================================
strategy_params = {
    "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70,
    "bb_period": 20, "bb_std": 2.0,
    "ma_short": 5, "ma_long": 20,
    "adx_period": 14, "adx_threshold": 25,
    "vol_spike_multiplier": 2.0,
    "stop_loss_pct": 3.0, "take_profit_pct": 5.0,
}

# ================================================================
#  사이드바
# ================================================================
with st.sidebar:
    st.title("코인 자동매매 봇")
    st.divider()

    # ── API 연결 ──
    st.subheader("업비트 API 연결")
    access_key = st.text_input("Access Key", type="password", key="ak")
    secret_key = st.text_input("Secret Key", type="password", key="sk")

    col_conn1, col_conn2 = st.columns(2)
    with col_conn1:
        connect_btn = st.button("연결", type="primary", use_container_width=True)
    with col_conn2:
        disconnect_btn = st.button("연결 해제", use_container_width=True)

    if connect_btn:
        if not access_key or not secret_key:
            st.error("Access Key / Secret Key를 모두 입력해주세요.")
        else:
            try:
                trader = AutoTrader(access_key, secret_key)
                # 예외를 삼키는 래퍼 대신 원본 응답으로 연결 유효성 검증
                raw_balances = trader.upbit.get_balances()
                if not isinstance(raw_balances, list):
                    error_msg = "잔고 조회 응답이 비정상입니다."
                    if isinstance(raw_balances, dict):
                        error_msg = raw_balances.get("error", {}).get("message", error_msg)
                    raise ValueError(error_msg)
                if raw_balances and isinstance(raw_balances[0], dict) and "error" in raw_balances[0]:
                    raise ValueError(raw_balances[0].get("error", {}).get("message", "API 인증 실패"))

                st.session_state.trader = trader
                st.session_state.connected = True
                st.success("연결 성공!")
            except Exception as e:
                st.session_state.trader = None
                st.session_state.connected = False
                st.session_state.auto_trading = False
                st.error(f"연결 실패: {e}")

    if disconnect_btn:
        if st.session_state.trader:
            st.session_state.trader.cleanup()
        st.session_state.trader = None
        st.session_state.connected = False
        st.session_state.auto_trading = False
        st.rerun()

    if st.session_state.connected:
        st.success("연결됨")
    else:
        st.info("API 키를 입력하고 연결하세요")
        st.caption("업비트 > 마이페이지 > Open API 관리\n에서 키를 발급받을 수 있습니다.")

    st.divider()

    # ── 매매 모드 선택 ──
    st.subheader("매매 모드")
    trade_mode = st.radio("매매 모드", ["단일 코인", "멀티코인"],
                          horizontal=True, key="trade_mode")

    # ── 코인 / 차트 설정 ──
    st.subheader("코인 / 차트 설정")
    tickers = cached_get_tickers()
    ticker_names = {t: t.replace("KRW-", "") for t in tickers}

    if trade_mode == "단일 코인":
        coin_search = st.text_input("코인 검색", placeholder="BTC, ETH, XRP...", key="coin_search")
        if coin_search:
            filtered = [t for t in tickers if coin_search.upper() in t.upper()]
        else:
            filtered = tickers

        if not filtered:
            st.warning(f"'{coin_search}' 검색 결과가 없습니다.")
            filtered = tickers

        default_idx = filtered.index("KRW-BTC") if "KRW-BTC" in filtered else 0
        selected_ticker = st.selectbox(
            "코인 선택", filtered,
            format_func=lambda x: ticker_names.get(x, x), index=default_idx,
        )
    else:
        st.info("멀티코인 모드: 25개 주요 코인을 자동 스캔합니다.")
        max_coins = st.slider("동시 보유 최대 코인 수", 1, 10, 5, key="max_coins")
        selected_ticker = "KRW-BTC"  # 차트 표시용 기본값

    interval_options = {
        "1분": "minute1", "3분": "minute3", "5분": "minute5",
        "15분": "minute15", "30분": "minute30", "1시간": "minute60",
        "4시간": "minute240", "일봉": "day", "주봉": "week",
    }
    selected_interval_name = st.selectbox("차트 주기", list(interval_options.keys()), index=5)
    selected_interval = interval_options[selected_interval_name]

    # ── 매매 설정 (연결 시에만) ──
    if st.session_state.connected:
        st.divider()
        st.subheader("매매 설정")

        invest_ratio = st.slider("투자 비율 (%)", 1, 100, 10, 1) / 100
        max_invest = st.number_input("1회 최대 투자금 (원)", 5000, 10_000_000, 100_000, 10_000)
        max_budget = st.number_input(
            "총 운용자금 한도 (원)", 0, 1_000_000_000, 0, 100_000,
            help="0 = 제한 없음. 계좌 잔고와 관계없이 이 금액 내에서만 매매합니다.",
        )
        check_interval = st.number_input("확인 주기 (초)", 30, 3600, 60, 30)

        with st.expander("전략 파라미터 조정"):
            st.caption("기본 지표")
            rsi_period = st.number_input("RSI 기간", 5, 50, 14)
            rsi_oversold = st.number_input("RSI 과매도", 10, 50, 30)
            rsi_overbought = st.number_input("RSI 과매수", 50, 95, 70)
            bb_period = st.number_input("볼린저밴드 기간", 5, 50, 20)
            bb_std = st.number_input("볼린저밴드 표준편차", 1.0, 3.0, 2.0, 0.1)
            ma_short = st.number_input("단기 이동평균", 3, 50, 5)
            ma_long = st.number_input("장기 이동평균", 10, 200, 20)

            st.caption("추세 / 거래량")
            adx_period = st.number_input("ADX 기간", 7, 50, 14)
            adx_threshold = st.number_input("ADX 추세 기준", 15, 40, 25)
            vol_spike = st.number_input("거래량 급증 배수", 1.5, 5.0, 2.0, 0.5)

            st.caption("리스크 관리")
            stop_loss = st.number_input("손절 (%)", 1.0, 20.0, 3.0, 0.5)
            take_profit = st.number_input("익절 (%)", 1.0, 50.0, 5.0, 0.5)

        strategy_params = {
            "rsi_period": rsi_period, "rsi_oversold": rsi_oversold,
            "rsi_overbought": rsi_overbought, "bb_period": bb_period,
            "bb_std": bb_std, "ma_short": ma_short, "ma_long": ma_long,
            "adx_period": adx_period, "adx_threshold": adx_threshold,
            "vol_spike_multiplier": vol_spike,
            "stop_loss_pct": stop_loss, "take_profit_pct": take_profit,
        }

        st.divider()

        # 자동매매 시작 / 중지
        if not st.session_state.auto_trading:
            if st.button("자동매매 시작", type="primary", use_container_width=True):
                if trade_mode == "멀티코인":
                    st.session_state.trader.start_multi(
                        interval=selected_interval,
                        check_interval=check_interval,
                        invest_ratio=invest_ratio,
                        max_invest_amount=max_invest,
                        max_coins=max_coins,
                        strategy_params=strategy_params,
                        max_total_budget=max_budget,
                    )
                else:
                    st.session_state.trader.start(
                        ticker=selected_ticker, interval=selected_interval,
                        check_interval=check_interval, invest_ratio=invest_ratio,
                        max_invest_amount=max_invest,
                        strategy_params=strategy_params,
                        max_total_budget=max_budget,
                    )
                st.session_state.auto_trading = True
                st.rerun()
        else:
            mode_label = "멀티코인" if st.session_state.trader.multi_mode else st.session_state.trader.ticker
            st.warning(f"자동매매 실행 중 ({mode_label})")
            if st.button("자동매매 중지", use_container_width=True):
                st.session_state.trader.stop()
                st.session_state.auto_trading = False
                st.rerun()

    st.divider()
    if st.button("새로고침", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ================================================================
#  메인 콘텐츠
# ================================================================
st.title("코인 자동매매 대시보드")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["대시보드", "차트 분석", "거래 내역", "시장 현황", "코인 스캐너"])

# ──────────────────────────────────────────────
#  탭 1 : 대시보드
# ──────────────────────────────────────────────
with tab1:
    if st.session_state.connected:
        trader = st.session_state.trader

        # 잔고 요약
        krw_balance = trader.get_krw_balance()
        balances = trader.get_balances()

        total_coin_value = 0.0
        total_buy_amount = 0.0
        coin_holdings = []
        unresolved_price_count = 0
        holding_rows = []
        ticker_keys = []

        for b in balances:
            if b["currency"] == "KRW":
                continue
            total_amt = float(b.get("balance", 0)) + float(b.get("locked", 0))
            if total_amt <= 0:
                continue
            avg_price = float(b.get("avg_buy_price", 0) or 0)
            ticker_key = f"KRW-{b['currency']}"
            holding_rows.append({
                "coin": b["currency"],
                "ticker": ticker_key,
                "amount": total_amt,
                "avg_price": avg_price,
            })
            ticker_keys.append(ticker_key)

        price_map = {}
        if ticker_keys:
            try:
                batch_prices = pyupbit.get_current_price(ticker_keys)
                if isinstance(batch_prices, dict):
                    for tk, pv in batch_prices.items():
                        if pv is not None:
                            price_map[tk] = float(pv)
                elif len(ticker_keys) == 1 and isinstance(batch_prices, (int, float)):
                    price_map[ticker_keys[0]] = float(batch_prices)
            except Exception:
                price_map = {}

        for row in holding_rows:
            total_amt = row["amount"]
            avg_price = row["avg_price"]
            ticker_key = row["ticker"]
            cur_price = price_map.get(ticker_key)
            if cur_price is None:
                try:
                    single_price = pyupbit.get_current_price(ticker_key)
                    if single_price is not None:
                        cur_price = float(single_price)
                except Exception:
                    cur_price = None

            buy_amt = total_amt * avg_price if avg_price > 0 else 0.0
            if cur_price is not None and cur_price > 0:
                eval_amt = total_amt * cur_price
                profit = eval_amt - buy_amt if buy_amt > 0 else 0.0
                profit_pct = (profit / buy_amt * 100) if buy_amt > 0 else 0.0
                total_coin_value += eval_amt
                total_buy_amount += buy_amt
                cur_price_text = f"{cur_price:,.0f}원"
                eval_amt_text = f"{eval_amt:,.0f}원"
                profit_pct_text = f"{profit_pct:+.2f}%"
                profit_text = f"{profit:+,.0f}원"
                sort_value = eval_amt
            else:
                unresolved_price_count += 1
                cur_price_text = "조회 실패"
                eval_amt_text = "-"
                profit_pct_text = "-"
                profit_text = "-"
                sort_value = buy_amt

            coin_holdings.append({
                "코인": row["coin"],
                "보유량": round(total_amt, 8),
                "매수평균가": f"{avg_price:,.0f}원" if avg_price > 0 else "-",
                "현재가": cur_price_text,
                "평가금액": eval_amt_text,
                "수익률": profit_pct_text,
                "수익금": profit_text,
                "_sort_value": sort_value,
                "_amount": total_amt,
                "_avg_price": avg_price,
                "_current_price": cur_price if cur_price is not None else 0.0,
                "_eval_amt": eval_amt if cur_price is not None else 0.0,
                "_profit_pct": profit_pct if cur_price is not None and buy_amt > 0 else 0.0,
                "_profit": profit if cur_price is not None and buy_amt > 0 else 0.0,
                "_price_missing": cur_price is None,
            })

        coin_holdings.sort(key=lambda x: x["_sort_value"], reverse=True)

        total_assets = krw_balance + total_coin_value
        unrealized_profit = total_coin_value - total_buy_amount
        unrealized_return = (unrealized_profit / total_buy_amount * 100) if total_buy_amount > 0 else 0.0

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("KRW 잔고", f"{krw_balance:,.0f}원")
        mc2.metric("코인 평가액", f"{total_coin_value:,.0f}원")
        mc3.metric("총 자산", f"{total_assets:,.0f}원")
        mc4.metric("미실현 손익", f"{unrealized_profit:+,.0f}원", f"{unrealized_return:+.2f}%")
        if unresolved_price_count > 0:
            st.caption(
                f"현재가 조회 실패 코인 {unresolved_price_count}개는 보유 목록에는 표시되지만 "
                "평가금액/손익 계산에서는 제외됩니다."
            )

        # 운용자금 한도 표시
        if trader.max_total_budget > 0:
            budget = trader.max_total_budget
            invested = trader._get_total_invested()
            available = trader.get_available_krw()
            usage_pct = min(invested / budget, 1.0) if budget > 0 else 0
            bar_color = "#00c853" if usage_pct < 0.7 else "#ff9800" if usage_pct < 0.9 else "#ff1744"
            st.markdown(
                f'<div class="budget-info">'
                f'<strong>운용자금 한도:</strong> {budget:,.0f}원 &nbsp;|&nbsp; '
                f'<strong>투자 중:</strong> {colored_pnl(f"{invested:,.0f}원") if invested > 0 else "0원"} &nbsp;|&nbsp; '
                f'<strong>추가 매수 가능:</strong> {available:,.0f}원'
                f'<div class="budget-bar" style="margin-top:8px">'
                f'<div class="budget-fill" style="background:{bar_color};width:{max(usage_pct * 100, 1):.0f}%">'
                f'{usage_pct:.0%} 사용</div></div></div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # 보유 코인
        if coin_holdings:
            st.subheader("보유 코인 현황")
            html = '<table class="coin-table"><tr>'
            for header in ["코인", "보유량", "매수평균가", "현재가", "평가금액", "수익률", "수익금"]:
                html += f'<th>{header}</th>'
            html += '</tr>'
            for h in coin_holdings:
                html += (
                    f'<tr><td><strong>{h["코인"]}</strong></td>'
                    f'<td>{h["보유량"]}</td>'
                    f'<td>{h["매수평균가"]}</td>'
                    f'<td>{h["현재가"]}</td>'
                    f'<td>{h["평가금액"]}</td>'
                    f'<td>{colored_pnl(h["수익률"])}</td>'
                    f'<td>{colored_pnl(h["수익금"])}</td></tr>'
                )
            html += '</table>'
            st.markdown(html, unsafe_allow_html=True)

            st.markdown("**보유 코인 빠른 주문**")
            allow_manual_fast_order = True
            if st.session_state.auto_trading:
                st.warning("자동매매 실행 중입니다. 빠른 주문은 봇 동작과 충돌할 수 있습니다.")
                allow_manual_fast_order = st.checkbox(
                    "자동매매 중에도 빠른 주문 허용",
                    value=False,
                    key="allow_manual_fast_order",
                )

            quick_buy_krw = st.number_input(
                "추가 매수 금액 (원)",
                min_value=5000,
                max_value=1_000_000_000,
                value=10000,
                step=5000,
                key="holding_quick_buy_krw",
                disabled=not allow_manual_fast_order,
            )
            confirm_sell_all = st.checkbox(
                "전량 매도 확인 (체크 시 '바로 매도' 버튼 활성화)",
                value=False,
                key="confirm_sell_all_holdings",
                disabled=not allow_manual_fast_order,
            )

            for h in coin_holdings:
                coin = h["코인"]
                ticker = f"KRW-{coin}"
                qc1, qc2, qc3 = st.columns([2, 1, 1])
                with qc1:
                    st.caption(f"{coin} | 보유량 {h['보유량']}개")
                with qc2:
                    if st.button(
                        f"{coin} 추가 매수",
                        use_container_width=True,
                        key=f"holding_buy_{coin}",
                        disabled=not allow_manual_fast_order,
                    ):
                        ok, msg = trader.manual_buy(ticker, quick_buy_krw)
                        if ok:
                            st.success(f"{coin} 추가 매수 완료")
                            st.rerun()
                        st.error(f"{coin} 매수 실패: {msg}")
                with qc3:
                    if st.button(
                        f"{coin} 바로 매도",
                        use_container_width=True,
                        key=f"holding_sell_{coin}",
                        disabled=(not allow_manual_fast_order) or (not confirm_sell_all),
                    ):
                        ok, msg = trader.manual_sell(ticker, 1.0)
                        if ok:
                            st.success(f"{coin} 전량 매도 완료")
                            st.rerun()
                        st.error(f"{coin} 매도 실패: {msg}")
        else:
            st.info("보유 중인 코인이 없습니다.")

        st.divider()

        # 현재 시그널
        st.subheader("현재 매매 시그널 분석")
        df_ohlcv = cached_get_ohlcv(selected_ticker, selected_interval)
        if df_ohlcv is not None and len(df_ohlcv) > 50:
            strategy = TradingStrategy(**strategy_params)
            # 공포탐욕지수 주입
            fng_data = cached_fear_greed()
            if fng_data:
                strategy.fear_greed_value = int(fng_data[0].get("value", 50))
            signal = strategy.get_signal(df_ohlcv)

            tag_map = {"BUY": ("매수", "buy-tag"), "SELL": ("매도", "sell-tag"), "HOLD": ("관망", "hold-tag")}
            txt, css = tag_map.get(signal["action"], ("관망", "hold-tag"))
            buy_th = signal.get("buy_threshold", 2)
            sell_th = signal.get("sell_threshold", -2)
            st.markdown(
                f'종합 판단: <span class="{css}">{txt}</span> '
                f'(스코어: {signal["score"]:+d} / 매수 {buy_th}+ 매도 {sell_th}-)',
                unsafe_allow_html=True,
            )

            # 시그널 카드 표시 (4열씩)
            sig_items = list(signal["signals"].items())
            for row_start in range(0, len(sig_items), 4):
                row = sig_items[row_start:row_start + 4]
                sig_cols = st.columns(len(row))
                for i, (name, (direction, detail)) in enumerate(row):
                    with sig_cols[i]:
                        st.markdown(f"**{name}**")
                        color = "red" if direction == "매수" else "blue" if direction == "매도" else "gray"
                        st.markdown(f":{color}[{direction}]")
                        st.caption(detail)

        # 봇 상태
        if st.session_state.auto_trading:
            st.divider()
            st.subheader("봇 상태")
            state = trader.get_state_snapshot()
            bc1, bc2, bc3, bc4 = st.columns(4)
            bc1.metric("상태", state.get("status") or "-")
            bc2.metric("마지막 확인", state.get("last_check_time") or "-")
            bc3.metric("체크 횟수", f"{state.get('check_count', 0)}회")
            bc4.metric("총 거래", f"{len(trader.get_trade_log())}건")

            # 봇의 실제 판단 표시
            if state.get("last_reason"):
                st.info(f"봇 판단: {state.get('last_reason')}")

            # ── 멀티코인 모드 상태 표시 ──
            multi_status = state.get("multi_status") or {}
            if state.get("multi_mode") and multi_status:
                st.subheader("멀티코인 스캔 결과")

                # 보유 코인 수익률
                held = trader.get_held_coins()
                if held:
                    st.markdown("**보유 코인**")
                    html = (
                        '<div style="max-height:420px;overflow-y:auto;border:1px solid rgba(255,255,255,0.12);'
                        'border-radius:8px;padding:4px 0">'
                        '<table class="coin-table"><tr>'
                    )
                    for header in ["코인", "수익률", "평가금액", "스코어", "상태"]:
                        html += f'<th>{header}</th>'
                    html += '</tr>'
                    for t, info in held.items():
                        try:
                            cur_p = pyupbit.get_current_price(t)
                            pnl = ((cur_p - info["avg_price"]) / info["avg_price"] * 100) if cur_p and info["avg_price"] > 0 else 0
                            eval_amt = info["amount"] * cur_p if cur_p else 0
                        except Exception:
                            pnl = 0
                            eval_amt = 0
                        status_info = multi_status.get(t, {})
                        score = status_info.get("score", "-")
                        score_html = colored_pnl(f"{score:+d}") if isinstance(score, int) else str(score)
                        html += (
                            f'<tr><td><strong>{t.replace("KRW-", "")}</strong></td>'
                            f'<td>{colored_pnl(f"{pnl:+.2f}%")}</td>'
                            f'<td>{eval_amt:,.0f}원</td>'
                            f'<td>{score_html}</td>'
                            f'<td>{status_info.get("reason", "-")}</td></tr>'
                        )
                    html += '</table></div>'
                    st.markdown(html, unsafe_allow_html=True)

                # 전체 스캔 결과 테이블
                with st.expander("전체 스캔 결과 (클릭하여 펼치기)"):
                    scan_rows = []
                    for t, s in multi_status.items():
                        action_display = {
                            "BUY": "매수", "SELL": "매도", "HOLD": "관망",
                            "BUY 후보": "매수 후보", "보유중": "보유중",
                            "SKIP": "건너뜀", "ERROR": "에러",
                        }.get(s["action"], s["action"])
                        score_val = s["score"]
                        scan_rows.append({
                            "코인": t.replace("KRW-", ""),
                            "판단": action_display,
                            "score_num": score_val if isinstance(score_val, int) else 0,
                            "스코어": f"{score_val:+d}" if isinstance(score_val, int) else str(score_val),
                            "사유": s["reason"],
                        })
                    scan_rows.sort(key=lambda x: x["score_num"], reverse=True)
                    html = '<table class="coin-table"><tr><th>코인</th><th>판단</th><th>스코어</th><th>사유</th></tr>'
                    for r in scan_rows:
                        html += (
                            f'<tr><td><strong>{r["코인"]}</strong></td>'
                            f'<td>{type_badge(r["판단"])}</td>'
                            f'<td>{colored_pnl(r["스코어"])}</td>'
                            f'<td>{r["사유"]}</td></tr>'
                        )
                    html += '</table>'
                    st.markdown(html, unsafe_allow_html=True)

            # ── 단일 코인 모드 시그널 상세 ──
            elif not state.get("multi_mode") and state.get("last_signal"):
                bot_sig = state.get("last_signal")
                bot_tag_map = {"BUY": ("매수", "buy-tag"), "SELL": ("매도", "sell-tag"), "HOLD": ("관망", "hold-tag")}
                bot_txt, bot_css = bot_tag_map.get(bot_sig["action"], ("관망", "hold-tag"))
                bot_buy_th = bot_sig.get("buy_threshold", 3)
                bot_sell_th = bot_sig.get("sell_threshold", -3)
                st.markdown(
                    f'봇 시그널: <span class="{bot_css}">{bot_txt}</span> '
                    f'(스코어: {bot_sig["score"]:+d} / 매수 {bot_buy_th}+ 매도 {bot_sell_th}-)',
                    unsafe_allow_html=True,
                )
                # 봇의 개별 시그널 중 매수/매도만 표시
                active_sigs = {k: v for k, v in bot_sig["signals"].items() if v[0] != "중립"}
                if active_sigs:
                    cols = st.columns(min(len(active_sigs), 4))
                    for i, (name, (direction, detail)) in enumerate(active_sigs.items()):
                        with cols[i % len(cols)]:
                            color = "red" if direction == "매수" else "blue"
                            st.markdown(f"**{name}**: :{color}[{direction}] {detail}")
    else:
        st.info("사이드바에서 업비트 API 키를 입력하고 연결하세요.")
        st.markdown("""
### 시작하기
1. **업비트 API 키 발급** - 업비트 > 마이페이지 > Open API 관리
2. **허용 IP 설정** - API 키 생성 시 본인 IP를 허용 목록에 추가
3. **Access Key / Secret Key 입력** - 사이드바에 입력 후 연결
4. **코인 선택 & 전략 설정** - 원하는 코인과 매매 전략 설정
5. **자동매매 시작** - 설정한 전략에 따라 자동 매매
        """)

# ──────────────────────────────────────────────
#  탭 2 : 차트 분석
# ──────────────────────────────────────────────
with tab2:
    chart_tickers = cached_get_tickers()
    chart_ticker_names = {t: t.replace("KRW-", "") for t in chart_tickers}
    chart_default = selected_ticker if selected_ticker in chart_tickers else "KRW-BTC"
    if chart_default not in chart_tickers and chart_tickers:
        chart_default = chart_tickers[0]

    chart_ctrl1, chart_ctrl2, chart_ctrl3 = st.columns([2, 3, 3])
    with chart_ctrl1:
        chart_coin_search = st.text_input(
            "차트 코인 검색",
            placeholder="BTC, ETH, XRP...",
            key="chart_coin_search",
        )
    with chart_ctrl2:
        if chart_coin_search:
            chart_filtered = [t for t in chart_tickers if chart_coin_search.upper() in t.upper()]
        else:
            chart_filtered = chart_tickers
        if not chart_filtered:
            chart_filtered = chart_tickers

        chart_idx = chart_filtered.index(chart_default) if chart_default in chart_filtered else 0
        chart_selected_ticker = st.selectbox(
            "차트 코인 선택",
            chart_filtered,
            index=chart_idx,
            format_func=lambda x: chart_ticker_names.get(x, x),
            key="chart_selected_ticker",
        )
    with chart_ctrl3:
        fav_col1, fav_col2 = st.columns(2)
        with fav_col1:
            if st.button("즐겨찾기 추가", use_container_width=True, key="chart_fav_add"):
                if chart_selected_ticker not in st.session_state.chart_favorites:
                    st.session_state.chart_favorites.append(chart_selected_ticker)
                st.rerun()
        with fav_col2:
            if st.button("즐겨찾기 제거", use_container_width=True, key="chart_fav_remove"):
                st.session_state.chart_favorites = [
                    t for t in st.session_state.chart_favorites if t != chart_selected_ticker
                ]
                st.rerun()

    valid_favorites = [t for t in st.session_state.chart_favorites if t in chart_tickers]
    if valid_favorites:
        fav_default_idx = (
            valid_favorites.index(chart_selected_ticker)
            if chart_selected_ticker in valid_favorites else 0
        )
        fav_ticker = st.selectbox(
            "즐겨찾기 바로가기",
            valid_favorites,
            index=fav_default_idx,
            format_func=lambda x: chart_ticker_names.get(x, x),
            key="chart_favorite_quick_pick",
        )
        if fav_ticker != chart_selected_ticker:
            chart_selected_ticker = fav_ticker

    st.subheader(f"{chart_ticker_names.get(chart_selected_ticker, chart_selected_ticker)} {selected_interval_name} 차트")

    chart_strategy = TradingStrategy(**strategy_params)
    df_chart = cached_get_ohlcv(chart_selected_ticker, selected_interval)

    if df_chart is not None and len(df_chart) > 0:
        fig = create_chart(df_chart, chart_strategy)
        st.plotly_chart(fig, use_container_width=True)

        latest = df_chart.iloc[-1]
        prev_close = df_chart.iloc[-2]["close"] if len(df_chart) > 1 else latest["close"]
        change_pct = ((latest["close"] - prev_close) / prev_close * 100) if prev_close else 0

        pc1, pc2, pc3, pc4 = st.columns(4)
        pc1.metric("현재가", f"{latest['close']:,.0f}원", f"{change_pct:+.2f}%")
        pc2.metric("고가", f"{latest['high']:,.0f}원")
        pc3.metric("저가", f"{latest['low']:,.0f}원")
        pc4.metric("거래량", f"{latest['volume']:,.4f}")
    else:
        st.warning("차트 데이터를 불러올 수 없습니다. 잠시 후 새로고침 해주세요.")

# ──────────────────────────────────────────────
#  탭 3 : 거래 내역
# ──────────────────────────────────────────────
with tab3:
    st.subheader("거래 내역")

    trade_log = load_trade_log_data()

    if trade_log:
        perf = calculate_trade_performance(trade_log)
        st.markdown("#### 자동매매 수익 요약 (실현 기준)")
        if perf["has_realized"]:
            pf1, pf2, pf3, pf4 = st.columns(4)
            pf1.metric("누적 실현손익", f'{perf["total_pnl"]:+,.0f}원', f'{perf["total_return_pct"]:+.2f}%')
            pf2.metric("누적 실현수익률", f'{perf["total_return_pct"]:+.2f}%')
            pf3.metric("오늘 실현손익", f'{perf["today_pnl"]:+,.0f}원')
            pf4.metric("이번달 실현손익", f'{perf["month_pnl"]:+,.0f}원')
            st.caption("수익률은 매도/손절/익절/트레일링 체결 건을 FIFO 원가 기준으로 계산합니다.")

            day_col, month_col = st.columns(2)
            with day_col:
                st.markdown("##### 일별 실현 성과")
                daily_view = perf["daily"].rename(columns={
                    "date": "일자", "trades": "청산건수",
                    "pnl": "실현손익(원)", "return_pct": "수익률(%)",
                    "cost": "원가(원)", "proceeds": "매도금액(원)",
                })
                st.dataframe(daily_view, use_container_width=True, hide_index=True)
            with month_col:
                st.markdown("##### 월별 실현 성과")
                monthly_view = perf["monthly"].rename(columns={
                    "month": "월", "trades": "청산건수",
                    "pnl": "실현손익(원)", "return_pct": "수익률(%)",
                    "cost": "원가(원)", "proceeds": "매도금액(원)",
                })
                st.dataframe(monthly_view, use_container_width=True, hide_index=True)
        else:
            st.info("아직 실현 손익을 계산할 매도 내역이 없습니다. (매수만 있는 상태)")

        st.divider()
        filter_options = ["전체", "매수", "매도", "손절", "익절", "트레일링", "매수실패", "매도실패", "ERROR"]
        filter_type = st.selectbox("유형 필터", filter_options)

        filtered = trade_log if filter_type == "전체" else [t for t in trade_log if t.get("type") == filter_type]

        if filtered:
            reversed_filtered = list(reversed(filtered))
            html = '<table class="coin-table"><tr>'
            for header in ["시간", "유형", "코인", "내용", "금액(원)", "가격", "시그널"]:
                html += f'<th>{header}</th>'
            html += '</tr>'
            for t in reversed_filtered[:100]:  # 최근 100건
                ticker_display = t.get("ticker", "").replace("KRW-", "")
                amount_str = f'{t.get("amount", 0):,.0f}' if t.get("amount", 0) else "-"
                price_str = f'{t.get("price", 0):,.0f}' if t.get("price", 0) else "-"
                score = t.get("signal_score", 0)
                score_str = colored_pnl(f"{score:+d}") if score else "-"
                html += (
                    f'<tr><td>{t.get("time", "")}</td>'
                    f'<td>{type_badge(t.get("type", ""))}</td>'
                    f'<td><strong>{ticker_display}</strong></td>'
                    f'<td>{t.get("message", "")}</td>'
                    f'<td>{amount_str}</td>'
                    f'<td>{price_str}</td>'
                    f'<td>{score_str}</td></tr>'
                )
            html += '</table>'
            if len(reversed_filtered) > 100:
                html += f'<p style="color:#aaa;font-size:0.85em">최근 100건만 표시 (전체 {len(reversed_filtered)}건)</p>'
            st.markdown(html, unsafe_allow_html=True)

            # CSV 다운로드는 전체 데이터
            df_trades = pd.DataFrame(reversed_filtered)
            display_cols = ["time", "type", "ticker", "message", "amount", "price", "signal_score"]
            rename_map = {
                "time": "시간", "type": "유형", "ticker": "코인",
                "message": "내용", "amount": "금액(원)", "price": "가격",
                "signal_score": "시그널",
            }
            available_cols = [c for c in display_cols if c in df_trades.columns]
            df_display = df_trades[available_cols].rename(columns=rename_map)
            csv = df_display.to_csv(index=False).encode("utf-8-sig")
            st.download_button("CSV 다운로드", csv, "trade_history.csv", "text/csv")
        else:
            st.info("해당 유형의 거래 내역이 없습니다.")

        if st.session_state.connected:
            confirm_clear_log = st.checkbox(
                "거래 내역 초기화 확인",
                value=False,
                key="confirm_clear_trade_log",
            )
            if st.button("거래 내역 초기화", disabled=not confirm_clear_log):
                st.session_state.trader.clear_trade_log()
                st.rerun()
    else:
        st.info("거래 내역이 없습니다. 자동매매를 시작하면 여기에 기록됩니다.")

# ──────────────────────────────────────────────
#  탭 4 : 시장 현황
# ──────────────────────────────────────────────
with tab4:
    st.subheader("시장 현황")

    fg_col, mkt_col = st.columns(2)

    # 공포탐욕지수
    with fg_col:
        fng_data = cached_fear_greed()
        if fng_data:
            cur_fng = fng_data[0]
            fng_val = int(cur_fng.get("value", 50))
            fng_cls = cur_fng.get("value_classification", "Neutral")
            cls_kr = {
                "Extreme Fear": "극도의 공포", "Fear": "공포",
                "Neutral": "중립", "Greed": "탐욕", "Extreme Greed": "극도의 탐욕",
            }

            fig_gauge = create_fear_greed_gauge(fng_val, fng_cls)
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown(f"**{cls_kr.get(fng_cls, fng_cls)}** ({fng_val}/100)")

            # 30일 추이
            if len(fng_data) > 1:
                fng_df = pd.DataFrame(fng_data)
                fng_df["value"] = fng_df["value"].astype(int)
                fng_df["date"] = pd.to_datetime(fng_df["timestamp"].astype(int), unit="s")
                fng_df = fng_df.sort_values("date")

                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(
                    x=fng_df["date"], y=fng_df["value"],
                    mode="lines+markers", name="공포탐욕지수",
                    line=dict(color="#f39c12", width=2), marker=dict(size=4),
                ))
                fig_trend.add_hline(y=50, line_dash="dash", line_color="gray")
                fig_trend.add_hline(y=25, line_dash="dot", line_color="#ff4444")
                fig_trend.add_hline(y=75, line_dash="dot", line_color="#44dd44")
                fig_trend.update_layout(
                    height=250, template="plotly_dark", title="30일 추이",
                    margin=dict(l=20, r=20, t=40, b=20), yaxis=dict(range=[0, 100]),
                )
                st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.warning("공포탐욕지수를 불러올 수 없습니다.")

    # 글로벌 시장 정보
    with mkt_col:
        st.markdown("#### 글로벌 시장 정보")
        overview = cached_market_overview()
        if overview:
            om1, om2 = st.columns(2)
            om1.metric("총 시가총액",
                        f"{format_krw(overview.get('total_market_cap_krw', 0))}원",
                        f"{overview.get('market_cap_change_24h', 0):+.2f}%")
            om2.metric("24시간 거래량",
                        f"{format_krw(overview.get('total_volume_krw', 0))}원")

            om3, om4 = st.columns(2)
            om3.metric("BTC 도미넌스", f"{overview.get('btc_dominance', 0):.1f}%")
            om4.metric("ETH 도미넌스", f"{overview.get('eth_dominance', 0):.1f}%")

            st.metric("활성 암호화폐 수", f"{overview.get('active_cryptos', 0):,}개")
        else:
            st.warning("시장 데이터를 불러올 수 없습니다.")

    st.divider()

    # 김치프리미엄 + 호가창
    kp_col, ob_col = st.columns(2)

    with kp_col:
        st.markdown("#### 김치프리미엄")
        coin_symbol = selected_ticker.split("-")[1] if "-" in selected_ticker else "BTC"
        binance_sym = f"{coin_symbol}USDT"
        kp_data = cached_kimchi_premium(selected_ticker, binance_sym)
        if kp_data:
            kp_pct = kp_data["premium_pct"]
            kp_color = "red" if kp_pct >= 3 else "blue" if kp_pct <= -3 else "gray"
            st.metric(
                f"{coin_symbol} 김치프리미엄",
                f"{kp_pct:+.2f}%",
                help="양수=국내가 비쌈, 음수=국내가 쌈",
            )
            kp1, kp2 = st.columns(2)
            kp1.metric("업비트", f"{kp_data['upbit_price']:,.0f}원")
            kp2.metric("바이낸스(원화환산)", f"{kp_data['binance_krw']:,.0f}원")
            st.caption(f"USD/KRW 환율: {kp_data['usd_krw']:,.0f}")
        else:
            st.warning("김치프리미엄 데이터를 불러올 수 없습니다.")

    with ob_col:
        st.markdown("#### 호가창 분석")
        ob_data = cached_orderbook(selected_ticker)
        if ob_data:
            ratio = ob_data["buy_ratio"]
            bar_color = "#ff4444" if ratio >= 0.6 else "#4444ff" if ratio <= 0.4 else "#888888"
            st.metric("매수/매도 비율", f"{ob_data['bid_ask_ratio']:.2f}")
            st.progress(ratio, text=f"매수 {ratio:.0%} | 매도 {1-ratio:.0%}")

            ob1, ob2 = st.columns(2)
            ob1.metric("매수벽 (대량주문)", f"{ob_data['whale_bid_count']}건")
            ob2.metric("매도벽 (대량주문)", f"{ob_data['whale_ask_count']}건")
            st.caption(f"스프레드: {ob_data['spread_pct']:.4f}% | "
                       f"상위3호가 매수 집중도: {ob_data['top3_bid_pct']:.1f}%")
        else:
            st.warning("호가창 데이터를 불러올 수 없습니다.")

    st.divider()

    # Top 10 코인
    st.subheader("시가총액 TOP 10")
    top_coins = cached_top_coins()
    if top_coins:
        html = '<table class="coin-table"><tr>'
        for header in ["#", "코인", "현재가", "24h 변동", "시가총액", "24h 거래량"]:
            html += f'<th>{header}</th>'
        html += '</tr>'
        for i, coin in enumerate(top_coins):
            change_24h = coin.get('price_change_percentage_24h', 0) or 0
            html += (
                f'<tr><td>{i + 1}</td>'
                f'<td><strong>{coin.get("name", "")} ({coin.get("symbol", "").upper()})</strong></td>'
                f'<td>{coin.get("current_price", 0):,.0f}원</td>'
                f'<td>{colored_pnl(f"{change_24h:+.2f}%")}</td>'
                f'<td>{format_krw(coin.get("market_cap", 0))}원</td>'
                f'<td>{format_krw(coin.get("total_volume", 0))}원</td></tr>'
            )
        html += '</table>'
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.warning("코인 데이터를 불러올 수 없습니다.")

# ──────────────────────────────────────────────
#  탭 5 : 코인 스캐너
# ──────────────────────────────────────────────
with tab5:
    st.subheader("코인 스캐너 - 유망 코인 자동 탐색")
    st.caption("주요 25개 코인의 기술적 지표를 분석하여 매수/매도 시그널 점수로 순위를 매깁니다.")
    scanner_perf = calculate_trade_performance(load_trade_log_data())
    if scanner_perf["has_realized"]:
        sf1, sf2, sf3 = st.columns(3)
        sf1.metric("누적 실현손익", f'{scanner_perf["total_pnl"]:+,.0f}원', f'{scanner_perf["total_return_pct"]:+.2f}%')
        sf2.metric("누적 실현수익률", f'{scanner_perf["total_return_pct"]:+.2f}%')
        sf3.metric("이번달 실현손익", f'{scanner_perf["month_pnl"]:+,.0f}원')
    else:
        st.caption("실현 손익 통계는 매도 체결이 누적되면 자동 표시됩니다.")

    if st.button("스캔 시작", type="primary", key="scan_btn"):
        st.cache_data.clear()

    scan_results = scan_coins()

    if scan_results:
        # 상위 매수 추천 / 하위 매도 경고
        buy_candidates = [r for r in scan_results if r["스코어"] >= 2]
        sell_warnings = [r for r in scan_results if r["스코어"] <= -2]

        if buy_candidates:
            st.success(f"매수 유망 코인: {', '.join(r['코인'] for r in buy_candidates[:5])}")
        if sell_warnings:
            st.error(f"매도 경고 코인: {', '.join(r['코인'] for r in sell_warnings[:5])}")
        if not buy_candidates and not sell_warnings:
            st.info("현재 강한 시그널을 보이는 코인이 없습니다. (관망장)")

        # 전체 결과 테이블
        html = '<table class="coin-table"><tr>'
        for header in ["#", "코인", "현재가", "24h", "스코어", "판단", "매수 시그널", "매도 시그널", "거래대금"]:
            html += f'<th>{header}</th>'
        html += '</tr>'
        for i, r in enumerate(scan_results):
            action_kr = {"BUY": "매수", "SELL": "매도", "HOLD": "관망"}.get(r["판단"], "관망")
            change_html = colored_pnl(f"{r['24h변동']:+.1f}%")
            score_html = colored_pnl(f"{r['스코어']:+d}")
            html += (
                f'<tr><td>{i + 1}</td>'
                f'<td><strong>{r["코인"]}</strong></td>'
                f'<td>{r["현재가"]:,.0f}원</td>'
                f'<td>{change_html}</td>'
                f'<td>{score_html}</td>'
                f'<td>{type_badge(action_kr)}</td>'
                f'<td style="font-size:0.85em">{r["매수시그널"]}</td>'
                f'<td style="font-size:0.85em">{r["매도시그널"]}</td>'
                f'<td>{format_krw(r["거래대금"])}원</td></tr>'
            )
        html += '</table>'
        st.markdown(html, unsafe_allow_html=True)

        st.divider()
        st.caption(
            "스코어 = 각 지표(RSI, MACD, 볼린저밴드, 이동평균, 거래량, ADX, StochRSI, "
            "OBV, 일목균형표, 변동성돌파, VWAP, RSI다이버전스)의 매수(+1)/매도(-1) 합산. "
            "높을수록 매수 시그널이 강합니다."
        )
    else:
        st.warning("코인 스캔 데이터를 불러올 수 없습니다.")

# ================================================================
#  매매 시 자동 새로고침
# ================================================================
@st.fragment(run_every=timedelta(seconds=5))
def trade_watcher():
    """매매 발생 시 자동으로 페이지 새로고침"""
    if not st.session_state.get("auto_trading"):
        return
    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_history.json")
    try:
        mtime = os.path.getmtime(log_file) if os.path.exists(log_file) else 0
    except Exception:
        mtime = 0
    if "last_log_mtime" not in st.session_state:
        st.session_state.last_log_mtime = mtime
    elif mtime > st.session_state.last_log_mtime:
        st.session_state.last_log_mtime = mtime
        st.rerun()

trade_watcher()

# ================================================================
#  푸터
# ================================================================
st.divider()
st.caption(
    f"마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
    "이 프로그램은 투자 조언이 아닙니다. 투자에 대한 책임은 본인에게 있습니다."
)
