from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# =========================================================
# ENV / CONFIG
# =========================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

UNIVERSE_CSV = os.getenv("UNIVERSE_CSV", "data/universe.csv")
OUTPUT_CSV = os.getenv(
    "OUTPUT_CSV", "output/oversold_but_not_broken_candidates.csv"
)

MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "0") or "0")  # 0 = unlimited

# 하드컷
MIN_PRICE = float(os.getenv("MIN_PRICE", "10"))
MIN_MARKET_CAP = float(os.getenv("MIN_MARKET_CAP", "500000000"))      # 5억 달러
MIN_DOLLAR_VOLUME_21D = float(os.getenv("MIN_DOLLAR_VOLUME_21D", "6000000"))  # 600만 달러

LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "420"))
TOP_OUTPUT = int(os.getenv("TOP_OUTPUT", "15"))

# 과매도 기준
DRAWDOWN_MIN_PCT = float(os.getenv("DRAWDOWN_MIN_PCT", "10"))       # 52주 고점 대비 최소 -10%
DRAWDOWN_MAX_PCT = float(os.getenv("DRAWDOWN_MAX_PCT", "35"))       # 52주 고점 대비 최대 -35%
TEN_DAY_RETURN_MAX = float(os.getenv("TEN_DAY_RETURN_MAX", "-6"))   # 최근 10일 <= -6%
TWENTY_MA_DISTANCE_MAX = float(os.getenv("TWENTY_MA_DISTANCE_MAX", "-8"))  # 20일선 대비 <= -8%
RSI_MAX = float(os.getenv("RSI_MAX", "40"))

# 구조 생존 기준
SIX_MONTH_RS_MIN = float(os.getenv("SIX_MONTH_RS_MIN", "-10"))   # 6개월 상대강도 최소
MAX_DISTANCE_BELOW_200DMA = float(os.getenv("MAX_DISTANCE_BELOW_200DMA", "-12"))  # 200일선 대비 -12% 초과 이탈 금지
RECENT_HIGH_LOOKBACK_DAYS = int(os.getenv("RECENT_HIGH_LOOKBACK_DAYS", "126"))    # 최근 6개월
RECENT_HIGH_TOLERANCE_PCT = float(os.getenv("RECENT_HIGH_TOLERANCE_PCT", "5"))    # 최근 6개월 안에 52주 고점의 95% 이상 근접

# 내부 등급 점수
W_DRAWDOWN = 30.0
W_10D_DROP = 25.0
W_20DMA_DISTANCE = 25.0
W_RSI = 20.0

GRADE_A_MIN = float(os.getenv("GRADE_A_MIN", "75"))
GRADE_B_MIN = float(os.getenv("GRADE_B_MIN", "60"))
GRADE_WATCH_MIN = float(os.getenv("GRADE_WATCH_MIN", "45"))

SPY_TICKER = "SPY"
SEND_TELEGRAM = os.getenv("SEND_TELEGRAM", "1") == "1"


# =========================================================
# DATA STRUCTURES
# =========================================================

@dataclass
class MarketRegime:
    spy_close: float
    spy_sma50: float
    spy_sma200: float
    passed: bool
    note: str


@dataclass
class OversoldRow:
    ticker: str
    name: str
    sector: str
    industry: str

    close: float
    market_cap: float
    dollar_volume_21d: float

    drawdown_from_52w_high_pct: float
    return_10d_pct: float
    distance_from_20dma_pct: float
    distance_from_50dma_pct: float
    distance_from_200dma_pct: float
    rsi_14: float
    rs_6m_pct: float

    above_200dma: bool
    sma50_above_sma200: bool
    recent_high_ok: bool
    structure_warning: str

    total_score: float
    grade: str

    rebound_trigger: float
    stop_suggestion: float


# =========================================================
# UTIL
# =========================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def send_telegram_message(text: str) -> None:
    if not SEND_TELEGRAM:
        return
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True,
    }
    try:
        requests.post(url, json=payload, timeout=20)
    except Exception:
        pass


def safe_text(v) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip()


def latest(series: pd.Series, n_back: int = 0) -> float:
    s = series.dropna()
    if len(s) <= n_back:
        return np.nan
    return float(s.iloc[-1 - n_back])


def safe_div(a: float, b: float, default: float = np.nan) -> float:
    if pd.isna(a) or pd.isna(b) or b == 0:
        return default
    return float(a) / float(b)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def linear_score_high_better(
    value: float, low: float, high: float, max_points: float
) -> float:
    if pd.isna(value):
        return 0.0
    if high == low:
        return max_points if value >= high else 0.0
    x = clamp((value - low) / (high - low), 0.0, 1.0)
    return x * max_points


def linear_score_low_better(
    value: float, low: float, high: float, max_points: float
) -> float:
    if pd.isna(value):
        return 0.0
    if high == low:
        return max_points if value <= low else 0.0
    x = clamp((high - value) / (high - low), 0.0, 1.0)
    return x * max_points


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def avg_dollar_volume(df: pd.DataFrame, window: int = 21) -> float:
    if "Close" not in df.columns or "Volume" not in df.columns:
        return np.nan
    return float((df["Close"] * df["Volume"]).tail(window).mean())


def load_universe(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Universe file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"ticker", "name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Universe CSV missing columns: {sorted(missing)}")

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["name"] = df["name"].map(safe_text)

    if "sector" in df.columns:
        df["sector"] = df["sector"].map(safe_text)
    else:
        df["sector"] = ""

    if "industry" in df.columns:
        df["industry"] = df["industry"].map(safe_text)
    else:
        df["industry"] = ""

    if "market_cap" in df.columns:
        df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
    else:
        df["market_cap"] = np.nan

    if MAX_SYMBOLS > 0:
        df = df.head(MAX_SYMBOLS).copy()

    return df.reset_index(drop=True)


def download_price_history(tickers: List[str], period_days: int) -> Dict[str, pd.DataFrame]:
    raw = yf.download(
        tickers=tickers,
        period=f"{period_days}d",
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    out: Dict[str, pd.DataFrame] = {}
    if raw.empty:
        return out

    if isinstance(raw.columns, pd.MultiIndex):
        for ticker in tickers:
            cols = raw.get(ticker)
            if cols is None:
                continue
            df = cols.copy()
            if "Close" not in df.columns:
                continue
            df = df.dropna(subset=["Close"]).copy()
            if not df.empty:
                out[ticker] = df
    else:
        t = tickers[0]
        df = raw.copy()
        if "Close" in df.columns:
            df = df.dropna(subset=["Close"]).copy()
            if not df.empty:
                out[t] = df

    return out


# =========================================================
# MARKET REGIME
# =========================================================

def compute_market_regime(price_map: Dict[str, pd.DataFrame]) -> MarketRegime:
    spy = price_map.get(SPY_TICKER)
    if spy is None or len(spy) < 220:
        raise RuntimeError("Missing or insufficient SPY history.")

    spy_close = latest(spy["Close"])
    spy_sma50 = float(spy["Close"].rolling(50).mean().iloc[-1])
    spy_sma200 = float(spy["Close"].rolling(200).mean().iloc[-1])

    passed = bool(spy_close > spy_sma200 and spy_sma50 > spy_sma200)
    note = "정상" if passed else "주의"

    return MarketRegime(
        spy_close=round(spy_close, 2),
        spy_sma50=round(spy_sma50, 2),
        spy_sma200=round(spy_sma200, 2),
        passed=passed,
        note=note,
    )


# =========================================================
# CORE
# =========================================================

def passes_hardcut(df: pd.DataFrame, market_cap_value: float) -> bool:
    if len(df) < 220:
        return False

    close = latest(df["Close"])
    if pd.isna(close) or close < MIN_PRICE:
        return False

    if not pd.isna(market_cap_value) and market_cap_value < MIN_MARKET_CAP:
        return False

    dv21 = avg_dollar_volume(df, 21)
    if pd.isna(dv21) or dv21 < MIN_DOLLAR_VOLUME_21D:
        return False

    return True


def classify_warning(
    distance_from_200dma_pct: float,
    sma50_above_sma200: bool,
    rs_6m_pct: float,
) -> str:
    warnings: List[str] = []

    if pd.notna(distance_from_200dma_pct) and distance_from_200dma_pct < 0:
        warnings.append("200일선 아래")

    if not sma50_above_sma200:
        warnings.append("50일선<200일선")

    if pd.notna(rs_6m_pct) and rs_6m_pct < 0:
        warnings.append("6개월RS음수")

    return ", ".join(warnings) if warnings else "없음"


def grade_from_score(score: float, warning_text: str) -> str:
    base = "DROP"
    if score >= GRADE_A_MIN:
        base = "A"
    elif score >= GRADE_B_MIN:
        base = "B"
    elif score >= GRADE_WATCH_MIN:
        base = "WATCH"

    if base == "DROP":
        return "DROP"

    if warning_text != "없음":
        if base == "A":
            return "B"
        if base == "B":
            return "WATCH"

    return base


def calc_row(
    ticker: str,
    name: str,
    sector: str,
    industry: str,
    df: pd.DataFrame,
    market_cap_value: float,
    spy_df: pd.DataFrame,
) -> Optional[OversoldRow]:
    close = latest(df["Close"])
    sma20 = float(df["Close"].rolling(20).mean().iloc[-1])
    sma50 = float(df["Close"].rolling(50).mean().iloc[-1])
    sma200 = float(df["Close"].rolling(200).mean().iloc[-1])

    if pd.isna(sma20) or pd.isna(sma50) or pd.isna(sma200):
        return None

    high_252 = float(df["High"].tail(252).max())
    if pd.isna(high_252) or high_252 <= 0:
        return None

    drawdown_pct = (1 - close / high_252) * 100.0
    close_10d_ago = latest(df["Close"], 10)
    return_10d_pct = (close / close_10d_ago - 1.0) * 100.0 if pd.notna(close_10d_ago) else np.nan
    dist_20_pct = (close / sma20 - 1.0) * 100.0
    dist_50_pct = (close / sma50 - 1.0) * 100.0
    dist_200_pct = (close / sma200 - 1.0) * 100.0

    rsi_14 = float(compute_rsi(df["Close"], 14).iloc[-1])

    # 6개월 RS
    joined = pd.DataFrame(
        {"stock_close": df["Close"], "spy_close": spy_df["Close"]}
    ).dropna()
    if len(joined) < 126:
        return None

    stock_6m = joined["stock_close"].iloc[-1] / joined["stock_close"].iloc[-126] - 1.0
    spy_6m = joined["spy_close"].iloc[-1] / joined["spy_close"].iloc[-126] - 1.0
    rs_6m_pct = (stock_6m - spy_6m) * 100.0

    # 최근 6개월 안에 52주 고점 근처까지 갔는지
    recent_high = float(df["High"].tail(RECENT_HIGH_LOOKBACK_DAYS).max())
    recent_high_ok = recent_high >= high_252 * (1 - RECENT_HIGH_TOLERANCE_PCT / 100.0)

    above_200dma = close >= sma200
    sma50_above_sma200 = sma50 >= sma200

    # 과매도 조건: 4개 중 2개 이상
    oversold_hits = 0
    if DRAWDOWN_MIN_PCT <= drawdown_pct <= DRAWDOWN_MAX_PCT:
        oversold_hits += 1
    if pd.notna(return_10d_pct) and return_10d_pct <= TEN_DAY_RETURN_MAX:
        oversold_hits += 1
    if pd.notna(dist_20_pct) and dist_20_pct <= TWENTY_MA_DISTANCE_MAX:
        oversold_hits += 1
    if pd.notna(rsi_14) and rsi_14 <= RSI_MAX:
        oversold_hits += 1

    if oversold_hits < 2:
        return None

    # 구조 생존 최소 조건
    if not recent_high_ok:
        return None
    if pd.isna(rs_6m_pct) or rs_6m_pct < SIX_MONTH_RS_MIN:
        return None
    if pd.isna(dist_200_pct) or dist_200_pct < MAX_DISTANCE_BELOW_200DMA:
        return None

    score_drawdown = linear_score_high_better(drawdown_pct, DRAWDOWN_MIN_PCT, DRAWDOWN_MAX_PCT, W_DRAWDOWN)
    score_10d = linear_score_low_better(return_10d_pct, -20.0, TEN_DAY_RETURN_MAX, W_10D_DROP)
    score_20d_dist = linear_score_low_better(dist_20_pct, -18.0, TWENTY_MA_DISTANCE_MAX, W_20DMA_DISTANCE)
    score_rsi = linear_score_low_better(rsi_14, 20.0, RSI_MAX, W_RSI)

    total_score = score_drawdown + score_10d + score_20d_dist + score_rsi
    warning_text = classify_warning(dist_200_pct, sma50_above_sma200, rs_6m_pct)
    grade = grade_from_score(total_score, warning_text)

    if grade == "DROP":
        return None

    recent_3d_high = float(df["High"].tail(3).max())
    rebound_trigger = recent_3d_high * 1.001
    stop_suggestion = close * 0.92

    return OversoldRow(
        ticker=ticker,
        name=name,
        sector=sector,
        industry=industry,

        close=round(close, 2),
        market_cap=float(market_cap_value) if not pd.isna(market_cap_value) else np.nan,
        dollar_volume_21d=round(avg_dollar_volume(df, 21), 2),

        drawdown_from_52w_high_pct=round(drawdown_pct, 2),
        return_10d_pct=round(return_10d_pct, 2) if pd.notna(return_10d_pct) else np.nan,
        distance_from_20dma_pct=round(dist_20_pct, 2),
        distance_from_50dma_pct=round(dist_50_pct, 2),
        distance_from_200dma_pct=round(dist_200_pct, 2),
        rsi_14=round(rsi_14, 2) if pd.notna(rsi_14) else np.nan,
        rs_6m_pct=round(rs_6m_pct, 2),

        above_200dma=bool(above_200dma),
        sma50_above_sma200=bool(sma50_above_sma200),
        recent_high_ok=bool(recent_high_ok),
        structure_warning=warning_text,

        total_score=round(total_score, 2),
        grade=grade,

        rebound_trigger=round(rebound_trigger, 2),
        stop_suggestion=round(stop_suggestion, 2),
    )


def build_radar(universe: pd.DataFrame, price_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    spy_df = price_map[SPY_TICKER]
    rows: List[Dict] = []

    for _, row in universe.iterrows():
        ticker = row["ticker"]
        name = row["name"]
        sector = safe_text(row.get("sector", ""))
        industry = safe_text(row.get("industry", ""))
        market_cap_value = row["market_cap"]

        if ticker == SPY_TICKER:
            continue

        df = price_map.get(ticker)
        if df is None:
            continue

        if not passes_hardcut(df, market_cap_value):
            continue

        result = calc_row(
            ticker=ticker,
            name=name,
            sector=sector,
            industry=industry,
            df=df,
            market_cap_value=market_cap_value,
            spy_df=spy_df,
        )
        if result is None:
            continue

        rows.append(asdict(result))

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    grade_rank = {"A": 3, "B": 2, "WATCH": 1}
    out["grade_rank"] = out["grade"].map(grade_rank).fillna(0)

    out = out.sort_values(
        by=[
            "grade_rank",
            "total_score",
            "drawdown_from_52w_high_pct",
            "return_10d_pct",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    return out.head(TOP_OUTPUT).copy()


# =========================================================
# OUTPUT
# =========================================================

def save_output(df: pd.DataFrame, path: str) -> None:
    ensure_parent_dir(path)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def build_summary_message(regime: MarketRegime, radar_df: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("Oversold but Not Broken Radar")
    lines.append(f"시각: {utc_now()}")
    lines.append("")
    lines.append("시장")
    lines.append(f"- 상태: {regime.note}")
    lines.append(
        f"- SPY: {regime.spy_close:.2f} / 50MA {regime.spy_sma50:.2f} / 200MA {regime.spy_sma200:.2f}"
    )
    lines.append("")

    if radar_df.empty:
        lines.append("후보 없음")
        return "\n".join(lines)

    lines.append(f"후보 수: {len(radar_df)}")
    lines.append("")

    for i in range(min(len(radar_df), TOP_OUTPUT)):
        row = radar_df.iloc[i]

        lines.append(f"[{i+1}] {row['ticker']} {row['name']}")
        lines.append(
            f"등급 {row['grade']} | 52주고점대비 -{row['drawdown_from_52w_high_pct']:.1f}% | "
            f"10일수익률 {row['return_10d_pct']:.1f}% | RSI {row['rsi_14']:.1f}"
        )
        lines.append(
            f"20일이격 {row['distance_from_20dma_pct']:.1f}% | "
            f"50일이격 {row['distance_from_50dma_pct']:.1f}% | "
            f"200일이격 {row['distance_from_200dma_pct']:.1f}%"
        )
        lines.append(
            f"6개월RS {row['rs_6m_pct']:.1f}% | 반등트리거 {row['rebound_trigger']:.2f} | "
            f"손절참고 {row['stop_suggestion']:.2f}"
        )
        lines.append(f"구조경고 {row['structure_warning']}")

        if i < min(len(radar_df), TOP_OUTPUT) - 1:
            lines.append("")

    return "\n".join(lines)


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    universe = load_universe(UNIVERSE_CSV)

    all_tickers = sorted(set([SPY_TICKER] + universe["ticker"].tolist()))
    print(f"[INFO] Downloading data for {len(all_tickers)} tickers...")

    price_map = download_price_history(all_tickers, LOOKBACK_DAYS)

    if SPY_TICKER not in price_map:
        raise RuntimeError("SPY download failed.")

    regime = compute_market_regime(price_map)
    radar_df = build_radar(universe, price_map)

    save_output(radar_df, OUTPUT_CSV)

    message = build_summary_message(regime, radar_df)
    print(message)
    send_telegram_message(message)

    print("")
    print(f"[INFO] Output saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
