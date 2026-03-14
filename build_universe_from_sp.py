from __future__ import annotations

import os
from io import StringIO
from typing import List

import pandas as pd
import requests


OUT_PATH = os.getenv("OUT_PATH", "data/universe.csv")
TIMEOUT = 20

HEADERS = {"User-Agent": "Mozilla/5.0"}

# S&P 1500 = S&P 500 + S&P 400 + S&P 600
WIKI_URLS = [
    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
]

ALLOW_DOTTED = {"BRK.B"}

EXCLUDE_SECTOR_KEYWORDS = [
    "REAL ESTATE",
]

EXCLUDE_INDUSTRY_KEYWORDS = [
    "REIT",
    "MORTGAGE REIT",
    "PROPERTY MANAGEMENT",
    "REAL ESTATE",
    "REAL ESTATE SERVICES",
    "REAL ESTATE DEVELOPMENT",
    "BIOTECH",
    "BIOTECHNOLOGY",
    "DRUG MANUFACTURERS",
    "PHARMACEUTICAL",
]

EXCLUDE_NAME_KEYWORDS = [
    "ACQUISITION",
    "CAPITAL TRUST",
    "CHINA",
    "ADR",
    "SPAC",
    "FUND",
    "ETF",
    "REIT",
    "TRUST",
    "WARRANT",
]

EXCLUDE_TICKER_CONTAINS = [
    "^",
    "/",
]

EXCLUDE_TICKER_SUFFIXES = [
    "W",
    "WS",
    "WT",
    "U",
    "UN",
    "R",
    "RT",
    "P",
    "PR",
]


def normalize_ticker(t: str) -> str:
    t = str(t).strip().upper()
    if t in ALLOW_DOTTED:
        return t
    return t.replace(".", "-")


def contains_any(text: str, keywords: List[str]) -> bool:
    text = str(text).upper()
    return any(k in text for k in keywords)


def fetch_table(url: str) -> pd.DataFrame:
    resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    if not tables:
        raise RuntimeError(f"No tables found at {url}")
    return tables[0]


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    ticker_col = None
    name_col = None
    sector_col = None
    industry_col = None

    for c in df.columns:
        lc = c.lower()
        if lc in ("symbol", "ticker"):
            ticker_col = c
        elif "security" in lc or "company" in lc or "name" in lc:
            if name_col is None:
                name_col = c
        elif "gics sector" in lc or lc == "sector":
            sector_col = c
        elif "gics sub-industry" in lc or "sub-industry" in lc or lc == "industry":
            industry_col = c

    if ticker_col is None:
        raise RuntimeError("Ticker column not found")
    if name_col is None:
        raise RuntimeError("Name column not found")
    if sector_col is None:
        raise RuntimeError("Sector column not found")
    if industry_col is None:
        raise RuntimeError("Industry column not found")

    out = pd.DataFrame()
    out["ticker"] = df[ticker_col].astype(str).map(normalize_ticker)
    out["name"] = df[name_col].astype(str).str.strip()
    out["market_cap"] = pd.NA
    out["security_type"] = "Common Stock"
    out["country"] = "United States"
    out["sector"] = df[sector_col].astype(str).str.strip()
    out["industry"] = df[industry_col].astype(str).str.strip()
    return out


def is_bad_ticker(ticker: str) -> bool:
    t = str(ticker).strip().upper()
    if not t:
        return True

    for bad in EXCLUDE_TICKER_CONTAINS:
        if bad in t:
            return True

    raw = t.replace("-", ".")
    if "." in raw and raw not in ALLOW_DOTTED:
        return True

    for suffix in EXCLUDE_TICKER_SUFFIXES:
        if t.endswith(f"-{suffix}") or t.endswith(f".{suffix}"):
            return True

    return False


def apply_filters(universe: pd.DataFrame) -> pd.DataFrame:
    universe = universe.copy()

    universe = universe[~universe["ticker"].map(is_bad_ticker)].copy()

    universe = universe[
        ~universe["name"].map(lambda x: contains_any(x, EXCLUDE_NAME_KEYWORDS))
    ].copy()

    universe = universe[
        ~universe["sector"].map(lambda x: contains_any(x, EXCLUDE_SECTOR_KEYWORDS))
    ].copy()

    universe = universe[
        ~universe["industry"].map(lambda x: contains_any(x, EXCLUDE_INDUSTRY_KEYWORDS))
    ].copy()

    return universe


def main() -> None:
    frames = []
    for url in WIKI_URLS:
        df = fetch_table(url)
        frames.append(standardize_columns(df))

    universe = pd.concat(frames, ignore_index=True)
    before_count = len(universe)

    universe = universe.drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    after_dedup_count = len(universe)

    universe = apply_filters(universe)
    final_count = len(universe)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    universe = universe[
        ["ticker", "name", "market_cap", "security_type", "country", "sector", "industry"]
    ].sort_values(["sector", "industry", "ticker"]).reset_index(drop=True)

    universe.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print("[INFO] S&P 1500 universe build complete")
    print(f"[INFO] Before concat filters: {before_count}")
    print(f"[INFO] After dedup: {after_dedup_count}")
    print(f"[INFO] Final saved: {final_count}")
    print(f"[INFO] Output: {OUT_PATH}")


if __name__ == "__main__":
    main()
