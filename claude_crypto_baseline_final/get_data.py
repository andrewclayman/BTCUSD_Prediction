import pathlib
from typing import Optional, Tuple

import pandas as pd
import yfinance as yf

DATA_DIR = pathlib.Path("data")


def _extract_close(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Return a 1-D Close series from a yfinance download result.
    Handles single-level columns and MultiIndex like ('Close', '^GSPC').
    Falls back to Adj Close if Close is missing.
    """
    if df is None or df.empty:
        return None

    # Ensure DatetimeIndex with UTC
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    else:
        df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")

    # Simple columns case
    if not isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns:
            return pd.to_numeric(df["Close"], errors="coerce")
        if "Adj Close" in df.columns:
            return pd.to_numeric(df["Adj Close"], errors="coerce")
        # Fallback to first numeric column
        for col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().any():
                return s
        return None

    # MultiIndex case: level 0 has price fields
    lvl0 = df.columns.get_level_values(0)
    target = "Close" if "Close" in lvl0 else ("Adj Close" if "Adj Close" in lvl0 else None)
    if target is None:
        # Pick first numeric subgroup
        for g in lvl0.unique():
            sub = df[g]
            if isinstance(sub, pd.DataFrame):
                for subc in sub.columns:
                    s = pd.to_numeric(sub[subc], errors="coerce")
                    if s.notna().any():
                        return s
            else:
                s = pd.to_numeric(sub, errors="coerce")
                if s.notna().any():
                    return s
        return None

    sub = df[target]
    if isinstance(sub, pd.Series):
        return pd.to_numeric(sub, errors="coerce")
    # Multiple tickers: choose the first column with data
    for c in sub.columns:
        s = pd.to_numeric(sub[c], errors="coerce")
        if s.notna().any():
            return s
    return None


def _save_close_series(close: Optional[pd.Series], out_csv: pathlib.Path, label: str) -> bool:
    if close is None or close.empty:
        print(f"[skip] No data for {label}")
        return False

    ser = pd.to_numeric(close.copy(), errors="coerce")
    # Normalize index to UTC DatetimeIndex
    idx = ser.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx, utc=True, errors="coerce")
    elif idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")

    out = pd.DataFrame(
        {
            "Date": idx.strftime("%Y-%m-%d"),
            "Close": ser.values,
        }
    )

    # Clear any index name collisions and ensure RangeIndex
    out.index.name = None
    out = out.reset_index(drop=True)

    # Clean and order
    out = (
        out.dropna()
           .drop_duplicates(subset=["Date"])
           .sort_values(by=["Date"])
           .reset_index(drop=True)
    )

    if out.empty:
        print(f"[skip] Empty after cleaning for {label}")
        return False

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"[ok] Saved {label}: {len(out):,} rows -> {out_csv}")
    return True


def _download_single(symbol: str, start: str = "2018-01-01", interval: str = "1d") -> Tuple[str, Optional[pd.DataFrame]]:
    try:
        df = yf.download(
            symbol,
            start=start,
            end=None,
            interval=interval,
            auto_adjust=False,
            progress=False,
            group_by="column",
        )
        if df is not None and not df.empty:
            return symbol, df
    except Exception as e:
        print(f"[warn] download failed for {symbol}: {e}")
    return symbol, None


def _download_try(symbols, start="2018-01-01", interval="1d") -> Tuple[Optional[str], Optional[pd.Series]]:
    if isinstance(symbols, str):
        symbols = [symbols]
    for s in symbols:
        sym, df = _download_single(s, start=start, interval=interval)
        if df is not None and not df.empty:
            close = _extract_close(df)
            if close is not None and close.notna().any():
                return s, close
    return None, None


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # S&P 500
    sym, close = _download_try("^GSPC", start="2018-01-01", interval="1d")
    _save_close_series(close, DATA_DIR / "sp500_daily.csv", label=f"S&P 500 ({sym or '^GSPC'})")

    # VIX
    sym, close = _download_try("^VIX", start="2018-01-01", interval="1d")
    _save_close_series(close, DATA_DIR / "vix_daily.csv", label=f"VIX ({sym or '^VIX'})")

    # Gold: prefer XAUUSD spot, fallback to COMEX futures GC=F
    sym, close = _download_try(["XAUUSD=X", "GC=F", "GLD", "IAU"], start="2018-01-01", interval="1d")
    if sym:
        _save_close_series(close, DATA_DIR / "gold_daily.csv", label=f"Gold ({sym})")
    else:
        print("[skip] No gold series available")

    # Dow Jones
    sym, close = _download_try("^DJI", start="2018-01-01", interval="1d")
    _save_close_series(close, DATA_DIR / "djia_daily.csv", label=f"DJIA ({sym or '^DJI'})")

    # Dollar index: prefer DXY, fallback to ICE alias or UUP ETF
    sym, close = _download_try(["^DXY", "DX-Y.NYB", "UUP"], start="2018-01-01", interval="1d")
    if sym in ("^DXY", "DX-Y.NYB"):
        _save_close_series(close, DATA_DIR / "dxy_daily.csv", label=f"Dollar Index ({sym})")
    elif sym == "UUP":
        _save_close_series(close, DATA_DIR / "uup_daily.csv", label="Dollar Index proxy (UUP)")
    else:
        print("[skip] No DXY/UUP series available")

    print("\nDone.")


if __name__ == "__main__":
    main()
