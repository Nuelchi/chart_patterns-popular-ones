import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    symbol: str
    interval: str
    outputsize: int
    out_name: str


def _fetch_twelvedata_time_series(symbol: str, interval: str, outputsize: int, api_key: str) -> pd.DataFrame:
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": str(outputsize),
        "format": "JSON",
        "apikey": api_key,
    }
    url = "https://api.twelvedata.com/time_series?" + urlencode(params)

    with urlopen(url, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    if "status" in payload and payload.get("status") == "error":
        raise RuntimeError(f"TwelveData error: {payload.get('message')}")

    values = payload.get("values") or []
    if not values:
        raise RuntimeError("No TwelveData values returned")

    df = pd.DataFrame(values)

    # TwelveData returns most recent first; reverse to chronological
    df = df.iloc[::-1].reset_index(drop=True)

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    return df


def fetch_and_save(specs: Iterable[DatasetSpec], data_dir: Path, api_key: str) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)

    for spec in specs:
        df = _fetch_twelvedata_time_series(
            symbol=spec.symbol,
            interval=spec.interval,
            outputsize=spec.outputsize,
            api_key=api_key,
        )
        out_path = data_dir / spec.out_name
        df.to_csv(out_path, index=False)
        print(f"Saved {spec.symbol} {spec.interval} -> {out_path} ({len(df)} rows)")


def main() -> None:
    api_key = os.getenv("TWELVE_DATA_API_KEY")
    if not api_key:
        raise SystemExit("TWELVE_DATA_API_KEY is not set")

    data_dir = Path(__file__).resolve().parents[1] / "data"

    specs = [
        # EURUSD
        DatasetSpec(symbol="EUR/USD", interval="4h", outputsize=2500, out_name="eurusd-4h-twelvedata.csv"),
        DatasetSpec(symbol="EUR/USD", interval="1h", outputsize=5000, out_name="eurusd-1h-twelvedata.csv"),
        DatasetSpec(symbol="EUR/USD", interval="30min", outputsize=5000, out_name="eurusd-30m-twelvedata.csv"),
        # XAUUSD (Gold)
        DatasetSpec(symbol="XAU/USD", interval="4h", outputsize=2500, out_name="xauusd-4h-twelvedata.csv"),
        DatasetSpec(symbol="XAU/USD", interval="1h", outputsize=5000, out_name="xauusd-1h-twelvedata.csv"),
        DatasetSpec(symbol="XAU/USD", interval="30min", outputsize=5000, out_name="xauusd-30m-twelvedata.csv"),
    ]

    fetch_and_save(specs=specs, data_dir=data_dir, api_key=api_key)


if __name__ == "__main__":
    main()
