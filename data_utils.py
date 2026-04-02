"""Shared helpers for data downloads and metadata generation."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from urllib import request

import pandas as pd


def ensure_directory(path):
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_metadata(metadata, path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=True, default=str)


def metadata_entry(source, frequency, unit, fallback=None, note=None):
    return {
        "source": source,
        "frequency": frequency,
        "unit": unit,
        "fallback": fallback,
        "note": note,
    }


def _read_json(url):
    http_request = request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "romer-study/1.0",
        },
    )
    with request.urlopen(http_request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_world_bank_indicator(countries, indicator, start_year, end_year):
    country_string = ";".join(countries)
    url = (
        "https://api.worldbank.org/v2/country/"
        f"{country_string}/indicator/{indicator}"
        f"?format=json&per_page=20000&date={start_year}:{end_year}"
    )
    payload = _read_json(url)
    rows = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
    records = []
    for row in rows:
        country = row.get("country") or {}
        records.append(
            {
                "country": country.get("value"),
                "countryiso3code": row.get("countryiso3code"),
                "date": int(row["date"]),
                "value": pd.to_numeric(row.get("value"), errors="coerce"),
                "indicator": indicator,
            }
        )
    return pd.DataFrame(records)


def fetch_world_bank_panel(countries, indicator_map, start_year, end_year):
    frames = []
    for indicator, alias in indicator_map.items():
        frame = fetch_world_bank_indicator(countries, indicator, start_year, end_year)
        if frame.empty:
            continue
        frame["indicator_alias"] = alias
        frames.append(frame)

    if not frames:
        raise RuntimeError("World Bank API returned no observations for the requested indicators.")

    combined = pd.concat(frames, ignore_index=True)
    panel = (
        combined.pivot_table(
            index=["country", "countryiso3code", "date"],
            columns="indicator_alias",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .sort_values(["countryiso3code", "date"])
    )
    panel.columns.name = None
    return panel


def _format_bcb_date(value):
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.strftime("%d/%m/%Y")
    if isinstance(value, str) and len(value) >= 10 and value[4] == "-" and value[7] == "-":
        return datetime.fromisoformat(value[:10]).strftime("%d/%m/%Y")
    return str(value)


def _coerce_date(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value)[:10]).date()


def _fetch_bcb_sgs_once(series_code, start_date=None, end_date=None):
    params = {"formato": "json"}
    if start_date is not None:
        params["dataInicial"] = _format_bcb_date(start_date)
    if end_date is not None:
        params["dataFinal"] = _format_bcb_date(end_date)
    query = "&".join(f"{key}={value}" for key, value in params.items())
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_code}/dados?{query}"
    payload = _read_json(url)
    frame = pd.DataFrame(payload)
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["data"], dayfirst=True)
    frame["value"] = pd.to_numeric(frame["valor"], errors="coerce")
    return frame[["date", "value"]].dropna()


def fetch_bcb_sgs_series(series_code, start_date=None, end_date=None):
    start = _coerce_date(start_date)
    end = _coerce_date(end_date)

    if start is None or end is None or (end - start).days <= 365:
        return _fetch_bcb_sgs_once(series_code, start_date=start_date, end_date=end_date)

    frames = []
    current_start = start
    while current_start <= end:
        current_end = date(min(current_start.year, end.year), 12, 31)
        current_end = min(current_end, end)
        frames.append(_fetch_bcb_sgs_once(series_code, start_date=current_start, end_date=current_end))
        current_start = date(current_end.year + 1, 1, 1)

    combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date")
    return combined.reset_index(drop=True)
