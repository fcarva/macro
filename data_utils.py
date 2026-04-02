"""Shared helpers for data downloads, official Brazil series, and metadata."""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from urllib import request

import pandas as pd
import sidrapy
from bcb import sgs


BRAZIL_SOURCE_MAP = {
    "sidra": {
        "cna_6784": {
            "dataset_id": "CNA 6784",
            "table_code": "6784",
            "frequency": "annual",
            "source": "IBGE SIDRA",
            "series": {
                "9808": "PIB - valores correntes",
                "9810": "PIB - variacao em volume",
                "9811": "PIB - deflator - variacao anual",
                "9812": "PIB per capita - valores correntes",
                "93": "Populacao residente",
            },
        },
        "cnt_1846": {
            "dataset_id": "CNT 1846",
            "table_code": "1846",
            "frequency": "quarterly",
            "source": "IBGE SIDRA",
            "variable_code": "585",
            "classification_code": "11255",
            "series": {
                "90707": "PIB a precos de mercado",
                "93404": "Despesa de consumo das familias",
                "93406": "Formacao bruta de capital fixo",
            },
        },
        "cnt_1620": {
            "dataset_id": "CNT 1620",
            "table_code": "1620",
            "frequency": "quarterly",
            "source": "IBGE SIDRA",
            "variable_code": "583",
            "classification_code": "11255",
            "series": {
                "90707": "PIB a precos de mercado",
                "93404": "Despesa de consumo das familias",
                "93406": "Formacao bruta de capital fixo",
            },
        },
    },
    "scn_annual": {
        "tab05": {
            "dataset_id": "SCN 2023 tab05",
            "source": "IBGE SCN annual workbook",
            "frequency": "annual",
            "revision_reference": "Sistema de Contas Nacionais 2023",
            "workbook_url": (
                "https://ftp.ibge.gov.br/Contas_Nacionais/"
                "Sistema_de_Contas_Nacionais/2023/tabelas_xls/sinoticas/tab05.xls"
            ),
            "section": "B - Otica da despesa",
            "series": {
                "90707": {
                    "row_label": "Total",
                    "series_name": "PIB a precos de mercado",
                    "series_code": "tab05:B:total:valor_corrente",
                },
                "93404": {
                    "row_label": "Despesa de consumo das familias",
                    "series_name": "Despesa de consumo das familias",
                    "series_code": "tab05:B:consumo_familias:valor_corrente",
                },
                "93406": {
                    "row_label": "Formacao bruta de capital fixo",
                    "series_name": "Formacao bruta de capital fixo",
                    "series_code": "tab05:B:fbcf:valor_corrente",
                },
            },
        }
    },
    "bcb_sgs": {
        "11": {
            "dataset_id": "BCB SGS",
            "series_code": 11,
            "series_name": "Selic nominal diaria",
            "frequency": "daily",
            "unit": "percent per day",
            "source": "BCB SGS",
        },
        "433": {
            "dataset_id": "BCB SGS",
            "series_code": 433,
            "series_name": "IPCA mensal",
            "frequency": "monthly",
            "unit": "percent per month",
            "source": "BCB SGS",
        },
    },
    "future_extensions": {
        "labor_input": ["population", "pea", "occupied"],
        "real_rate_input": ["selic_ipca", "ntnb_real_yield"],
        "human_capital_input": ["none", "future_pnad_schooling"],
    },
}


def ensure_directory(path):
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_metadata(metadata, path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=True, default=str)


def metadata_entry(
    source,
    frequency,
    unit,
    fallback=None,
    note=None,
    dataset_id=None,
    series_code=None,
    proxy_used=None,
    aggregation=None,
    revision_reference=None,
    concept_note=None,
    **extra_fields,
):
    entry = {
        "source": source,
        "frequency": frequency,
        "unit": unit,
        "fallback": fallback,
        "note": note,
        "dataset_id": dataset_id,
        "series_code": series_code,
        "proxy_used": proxy_used,
        "aggregation": aggregation,
        "revision_reference": revision_reference,
        "concept_note": concept_note,
    }
    for key, value in extra_fields.items():
        if value is not None:
            entry[key] = value
    return {key: value for key, value in entry.items() if value is not None}


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


def _coerce_date(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value)[:10]).date()


def fetch_bcb_sgs_series(series_code, start_date=None, end_date=None):
    start = _coerce_date(start_date)
    end = _coerce_date(end_date)

    def _fetch_chunk(chunk_start, chunk_end):
        frame = sgs.get({"value": int(series_code)}, start=chunk_start, end=chunk_end)
        if frame.empty:
            return pd.DataFrame(columns=["date", "value"])
        frame = frame.reset_index()
        frame = frame.rename(columns={frame.columns[0]: "date", "value": "value"})
        frame["date"] = pd.to_datetime(frame["date"])
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        return frame[["date", "value"]].dropna()

    if start is None or end is None or (end - start).days <= 3652:
        return _fetch_chunk(start, end).sort_values("date").reset_index(drop=True)

    frames = []
    current_start = start
    while current_start <= end:
        current_end = min(end, date(current_start.year + 9, 12, 31))
        frames.append(_fetch_chunk(current_start, current_end))
        current_start = current_end + timedelta(days=1)

    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)


def normalize_text(value):
    text = "" if value is None or pd.isna(value) else str(value)
    text = " ".join(text.replace("\n", " ").split())
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return text.strip()


def normalize_sidra_tidy(
    frame,
    *,
    dataset_id,
    source,
    frequency,
    period_code_col,
    period_label_col,
    series_code_col,
    series_name_col,
    unit_col="MN",
    value_col="V",
    extra_columns=None,
):
    extra_columns = {} if extra_columns is None else dict(extra_columns)
    working = frame.copy()
    tidy = pd.DataFrame(
        {
            "period": working[period_code_col].astype(str),
            "period_label": working[period_label_col].astype(str),
            "series_id": working[series_code_col].astype(str),
            "series_name": working[series_name_col].map(normalize_text),
            "value": pd.to_numeric(working[value_col], errors="coerce"),
            "unit": working[unit_col].map(normalize_text),
            "source": source,
            "dataset_id": dataset_id,
            "frequency": frequency,
        }
    )
    for key, value in extra_columns.items():
        if isinstance(value, str) and value in working.columns:
            tidy[key] = working[value].map(normalize_text)
        else:
            tidy[key] = value
    return tidy.dropna(subset=["value"]).reset_index(drop=True)


def fetch_sidra_table_tidy(
    *,
    table_code,
    dataset_id,
    source,
    frequency,
    territorial_level="1",
    ibge_territorial_code="1",
    variable="all",
    classifications=None,
    period="all",
    period_code_col="D2C",
    period_label_col="D2N",
    series_code_col="D3C",
    series_name_col="D3N",
    extra_columns=None,
):
    frame = sidrapy.get_table(
        table_code=table_code,
        territorial_level=territorial_level,
        ibge_territorial_code=ibge_territorial_code,
        variable=variable,
        classifications=classifications,
        period=period,
        header="n",
    )
    return normalize_sidra_tidy(
        frame,
        dataset_id=dataset_id,
        source=source,
        frequency=frequency,
        period_code_col=period_code_col,
        period_label_col=period_label_col,
        series_code_col=series_code_col,
        series_name_col=series_name_col,
        extra_columns=extra_columns,
    )


def fetch_brazil_cna_6784_annual(period="all"):
    spec = BRAZIL_SOURCE_MAP["sidra"]["cna_6784"]
    return fetch_sidra_table_tidy(
        table_code=spec["table_code"],
        dataset_id=spec["dataset_id"],
        source=spec["source"],
        frequency=spec["frequency"],
        variable="9808,9810,9811,9812,93",
        period=period,
        period_code_col="D2C",
        period_label_col="D2N",
        series_code_col="D3C",
        series_name_col="D3N",
    )


def fetch_brazil_cnt_1846_quarterly(period="all"):
    spec = BRAZIL_SOURCE_MAP["sidra"]["cnt_1846"]
    return fetch_sidra_table_tidy(
        table_code=spec["table_code"],
        dataset_id=spec["dataset_id"],
        source=spec["source"],
        frequency=spec["frequency"],
        variable=spec["variable_code"],
        classifications={spec["classification_code"]: "all"},
        period=period,
        period_code_col="D2C",
        period_label_col="D2N",
        series_code_col="D4C",
        series_name_col="D4N",
        extra_columns={
            "value_variable_code": "D3C",
            "value_variable_name": "D3N",
        },
    )


def fetch_brazil_cnt_1620_quarterly(period="all"):
    spec = BRAZIL_SOURCE_MAP["sidra"]["cnt_1620"]
    return fetch_sidra_table_tidy(
        table_code=spec["table_code"],
        dataset_id=spec["dataset_id"],
        source=spec["source"],
        frequency=spec["frequency"],
        variable=spec["variable_code"],
        classifications={spec["classification_code"]: "all"},
        period=period,
        period_code_col="D2C",
        period_label_col="D2N",
        series_code_col="D4C",
        series_name_col="D4N",
        extra_columns={
            "value_variable_code": "D3C",
            "value_variable_name": "D3N",
        },
    )


def filter_tidy_series(frame, series_ids):
    series_ids = {str(series_id) for series_id in series_ids}
    return frame.loc[frame["series_id"].astype(str).isin(series_ids)].copy()


def add_year_column(frame):
    working = frame.copy()
    working["year"] = working["period"].astype(str).str.slice(0, 4).astype(int)
    return working


def aggregate_quarterly_to_annual(frame, aggregation="sum"):
    if aggregation not in {"sum", "mean", "last"}:
        raise ValueError("aggregation must be one of {'sum', 'mean', 'last'}.")
    working = add_year_column(frame)
    metadata_columns = [column for column in working.columns if column not in {"period", "period_label", "value"}]
    grouped = (
        working.groupby(metadata_columns, dropna=False)["value"]
        .agg(aggregation)
        .reset_index()
        .rename(columns={"year": "period"})
    )
    grouped["period"] = grouped["period"].astype(int).astype(str)
    grouped["period_label"] = grouped["period"]
    grouped["frequency"] = "annual"
    grouped["aggregation"] = f"quarterly_to_annual_{aggregation}"
    ordered_columns = [
        "period",
        "period_label",
        "series_id",
        "series_name",
        "value",
        "unit",
        "source",
        "dataset_id",
        "frequency",
        "aggregation",
    ]
    remaining_columns = [column for column in grouped.columns if column not in ordered_columns]
    return grouped[ordered_columns + remaining_columns].sort_values(["series_id", "period"]).reset_index(drop=True)


@lru_cache(maxsize=4)
def _read_excel_table(url):
    return pd.read_excel(url, header=None)


def _extract_current_value_columns(table):
    header = table.iloc[4]
    year_columns = {}
    for column_index, value in header.items():
        label = normalize_text(value).lower()
        match = re.match(r"^(\d{4}) valor corrente$", label)
        if match:
            year_columns[int(match.group(1))] = column_index
    if not year_columns:
        raise ValueError("Could not identify annual current-value columns in the SCN workbook.")
    return year_columns


def _find_row_within_section(table, section_label, row_label):
    labels = table.iloc[:, 0].map(lambda value: normalize_text(value).lower())
    section_key = normalize_text(section_label).lower()
    row_key = normalize_text(row_label).lower()
    section_positions = labels.index[labels == section_key].tolist()
    if not section_positions:
        raise KeyError(f"Section '{section_label}' not found in the SCN workbook.")
    section_start = section_positions[0]
    next_sections = [index for index, value in labels.items() if re.match(r"^[a-z] - ", value) and index > section_start]
    section_end = next_sections[0] if next_sections else len(labels)
    section_slice = labels.iloc[section_start:section_end]
    matches = section_slice.index[section_slice == row_key].tolist()
    if not matches:
        raise KeyError(f"Row '{row_label}' not found inside section '{section_label}'.")
    return matches[0]


def fetch_brazil_scn_annual_current():
    spec = BRAZIL_SOURCE_MAP["scn_annual"]["tab05"]
    workbook = _read_excel_table(spec["workbook_url"]).copy()
    current_value_columns = _extract_current_value_columns(workbook)

    frames = []
    for series_id, series_spec in spec["series"].items():
        row_index = _find_row_within_section(workbook, spec["section"], series_spec["row_label"])
        records = []
        for year, column_index in current_value_columns.items():
            records.append(
                {
                    "period": str(year),
                    "period_label": str(year),
                    "series_id": str(series_id),
                    "series_name": series_spec["series_name"],
                    "value": pd.to_numeric(workbook.iat[row_index, column_index], errors="coerce"),
                    "unit": "1_000_000 BRL",
                    "source": spec["source"],
                    "dataset_id": spec["dataset_id"],
                    "frequency": spec["frequency"],
                    "series_code": series_spec["series_code"],
                    "revision_reference": spec["revision_reference"],
                }
            )
        frames.append(pd.DataFrame(records))
    return pd.concat(frames, ignore_index=True).dropna(subset=["value"]).sort_values(["series_id", "period"]).reset_index(drop=True)


def compute_validation_residuals(annualized_quarterly, annual_official):
    left = add_year_column(annualized_quarterly).rename(columns={"value": "quarterly_sum"})
    right = add_year_column(annual_official).rename(columns={"value": "annual_official"})
    join_columns = ["year", "series_id"]
    merged = left.merge(
        right[join_columns + ["annual_official"]],
        on=join_columns,
        how="inner",
    )
    merged["residual"] = merged["quarterly_sum"] - merged["annual_official"]
    merged["residual_pct_of_annual"] = 100.0 * merged["residual"] / merged["annual_official"]
    merged["period"] = merged["year"].astype(str)
    merged["period_label"] = merged["period"]
    ordered_columns = [
        "period",
        "period_label",
        "year",
        "series_id",
        "series_name",
        "quarterly_sum",
        "annual_official",
        "residual",
        "residual_pct_of_annual",
        "unit",
        "source",
        "dataset_id",
    ]
    remaining_columns = [column for column in merged.columns if column not in ordered_columns]
    return merged[ordered_columns + remaining_columns].sort_values(["series_id", "year"]).reset_index(drop=True)
