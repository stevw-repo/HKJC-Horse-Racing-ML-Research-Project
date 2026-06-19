"""HKO open-data daily-climate CSV: station mapping + CSV parsing.

The endpoint ``opendata.php?dataType=<CLMTEMP|CLMMAXT|CLMMINT>&rformat=csv&station=<code>``
returns the full daily history for a station (back to 1884) as
``Year,Month,Day,Value,Completeness`` rows, with a BOM and bilingual title/legend lines.
Only temperature series are exposed per station; humidity/rainfall are not (PLAN.md §0).
"""

from __future__ import annotations

from datetime import date

# venue -> nearest HKO station (PLAN.md §6 default: urban/HKO HQ for HV, Sha Tin for ST).
VENUE_STATION = {"HV": "HKO", "ST": "SHA"}
# HKO dataType -> our field name.
DATATYPES = {"CLMTEMP": "mean_temp", "CLMMAXT": "max_temp", "CLMMINT": "min_temp"}


def weather_url(api_base: str, station: str, data_type: str) -> str:
    """Full-history CSV URL for one station + climate series."""
    return f"{api_base}/opendata.php?dataType={data_type}&rformat=csv&station={station}"


def parse_climate_csv(text: str) -> dict[date, float]:
    """Parse a daily-climate CSV into ``{date: value}``, skipping unavailable (``***``) days."""
    out: dict[date, float] = {}
    for line in text.splitlines():
        parts = [p.strip().strip('"') for p in line.split(",")]
        if len(parts) < 4:
            continue
        year, month, day, value = parts[0], parts[1], parts[2], parts[3]
        if not (year.isdigit() and month.isdigit() and day.isdigit()):
            continue
        try:
            out[date(int(year), int(month), int(day))] = float(value)
        except ValueError:
            continue  # "***" (unavailable) or blank
    return out
