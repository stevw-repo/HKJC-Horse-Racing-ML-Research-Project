# scrapers/results_scraper.py
import datetime
import logging
import re

import pandas as pd

from config import RAW_RESULTS_DIR, RESULTS_BASE_URL, RESULTS_START_DATE, VALID_VENUES
from scrapers.base_scraper import BaseScraper
from storage.parquet_store import read, write

logger = logging.getLogger(__name__)

_SPECIAL_CODES = {"WV", "ML", "DISQ", "DNF", "UR", "PU", "FE", "TNP", "SCR"}


class ResultsScraper(BaseScraper):
    """Scrape HKJC historical results HTML pages, storing one parquet per year."""

    def scrape(self, start_date: str = None, end_date: str = None, **kwargs):
        start = datetime.date.fromisoformat(start_date or RESULTS_START_DATE)
        end   = datetime.date.fromisoformat(
            end_date or datetime.date.today().isoformat()
        )
        existing = self._existing_dates()
        cursor   = start
        while cursor <= end:
            if cursor.isoformat() not in existing:
                try:
                    self._scrape_date(cursor)
                except Exception as exc:
                    logger.warning("Skip %s: %s", cursor, exc)
            cursor += datetime.timedelta(days=1)

    # ── Private ───────────────────────────────────────────────────────────────

    def _existing_dates(self) -> set:
        dates: set = set()
        for fp in RAW_RESULTS_DIR.glob("results_*.parquet"):
            try:
                df = read(fp)
                dates.update(
                    pd.to_datetime(df["race_date"]).dt.date.astype(str).tolist()
                )
            except Exception:
                pass
        return dates

    def _scrape_date(self, race_date: datetime.date):
        rows = []
        for venue in sorted(VALID_VENUES):
            for race_no in range(1, 14):
                try:
                    race_rows = self._scrape_race(race_date, venue, race_no)
                    if not race_rows:
                        break          # no more races at this venue today
                    rows.extend(race_rows)
                except Exception as exc:
                    logger.debug("No data %s %s R%d: %s", venue, race_date, race_no, exc)
                    break

        if not rows:
            return
        df   = pd.DataFrame(rows)
        year = race_date.year
        fp   = RAW_RESULTS_DIR / f"results_{year}.parquet"
        if fp.exists():
            existing_df = read(fp)
            df = pd.concat([existing_df, df], ignore_index=True)
            df = df.drop_duplicates(
                subset=["race_date", "venue", "race_no", "horse_no"], keep="last"
            )
        write(df, fp)
        logger.info("Saved %d rows — %s", len(rows), race_date)

    def _scrape_race(self, race_date: datetime.date, venue: str,
                     race_no: int) -> list:
        params = {
            "racedate":   race_date.strftime("%Y/%m/%d"),
            "Racecourse": venue,
            "RaceNo":     str(race_no),
        }
        soup = self.get_html(RESULTS_BASE_URL, params=params)

        # Detect empty / no-race page
        if soup.find("div", string=re.compile(r"no.*information|not.*available", re.I)):
            logger.info("No data found for %s %s R%d.", venue, race_date, race_no)
            #print(f"No data found for {venue} {race_date} R{race_no}.")
            return []

        header = self._parse_header(soup, race_date, venue, race_no)

        table = soup.find("div", class_=re.compile(r"performance"))#, re.I))
        # if not table:
        #     table = soup.find("table")
        if not table:
            return []

        rows = []
        trs  = table.find_all("tr")
        for tr in trs[1:]:
            cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
            if len(cells) < 5:
                continue

            horse_link = tr.find("a", href=re.compile(r"horseid=", re.I))
            horse_id   = None
            if horse_link:
                m = re.search(r"horseid=([A-Za-z]{2}_\d+_\w+)",
                              horse_link.get("href", ""), re.I)
                if m:
                    horse_id = m.group(1).upper()
            # print(cells)
            placing_raw           = cells[0].strip().upper()
            placing = _parse_placing(placing_raw)
            row = {**header,
                   "placing":                   placing,
                #    "placing_code":              placing_code,
                   "horse_no":                  _safe_int(cells[1] if len(cells) > 1 else ""),
                   "horse_name":                cells[2].strip() if len(cells) > 2 else None,
                   "horse_id":                  horse_id,
                   "jockey_name":               cells[3].strip() if len(cells) > 3 else None,
                   "trainer_name":              cells[4].strip() if len(cells) > 4 else None,
                   "actual_weight_lbs":         _safe_int(cells[5] if len(cells) > 5 else ""),
                   "declared_horse_weight_lbs": _safe_int(cells[6] if len(cells) > 6 else ""),
                   "draw":                      _safe_int(cells[7] if len(cells) > 7 else ""),
                   "lbw":                       _safe_lbw(cells[8] if len(cells) > 8 else ""),
                   "win_odds":                  _safe_float(cells[11] if len(cells) > 11 else ""),
                   "gear":                      None, #cells[11].strip() if len(cells) > 11 else "",
                   "finish_time_sec":           int(cells[10].split(':')[0]) * 60 + float(cells[10].split(':')[1]) if len(cells) > 10 else "",
                   "sectional_times":           None,
                   }
            rows.append(row)
        print(rows)
        return rows

    @staticmethod
    def _parse_header(soup, race_date, venue, race_no) -> dict:
        info = {
            "race_date":  race_date.isoformat(),
            "venue":      venue,
            "race_no":    race_no,
            "race_index": None,
            "race_name":  None,
            "race_class": None,
            "distance_m": None,
            "going":      None,
            "course":     None,
            "prize_hkd":  None,
        }
        hdr = (soup.find(class_=re.compile(r"race.?tab|race_tab", re.I)))
        if not hdr:
            return info
        text = hdr.get_text(" ", strip=True)
        m = re.search(r"(\d{3,5})\s*[Mm]", text)
        if m:
            info["distance_m"] = int(m.group(1))
        for cls_str in ["Group 1", "Group 2", "Group 3", "Listed",
                        "Class 1", "Class 2", "Class 3", "Class 4",
                        "Class 5", "Griffin"]:
            if cls_str.lower() in text.lower():
                info["race_class"] = cls_str
                break
        for going_str in ["GOOD TO FIRM", "GOOD TO YIELDING", "FIRM",
                          "GOOD", "YIELDING", "SOFT", "HEAVY"]:
            if going_str in text.upper():
                info["going"] = going_str
                break
        for course_str in ['TURF - "A"', 'TURF - "A+2"', 'TURF - "A+3"', 
                           'TURF - "B"', 'TURF - "B+2"', 'TURF - "B+3"',
                           'TURF - "C"', 'TURF - "C+2"', 'TURF - "C+3"', 
                           'ALL WEATHER TRACK']:
            if course_str in text.upper():
                info["course"] = course_str.replace("TURF - ", "").replace('"', "")
                break
        m = re.search(r"HK\$\s*([\d,]+)", text)
        if m:
            info["prize_hkd"] = float(m.group(1).replace(",", ""))
        return info
    


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_placing(raw: str):
    pla = raw.strip().upper()
    if raw.find(" DH") != -1:
        pla = raw.replace(" DH", "")
        print(pla)
    try:
        return int(pla)
    except ValueError:
        return None


def _safe_int(val: str):
    try:
        return int(str(val).strip().replace(",", ""))
    except (ValueError, TypeError):
        return None


def _safe_float(val: str):
    try:
        return float(str(val).strip().replace(",", ""))
    except (ValueError, TypeError):
        return None
    
def _safe_lbw(val: str):
    try:
        if val == "-":
            lbw = 0.0
        elif val == "SH" or val == "N":
            lbw = 0.0
        else:
            if val.find("-") == -1:
                if val.find("/") == -1:
                    lbw = float(val)
                else:
                    frac = str(val).strip().split("/")
                    lbw = float(frac[0])/float(frac[1])
            else:
                temp = str(val).strip().split("-")
                lbw = float(temp[0]) + float(temp[1].split("/")[0])/float(temp[1].split("/")[1])
        return lbw
    except (ValueError, TypeError):
        return None